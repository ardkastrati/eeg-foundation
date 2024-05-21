# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# RoPE: https://github.com/naver-ai/rope-vit/, https://arxiv.org/abs/2403.13298
# --------------------------------------------------------


import glob
import os
import sys
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functools import partial

import torch
import torch.nn as nn
import lightning as L

from timm.models.vision_transformer import Mlp as Mlp

from src.models.components.vit_rope import (
    RoPEAttention,
    RoPE_Layer_scale_init_Block,
    compute_axial_cis,
)

from src.utils.rope_utils import PatchEmbed, random_masking_new


class MaskedAutoencoderViTRoPE(nn.Module):
    def __init__(
        self,
        # General
        channel_names_stor_dir,
        in_chans=1,
        patch_size=16,
        mask_ratio=0.15,
        # Encoder
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_block_layers=RoPE_Layer_scale_init_Block,
        encoder_num_heads=6,
        encoder_mlp_ratio=4,
        encoder_qkv_bias=True,
        encoder_qk_scale=None,
        encoder_drop_rate=0.0,
        encoder_attn_drop_rate=0.0,
        encoder_drop_path_rate=0.0,
        encoder_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        encoder_act_layer=nn.GELU,
        encoder_attention_block=RoPEAttention,
        encoder_mlp_block=Mlp,
        encoder_init_scale=1e-4,
        encoder_rope_theta=100.0,
        # Decoder
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_block_layers=RoPE_Layer_scale_init_Block,
        decoder_num_heads=16,
        decoder_mlp_ratio=4,
        decoder_qkv_bias=True,
        decoder_qk_scale=None,
        decoder_drop_rate=0.0,
        decoder_attn_drop_rate=0.0,
        decoder_drop_path_rate=0.0,
        decoder_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        decoder_act_layer=nn.GELU,
        decoder_attention_block=RoPEAttention,
        decoder_mlp_block=Mlp,
        decoder_init_scale=1e-4,
        decoder_rope_theta=100.0,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # ====Init Encoder==============================================================================================================

        # Encoder: patch embedding
        self.patch_embed = PatchEmbed(patch_size, in_chans, encoder_embed_dim)

        # Encoder: cls token (map)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))

        chn_names_file_paths = glob.glob(
            os.path.join(channel_names_stor_dir, "channel_set*.json")
        )
        chn_names = []
        for chn_names_file_path in chn_names_file_paths:
            with open(chn_names_file_path, "r") as f:
                chn_names_new = json.load(f)
                chn_names.extend(chn_names_new)
        chn_names.append(None)
        print(
            "[MaskedAutoencoderViTRoPE.__init__] chn_names:", chn_names, file=sys.stderr
        )
        self.cls_token_map = {
            chn: nn.Parameter(torch.zeros(1, 1, encoder_embed_dim)) for chn in chn_names
        }

        # Encoder: transformer blocks (with RoPE)
        self.encoder_blocks = nn.ModuleList(
            [
                encoder_block_layers(
                    dim=encoder_embed_dim,
                    num_heads=encoder_num_heads,
                    mlp_ratio=encoder_mlp_ratio,
                    qkv_bias=encoder_qkv_bias,
                    qk_scale=encoder_qk_scale,
                    drop=encoder_drop_rate,
                    attn_drop=encoder_attn_drop_rate,
                    drop_path=encoder_drop_path_rate,
                    norm_layer=encoder_norm_layer,
                    act_layer=encoder_act_layer,
                    Attention_block=encoder_attention_block,
                    Mlp_block=encoder_mlp_block,
                    init_values=encoder_init_scale,
                )
                for _ in range(encoder_depth)
            ]
        )

        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_heads = encoder_num_heads
        self.encoder_rope_theta = encoder_rope_theta

        # Encoder: normalization layer
        self.encoder_norm = encoder_norm_layer(encoder_embed_dim)

        # ====Init Decoder==============================================================================================================

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # TODO: what about the mask token, now that we don't need to fill the masked patches anymore?
        #  (we pass the masked patches just as 0s to the decoder too (together with unmasked patches),
        #   which we need to do because of the RoPE, which expects the full spectrogram.
        #   we did not need to do that with absolute positional embeddings)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Initialize a series of Transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                decoder_block_layers(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=decoder_mlp_ratio,
                    qkv_bias=decoder_qkv_bias,
                    qk_scale=decoder_qk_scale,
                    drop=decoder_drop_rate,
                    attn_drop=decoder_attn_drop_rate,
                    drop_path=decoder_drop_path_rate,
                    norm_layer=decoder_norm_layer,
                    act_layer=decoder_act_layer,
                    Attention_block=decoder_attention_block,
                    Mlp_block=decoder_mlp_block,
                    init_values=decoder_init_scale,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_rope_theta = decoder_rope_theta

        self.decoder_norm = decoder_norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )

        # ====Init Weights==============================================================================================================

        self.initialize_weights()

    # == Forward pass ===================================================================================================================

    def forward_encoder(self, x, chn_names, mask_ratio):
        """ """
        B, C, H, W = x.shape

        # Encoder: patch embedding (flatten patches to a sequence)
        x = self.patch_embed(x)
        print("[forward_encoder] after patch_embed:", x.shape, "(B, N, D)")

        # Encoder: mask some patches
        x, mask, ids_restore = random_masking_new(x, mask_ratio)
        print("[forward_encoder] after random_masking_new:", x.shape, "(B, N, D)")

        # Encoder: add class token
        # cls_tokens = torch.stack([self.cls_token_map[chn] for chn in chn_names])
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        print("[forward_encoder] after cat cls_tokens:", x.shape, "(B, N, D)")

        # TODO: (?) channel token

        # Encoder: recompute freqs_cis each batch (for simplicity)
        # OPTIMIZE: .to() call is slow, same for forward_decoder
        freqs_cis = compute_axial_cis(
            dim=self.encoder_embed_dim // self.encoder_num_heads,
            end_x=W // self.patch_size,
            end_y=H // self.patch_size,
            theta=self.encoder_rope_theta,
        )
        freqs_cis = freqs_cis.to(x.device)
        print("[forward_encoder] freqs_cis.shape:", freqs_cis.shape, "(N, ?)")

        # Encoder: apply the encoder blocks
        for _, blk in enumerate(self.encoder_blocks):
            x = blk(x, freqs_cis=freqs_cis)
        print("[forward_encoder] after rope blocks:", x.shape, "(B, N, D)")

        # Encoder: normalize the output
        x = self.encoder_norm(x)
        print("[forward_encoder] after rope norm:", x.shape, "(B, N, D)")

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, H, W):

        # Decoder: embed the encoder output
        x = self.decoder_embed(x)
        print("[forward_decoder] after decoder_embed:", x.shape, "(B, N, D')")

        # Decoder: recompute freqs_cis each batch (for simplicity)
        freqs_cis = compute_axial_cis(
            dim=self.decoder_embed_dim // self.decoder_num_heads,
            end_x=W // self.patch_size,
            end_y=H // self.patch_size,
            theta=self.decoder_rope_theta,
        )
        freqs_cis = freqs_cis.to(x.device)

        # Decoder: apply the decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x, freqs_cis=freqs_cis)
        print("[forward_decoder] after rope decoder blocks:", x.shape, "(B, N, D')")

        # Decoder: normalize the output
        x = self.decoder_norm(x)
        print("[forward_decoder] after decoder_norm:", x.shape, "(B, N, D')")

        # Decoder: predict the reconstruction
        pred = self.decoder_pred(x)
        print(
            "[forward_decoder] after decoder_pred:",
            pred.shape,
            "(B, N, patch_size**2 * in_chans)",
        )

        # Decoder: remove the cls token
        pred = pred[:, 1:, :]
        print("[forward_decoder] after cls_token removal:", pred.shape, "?")

        return pred

    def forward_loss(self, batch, pred):

        B, C, H, W = batch.shape

        # Patchify the batch (i.e. transform it to a sequence of patches, similar to patch_embed)
        target = self.patchify(batch, B, H, W)
        print("[forward_loss] target.shape:", target.shape)
        print("[forward_loss] pred.shape:", pred.shape)

        # Calculate the squared error
        loss = (pred - target) ** 2
        print("[forward_loss] loss.shape:", loss.shape)

        # Compute the mean loss over the last dimension
        loss = loss.mean(dim=-1)
        print("[forward_loss] loss.shape after mean(dim=-1):", loss.shape)

        # Compute the mean loss over all patches (second dimension)
        # NOTE: in the original code we only computed the mean for the masked patches
        #  and ignored the others. This is not the case here.
        mean_loss = loss.mean()
        print("[forward_loss] mean_loss:", mean_loss.item())

        return mean_loss

    def forward(self, batch):

        chn_names = batch["chn_list"]
        batch = batch["batch"]

        B, C, H, W = batch.shape

        # == Encoder pass of model ==
        batch_emb, _, masked_indices = self.forward_encoder(
            batch, chn_names, self.mask_ratio
        )

        # == Decoder pass of model ==
        flattened_pred = self.forward_decoder(batch_emb, masked_indices, H, W)

        # == Loss calculation ==
        loss_recon = self.forward_loss(batch, flattened_pred)

        return loss_recon, flattened_pred, masked_indices

    # == Helpers ========================================================================================================================

    def initialize_weights(self):
        """
        Ensures that the model starts training from a reasonable state.
        1. Patch embeddings
        2. Class and mask tokens
        """

        # --------------------------------------------------------------------------
        # Patch Embedding Weight Initialization

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # --------------------------------------------------------------------------
        # Class Token and Mask Token Initialization

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=0.02)

        # --------------------------------------------------------------------------
        # Custom nn.Linear and nn.LayerNorm Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x, B, H, W):
        p = self.patch_size
        x = x.reshape(shape=(B, 1, H // p, p, W // p, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(B, H * W // (p**2), p**2))
        return x

    def unpatchify(self, x, B, H, W):
        p = self.patch_size
        x = x.reshape(shape=(B, H // p, W // p, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, 1, H, W))
        return x
