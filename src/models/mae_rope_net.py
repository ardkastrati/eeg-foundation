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
    FlexibleRoPEAttention,
    Flexible_RoPE_Layer_scale_init_Block,
    compute_axial_cis,
)

from src.utils.rope_utils import PatchEmbed, random_masking_smart


class ParameterWrapper(nn.Module):
    def __init__(self, parameter):
        super(ParameterWrapper, self).__init__()
        self.param = nn.Parameter(parameter)

    def forward(self):
        return self.param


class MaskedAutoencoderViTRoPE(nn.Module):
    def __init__(
        self,
        # General
        channel_names_path,
        in_chans=1,
        patch_size=16,
        mask_ratio=0.15,
        # freqs_cis-specific
        max_sr=1000,
        max_dur=3600,
        max_win_size=8,
        min_win_size=1 / 2,
        win_shift_factor=1 / 4,
        # Encoder
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_block_layers=Flexible_RoPE_Layer_scale_init_Block,
        encoder_num_heads=6,
        encoder_mlp_ratio=4,
        encoder_qkv_bias=True,
        encoder_qk_scale=None,
        encoder_drop_rate=0.0,
        encoder_attn_drop_rate=0.0,
        encoder_drop_path_rate=0.0,
        encoder_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        encoder_act_layer=nn.GELU,
        encoder_attention_block=FlexibleRoPEAttention,
        encoder_mlp_block=Mlp,
        encoder_init_scale=1e-4,
        encoder_rope_theta=100.0,
        # Decoder
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_block_layers=Flexible_RoPE_Layer_scale_init_Block,
        decoder_num_heads=16,
        decoder_mlp_ratio=4,
        decoder_qkv_bias=True,
        decoder_qk_scale=None,
        decoder_drop_rate=0.0,
        decoder_attn_drop_rate=0.0,
        decoder_drop_path_rate=0.0,
        decoder_norm_layer=partial(nn.LayerNorm, eps=1e-6),
        decoder_act_layer=nn.GELU,
        decoder_attention_block=FlexibleRoPEAttention,
        decoder_mlp_block=Mlp,
        decoder_init_scale=1e-4,
        decoder_rope_theta=100.0,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # ====Init freqs_cis============================================================================================================

        self.max_sr = max_sr
        self.max_dur = max_dur

        self.max_win_size = max_win_size
        # self.max_win_shift = max_win_size * win_shift_factor
        self.min_win_size = min_win_size
        self.min_win_shift = min_win_size * win_shift_factor
        self.win_shift_factor = win_shift_factor

        # This is after Fourier transform (i.e. for spectrograms)
        self.max_y_datapoints = max_sr // 2 * max_win_size
        self.max_y_patches = int(self.max_y_datapoints // patch_size)

        self.max_x_datapoints_per_second = 1 / self.min_win_shift
        self.max_x_datapoints = max_dur * self.max_x_datapoints_per_second
        self.max_x_patches = int(self.max_x_datapoints // patch_size)

        # ====Init Encoder==============================================================================================================

        # Encoder: patch embedding
        self.patch_embed = PatchEmbed(patch_size, in_chans, encoder_embed_dim)

        # Encoder: cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))

        # Encoder: mean layer
        self.mean_embed = nn.Linear(
            in_features=self.max_y_datapoints,
            out_features=encoder_embed_dim,
        )

        # Encoder: encoding for each channel + separation token
        with open(channel_names_path, "r") as f:
            chn_names = json.load(f).values()
        print(
            "[MaskedAutoencoderViTRoPE.__init__] chn_names:", chn_names, file=sys.stderr
        )
        self.channel_encoding_map = nn.ModuleDict(
            {
                chn: ParameterWrapper(torch.zeros(1, 1, encoder_embed_dim))
                for chn in chn_names
            }
        )

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

        self.encoder_freqs_cis = compute_axial_cis(
            dim=self.encoder_embed_dim // self.encoder_num_heads,
            end_x=self.max_x_patches,
            end_y=self.max_y_patches,
            theta=self.encoder_rope_theta,
        )

        # Encoder: normalization layer
        self.encoder_norm = encoder_norm_layer(encoder_embed_dim)

        # ====Init Decoder==============================================================================================================

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

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

        self.decoder_freqs_cis = compute_axial_cis(
            dim=self.decoder_embed_dim // self.decoder_num_heads,
            end_x=self.max_x_patches,
            end_y=self.max_y_patches,
            theta=self.decoder_rope_theta,
        )

        self.decoder_norm = decoder_norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )

        # ====Init Weights==============================================================================================================

        self.initialize_weights()

    # == Forward pass ===================================================================================================================

    def forward_encoder(self, x, means, channels, win_size, mask_ratio):
        """ """
        B, C, H, W = x.shape
        print("[forward_encoder] before patch_embed:", x.shape, "(B, C, H, W)")

        # Encoder: patch embedding (flatten patches to a sequence)
        x = self.patch_embed(x)
        B, N, D = x.shape
        print("[forward_encoder] after patch_embed:", x.shape, "(B, N, D)")

        # Encoder: overlay all patches with channel encodings
        channel_encodings = torch.zeros(B, N, D, device=x.device)
        for b in range(B):
            for n in range(N):
                channel = int(channels[b, n].item())
                encoding = self.channel_encoding_map[channel]
                channel_encodings[b, n] = encoding
        x = x + channel_encodings

        # == Prepend patch sequences with metadata
        nr_meta_patches = 0

        # Encoder: prepend mean patches
        B, M, _ = means.shape
        means = means.reshape(B * M, -1)
        means = self.mean_embed(means)
        means = means.reshape(B, M, -1)
        x = torch.cat((means, x), dim=1)
        nr_meta_patches += M

        # Encoder: prepend cls token
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        nr_meta_patches += 1
        print("[forward_encoder] after cat cls_tokens:", x.shape, "(B, N, D)")

        # Encoder: randomly mask some patches (exluding metadata patches)
        x, masked_indices, mask = random_masking_smart(x, mask_ratio, nr_meta_patches)
        print("[forward_encoder] after random_masking_smart:", x.shape, "(B, N, D)")

        # Encoder: select correct rotation information for the attention layers
        freqs_cis = self.select_freqs_cis(
            self.encoder_freqs_cis, H, W, win_size, x.device
        )
        print(
            "[forward_encoder] freqs_cis.shape:",
            freqs_cis.shape,
            "(N, D // num_heads // 2)",
        )

        # Encoder: apply the encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x, freqs_cis=freqs_cis, nr_meta_tokens=nr_meta_patches)
        print("[forward_encoder] after rope blocks:", x.shape, "(B, N, D)")

        # Encoder: normalize the output
        x = self.encoder_norm(x)
        print("[forward_encoder] after rope norm:", x.shape, "(B, N, D)")

        return x, masked_indices, mask, nr_meta_patches

    def forward_decoder(self, x, nr_meta_patches, H, W, win_size):

        # Decoder: embed the encoder output
        x = self.decoder_embed(x)
        print("[forward_decoder] after decoder_embed:", x.shape, "(B, N, D')")

        # Decoder: recompute freqs_cis each batch (for simplicity)
        freqs_cis = self.select_freqs_cis(
            self.decoder_freqs_cis, H, W, win_size, x.device
        )
        print(
            "[forward_decoder] freqs_cis.shape:",
            freqs_cis.shape,
            "(N, dec_d_head // 2)",
        )

        # Decoder: apply the decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x, freqs_cis=freqs_cis, nr_meta_patches=nr_meta_patches)
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

        # Decoder: remove the metadata patches
        pred = pred[:, nr_meta_patches:, :]
        print("[forward_decoder] after cls_token removal:", pred.shape, "?")

        return pred

    def forward_loss(self, batch, pred, mask):

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

        loss = loss[mask].view(B, -1)
        print("[forward_loss] loss.shape after mask:", loss.shape)

        mean_loss = loss.mean()

        return mean_loss

    def forward(self, batch):

        channels = batch["channels"]
        row_means = batch["means"]
        win_size = batch["win_size"]
        batch = batch["batch"]

        B, C, H, W = batch.shape

        # == Encoder pass of model ==
        batch_emb, masked_indices, mask, nr_meta_patches = self.forward_encoder(
            x=batch,
            row_means=row_means,
            channels=channels,
            win_size=win_size,
            mask_ratio=self.mask_ratio,
        )

        # == Decoder pass of model ==
        flattened_pred = self.forward_decoder(
            x=batch_emb,
            nr_meta_patches=nr_meta_patches,
            H=H,
            W=W,
            win_size=win_size,
        )

        # == Loss calculation ==
        loss_recon = self.forward_loss(batch, flattened_pred, mask)

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
        for wrapper in self.cls_token_map.values():
            torch.nn.init.normal_(wrapper.param, std=0.02)
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

    def select_freqs_cis(self, freqs_cis, H, W, win_size, device):
        h = H // self.patch_size
        w = W // self.patch_size

        win_shift = win_size * self.win_shift_factor

        # 1. Compute selection parameters
        y_nr_patches = h
        y_jump = int(self.max_win_size / win_size)

        x_nr_patches = w
        x_jump = int(win_shift / self.min_win_shift)

        print(f"[select_freqs_cis] h: {h}, w: {w}")
        print(
            f"[select_freqs_cis] y_nr_patches: {y_nr_patches}, x_nr_patches: {x_nr_patches}"
        )
        print(f"[select_freqs_cis] y_jump: {y_jump}, x_jump: {x_jump}")

        # 2. Select the freqs_cis rows
        # freqs_cis.shape = (N, d_head/2), where d_head = embed_dim // num_heads and /2 due to complex numbers
        freqs_cis_selected = []
        for i in range(y_nr_patches):
            row_start = i * self.max_x_patches * y_jump
            for j in range(x_nr_patches):
                freqs_cis_selected.append(freqs_cis[row_start + j * x_jump])
        freqs_cis_selected = torch.stack(freqs_cis_selected)

        # Send the newly created tensor to the same device as freqs_cis
        freqs_cis_selected = freqs_cis_selected.to(device)
        print("[select_freqs_cis] freqs_cis.device:", freqs_cis.device)
        print(
            "[select_freqs_cis] freqs_cis_selected.device:", freqs_cis_selected.device
        )

        print("[select_freqs_cis] freqs_cis_selected.shape:", freqs_cis_selected.shape)
        # assert freqs_cis_selected.shape[0] == h * w

        return freqs_cis_selected
