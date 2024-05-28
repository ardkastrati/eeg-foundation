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

from src.models.mae_rope_encoder import EncoderViTRoPE
from src.models.mae_rope_decoder import DecoderViTRoPE

from src.utils.rope_utils import PatchEmbed, random_masking_smart


class ParameterWrapper(nn.Module):
    def __init__(self, parameter):
        super(ParameterWrapper, self).__init__()
        self.param = nn.Parameter(parameter)

    def forward(self):
        return self.param


class ModularMaskedAutoencoderViTRoPE(nn.Module):
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

        # ====Init freqs_cis============================================================================================================

        self.mask_ratio = mask_ratio
        self.max_sr = max_sr
        self.max_dur = max_dur

        self.max_win_size = max_win_size
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

        self.encoder = EncoderViTRoPE(
            channel_names_path=channel_names_path,
            in_chans=in_chans,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            max_sr=max_sr,
            max_dur=max_dur,
            max_win_size=max_win_size,
            min_win_size=min_win_size,
            win_shift_factor=win_shift_factor,
            encoder_embed_dim=encoder_embed_dim,
            encoder_depth=encoder_depth,
            encoder_block_layers=encoder_block_layers,
            encoder_num_heads=encoder_num_heads,
            encoder_mlp_ratio=encoder_mlp_ratio,
            encoder_qkv_bias=encoder_qkv_bias,
            encoder_qk_scale=encoder_qk_scale,
            encoder_drop_rate=encoder_drop_rate,
            encoder_attn_drop_rate=encoder_attn_drop_rate,
            encoder_drop_path_rate=encoder_drop_path_rate,
            encoder_norm_layer=encoder_norm_layer,
            encoder_act_layer=encoder_act_layer,
            encoder_attention_block=encoder_attention_block,
            encoder_mlp_block=encoder_mlp_block,
            encoder_init_scale=encoder_init_scale,
            encoder_rope_theta=encoder_rope_theta,
        )

        # ====Init Decoder==============================================================================================================

        self.decoder = DecoderViTRoPE(
            channel_names_path=channel_names_path,
            in_chans=in_chans,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            max_sr=max_sr,
            max_dur=max_dur,
            max_win_size=max_win_size,
            min_win_size=min_win_size,
            win_shift_factor=win_shift_factor,
            encoder_embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_block_layers=decoder_block_layers,
            decoder_num_heads=decoder_num_heads,
            decoder_mlp_ratio=decoder_mlp_ratio,
            decoder_qkv_bias=decoder_qkv_bias,
            decoder_qk_scale=decoder_qk_scale,
            decoder_drop_rate=decoder_drop_rate,
            decoder_attn_drop_rate=decoder_attn_drop_rate,
            decoder_drop_path_rate=decoder_drop_path_rate,
            decoder_norm_layer=decoder_norm_layer,
            decoder_act_layer=decoder_act_layer,
            decoder_attention_block=decoder_attention_block,
            decoder_mlp_block=decoder_mlp_block,
            decoder_init_scale=decoder_init_scale,
            decoder_rope_theta=decoder_rope_theta,
        )

        # ====Init Weights==============================================================================================================

        self.initialize_weights()

    # == Weight Init ========================================================================================================================

    def initialize_weights(self):
        """
        Ensures that the model starts training from a reasonable state.
        1. Patch embeddings
        2. Class and mask tokens
        """

        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder.cls_token, std=0.02)

        for wrapper in self.encoder.channel_encoding_map.values():
            torch.nn.init.normal_(wrapper.param, std=0.02)

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

    # == Forward pass ===================================================================================================================

    def forward_loss(self, target, meta_patches, nr_meta_patches, pred, mask):

        B, C, H, W = target.shape

        # Patchify the batch (i.e. transform it to a sequence of patches, similar to patch_embed)
        target = self.patchify(target, B, H, W)
        # print("[forward_loss] target.shape:", target.shape)
        # print("[forward_loss] pred.shape:", pred.shape)

        # Remove the cls token for loss computation
        pred_meta = pred[:, :nr_meta_patches, :]
        pred = pred[:, nr_meta_patches:, :]
        mask = mask[:, nr_meta_patches:]

        # Calculate the squared error
        loss = (pred - target) ** 2
        # print("[forward_loss] loss.shape:", loss.shape)

        # Compute the mean loss over the last dimension
        loss = loss.mean(dim=-1)
        # print("[forward_loss] loss.shape after mean(dim=-1):", loss.shape)

        loss = loss[mask].view(B, -1)
        # print("[forward_loss] loss.shape after mask:", loss.shape)

        mean_loss = loss.mean()

        return mean_loss

    def forward(self, batch):

        channels = batch["channels"]
        means = batch["means"]
        stds = batch["stds"]
        win_size = batch["win_size"]
        x = batch["batch"]

        B, C, H, W = x.shape

        # == Encoder pass of model ==
        x_emb, meta_patches, mask, nr_meta_patches = self.encoder(
            x=x,
            means=means,
            stds=stds,
            channels=channels,
            win_size=win_size,
            mask_ratio=self.mask_ratio,
        )

        # == Decoder pass of model ==
        x_pred = self.decoder(
            x=x_emb,
            nr_meta_patches=nr_meta_patches,
            H=H,
            W=W,
            win_size=win_size,
        )

        # == Loss calculation ==
        loss_recon = self.forward_loss(
            target=x,
            meta_patches=meta_patches,
            nr_meta_patches=nr_meta_patches,
            pred=x_pred,
            mask=mask,
        )

        x_pred = x_pred[:, nr_meta_patches:, :]
        mask = mask[:, nr_meta_patches:]

        return loss_recon, x_pred, mask

    # == Helpers ========================================================================================================================

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
