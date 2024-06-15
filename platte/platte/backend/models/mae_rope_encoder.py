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
    select_freqs_cis,
)

from src.utils.rope_utils import PatchEmbed, random_masking_smart


class ParameterWrapper(nn.Module):
    def __init__(self, parameter):
        super(ParameterWrapper, self).__init__()
        self.param = nn.Parameter(parameter)

    def forward(self):
        return self.param


class EncoderViTRoPE(nn.Module):
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
    ):
        super().__init__()

        self.in_chans = in_chans
        self.patch_size = patch_size

        # ====Init freqs_cis============================================================================================================

        self.mask_ratio = mask_ratio
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

        # Encoder: mean and std layer
        self.mean_embed = nn.Linear(
            in_features=self.max_y_datapoints,
            out_features=encoder_embed_dim // 2,
        )

        self.std_embed = nn.Linear(
            in_features=self.max_y_datapoints,
            out_features=encoder_embed_dim // 2,
        )

        # Encoder: encoding for each channel + separation token
        with open(channel_names_path, "r") as f:
            self.chn_names = json.load(f)
            self.reversed_chn_names = {
                value: key for key, value in self.chn_names.items()
            }

        self.channel_encoding_map = nn.ModuleDict(
            {
                chn: ParameterWrapper(torch.zeros(1, 1, encoder_embed_dim))
                for chn in self.chn_names.keys()
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

    # == Forward pass ===================================================================================================================

    def forward(self, x, means, stds, channels, win_size, mask_ratio):
        """ """
        B, C, H, W = x.shape
        # print("[forward_encoder] NaN in x:", torch.isnan(x).any())
        # print("[forward_encoder] before patch_embed:", x.shape, "(B, C, H, W)")

        # Encoder: patch embedding (flatten patches to a sequence)
        x = self.patch_embed(x)
        B, N, D = x.shape
        # print("[forward_encoder] after patch_embed:", x.shape, "(B, N, D)")

        # Encoder: overlay all patches with channel encodings
        channel_encodings = torch.zeros(B, N, D, device=x.device)
        for b in range(B):
            for n in range(N):
                channel = int(channels[b, n].item())
                channel_name = self.reversed_chn_names[channel]
                encoding = self.channel_encoding_map[channel_name].param
                channel_encodings[b, n] = encoding
        x = x + channel_encodings

        # == Prepend patch sequences with metadata
        nr_meta_patches = 0

        # Encoder: prepend mean patches
        B, M, _ = means.shape
        B, S, _ = stds.shape
        means = means.reshape(B * M, -1)
        means = self.mean_embed(means)
        means = means.reshape(B, M, -1)
        stds = stds.reshape(B * S, -1)
        stds = self.mean_embed(stds)
        stds = stds.reshape(B, S, -1)
        normalization_patches = torch.cat((means, stds), dim=2)
        x = torch.cat((normalization_patches, x), dim=1)
        nr_meta_patches += M

        # Encoder: prepend cls token
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        nr_meta_patches += 1
        # print("[forward_encoder] after cat cls_tokens:", x.shape, "(B, N, D)")

        # Keep for reconstruction loss
        meta_patches = x[:, :nr_meta_patches, :]

        # Encoder: randomly mask some patches (exluding metadata patches)
        x, mask = random_masking_smart(x, mask_ratio, nr_meta_patches)
        # print("[forward_encoder] after random_masking_smart:", x.shape, "(B, N, D)")

        # Encoder: select correct rotation information for the attention layers
        freqs_cis = select_freqs_cis(
            self, self.encoder_freqs_cis, H, W, win_size, x.device
        )
        # print(
        #     "[forward_encoder] freqs_cis.shape:",
        #     freqs_cis.shape,
        #     "(N, D // num_heads // 2)",
        # )

        # Encoder: apply the encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x, freqs_cis=freqs_cis, nr_meta_tokens=nr_meta_patches)
        # print("[forward_encoder] after rope blocks:", x.shape, "(B, N, D)")

        # Encoder: normalize the output
        x = self.encoder_norm(x)
        # print("[forward_encoder] after rope norm:", x.shape, "(B, N, D)")

        return x, meta_patches, mask, nr_meta_patches
