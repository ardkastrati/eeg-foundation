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


class DecoderViTRoPE(nn.Module):
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

    # == Forward pass ===================================================================================================================

    def forward(self, x, nr_meta_patches, H, W, win_size):

        # Decoder: embed the encoder output
        x = self.decoder_embed(x)
        # print("[forward_decoder] NaN in x:", torch.isnan(x).any())
        # print("[forward_decoder] after decoder_embed:", x.shape, "(B, N, D')")

        # Decoder: recompute freqs_cis each batch (for simplicity)
        freqs_cis = select_freqs_cis(
            self, self.decoder_freqs_cis, H, W, win_size, x.device
        )
        # print(
        #     "[forward_decoder] freqs_cis.shape:",
        #     freqs_cis.shape,
        #     "(N, dec_d_head // 2)",
        # )

        # Decoder: apply the decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x, freqs_cis=freqs_cis, nr_meta_tokens=nr_meta_patches)
        # print("[forward_decoder] after rope decoder blocks:", x.shape, "(B, N, D')")

        # Decoder: normalize the output
        x = self.decoder_norm(x)
        # print("[forward_decoder] after decoder_norm:", x.shape, "(B, N, D')")

        # Decoder: predict the reconstruction
        pred = self.decoder_pred(x)
        # print(
        #     "[forward_decoder] after decoder_pred:",
        #     pred.shape,
        #     "(B, N, patch_size**2 * in_chans)",
        # )

        return pred
