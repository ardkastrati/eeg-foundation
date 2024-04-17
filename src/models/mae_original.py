# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from json import encoder

import json
import torch
import torch.nn as nn

# from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import Block
from src.utils.pos_embed import (
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_flexible,
    get_1d_sincos_pos_embed_from_grid,
)
from src.utils.misc import concat_all_gather
from src.utils.patch_embed import PatchEmbed_new, PatchEmbed_org
from timm.models.swin_transformer import SwinTransformerBlock


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone, specifically for EEG data

    Workflow of the MAE:
    - forward:
        - encapsulates the entire forward pass of the model
        - including encoding the input in `forward_encoder()`
        - decoding the representations in `forward_decoder()`
        - calculating the loss in `forward_loss()`
    - forward_encoder:
        - processes the input data through the encoder part of the model
        - including random masking of image patches
        - forward_encoder_no_mask:
            - does not employ the masking technique
            - used for tasks where masking is not desired
    - forward_decoder:
        - processes encoded representations through decoder
    - forward_loss:
        - reconstruction loss by comparing the predicted patch embeddings against the original patches (obtained via patchify)
    - forward_entire_edf: extension of the forward method designed to process entire EDF files
    """

    def __init__(
        self,
        img_size=(128, 1024),
        patch_size=16,
        stride=10,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        audio_exp=True,
        alpha=0.0,
        temperature=0.2,
        mode=0,
        contextual_depth=8,
        use_custom_patch=False,
        split_pos=False,
        pos_trainable=False,
        use_nce=False,
        beta=4.0,
        decoder_mode=0,
        mask_t_prob=0.6,
        mask_f_prob=0.5,
        mask_2d=False,
        epoch=0,
        no_shift=False,
        use_channel_emb=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.audio_exp = audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # --------------------------------------------------------------------------
        # EEG specifics

        # generate an embedding for each channel name.
        # - data from each channel corresponds to a specific location on the head
        # - learning distinct embeddings for each channel makes it possible to
        #   exploit the unique characteristics of the signals from different parts of the scalp

        with open(
            "/home/maxihuber/eeg-foundation/src/data/edf_index/channel_json", "r"
        ) as channel_file:
            self.all_channels = json.load(channel_file)

        self.channel_embed = {}  # store embeddings for each channel in a dictionary
        for channel in self.all_channels:  # zero-initialize each channel embedding
            # they are marked as nn.Parameter and will be learned during training
            self.channel_embed[channel] = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        # Choice for two patch embedding methods: PatchEmbed_new and PatchEmbed_org
        if use_custom_patch:
            print(
                f"Use custom patch_emb with patch size: {patch_size}, stride: {stride}"
            )
            self.patch_embed = PatchEmbed_new(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                stride=stride,
            )
        else:
            self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)

        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches

        # Class token:
        # - learnable parameter
        # - added to the sequence of patch embeddings
        #  . here, in non-classification context: capture global information about input
        #  . in transformers (for classification): state is used for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self.split_pos = split_pos # not useful

        # Positional embedding:
        # - learnable parameter matrix
        # - adds positional information to the patch embeddings
        # - size: nr of patches + class token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable
        )  # fixed sin-cos embedding (? pasc comment)

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth

        # Initialize sequence of transformer blocks
        # These layers collectively form the transformer encoder
        # The transformer encoder processes the sequence of patch embeddings
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)  # Final normalization layer

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        # Linear projection of encoder embedding dimension to decoder embedding dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Mask token:
        # - learnable parameter
        # - represents embedding for masked patches, i.e. serves as a
        #    placeholder for masked patches in the decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Positional embedding:
        # - learnable parameter matrix
        # - adds positional information to the patch embeddings (order & location)
        # - size: nf or patches + class token
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=pos_trainable,
        )  # fixed sin-cos embedding (? pasc comment)

        self.no_shift = no_shift
        self.decoder_mode = decoder_mode  # different architectures within the decoder

        if (
            self.use_custom_patch
        ):  # overlapped patches as in AST. Similar performance yet compute heavy
            window_size = (6, 6)
            feat_size = (102, 12)
        else:
            window_size = (4, 4)
            feat_size = (self.img_size[0] // patch_size, self.img_size[1] // patch_size)
        if self.decoder_mode == 1:
            decoder_modules = []
            # Initialize a series of Swin Transformers
            for index in range(16):  # why fixed 16 and not decoder_depth?
                if self.no_shift:
                    shift_size = (0, 0)
                else:
                    if (index % 2) == 0:
                        shift_size = (0, 0)
                    else:
                        shift_size = (2, 0)
                    # shift_size = tuple([0 if ((index % 2) == 0) else w // 2 for w in window_size])
                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=decoder_embed_dim,
                        num_heads=16,
                        feat_size=feat_size,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        drop=0.0,
                        drop_attn=0.0,
                        drop_path=0.0,
                        extra_norm=False,
                        sequential_attn=False,
                        norm_layer=norm_layer,  # nn.LayerNorm,y
                    )
                )
            self.decoder_blocks = nn.ModuleList(decoder_modules)
        else:
            # Initialize a series of Transformer blocks
            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for i in range(decoder_depth)
                ]
            )

        # Final normalization layer
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Maps the decoder's output back to the original data space, i.e.
        # reconstructs the original patches
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.stride = stride

        # Audio experiment specifics
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        # Layer that applies the logarithm of the softmax function (numerically more stable)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Masking probabilities (time and frequency)
        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob
        self.mask_2d = mask_2d

        self.epoch = epoch

        self.initialize_weights()

    def initialize_weights(self):
        """
        Ensures that the model starts training from a reasonable state.
        1. Positional embeddings
        2. Patch embeddings
        3. Class and mask tokens
        4.
        """

        # --------------------------------------------------------------------------
        # Positional Embeddings Initialization

        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(
                self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True
            )
        else:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.patch_embed.num_patches**0.5),
                cls_token=True,
            )

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.audio_exp:
            decoder_pos_embed = get_2d_sincos_pos_embed_flexible(
                self.decoder_pos_embed.shape[-1],
                self.patch_embed.patch_hw,
                cls_token=True,
            )
        else:
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1],
                int(self.patch_embed.num_patches**0.5),
                cls_token=True,
            )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # --------------------------------------------------------------------------
        # Patch Embedding Weight Initialization

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # --------------------------------------------------------------------------
        # Class Token and Mask Token Initialization

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        L = (H/p)*(W/p)
        """
        p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        if self.audio_exp:
            if self.use_custom_patch:  # overlapped patch
                h, w = self.patch_embed.patch_hw
                x = imgs.unfold(2, self.patch_size, self.stride).unfold(
                    3, self.patch_size, self.stride
                )  # n,1,H,W -> n,1,h,w,p,p
                x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
                # x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                # x = torch.einsum('nchpwq->nhwpqc', x)
                # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            else:
                h = imgs.shape[2] // p
                w = imgs.shape[3] // p
                # h,w = self.patch_embed.patch_hw
                x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                x = torch.einsum("nchpwq->nhwpqc", x)
                x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else:
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum("nchpwq->nhwpqc", x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        specs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        # Pascal: adjusted for our spectrogram size
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        specs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return specs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch:  # overlapped patch
            T = 101
            F = 12
        else:
            T = 64
            F = 8
        # x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(
            noise_t, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1)
        ids_keep_t = ids_shuffle_t[:, :len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(
            noise_f, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1)
        ids_keep_f = ids_shuffle_f[:, :len_keep_f]  #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:, :len_keep_f] = 0
        mask_f = (
            torch.gather(mask_f, dim=1, index=ids_restore_f)
            .unsqueeze(1)
            .repeat(1, T, 1)
        )  # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:, :len_keep_t] = 0
        mask_t = (
            torch.gather(mask_t, dim=1, index=ids_restore_t)
            .unsqueeze(1)
            .repeat(1, F, 1)
            .permute(0, 2, 1)
        )  # N,T,F
        mask = 1 - (1 - mask_t) * (1 - mask_f)  # N, T, F

        # get masked x
        # TODO: study why the .to(device) call is necessary here and it if could be done differently
        id2res = torch.Tensor(list(range(N * T * F))).reshape(N, T, F).to(x.device)
        id2res = id2res + 999 * mask  # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep = id2res2.flatten(start_dim=1)[:, : len_keep_f * len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, mask_2d=False):
        """ """

        # embed patches (pass through patch_embed layer)
        # i.e. transform input images into a sequence of flattened patches
        x = self.patch_embed(x)

        channels = []

        # add pos embed (exclude cls token)
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_2d:  # apply both time and frequency masking
            x, mask, ids_restore = self.random_masking_2d(
                x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob
            )
        else:  # apply unstructured random masking
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token (isn't it prepend here) ?
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # append channel token, why not done?
        """
        channel_tokens = []
        for ch in channels:
            channel_tokens.append(self.channel_embed[ch] + self.pos_embed[:, :1, :])
        x = torch.cat(channel_tokens, x)
        """

        # the sequence of patches is processed through the transformer blocks
        # applying self-attention
        for blk in self.blocks:
            x = blk(x)

        # output sequence is passed through normalization layer
        x = self.norm(x)

        # emb = self.encoder_emb(x)

        return x, mask, ids_restore, None

    def forward_encoder_no_mask(self, x):
        """ """

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        contextual_embs = []
        for n, blk in enumerate(self.blocks):
            x = blk(x)
            if n > self.contextual_depth:
                contextual_embs.append(self.norm(x))
        # x = self.norm(x)
        contextual_emb = torch.stack(contextual_embs, dim=0).mean(dim=0)

        return contextual_emb

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        if self.decoder_mode != 0:
            B, L, D = x.shape
            x = x[:, 1:, :]
            if self.use_custom_patch:
                x = x.reshape(B, 101, 12, D)
                x = torch.cat([x, x[:, -1, :].unsqueeze(1)], dim=1)  # hack
                x = x.reshape(B, 1224, D)
        if self.decoder_mode > 3:  # mvit
            x = self.decoder_blocks(x)
        else:
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        if self.decoder_mode != 0:
            if self.use_custom_patch:
                pred = pred.reshape(B, 102, 12, 256)
                pred = pred[:, :101, :, :]
                pred = pred.reshape(B, 1212, 256)
            else:
                pred = pred
        else:
            pred = pred[:, 1:, :]
        return pred, None, None  # emb, emb_pixel

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = loss.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.5):

        # Encoder pass of model
        # emb_enc: encoded representations of the input images after applying the transformer encoder blocks
        # mask: binary mask (which patches were masked)
        # ids_restore: indices for restoring original order of patches after shuffling for masking
        emb_enc, mask, ids_restore, _ = self.forward_encoder(
            imgs, mask_ratio, mask_2d=self.mask_2d
        )

        # Decoder pass of model
        # pred: predicted reconstruction (tensor of same shape as emb_enc)
        pred, _, _ = self.forward_decoder(emb_enc, ids_restore)  # [N, L, p*p*3]

        # Computing reconstruction error
        loss_recon = self.forward_loss(
            imgs, pred, mask, norm_pix_loss=self.norm_pix_loss
        )
        # According to Pascal, this is an artefact from the VisionTransformer Architecture, but we're not using it rn!
        # Each batch spent about 1/4th of its time moving this little tensor to the GPU due to the .cuda() call
        # and in MAEModule.forward we never use it!!
        # loss_contrastive = torch.FloatTensor([0.0]).cuda()
        loss_contrastive = 0.0

        return loss_recon, pred, mask, loss_contrastive

    def freeze_encoder(self):

        # freeze weights of encoder blocks
        self.blocks.requires_grad = False

    def unfreeze_encoder(self):

        # unfreeze weights of encoder blocks
        self.blocks.requires_grad = True

    def forward_finetune(self, x):

        # input is batchsize * [#channels]
        predictions = torch.zeros(len(x))

        for i, sample in enumerate(x):

            num_channels = len(sample)
            emb_sum = torch.zeros(self.embed_dim)
            emb_sum = emb_sum.cuda()

            for spg in sample:

                spg = spg.unsqueeze(0)
                emb = self.forward_encoder_no_mask(spg)

                emb = emb[0][0]

                emb_sum = emb_sum + emb

            emb_sum = emb_sum / num_channels

            pred = self.mlp1(emb_sum)

            pred = torch.relu(pred)
            pred = self.mlp2(pred)
            pred = self.sigmoid(pred)
            predictions[i] = pred

        return predictions

    def forward_entire_edf(self, edf_data, mask_ratio=0.8):

        # takes as input an entire edf file, separated into channels. each channel is separated into 1024,128 spectrograms.
        # input is [channels, spectrograms (a mapping from channel name to a list of spectrograms)]

        channels, spectrograms = edf_data

        # a mappping from channel name to the embeddings produced by the encoder
        embed_per_channel = {}

        # feed the spectrograms from each channel as a batch to the encoder.
        for channel in channels:

            spgs = spectrograms[channel]

            # pass spectrograms through forward_encoder, treated like a batch

            emb_enc, mask, ids_restore, _ = self.forward_encoder(
                spgs, mask_ratio, mask_2d=self.mask_2d
            )
            embed_per_channel[channel] = emb_enc, mask, ids_restore

        # for each channel, compute the reconstruction loss
        total_loss = torch.tensor(0.0)

        for channel in channels:

            # run the generated embeddings through the decoder
            spgs = spectrograms[channel]
            emb_enc, mask, ids_restore = embed_per_channel[channel]
            pred, _, _ = self.forward_decoder(emb_enc, ids_restore)
            # compute the reconstruction loss and add it up
            loss_recon = self.forward_loss(
                spgs, pred, mask, norm_pix_loss=self.norm_pix_loss
            )
            total_loss = total_loss + loss_recon

        return total_loss


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended architectures
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

if __name__ == "__main__":
    MaskedAutoencoderViT()
