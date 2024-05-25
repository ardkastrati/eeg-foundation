from itertools import repeat

import torch
import torch.nn as nn

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


# ====Patch Embedding==============================================================================================================


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=16, in_chans=1, embed_dim=384):
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ====Random Masking==============================================================================================================


def random_masking_smart(batch, mask_ratio, nr_meta_tokens):
    B, N, D = batch.shape

    num_tokens_to_mask = int((N - nr_meta_tokens) * mask_ratio)

    rand_indices, _ = (
        torch.rand(B, N - nr_meta_tokens, device=batch.device)
        .argsort(dim=1)[:, :num_tokens_to_mask]
        .sort(dim=1)
    )
    rand_indices += nr_meta_tokens

    # Boolean mask
    mask = torch.zeros(B, N, dtype=torch.bool, device=batch.device)
    mask.scatter_(1, rand_indices, True)

    # Zero out the patches at the chosen indices
    batch.masked_fill_(mask.unsqueeze(-1), 0)

    # Extract the masked elements
    # masked_batch = batch[mask].view(B, num_tokens_to_mask, D)

    return batch, rand_indices, mask


# def random_masking_smart(batch, mask_ratio, nr_meta_tokens):

#     B, N, D = batch.shape

#     # Number of patches to mask
#     num_masked = int((N - nr_meta_tokens) * mask_ratio)

#     # Generate random indices to mask
#     rand_indices = torch.rand(B, N - nr_meta_tokens, device=batch.device).argsort(dim=1)
#     rand_indices += nr_meta_tokens

#     mask_indices = rand_indices[:, :num_masked]

#     # Zero out the patches at the chosen indices
#     batch.scatter_(1, mask_indices.unsqueeze(-1).expand(-1, -1, D), 0)

#     return batch, mask_indices
