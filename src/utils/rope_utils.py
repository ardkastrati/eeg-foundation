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


def random_masking_smart(x, mask_ratio, nr_meta_tokens):
    B, N, D = x.shape

    num_tokens_to_mask = int((N - nr_meta_tokens) * mask_ratio)

    rand_indices, _ = (
        torch.rand(B, N - nr_meta_tokens, device=x.device)
        .argsort(dim=1)[:, :num_tokens_to_mask]
        .sort(dim=1)
    )
    rand_indices += nr_meta_tokens

    # Boolean mask
    mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
    mask.scatter_(1, rand_indices, True)

    # Zero out the patches at the chosen indices
    x = x.masked_fill(mask.unsqueeze(-1), 0)

    return x, mask
