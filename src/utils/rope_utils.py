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


def random_masking_new(batch, mask_ratio):
    # Compute the number of patches
    B, N, D = batch.shape  # B: batch size, N: number of patches, D: embedding dimension
    # print("batch\n", batch)

    num_masked = int(N * mask_ratio)  # Number of patches to mask

    # Generate random indices to mask
    rand_indices = torch.rand(B, N, device=batch.device).argsort(dim=1)
    mask_indices = rand_indices[:, :num_masked]

    # Gather the values of the patches to be masked before masking them
    # masked_patches = torch.gather(
    #     batch, 1, mask_indices.unsqueeze(-1).expand(-1, -1, D)
    # )

    # Zero out the patches at the chosen indices
    batch.scatter_(1, mask_indices.unsqueeze(-1).expand(-1, -1, D), 0)
    # print("=" * 100)
    # print("batch\n", batch)
    return batch, None, mask_indices
