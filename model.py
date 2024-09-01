import torch
from torch import nn, Tensor
from audio_xlstm.lstm_modules import mLSTM


class AudioPatcher(nn.Module):
    def __init__(
        self,
        patch_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.shape

        num_patches = s // self.patch_size

        # Reshape into patches
        patches = x.unfold(
            dimension=2, size=self.patch_size, step=self.patch_size
        )

        # Reshape -> (b, num_patches, channels * patch_size)
        patches = patches.permute(
            0,
            2,
            1,
            3,
        ).reshape(b, num_patches, -1)

        return patches


class AudioEmbedder(nn.Module):
    def __init__(self, dim: int, embedding_dim: int):
        super().__init__()
        self.dim = dim
        self.embedding_dim = embedding_dim

        # Layer
        self.linear = nn.Linear(dim, embedding_dim)

    def forward(self, patches: Tensor) -> Tensor:
        return self.linear(patches)


def mask_and_add_cls_token(
    x: Tensor,
    num_masked_patches: int,
    cls_token: Tensor,
    mask_token: Tensor,
):
    """
    Mask 50% of the input patches and add a class token at the beginning.

    Args:
        x (torch.Tensor): Input patches of shape (batch_size, num_patches, dim).
        num_masked_patches (int): Number of patches to mask.
        cls_token (torch.Tensor): Class token of shape (1, 1, dim).
        mask_token (torch.Tensor): Mask token of shape (1, 1, dim).

    Returns:
        torch.Tensor: Output tensor with masked patches and added class token.
    """
    batch_size, num_patches, dim = x.shape

    # Add class token at the beginning
    cls_token = cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_token, x), dim=1)

    # Randomly select patches to mask
    mask_indices = (
        torch.randperm(num_patches)[:num_masked_patches] + 1
    )  # +1 to skip cls token
    x[:, mask_indices] = mask_token.expand(
        batch_size, num_masked_patches, dim
    )

    return x


class Spectra(nn.Module):
    def __init__(
        self,
        dim: int = None,
        depth: int = 1,
        heads: int = 1,
        dim_head: int = 64,
        patch_size: int = 16,
        p_factor: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.p_factor = p_factor
        self.patch_size_for_m = patch_size / 2

        # Patcher
        self.patcher = AudioPatcher(patch_size=patch_size)

        # Patch Embedder
        self.patch_embedder = AudioEmbedder(dim, dim)

        # Norm
        self.norm = nn.LayerNorm(dim)

        # silu
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        # b, s, d = x.shape

        # Patch
        patch = self.patcher(x)
        b, s, d = patch.shape
        print(patch.shape)

        # Patch Embed
        embedded_patches = AudioEmbedder(d, d)(patch)
        print(embedded_patches.shape)

        # Mask and add cls token
        embedded_patches = mask_and_add_cls_token(
            embedded_patches,
            num_masked_patches=4,
            cls_token=torch.zeros(1, 1, d),
            mask_token=torch.zeros(1, 1, d),
        )
        print(embedded_patches.shape)

        # Norm
        residual = embedded_patches
        embedded_patches = nn.LayerNorm(d)(embedded_patches)

        # 2 Pathways
        first_path = nn.Linear(d, d)(embedded_patches)
        second_path = nn.Linear(d, d)(embedded_patches)

        # Activation
        second_path_way_act = self.act(second_path)

        # First Pathway Mstm
        # FLIP before
        hid = (
            torch.zeros(first_path.shape),
            torch.zeros(first_path.shape),
            torch.zeros(first_path.shape),
        )
        first_path_tmed, _ = mLSTM(
            d, self.heads, self.dim_head, self.p_factor
        )(first_path, hid)
        print(first_path_tmed.shape)
        

        # Elementwise sum with both branches
        merged_paths = first_path_tmed * second_path_way_act

        # Projection
        merged_paths = nn.Linear(d, d)(merged_paths)

        # Element wise with the residual

        return merged_paths + residual


# Example
model = Spectra(dim=128, depth=1, heads=1, dim_head=64, patch_size=16)

input = torch.randn(1, 1024, 128)
output = model(input)
print(output.shape)
