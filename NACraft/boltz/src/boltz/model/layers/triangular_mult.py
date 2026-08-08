import torch
from torch import Tensor, nn
import os
from boltz.model.layers import initialize as init
import cuequivariance_torch as cueq


USE_CUEQ = os.getenv("BOLTZ_USE_CUEQ", "0").lower() in ("1", "true", "yes", "on")
print(f"USING TRIMULT CUEQ KERNELS SET TO {str(USE_CUEQ).upper()}")


class TriangleMultiplicationOutgoing(nn.Module):
    """TriangleMultiplicationOutgoing."""

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        init.bias_init_one_(self.norm_in.weight)
        init.bias_init_zero_(self.norm_in.bias)

        init.lecun_normal_init_(self.p_in.weight)
        init.gating_init_(self.g_in.weight)

        init.bias_init_one_(self.norm_out.weight)
        init.bias_init_zero_(self.norm_out.bias)

        init.final_init_(self.p_out.weight)
        init.gating_init_(self.g_out.weight)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if USE_CUEQ:
            x = _cueq_triangle_mult(
                z=x,
                g_in_weight=self.g_in.weight,
                p_in_weight=self.p_in.weight,
                _outgoing=True,
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
            )
            return x
        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        x = torch.einsum("bikd,bjkd->bijd", a, b)

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x


class TriangleMultiplicationIncoming(nn.Module):
    """TriangleMultiplicationIncoming."""

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        init.bias_init_one_(self.norm_in.weight)
        init.bias_init_zero_(self.norm_in.bias)

        init.lecun_normal_init_(self.p_in.weight)
        init.gating_init_(self.g_in.weight)

        init.bias_init_one_(self.norm_out.weight)
        init.bias_init_zero_(self.norm_out.bias)

        init.final_init_(self.p_out.weight)
        init.gating_init_(self.g_out.weight)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        
        if USE_CUEQ:
            x = _cueq_triangle_mult(
                z=x,
                g_in_weight=self.g_in.weight,
                p_in_weight=self.p_in.weight,
                _outgoing=False,
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
            )
            return x
        
        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        x = torch.einsum("bkid,bkjd->bijd", a, b)

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x


def _cueq_triangle_mult(
    z: torch.Tensor,
    g_in_weight: torch.Tensor,
    p_in_weight: torch.Tensor,
    _outgoing: bool,
    mask: torch.Tensor | None,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
) -> torch.Tensor:
    ##VS: similar issue here as to the cueq triangle attention
    ## kernel, we need to reshape the input so that batch and
    ## n_tmpl are combined into a single dimension.

    ## only hidden dimension multiple of 32 is supported for now
    if z.shape[-1] % 32 != 0:
        raise ValueError(
            "CUEQ triangle multiplicative update only supports "
            "channel dimension multiple of 32, got: "
            f"{z.shape[-1]}"
        )

    is_batched_input = False
    if len(z.shape) > 4:
        assert len(z.shape) == 5, (
            "CUEQ triangle multiplicative update only supports "
            f"max 5 input dimensions, got: {len(z.shape)}"
        )
        is_batched_input = True
        batch, n_tmpl, n_res, _, c_in = z.shape
        z = z.view(batch * n_tmpl, *z.shape[2:])
        mask = mask.view(batch * n_tmpl, *mask.shape[2:]) if mask is not None else None

    x = cueq.triangle_multiplicative_update(
        z,
        direction="outgoing" if _outgoing else "incoming",
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        g_in_weight=g_in_weight,
        p_in_weight=p_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=1e-5,
    )
    if is_batched_input:
        x = x.view(batch, n_tmpl, *x.shape[1:])
    return x