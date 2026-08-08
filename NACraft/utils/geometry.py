import torch
import numpy as np

# from .rigid_utils import Rigid, Rotation
# from . import residue_constants as rc
from .tensor_utils import batched_gather

@torch.cuda.amp.autocast(False)
def compute_rmsd(a, b, weights=None, reduce=True):
    B = a.shape[:-2]
    N = a.shape[-2]

    if weights is None:
        weights = a.new_ones(*B, N)

    b = rmsdalign(a, b, weights)
    sqdist = torch.square(a - b).sum(-1)
    if reduce:
        return torch.sqrt((sqdist * weights).sum(-1) / weights.sum(-1))
    else:
        return torch.sqrt(sqdist)

@torch.cuda.amp.autocast(False)
def compute_pseudo_tm(a, b, weights=None):
    B = a.shape[:-2]
    N = a.shape[-2]

    if weights is None:
        weights = a.new_ones(*B, N)

    b = rmsdalign(a, b, weights)
    L = weights.sum(-1)
    d0 = 1.24*(L-15)**(1/3)-1.8
    
    dis = torch.square(a - b).sum(-1).sqrt()
    tm = 1/(1+torch.square(dis/d0[...,None]))
    
    return (tm * weights).sum(-1) / weights.sum(-1)

# https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
@torch.cuda.amp.autocast(False)
def rmsdalign(
    a, b, weights=None, demean=True, a_origin=None, b_origin=None
):  # alignes B to A  # [*, N, 3]
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights is None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    if demean:
        a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
        a = a - a_mean
        b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
        b = b - b_mean
    if a_origin is not None:
        a = a - a_origin
    if b_origin is not None:
        b = b - b_origin
    B = torch.einsum("...ji,...jk->...ik", weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    sgn = torch.sign(torch.linalg.det((u @ vh).cpu())).to(u.device)  # ugly workaround
    s[..., -1] *= sgn
    u[..., :, -1] *= sgn.unsqueeze(-1)
    C = u @ vh  # c rotates B to A
    if demean:
        return b @ C.mT + a_mean
    elif a_origin is not None:
        return b @ C.mT + a_origin
    else:
        return b @ C.mT

def kabsch(P, Q):
    P, Q = P.float(), Q.float()
    assert P.shape == Q.shape, "Matrix dimensions must match"

    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    p = P - centroid_P
    q = Q - centroid_Q
    
    H = torch.matmul(q.T, p)

    U, S, Vt = torch.linalg.svd(H)

    if torch.det(Vt.T @ U.T) < 0:
        Vt[-1, :] *= -1.0

    R = Vt.T @ U.T
    t = centroid_P - centroid_Q @ R

    return R, t


def rg(X):
    return torch.sqrt(X.var(dim=0, unbiased=False).sum())
