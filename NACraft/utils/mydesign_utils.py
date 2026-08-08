import pickle
import os
import numpy as np
import torch
from boltz.data.types import MSA, Connection, Input, Structure, Interface
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb


def save_structs(structs, outdir, prefix=""):
    os.makedirs(outdir, exist_ok=True)

    for output, struct_list, state_idx in structs:
        pkl_path = os.path.join(outdir, f"{prefix}state{state_idx}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(output, f)

        for j, struct in enumerate(struct_list):
            base = os.path.join(outdir, f"{prefix}state{state_idx}_sample{j}")
            with open(base + ".pdb", "w") as f:
                f.write(to_pdb(struct))
            with open(base + ".cif", "w") as f:
                f.write(to_mmcif(struct))


def get_batch(target, max_seqs=4096, keep_record=False):
    target_id = target.record.id
    structure = target.structure

    structure = Structure(
        atoms=structure.atoms,
        bonds=structure.bonds,
        residues=structure.residues,
        chains=structure.chains,
        connections=structure.connections.astype(Connection),
        interfaces=structure.interfaces,
        mask=structure.mask,
    )

    msas = {}
    for chain in target.record.chains:
        msa_id = chain.msa_id
        if msa_id != -1:
            msa = np.load(msa_id)
            msas[chain.chain_id] = MSA(**msa)

    input = Input(structure, msas)

    tokenizer = BoltzTokenizer()
    tokenized = tokenizer.tokenize(input)
    featurizer = BoltzFeaturizer()

    batch = featurizer.process(
        tokenized,
        training=False,
        max_atoms=None,
        max_tokens=None,
        max_seqs=max_seqs,
        pad_to_max_seqs=False,
        symmetries={},
        compute_symmetries=False,
        inference_binder=None,
        inference_pocket=None,
    )

    if keep_record:
        batch["record"] = target.record

    return batch, structure


def get_mid_points(pdistogram):
    boundaries = torch.linspace(2, 22.0, 63)
    lower = torch.tensor([1.0])
    upper = torch.tensor([22.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(pdistogram.device)

    return mid_points


def get_con_loss(
    dgram,
    dgram_bins,
    num=None,
    seqsep=None,
    num_pos=float("inf"),
    cutoff=None,
    binary=False,
    mask_1d=None,
    mask_1b=None,
):
    con_loss = _get_con_loss(dgram, dgram_bins, cutoff, binary)
    idx = torch.arange(dgram.shape[1])
    offset = idx[:, None] - idx[None, :]
    m = (torch.abs(offset) >= seqsep).to(dgram.device)
    if mask_1d is None:
        mask_1d = torch.ones(m.shape[0])
    if mask_1b is None:
        mask_1b = torch.ones(m.shape[0])

    m = torch.logical_and(m, mask_1b)
    p = min_k(con_loss, num, m).to(dgram.device)
    p = min_k(p, num_pos, mask_1d).to(dgram.device)
    return p


def _get_con_loss(dgram, dgram_bins, cutoff=None, binary=False):
    if cutoff is None:
        cutoff = dgram_bins[-1]
    bins = dgram_bins < cutoff
    px = torch.softmax(dgram, dim=-1)
    px_ = torch.softmax(dgram - 1e7 * (~bins), dim=-1)
    con_loss_cat_ent = -(px_ * torch.log_softmax(dgram, dim=-1)).sum(-1)
    con_loss_bin_ent = -torch.log((bins * px + 1e-8).sum(-1))

    return binary * con_loss_bin_ent + (1 - binary) * con_loss_cat_ent


def _get_helix_loss(
    dgram, dgram_bins, offset=None, mask_2d=None, binary=False, **kwargs
):
    x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=binary)
    if offset is None:
        if mask_2d is None:
            return x.diagonal(offset=3).mean()
        else:
            mask_2d = mask_2d.float()
            return (x * mask_2d).diagonal(offset=3, dim1=-2, dim2=-1).sum() / (
                torch.diagonal(mask_2d, offset=3, dim1=-2, dim2=-1).sum() + 1e-8
            )
    else:
        mask = (offset == 3).float()
        if mask_2d is not None:
            mask = mask * mask_2d.float()
        return (x * mask).sum() / (mask.sum() + 1e-8)


def _shift2d(x, di, dj):
    *batch, L, _ = x.shape
    y = torch.zeros_like(x)
    i_src = slice(max(0, -di), min(L, L - di))
    j_src = slice(max(0, -dj), min(L, L - dj))
    i_dst = slice(max(0, di),  min(L, L + di))
    j_dst = slice(max(0, dj),  min(L, L + dj))
    y[..., i_dst, j_dst] = x[..., i_src, j_src]
    return y


def _get_sheet_loss(
    dgram, dgram_bins, offset=None, mask_2d=None, binary=False,
    shift: int = 2, use_parallel: bool = True, use_antiparallel: bool = True,
    min_sep: int = 3
):
    x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=binary)

    if mask_2d is None:
        mask_2d = torch.ones_like(x, dtype=x.dtype)
    else:
        mask_2d = mask_2d.float()

    if offset is None:
        L = x.shape[-1]
        i = torch.arange(L, device=x.device)
        offset = (i[:, None] - i[None, :]).abs()
        for _ in range(x.dim() - 2):
            offset = offset.unsqueeze(0).expand(x.shape[:-2] + offset.shape)

    sep_mask = (offset >= min_sep).float()
    mask_2d = mask_2d * sep_mask

    par_x = _shift2d(x, +shift, +shift)
    par_m = _shift2d(mask_2d, +shift, +shift)
    anti_x = _shift2d(x, +shift, -shift)
    anti_m = _shift2d(mask_2d, +shift, -shift)

    eps = 1e-8
    score = x.new_tensor(0.0)
    denom = x.new_tensor(0.0)

    if use_parallel:
        score = score + (x * par_x * mask_2d * par_m).sum()
        denom = denom + (mask_2d * par_m).sum() + eps

    if use_antiparallel:
        score = score + (x * anti_x * mask_2d * anti_m).sum()
        denom = denom + (mask_2d * anti_m).sum() + eps

    return score / denom


def min_k(x, k=1, mask=None):
    if mask is not None:
        mask = mask.bool()
    y = torch.sort(x if mask is None else torch.where(mask, x, float("nan")))[0]
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def norm_seq_grad(grad, chain_mask):
    chain_mask = chain_mask.bool()
    masked_grad = grad[:, chain_mask.squeeze(0), :]
    eff_L = (masked_grad.pow(2).sum(-1, keepdim=True) > 0).sum(-2, keepdim=True)
    gn = masked_grad.norm(dim=(-1, -2), keepdim=True)
    return grad * torch.sqrt(torch.tensor(eff_L)) / (gn + 1e-7)


def run_model(boltz_model, batch, predict_args):
    boltz_model.predict_args = predict_args
    return boltz_model.predict_step(batch, batch_idx=0, dataloader_idx=0)


class Annealer:
    def __init__(
        self,
        soft=1,
        e_soft=1,
        temp=1,
        e_temp=1,
        hard=1,
        e_hard=1,
        step=1,
        e_step=1,
        num_optimizing_binder_pos=1,
        e_num_optimizing_binder_pos=1,
        iters=100,
        lr=0.1,
    ):
        m = {
            "soft": [soft, e_soft],
            "temp": [temp, e_temp],
            "hard": [hard, e_hard],
            "step": [step, e_step],
            "num_optimizing_binder_pos": [
                num_optimizing_binder_pos,
                e_num_optimizing_binder_pos,
            ],
        }
        self.m = {k: [s, (s if e is None else e)] for k, (s, e) in m.items()}
        self.lr = lr
        self.iters = iters

    def __iter__(self):
        import tqdm

        for i in tqdm.trange(self.iters):
            opt = {}
            for k, (s, e) in self.m.items():
                if k == "temp":
                    opt[k] = e + (s - e) * (1 - (i) / self.iters) ** 2
                else:
                    v = s + (e - s) * ((i) / self.iters)
                    if k == "step":
                        step = v
                    opt[k] = v

            lr_scale = step * ((1 - opt["soft"]) + (opt["soft"] * opt["temp"]))
            num_optimizing_binder_pos = int(opt["num_optimizing_binder_pos"])

            opt["lr_rate"] = self.lr * lr_scale
            yield opt
