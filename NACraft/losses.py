import numpy as np
import torch
import torch.nn.functional as F
from utils.mydesign_utils import get_mid_points, get_con_loss, _get_helix_loss, _get_sheet_loss
from utils import na_constants

class MotifLoss:
    def __init__(self, motif):
        self.motif = motif
        cb_pos = motif['cb_pos']
        dmat = np.square(cb_pos[None] - cb_pos[:,None]).sum(-1)**0.5
        motif_mask = motif['motif_mask']
        dmat[~motif_mask,:] = dmat[:,~motif_mask] = 0
        self.dmat = dmat

    def evaluate(self, dict_out, device, opt=None):
        pdist = dict_out['pdistogram'].to(device)

        mid_pts = get_mid_points(pdist).to(device)
        motif_dmat = torch.from_numpy(self.dmat).to(device)
        pdist = pdist[:,:len(motif_dmat),:len(motif_dmat)]
        motif_dmat_mask = (motif_dmat > 1e-3) & (motif_dmat < 22)

        motif_mse_loss = (pdist.softmax(-1) * (mid_pts - motif_dmat[...,None])**2).sum(-1)
        motif_mse_loss = (motif_mse_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()

        return motif_mse_loss


class AntiMotifLoss:
    def __init__(self, motif):
        self.motif = motif
        cb_pos = motif['cb_pos']
        dmat = np.square(cb_pos[None] - cb_pos[:,None]).sum(-1)**0.5
        motif_mask = motif['motif_mask']
        dmat[~motif_mask,:] = dmat[:,~motif_mask] = 0
        self.dmat = dmat

    def evaluate(self, dict_out, device, opt=None):
        pdist = dict_out['pdistogram'].to(device)

        mid_pts = get_mid_points(pdist).to(device)
        motif_dmat = torch.from_numpy(self.dmat).to(device)
        pdist = pdist[:,:len(motif_dmat),:len(motif_dmat)]
        motif_dmat_mask = (motif_dmat > 1e-3) & (motif_dmat < 22)

        motif_mse_loss = (pdist.softmax(-1) * (mid_pts - motif_dmat[...,None])**2).sum(-1)
        motif_mse_loss = (motif_mse_loss * motif_dmat_mask).sum() / motif_dmat_mask.sum()

        return -0.5*motif_mse_loss

class ContactLoss:
    def __init__(self, polymer_type="protein"):
        self.polymer_type = polymer_type

    def evaluate(self, dict_out, device, opt=None):
        mol_type_val = na_constants.get_mol_type(self.polymer_type)
        chain_mask = (dict_out['mol_type'] == mol_type_val).to(device)
        pdist = dict_out['pdistogram'].to(device)
        mid_pts = get_mid_points(pdist).to(device)
        seqsep = 9 if self.polymer_type == "protein" else 4  # NA has shorter persistence length
        con_loss = get_con_loss(
            pdist,
            mid_pts,
            num=1,
            seqsep=seqsep,
            cutoff=14.,
            binary=False,
            mask_1d=chain_mask,
            mask_1b=chain_mask,
        )
        return con_loss

class HelixBiasLoss:
    def __init__(self, strength: float = 0.0):
        self.strength = strength

    def evaluate(self, dict_out, device, opt=None):
        if self.strength == 0.0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        chain_mask = (dict_out["mol_type"] == 0).to(device)
        pdist = dict_out["pdistogram"].to(device)
        mid_pts = get_mid_points(pdist).to(device)

        mask_2d = chain_mask[:, :, None] * chain_mask[:, None, :]
        helix_loss = _get_helix_loss(pdist, mid_pts, offset=None, mask_2d=mask_2d, binary=True)

        return self.strength * helix_loss

class ConfChangeLoss:
    def __init__(self, strength=1.0, eps=1e-6, stable=False, topk=1, polymer_type="protein"):
        self.strength = strength
        self.eps = eps
        self.stable = stable
        self.topk = topk
        self.polymer_type = polymer_type

    def _jsd(self, p, q):
        p = p.clamp(min=self.eps)
        q = q.clamp(min=self.eps)
        m = 0.5 * (p + q)
        m = m.clamp(min=self.eps)

        return 0.5 * (
            p * (p.log() - m.log()) +
            q * (q.log() - m.log())
        ).sum(dim=-1)

    def evaluate(self, dict_out, device, opt=None):
        mol_type_val = na_constants.get_mol_type(self.polymer_type)

        mask0 = dict_out[0]["mol_type"] == mol_type_val
        mask1 = dict_out[1]["mol_type"] == mol_type_val

        p0 = dict_out[0]["pdistogram"].softmax(dim=-1)
        p1 = dict_out[1]["pdistogram"].softmax(dim=-1)

        p0 = p0[:, mask0[0]][:, :, mask0[0]]
        p1 = p1[:, mask1[0]][:, :, mask1[0]]

        jsd = self._jsd(p0, p1)

        if self.stable:
            score = jsd.mean()
        else:
            score = jsd.max(-1).values.mean()

        return -self.strength * score

class LigandContactLoss:
    def __init__(self, idx=None, strength=1.0, polymer_type="protein"):
        self.idx = idx
        self.strength = strength
        self.polymer_type = polymer_type

    def evaluate(self, dict_out, device, opt=None):
        chain_mask = (dict_out['asym_id'] == 0).to(device)
        if self.idx is None:
            i_chain_mask = (dict_out['asym_id'] != 0).to(device)
        else:
            i_chain_mask = (dict_out['asym_id'] == self.idx).to(device)
        pdist = dict_out['pdistogram'].to(device)
        mid_pts = get_mid_points(pdist).to(device)
        i_con_loss = get_con_loss(
            pdist,
            mid_pts,
            num=2,
            seqsep=0,
            num_pos=int(opt["num_optimizing_binder_pos"]),
            cutoff=20.,
            binary=False,
            mask_1d=chain_mask,
            mask_1b=i_chain_mask,
        )

        return self.strength * i_con_loss


class AntiLigandContactLoss:
    def __init__(self, strength=1.0, idx=None, polymer_type="protein"):
        self.idx = idx
        self.strength = strength
        self.polymer_type = polymer_type

    def evaluate(self, dict_out, device, opt=None):
        chain_mask = (dict_out['asym_id'] == 0).to(device)
        if self.idx is None:
            i_chain_mask = (dict_out['asym_id'] != 0).to(device)
        else:
            i_chain_mask = (dict_out['asym_id'] == self.idx).to(device)
        pdist = dict_out['pdistogram'].to(device)
        mid_pts = get_mid_points(pdist).to(device)
        i_con_loss = get_con_loss(
            pdist,
            mid_pts,
            num=2,
            seqsep=0,
            num_pos=int(opt["num_optimizing_binder_pos"]),
            cutoff=20.,
            binary=False,
            mask_1d=chain_mask,
            mask_1b=i_chain_mask,
        )

        return -self.strength * i_con_loss

class SheetBiasLoss:
    def __init__(self, strength: float = 0.0):
        self.strength = strength

    def evaluate(self, dict_out, device, opt=None):
        if self.strength == 0.0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        chain_mask = (dict_out["mol_type"] == 0).to(device)
        pdist = dict_out["pdistogram"].to(device)
        mid_pts = get_mid_points(pdist).to(device)

        mask_2d = chain_mask[:, :, None] * chain_mask[:, None, :]
        sheet_loss = _get_sheet_loss(pdist, mid_pts, offset=None, mask_2d=mask_2d, binary=True)

        return self.strength * sheet_loss

class SequenceSimilarityLoss:
    def __init__(self, target_sequence, strength=1.0, polymer_type="protein"):
        self.target_sequence = target_sequence
        self.strength = strength
        self.polymer_type = polymer_type

    def evaluate(self, dict_out, device, opt=None):
        probs = dict_out['restype']['soft'].to(device)

        if self.polymer_type == "protein":
            aa_order = "ARNDCQEGHILKMFPSTWYV"
            aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
            probs = probs[..., 2:22]
            target_idx = torch.tensor(
                [aa_to_idx[aa] for aa in self.target_sequence],
                device=device,
            )
        elif self.polymer_type == "rna":
            na_order = "AGCU"
            na_to_idx = {b: i for i, b in enumerate(na_order)}
            probs = probs[..., 23:27]
            target_idx = torch.tensor(
                [na_to_idx[b] for b in self.target_sequence],
                device=device,
            )
        elif self.polymer_type == "dna":
            na_order = "AGCT"
            na_to_idx = {b: i for i, b in enumerate(na_order)}
            probs = probs[..., 28:32]
            target_idx = torch.tensor(
                [na_to_idx[b] for b in self.target_sequence],
                device=device,
            )

        p_tgt = probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)
        return -self.strength * p_tgt.mean()

class RadiusOfGyrationLoss:
    def __init__(self, strength=1.0):
        self.strength = strength

    def evaluate(self, dict_out, device, opt=None):
        pdist = dict_out['pdistogram'].to(device).softmax(dim=-1)
        mid_pts = get_mid_points(pdist).to(device)

        L = pdist.size(1)
        Ed2 = (pdist * (mid_pts ** 2)).sum(-1)
        Ed2 = Ed2 * (~torch.eye(L, device=device, dtype=torch.bool)).unsqueeze(0)
        rg  = torch.sqrt(Ed2.sum((1, 2)) / (2.0 * L * L) + 1e-8)
        rg_th = 2.38 * (L ** 0.365)
        loss = F.elu(rg - rg_th).mean()

        return self.strength * loss
