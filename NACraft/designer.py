from boltz.data.parse.schema import parse_boltz_schema
import torch
import copy
import numpy as np
import pickle
import csv
import json
from typing import Optional
from utils.mydesign_utils import get_batch, run_model, Annealer, norm_seq_grad
import os,time
from utils.tied_lmpnn import perform_tied_lmpnn_redesign
from utils.na_mpnn_utils import perform_tied_nampnn_redesign
from utils import mydesign_utils, na_constants, af3_utils
from utils.model_assets import resolve_boltz_asset
from losses import *
from pathlib import Path

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32= os.getenv("BOLTZ_USE_CUEQ", "1").lower() in ("1", "true", "yes", "on") # for cueq kernels



with open(resolve_boltz_asset("ccd.pkl"), "rb") as f:
    CCD_LIB = pickle.load(f)


OPTIMIZATION_TRACE_COLUMNS = [
    "global_step",
    "stage",
    "stage_step",
    "sequence",
    "total_loss",
    "loss_name",
    "state",
    "weight",
    "raw_loss",
    "weighted_loss",
]


def _format_trace_state(state):
    if isinstance(state, list):
        return json.dumps(state)
    return str(state)


def format_optimization_trace_rows(
    global_step,
    stage,
    stage_step,
    sequence,
    total_loss,
    losses,
):
    if not losses:
        losses = [
            {
                "name": "",
                "state": "",
                "weight": "",
                "raw_loss": "",
                "weighted_loss": "",
            }
        ]

    rows = []
    for loss in losses:
        rows.append(
            {
                "global_step": global_step,
                "stage": stage,
                "stage_step": stage_step,
                "sequence": sequence,
                "total_loss": total_loss,
                "loss_name": loss["name"],
                "state": _format_trace_state(loss["state"]),
                "weight": loss["weight"],
                "raw_loss": loss["raw_loss"],
                "weighted_loss": loss["weighted_loss"],
            }
        )
    return rows


def write_optimization_trace(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=OPTIMIZATION_TRACE_COLUMNS, delimiter="\t"
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def revcomp(seq: str) -> str:
    complement = str.maketrans("ACGT", "TGCA")
    return seq.upper().translate(complement)[::-1]
    
def get_batch_with_ligands(seq, ligands=None, device="cuda", polymer_type="protein",
                           presearch_lookup=None):
    polymer_key = na_constants.get_polymer_key(polymer_type)
    polymer_entry = {
        polymer_key: {
            "id": ["A"],
            "sequence": seq,
        }
    }
    # Protein uses MSA; RNA/DNA do not
    if polymer_type == "protein":
        # Presearch hit → real MSA. Boltz needs a pre-processed .npz path
        # (added by MultistateDesigner._prepare_boltz_msa_cache) because
        # get_batch() calls np.load() directly. Templates are not supported
        # by Boltz's input schema, so we only inject MSA here.
        hit = presearch_lookup.get(seq) if presearch_lookup else None
        polymer_entry[polymer_key]["msa"] = hit.get("boltzMsaPath", "empty") if hit else "empty"

    data = {
        "version": 1,
        "sequences": [polymer_entry],
    }
    ALPHABET = "BCDEFGHIJKLMNOPQRSTUVWXYZ"
    for ligand in ligands:
        if ligand is not None:
            assert isinstance(ligand, tuple) and len(ligand) == 2, "ligand must be a (value, mol_type) tuple"
            ligand, mol_type = ligand
            if mol_type == "ligand":
                data["sequences"].append({
                    "ligand": {
                        "id": [ALPHABET[0]],
                        "smiles": ligand,
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type == "ccd":
                data["sequences"].append({
                    "ligand": {
                        "id": [ALPHABET[0]],
                        "ccd": ligand,
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type == "protein":
                hit = presearch_lookup.get(ligand) if presearch_lookup else None
                msa_val = hit.get("boltzMsaPath", "empty") if hit else "empty"
                data["sequences"].append({
                    "protein": {
                        "id": [ALPHABET[0]],
                        "sequence": ligand,
                        "msa": msa_val,
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type in "rna":
                data["sequences"].append({
                    mol_type: {
                        "id": [ALPHABET[0]],
                        "sequence": ligand,
                    }
                })
                ALPHABET = ALPHABET[1:]
            elif mol_type in "dna":
                data["sequences"].append({
                    "dna": {
                        "id": [ALPHABET[0]],
                        "sequence": ligand,
                    }
                })
                data["sequences"].append({
                    "dna": {
                        "id": [ALPHABET[1]],
                        "sequence": revcomp(ligand),
                    }
                })
                ALPHABET = ALPHABET[2:]
            else:
                raise ValueError(f"Unsupported mol_type: {mol_type!r}. Valid types: ligand, ccd, protein, dna, rna")
        else:
            raise ValueError(f"Unsupported mol_type: {mol_type}")

    target = parse_boltz_schema(None, data, CCD_LIB)
    batch, structure = get_batch(target)
    batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}

    # When ALL chains are in single-sequence mode (no real MSA), the featurizer
    # produces depth-1 MSA tensors that need to be coerced into the shape the
    # trunk expects (the original NACraft overrides below). When a real MSA is
    # loaded, the featurizer already produces correctly-shaped tensors and these
    # overrides would clobber them with the wrong depth — skip them.
    msa_depth = batch["msa"].shape[1] if batch["msa"].dim() >= 4 else 1
    if msa_depth <= 1:
        # batch["msa"] = batch["res_type_logits"].unsqueeze(0).to(device)
        batch["msa_paired"] = torch.ones(
            batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
        ).to(device)
        batch["deletion_value"] = torch.zeros(
            batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
        ).to(device)
        batch["has_deletion"] = torch.full(
            (batch["res_type"].shape[0], 1, batch["res_type"].shape[1]), False
        ).to(device)
        batch["msa_mask"] = torch.ones(
            batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
        ).to(device)
        batch["profile"] = batch["msa"].float().mean(dim=0).to(device)
        batch["deletion_mean"] = torch.zeros(batch["deletion_mean"].shape).to(device)
    batch["res_type"] = batch["res_type"].float()

    return batch, structure



        
class MultistateDesigner:
    def __init__(self, num_states=1, radius_gyr=False, visualize=False, polymer_type="protein",
                 predictor="boltz", af3_cfg=None):
        self.ligands = [[] for _ in range(num_states)]
        self.motifs = []
        self.losses = []
        self.loss_log = []
        self.last_loss_details = []
        self.last_total_loss = None
        self.optimization_step = 0
        self.radius_gyr = radius_gyr
        self.polymer_type = polymer_type
        self.predictor = predictor
        self.af3_cfg = af3_cfg or {}
        # Presearch cache (seq → {unpairedMsaPath, pairedMsaPath, templates}).
        # Populated via af3_cfg["presearch_lookup"]; consumed by both Boltz and
        # AF3 final-prediction paths to inject real MSA/templates for target
        # protein chains. None = de novo single-sequence mode (default).
        self.presearch_lookup = self.af3_cfg.get("presearch_lookup")
        if self.presearch_lookup:
            self.presearch_lookup = self._prepare_boltz_msa_cache(self.presearch_lookup)

        self.visualize = visualize
        self.pseudo_logit_traj = []

    def _prepare_boltz_msa_cache(self, lookup: dict) -> dict:
        """Pre-process presearch .a3m files into Boltz's .npz format.

        Boltz's `parse_boltz_schema` accepts a path in the `msa` field, but
        the downstream `get_batch()` calls `np.load(msa_id)` directly — which
        only works on .npz files (processed MSAs), not raw .a3m text. Boltz's
        `main.py` does this preprocessing as a separate step; since NACraft
        calls `parse_boltz_schema` directly, we replicate it here.

        Side effect: adds a `boltzMsaPath` key to each lookup entry (the .npz
        path), leaving `unpairedMsaPath` untouched so AF3 still sees the .a3m.
        The cache is content-addressed by source path + size + mtime.
        """
        import hashlib
        from pathlib import Path
        try:
            from boltz.data.parse.a3m import parse_a3m
        except ImportError as e:
            print(f"[presearch] boltz MSA parser unavailable ({e}); "
                  f"skipping .npz cache — Boltz runs will fall back to no MSA", flush=True)
            return lookup

        cache_root = Path(
            self.af3_cfg.get("boltz_msa_cache_dir", ".cache/boltz_msa")
        )
        cache_root.mkdir(parents=True, exist_ok=True)

        new_lookup = {}
        for seq, entry in lookup.items():
            new_entry = dict(entry)
            a3m_path = Path(entry["unpairedMsaPath"])
            if not a3m_path.exists():
                print(f"[presearch] WARNING: a3m missing at {a3m_path}, "
                      f"Boltz will use single-sequence mode for this target", flush=True)
                new_lookup[seq] = new_entry
                continue
            stat = a3m_path.stat()
            key = hashlib.sha1(f"{a3m_path}|{stat.st_size}|{stat.st_mtime}".encode()).hexdigest()[:16]
            npz_path = cache_root / f"{a3m_path.stem}_{key}.npz"
            if not npz_path.exists():
                t0 = time.perf_counter()
                # Cap MSA depth at parse time. Boltz's own CLI default is
                # max_msa_seqs=8192, but NACraft's direct get_batch path
                # bypasses that gate. Limit depth before optimization so large
                # target MSAs do not dominate every gradient step.
                msa_obj = parse_a3m(a3m_path, taxonomy=None, max_seqs=2048)
                msa_obj.dump(npz_path)
                print(f"[presearch] cached {a3m_path.name} → {npz_path.name} "
                      f"({len(msa_obj.sequences)} seqs, {time.perf_counter()-t0:.1f}s)", flush=True)
            new_entry["boltzMsaPath"] = str(npz_path)
            new_lookup[seq] = new_entry
        return new_lookup
        
        
    def add_ligand(self, ligand, state):
        if type(ligand) is list:
            self.ligands[state].extend(ligand)
        else:
            self.ligands[state].append(ligand)
            
    def add_motif(self, motif):
        self.motifs.append(motif)

    def add_loss(self, loss, state, weight=1.0):
        self.losses.append((loss, state, weight))
        
    def initialize(self, length, device='cuda', seq=None, seq_weight=0.0, contact_loss=True, init_seq=None):

        self.length = length
        self.device = device
        self.invalid_idx = na_constants.get_invalid_indices(self.polymer_type)
        self.unk_idx = na_constants.get_unk_token_id(self.polymer_type)

        if self.polymer_type == "protein":
            alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
            num_classes = 33  # full Boltz vocab
            mask_after = 22
        elif self.polymer_type == "rna":
            alphabet = list("XXARNDCQEGHILKMFPSTWYV-")  # keep for compatibility; actual display via na_constants
            num_classes = 33
            mask_after = None  # don't use old mask scheme
        elif self.polymer_type == "dna":
            alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
            num_classes = 33
            mask_after = None
        else:
            raise ValueError(f"Unknown polymer_type: {self.polymer_type}")

        z = torch.distributions.Gumbel(0, 1).sample((length, 33)).to(device)
        # Mask invalid tokens
        z[..., self.invalid_idx] = -np.inf
        self.logits = z.softmax(-1)

        ### build the motif mask ###
        self.fixed_mask = torch.zeros(length, dtype=bool, device=device)
        self.fixed_aa = torch.zeros_like(self.logits)

        unk_char = "X" if self.polymer_type == "protein" else "N"
        if init_seq is not None:
            if len(init_seq) != length:
                raise ValueError(
                    f"init_seq length {len(init_seq)} does not match design length {length}"
                )
            if self.polymer_type == "rna":
                seed_ids = [na_constants.RNA_TOKEN_IDS.get(c.upper(), 27) for c in init_seq]
            elif self.polymer_type == "dna":
                seed_ids = [na_constants.DNA_LETTER_TO_TOKEN_ID.get(c.upper(), 32) for c in init_seq]
            else:
                prot_alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
                seed_ids = [prot_alphabet.index(c) if c in prot_alphabet else 1 for c in init_seq]
            seed_onehot = torch.nn.functional.one_hot(
                torch.tensor(seed_ids), num_classes=33
            ).float().to(device)
            self.logits = 0.9 * seed_onehot + 0.1 * self.logits
            start_seq = list(init_seq)
            print(f"[init_seq] logits seeded from {length}-nt sequence", flush=True)
        else:
            start_seq = [unk_char] * length

        for i, motif in enumerate(self.motifs):
            if motif is not None:
                motif_mask = torch.from_numpy(motif['motif_mask']).to(device)

                self.fixed_mask |= motif_mask
                # Convert motif sequence to Boltz token indices
                motif_seq_indices = []
                for c in motif['motif_seq']:
                    if self.polymer_type == "protein":
                        motif_seq_indices.append(alphabet.index(c))
                    elif self.polymer_type == "rna":
                        motif_seq_indices.append(na_constants.RNA_TOKEN_IDS.get(c, 27))
                    elif self.polymer_type == "dna":
                        motif_seq_indices.append(na_constants.DNA_LETTER_TO_TOKEN_ID.get(c, 32))

                motif_seq_onehot = torch.nn.functional.one_hot(
                    torch.tensor(motif_seq_indices), num_classes=33
                )
                self.fixed_aa[motif_mask] = motif_seq_onehot.to(device)[motif_mask].float()
                for j in range(length):
                    if motif['motif_mask'][j]:
                        start_seq[j] = motif['motif_seq'][j]
                
        self.logits = torch.where(
            self.fixed_mask[...,None], 
            self.fixed_aa,
            self.logits
        )

        ## make the boltz batch objects
        self.batches = []
        for i, ligs in enumerate(self.ligands):
            self.batches.append(
                get_batch_with_ligands(''.join(start_seq), ligs, device, polymer_type=self.polymer_type)[0]
            )
            if contact_loss:
                self.add_loss(ContactLoss(), state=i)
            if self.radius_gyr:
                self.add_loss(RadiusOfGyrationLoss(), state=i)

        

    def get_seq(self):
        logits = self.logits.clone()
        logits[..., self.invalid_idx] = -float("inf")
        indices = logits.argmax(-1)

        if self.polymer_type == "protein":
            alphabet = list("XXARNDCQEGHILKMFPSTWYV-")
            return "".join(alphabet[i.item()] for i in indices)
        elif self.polymer_type == "rna":
            return "".join(na_constants.RNA_TOKEN_ID_TO_LETTER.get(i.item(), "N") for i in indices)
        elif self.polymer_type == "dna":
            return "".join(na_constants.DNA_TOKEN_ID_TO_LETTER.get(i.item(), "N") for i in indices)


    def get_final_structs(self, boltz_model, samples: Optional[int] = 5, set_seq: Optional[str] = None):
        predict_args={
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": samples,
            "write_confidence_summary": True,
            "write_full_pae": True,
            "write_full_pde": True,
        }
        results = []
        for i, ligands in enumerate(self.ligands):
            seq = self.get_seq() if set_seq is None else set_seq
            new_batch, new_struct = get_batch_with_ligands(
                seq, ligands, polymer_type=self.polymer_type,
                presearch_lookup=self.presearch_lookup,
            )

            output = run_model(boltz_model, new_batch, predict_args)
            coords_all = output["coords"]

            struct_list = []
            for j in range(coords_all.shape[0]):
                struct_copy = copy.deepcopy(new_struct)
                struct_copy.atoms["coords"] = (
                    coords_all[j, : len(new_struct.atoms)].cpu().numpy()
                )
                struct_list.append(struct_copy)

            results.append((output, struct_list, i))

        return results

    def get_final_structs_af3(self, samples: Optional[int] = None, set_seq: Optional[str] = None,
                              output_root: Optional[str] = None, name_prefix: Optional[str] = None,
                              save_dir: Optional[str] = None):
        """Final structure prediction via AlphaFold3 (alternative to Boltz-1 path).

        Per state: build AF3 input JSON → apptainer exec → parse CIF + confidences.
        Returns list of (output_dict, struct_list, state_idx) matching Boltz path's shape.
        Differentiable sequence optimization still uses Boltz-1; this method
        only replaces the post-optimization structure-generation step.

        output_dict contains the AF3 metrics in the same key schema as Boltz's
        (confidence_score, iptm, ptm, complex_iplddt, complex_plddt, complex_ipde)
        plus a `_predictor: 'af3'` tag for traceability.

        If `save_dir` is given, also writes:
          <save_dir>/state<i>.pkl                 - metrics dict
          <save_dir>/state<i>_sample<j>.cif/.pdb  - per-sample structures
        matching the layout expected by downstream packaging scripts.
        """
        ns = samples if samples is not None else self.af3_cfg.get("num_samples", 5)
        results = []
        for i, ligands in enumerate(self.ligands):
            seq = self.get_seq() if set_seq is None else set_seq
            name = f"{name_prefix or 'nacraft'}_state{i}"
            state_out = os.path.join(output_root or ".", f"af3_state{i}_raw")
            os.makedirs(state_out, exist_ok=True)

            t0 = time.perf_counter()
            print(f"[AF3] state {i}: predicting {ns} samples for seq len={len(seq)}, "
                  f"ligands={len(ligands)}")
            struct_list, metrics = af3_utils.predict_complex(
                name=name,
                seq=seq,
                ligands=ligands,
                polymer_type=self.polymer_type,
                output_dir=state_out,
                cfg=self.af3_cfg,
                exp_name=name,
                num_samples=ns,
                presearch_lookup=self.presearch_lookup,
            )

            # Pack into output_dict with same key schema as Boltz path
            import torch as _torch
            output = {
                "confidence_score": _torch.tensor(metrics["confidence_score"]),
                "iptm":             _torch.tensor(metrics["iptm"]),
                "ptm":              _torch.tensor(metrics["ptm"]),
                "complex_iplddt":   _torch.tensor(metrics["complex_iplddt"]),
                "complex_plddt":    _torch.tensor(metrics["complex_plddt"]),
                "complex_ipde":     _torch.tensor(metrics["complex_ipde"]),
                "_predictor":       self.predictor,
            }

            # Write to disk if save_dir provided (matches Boltz save_structs layout)
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                pkl_path = os.path.join(save_dir, f"state{i}.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(output, f)
                for j, struct in enumerate(struct_list):
                    base = os.path.join(save_dir, f"state{i}_sample{j}")
                    try:
                        struct.write_pdb(base + ".pdb")
                    except Exception:
                        # gemmi fallback
                        struct.write_minimal_pdb(base + ".pdb")
                    struct.make_mmcif_document().write_file(base + ".cif")

            results.append((output, struct_list, i))
            print(f"[AF3] state {i}: done in {time.perf_counter() - t0:.1f}s "
                  f"(conf={metrics['confidence_score'].tolist()}, "
                  f"iptm={metrics['iptm'].tolist()})")
        return results

    def get_restype_from_logits(self, res_type_logits, opt, alpha=2.0):
        device = res_type_logits.device
        logits = alpha * res_type_logits

        X = logits - torch.sum(
            torch.eye(logits.shape[-1])[self.invalid_idx],
            dim=0,
        ).to(device) * (1e10)
        soft = torch.softmax(X / opt["temp"], dim=-1)  # probs
        hard = torch.zeros_like(soft).scatter_(
            -1, soft.max(dim=-1, keepdim=True)[1], 1.0
        )  # one hot
        hard = (hard - soft).detach() + soft # carries same grad

        pseudo = (
            opt["soft"] * soft + (1 - opt["soft"]) * res_type_logits
        )  # interp between probs and logits
        pseudo = (
            opt["hard"] * hard + (1 - opt["hard"]) * pseudo
        )  # interp between on hot and the above
    
        return {'soft': soft, 'hard': hard, 'pseudo': pseudo}

    def get_loss(self, restype, boltz_model, opt, verbose=False):
        total_loss = 0
        loss_dict = []

        boltz_out = []
        # prep for boltz
        for state_idx, batch in enumerate(self.batches):
            batch['res_type'] = torch.cat([
                restype['pseudo'][None],
                batch['res_type'][:,len(restype['pseudo']):].detach()
            ], 1)
            batch["msa"] = batch["res_type"].unsqueeze(0).detach()
            batch["profile"] = batch["msa"].float().mean(dim=0).detach()

            dict_out = boltz_model.get_distogram(batch)[0]
            boltz_out.append(batch | dict_out | {'restype': restype})


        for loss, state, weight in self.losses:
            if type(state) is list:
                readout = [boltz_out[s] for s in state]
            else:
                readout = boltz_out[state]
            this_loss = loss.evaluate(readout, boltz_model.device, opt)

            loss_dict.append((type(loss).__name__, state, this_loss.item()))
            total_loss = total_loss + weight * this_loss

        self.last_loss_details = [
            {
                "name": name,
                "state": state,
                "weight": weight,
                "raw_loss": raw_loss,
                "weighted_loss": weight * raw_loss,
            }
            for (name, state, raw_loss), (_, _, weight) in zip(loss_dict, self.losses)
        ]
        self.last_total_loss = float(total_loss.detach().item())

        if verbose:
            print(loss_dict)
            print(self.get_seq())

        self.loss_log.append(loss_dict)

        return total_loss

    def do_iter(
        self,
        boltz_model,
        opt,
        pre_run=False,
        verbose=False,
        stage=None,
        stage_step=None,
        trace_path=None,
    ):

        self.logits.requires_grad = True
        restype = self.get_restype_from_logits(self.logits, opt)
        restype = {
            k: torch.where(self.fixed_mask[...,None], self.fixed_aa, v)\
            for k, v in restype.items()
        }
        
        if self.visualize:
            with torch.no_grad():
                self.pseudo_logit_traj.append(restype["pseudo"].detach().cpu().clone())
       
        loss = self.get_loss(restype, boltz_model, opt, verbose=verbose)
        # loss = restype.sum()
    
        loss.backward()
        if verbose: print('total_loss', loss)
        
        
        with torch.no_grad():
            self.logits.grad[self.fixed_mask] = 0
            self.logits.grad[..., self.invalid_idx] = 0
            self.logits.grad = norm_seq_grad(
                self.logits.grad[None], 
                torch.ones_like(self.logits[:,0])
            )[0]
        
            self.logits -= opt["lr_rate"] * self.logits.grad
        self.logits.grad = None

        if trace_path is not None:
            rows = format_optimization_trace_rows(
                global_step=self.optimization_step,
                stage=stage or "",
                stage_step=stage_step if stage_step is not None else "",
                sequence=self.get_seq(),
                total_loss=self.last_total_loss,
                losses=self.last_loss_details,
            )
            write_optimization_trace(trace_path, rows)
        self.optimization_step += 1
        
                

    def optimize(
        self,
        boltz_model,
        verbose=False,
        debug=False,
        trace_dir=None,
        early_stopping=False,
        post_early_stop_only=False,
    ):
        trace_path = None
        if trace_dir is not None:
            trace_path = os.path.join(trace_dir, "optimization_trace.tsv")
            if os.path.exists(trace_path):
                os.remove(trace_path)
            self.optimization_step = 0

        global_best = float("inf")
        global_no_improve = 0
        stage_best = float("inf")
        stage_no_improve = 0
        monitored_stage = None

        def should_early_stop(stage):
            """Monitor only the exploration stage."""
            nonlocal global_best, global_no_improve
            nonlocal stage_best, stage_no_improve, monitored_stage
            if not early_stopping or stage != "exploration":
                return False
            if monitored_stage != stage:
                monitored_stage = stage
                stage_best = float("inf")
                stage_no_improve = 0
            loss = float(self.last_total_loss)
            if loss < stage_best:
                stage_best = loss
                stage_no_improve = 0
            else:
                stage_no_improve += 1
            if loss < global_best:
                global_best = loss
                global_no_improve = 0
            else:
                global_no_improve += 1
            if stage_no_improve >= 10 or global_no_improve >= 30:
                reason = (
                    "stage_patience=10"
                    if stage_no_improve >= 10
                    else "global_patience=30"
                )
                print(
                    f"[early_stopping] stage stop at global_step={self.optimization_step} "
                    f"stage={stage} loss={loss:.8f} global_best={global_best:.8f} "
                    f"stage_no_improve={stage_no_improve} "
                    f"global_no_improve={global_no_improve} reason={reason}",
                    flush=True,
                )
                return True
            return False

        if post_early_stop_only:
            print("continuation: 20 annealing iterations")
            for stage_step, opt in enumerate(Annealer(
                e_temp=0.01,
                hard=0,
                e_hard=0,
                num_optimizing_binder_pos=8,
                e_num_optimizing_binder_pos=12,
                iters=20,
            )):
                self.do_iter(
                    boltz_model, opt, verbose=verbose,
                    stage="annealing_down", stage_step=stage_step,
                    trace_path=trace_path,
                )
            print("continuation: 10 argmax iterations")
            for stage_step, opt in enumerate(Annealer(
                temp=0.01,
                e_temp=0.01,
                num_optimizing_binder_pos=12,
                e_num_optimizing_binder_pos=16,
                iters=10,
            )):
                self.do_iter(
                    boltz_model, opt, verbose=verbose,
                    stage="argmax", stage_step=stage_step,
                    trace_path=trace_path,
                )
            return

        print(f"stage 1: warmup")

        for stage_step, opt in enumerate(Annealer(hard=0, e_hard=0, iters=30, lr=0.2)):
            self.do_iter(
                boltz_model,
                opt,
                pre_run=True,
                verbose=verbose,
                stage="warmup",
                stage_step=stage_step,
                trace_path=trace_path,
            )
        if debug: return
        
        with torch.no_grad():
            self.logits = self.get_restype_from_logits(self.logits, opt)['pseudo']
    
        print(f"stage 2: exploration")
        exploration_stopped = False
        for stage_step, opt in enumerate(Annealer(
            soft=0, 
            e_soft=1,
            hard=0,
            e_hard=0,
            e_num_optimizing_binder_pos=8,
            iters=100,
        )):
            self.do_iter(
                boltz_model,
                opt,
                verbose=verbose,
                stage="exploration",
                stage_step=stage_step,
                trace_path=trace_path,
            )
            if should_early_stop("exploration"):
                exploration_stopped = True
                print(
                    "[early_stopping] exploration stopped; proceeding through "
                    "20 annealing iterations and 10 argmax iterations",
                    flush=True,
                )
                break

        with torch.no_grad():
            self.logits = 2 * self.logits

        annealing_iters = 20 if exploration_stopped else 100
        print(f"stage 3: annealing down ({annealing_iters} iterations)")
        for stage_step, opt in enumerate(Annealer(
            e_temp=0.01,
            hard=0,
            e_hard=0,
            num_optimizing_binder_pos=8,
            e_num_optimizing_binder_pos=12,
            iters=annealing_iters,
        )):
            self.do_iter(
                boltz_model,
                opt,
                verbose=verbose,
                stage="annealing_down",
                stage_step=stage_step,
                trace_path=trace_path,
            )
        
        print(f"stage 4: argmax")
        for stage_step, opt in enumerate(Annealer(
            temp=0.01,
            e_temp=0.01,
            num_optimizing_binder_pos=12,
            e_num_optimizing_binder_pos=16,
            iters=10,
        )):
            self.do_iter(
                boltz_model,
                opt,
                verbose=verbose,
                stage="argmax",
                stage_step=stage_step,
                trace_path=trace_path,
            )
            

    def do_lmpnn_redesign(self, boltz_model, design_dir, structs, num_seqs=1):
        print("Running LigandMPNN redesign...")
        t0 = time.perf_counter()
        
        try:
            motif_indices = self.fixed_mask.nonzero(as_tuple=True)[0].tolist()
        except Exception:
            motif_indices = None

        lmpnn_seqs, fasta_path, best_sample_idx_by_state = perform_tied_lmpnn_redesign(
            design_dir=design_dir,
            state_results=structs,
            num_seqs=num_seqs,
            motif_indices=motif_indices,
        )

        regen_dir = os.path.join(design_dir, "lmpnn", "boltz_regen")
        os.makedirs(regen_dir, exist_ok=True)

        for seq_idx, seq in enumerate(lmpnn_seqs):
            print(f"Regenerating structures for LigandMPNN seq {seq_idx}")
            regen_structs = self.get_final_structs(boltz_model, samples=5, set_seq=seq)
            mydesign_utils.save_structs(regen_structs, regen_dir, prefix=f"lmpnn_seq{seq_idx}_")

        t1 = time.perf_counter()
        print(f"LigandMPNN redesign completed in {t1 - t0:.1f}s")

        return lmpnn_seqs, fasta_path, regen_dir

    def do_nampnn_redesign(self, boltz_model, design_dir, structs, num_seqs=1):
        print("Running NA-MPNN redesign...")
        t0 = time.perf_counter()

        try:
            motif_indices = self.fixed_mask.nonzero(as_tuple=True)[0].tolist()
        except Exception:
            motif_indices = None

        nampnn_seqs, fasta_path, best_sample_idx_by_state = perform_tied_nampnn_redesign(
            design_dir=design_dir,
            state_results=structs,
            polymer_type=self.polymer_type,
            num_seqs=num_seqs,
            motif_indices=motif_indices,
        )

        regen_dir = os.path.join(design_dir, "nampnn", "boltz_regen")
        os.makedirs(regen_dir, exist_ok=True)

        for seq_idx, seq in enumerate(nampnn_seqs):
            print(f"Regenerating structures for NA-MPNN seq {seq_idx}")
            regen_structs = self.get_final_structs(boltz_model, samples=5, set_seq=seq)
            mydesign_utils.save_structs(regen_structs, regen_dir, prefix=f"nampnn_seq{seq_idx}_")

        t1 = time.perf_counter()
        print(f"NA-MPNN redesign completed in {t1 - t0:.1f}s")

        return nampnn_seqs, fasta_path, regen_dir


    def save_visualization_info(self, design_dir):
        save_path = os.path.join(design_dir, "visualization_info.pt")
        torch.save(
            {
                "pseudo_logit_traj": self.pseudo_logit_traj,
                "loss_log": self.loss_log,
                "final_seq": self.get_seq(),
                "num_iters": len(self.pseudo_logit_traj),
            },
            save_path,
        )

        print(f"Saved visualization information (logits, loss, etc.) to: {save_path}")
