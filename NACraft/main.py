import pickle, os, shutil, sys, yaml
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_BOLTZ_SRC = _MODULE_DIR / "boltz" / "src"
for _path in (_MODULE_DIR, _BOLTZ_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
from dataclasses import asdict

# Two compatibility fixes for torch.load on the boltz1_conf.ckpt checkpoint:
#  (1) PyTorch 2.6+ defaults weights_only=True; the checkpoint bundles many
#      omegaconf internal classes that aren't on the safe-globals list.
#  (2) On some hosts torch.load(..., map_location="cuda") gets SIGTERM'd mid
#      load (likely a resource manager reacting to the GPU-side allocation).
#      Loading to CPU first, then letting lightning move the state_dict, is
#      robust. We trust this checkpoint.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    if isinstance(kwargs.get("map_location"), str) and "cuda" in kwargs["map_location"]:
        kwargs["map_location"] = "cpu"
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
import time
import argparse
from utils import motif_utils, mydesign_utils, na_constants
from utils.model_assets import resolve_boltz_asset
from designer import MultistateDesigner
from losses import (
    MotifLoss, AntiMotifLoss, ContactLoss, HelixBiasLoss,
    ConfChangeLoss, LigandContactLoss, AntiLigandContactLoss,
    SheetBiasLoss, SequenceSimilarityLoss, RadiusOfGyrationLoss,
)

device = "cuda"


def _rna_sequence_from_pdb(path):
    """Recover RNA chain A from a saved Boltz PDB."""
    residues = {}
    with open(path) as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or line[21] != "A":
                continue
            name = line[17:20].strip().upper()
            if name not in {"A", "C", "G", "U"}:
                continue
            residues.setdefault((line[22:26], line[26]), name)
    if not residues:
        raise RuntimeError(f"no RNA chain A residues found in {path}")
    return "".join(residues.values())


def _init_boltz(recycles=0):
    predict_args = {
        "recycling_steps": recycles,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": True,
    }
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = 1.638
    checkpoint_path = resolve_boltz_asset("boltz1_conf.ckpt")
    model = Boltz1.load_from_checkpoint(
        str(checkpoint_path),
        strict=False,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        structure_prediction_training=True,
        no_msa=False,
        no_atom_encoder=False,
    ).to(device).eval().requires_grad_(False)
    model_parallel_devices = os.environ.get("NACRAFT_BOLTZ_DEVICES")
    if model_parallel_devices:
        names = tuple(name.strip() for name in model_parallel_devices.split(",") if name.strip())
        if len(names) not in (2, 3, 4, 5):
            raise ValueError(
                "NACRAFT_BOLTZ_DEVICES must contain two to five devices, "
                "e.g. cuda:0,cuda:1,cuda:2,cuda:3,cuda:4"
            )
        from experiments.OGA.scripts.multigpu_boltz import dispatch_oga_boltz

        model = dispatch_oga_boltz(model, names)
    return model


def _build_loss(loss_cfg, motif_templates, polymer_type="protein"):
    t = loss_cfg["type"]
    if t in ("MotifLoss", "AntiMotifLoss"):
        motif = motif_templates[loss_cfg["motif"]]
        return MotifLoss(motif) if t == "MotifLoss" else AntiMotifLoss(motif)
    if t == "ConfChangeLoss":
        return ConfChangeLoss(strength=loss_cfg.get("strength", 1.0), polymer_type=polymer_type)
    if t in ("LigandContactLoss", "AntiLigandContactLoss"):
        cls = LigandContactLoss if t == "LigandContactLoss" else AntiLigandContactLoss
        return cls(idx=loss_cfg.get("idx"), strength=loss_cfg.get("strength", 1.0), polymer_type=polymer_type)
    if t == "HelixBiasLoss":
        # Skip for NA — no secondary structure bias
        if polymer_type != "protein":
            return None
        return HelixBiasLoss(strength=loss_cfg.get("strength", 0.0))
    if t == "SheetBiasLoss":
        # Skip for NA — no beta-sheet equivalent
        if polymer_type != "protein":
            return None
        return SheetBiasLoss(strength=loss_cfg.get("strength", 0.0))
    if t == "RadiusOfGyrationLoss":
        return RadiusOfGyrationLoss(strength=loss_cfg.get("strength", 1.0))
    if t == "SequenceSimilarityLoss":
        return SequenceSimilarityLoss(loss_cfg["target_sequence"], strength=loss_cfg.get("strength", 1.0), polymer_type=polymer_type)
    if t == "ContactLoss":
        return ContactLoss(polymer_type=polymer_type)
    raise ValueError(f"Unknown loss type: {t!r}")


def build_designer(
    config,
    length_override=None,
    visualize=False,
    init_seq=None,
    device="cuda",
):
    num_states = config["num_states"]
    polymer_type = config.get("polymer_type", "protein")
    predictor = config.get("predictor", "boltz")
    if predictor not in {"boltz", "af3"}:
        raise ValueError(
            f"Unknown predictor: {predictor!r}; expected 'boltz' or 'af3'. "
            "Sequence optimization always uses Boltz-1 distogram gradients."
        )
    af3_cfg = config.get("af3", {})
    designer = MultistateDesigner(
        num_states=num_states,
        visualize=visualize,
        polymer_type=polymer_type,
        predictor=predictor,
        af3_cfg=af3_cfg,
    )

    motif_names = config.get("motifs", [])
    if motif_names:
        motif_templates = motif_utils.get_motif_scaffold_templates(
            paths=[f"motifs/{m}.pdb" for m in motif_names],
            target_length=length_override or config.get("length"),
            polymer_type=polymer_type,
        )
        length = len(motif_templates[0]["motif_mask"])
    else:
        motif_templates = []
        length = length_override or config.get("length")
        if length is None:
            raise ValueError("No motifs specified and no length set — provide 'length' in the config or pass --length")

    for motif in motif_templates:
        designer.add_motif(motif)

    for i, state_cfg in enumerate(config.get("states", [])):
        if i >= num_states:
            break
        for lig_str in (state_cfg or []):
            mol_type, value = lig_str.split(":", 1)
            designer.add_ligand((value, mol_type), state=i)

    for loss_cfg in config.get("losses", []):
        loss = _build_loss(loss_cfg, motif_templates, polymer_type=polymer_type)
        if loss is None:
            continue  # skip losses that don't apply (e.g., SS bias for NA)
        designer.add_loss(loss, state=loss_cfg.get("state", 0), weight=loss_cfg.get("weight", 1.0))

    contact_loss = config.get("contact_loss", True)
    designer.initialize(
        length=length,
        contact_loss=contact_loss,
        init_seq=init_seq,
        device=device,
    )
    return designer, motif_names, motif_templates


def run(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Optional presearch MSA injection — same pattern as rescore_af3.py.
    # Loads index.json keyed by protein sequence and threads it through
    # designer.presearch_lookup so both Boltz design + final-prediction
    # paths see real MSAs for target protein chains.
    if getattr(args, "presearch_json", None):
        from utils.af3_utils import load_presearch
        presearch_lookup = load_presearch(
            args.presearch_json, out_dir=args.presearch_outdir)
        config.setdefault("af3", {})["presearch_lookup"] = presearch_lookup
        print(f"[presearch] loaded {len(presearch_lookup)} entries from "
              f"{args.presearch_json}", flush=True)

    motif_names = config.get("motifs", [])
    out_dir = os.path.join(args.outpath, "_".join(motif_names)) if motif_names else args.outpath
    os.makedirs(out_dir, exist_ok=True)

    config_name = os.path.splitext(os.path.basename(args.config))[0]

    for design in range(args.worker_id, args.num_designs, args.num_workers):
        print(f"\nStarting {config_name} design {design+1}/{args.num_designs}")

        if args.skip_existing:
            design_dir_check = os.path.join(out_dir, f"design{design}")
            sentinel = os.path.join(design_dir_check, "state0_sample0.cif")
            if os.path.exists(sentinel):
                print(f"  SKIP design{design} (already has {sentinel})")
                continue

        boltz_model = _init_boltz(args.recycles)
        init_seq = args.init_seq or config.get("init_seq")
        if args.resume_oga_post_early_stop:
            prior_pdb = os.path.join(out_dir, f"design{design}", "state0_sample0.pdb")
            init_seq = _rna_sequence_from_pdb(prior_pdb)
            prior_trace = os.path.join(out_dir, f"design{design}", "optimization_trace.tsv")
            archived_trace = os.path.join(
                out_dir, f"design{design}", "optimization_trace_pre_continuation.tsv"
            )
            if os.path.exists(prior_trace) and not os.path.exists(archived_trace):
                shutil.copy2(prior_trace, archived_trace)
            print(f"  continuing design{design} from saved RNA sequence {init_seq}")
        designer, motif_names, motif_templates = build_designer(
            config, length_override=args.length, visualize=args.visualize,
            init_seq=init_seq,
        )

        design_dir = os.path.join(out_dir, f"design{design}")
        os.makedirs(design_dir, exist_ok=True)

        t0 = time.perf_counter()
        print("Optimizing sequence...")
        designer.optimize(
            boltz_model,
            verbose=args.verbose,
            debug=args.debug,
            trace_dir=design_dir,
            early_stopping=args.early_stopping,
            post_early_stop_only=args.resume_oga_post_early_stop,
        )
        print(f"Optimization done in {time.perf_counter() - t0:.1f} sec")

        for i, motif_name in enumerate(motif_names):
            with open(os.path.join(design_dir, f"{motif_name}_spec.pkl"), "wb") as f:
                pickle.dump(motif_templates[i], f)

        print("Saving structures...")
        t2 = time.perf_counter()
        if designer.predictor == "af3":
            structs = designer.get_final_structs_af3(
                output_root=os.path.join(out_dir, f"design{design}_af3raw"),
                name_prefix=f"{config_name}_design{design}",
                save_dir=design_dir,
            )
        elif designer.predictor == "boltz":
            structs = designer.get_final_structs(boltz_model)
            mydesign_utils.save_structs(structs, design_dir)
        else:
            raise ValueError(
                f"Unknown predictor: {designer.predictor!r} "
                "(expected 'boltz' or 'af3')"
            )
        print(f"Structure generation took {time.perf_counter() - t2:.1f} sec")

        if args.ligandmpnn_seqs > 0:
            if designer.polymer_type in ("rna", "dna"):
                designer.do_nampnn_redesign(boltz_model, design_dir, structs, num_seqs=args.ligandmpnn_seqs)
            else:
                designer.do_lmpnn_redesign(boltz_model, design_dir, structs, num_seqs=args.ligandmpnn_seqs)

        print(f"Finished design {design+1} in {time.perf_counter() - t0:.1f} sec total")

        if args.visualize:
            designer.save_visualization_info(design_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a NACraft YAML design config (see NACraft/experiments/).",
    )
    parser.add_argument("--num_designs", type=int, default=1)
    parser.add_argument("-o", "--outpath", type=str, default="./out/")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--recycles", default=0, type=int)
    parser.add_argument("--length", default=None, type=int, help="Override design length from config")
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--worker_id", default=0, type=int)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip designs whose output dir already contains state0_sample0.cif")
    parser.add_argument("--ligandmpnn_seqs", default=0, type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--presearch-json", default=None,
                        help="Path to af3_presearch/out/index.json. Injects real MSA "
                             "into target protein chains during design + final prediction.")
    parser.add_argument("--presearch-outdir", default=None,
                        help="Override path prefix for presearch files (defaults to "
                             "the index.json's parent dir).")
    parser.add_argument("--init-seq", default=None,
                        help="Seed sequence used to initialize relaxed logits. Length must "
                             "match the design length; used by similarity-guided design.")
    parser.add_argument(
        "--early_stopping", "--early-stopping", action="store_true",
        help=("During exploration only, stop after 10 within-stage or 30 "
              "global non-improving iterations, then run 20 annealing and "
              "10 argmax iterations."),
    )
    parser.add_argument(
        "--resume-oga-post-early-stop", action="store_true",
        help=("Recover RNA from an existing state0_sample0.pdb and run only "
              "20 annealing plus 10 argmax iterations."),
    )
    args = parser.parse_args()
    run(args)
