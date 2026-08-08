import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
from boltz.data.write.pdb import to_pdb

from dataclasses import asdict


from utils.mydesign_utils import save_structs
from utils import motif_utils, na_constants
from main import build_designer
from designer import get_batch_with_ligands
import copy
import yaml



device = "cuda"


def init_boltz(recycles=0):
    predict_args = {
        "recycling_steps": recycles,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    diffusion_params = BoltzDiffusionParams()
    boltz_model = Boltz1.load_from_checkpoint(
        "boltz/boltz1_conf.ckpt",
        strict=False,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        structure_prediction_training=True,
        no_msa=False,
        no_atom_encoder=False,
    ).eval().requires_grad_(False)

    return boltz_model


def rebuild_designer(config_path, length_override=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    designer, _, _ = build_designer(config, length_override=length_override)
    return designer



def get_trajectory_structures(boltz_model, designer, pseudo, traj_dir, idx, final_coords):
    os.makedirs(traj_dir, exist_ok=True)

    pseudo = pseudo.to(device)
    L = pseudo.shape[0]

    for state, ligs in enumerate(designer.ligands):

        pdb_file = os.path.join(traj_dir, f"vis_state{state}.pdb")
        ref_coords = final_coords[state]

        dummy_seq = "X" * L if designer.polymer_type == "protein" else "N" * L
        batch, template_struct = get_batch_with_ligands(
            dummy_seq, ligs, device=device, polymer_type=designer.polymer_type
        )

        batch["res_type"][0, :L, :] = pseudo
        batch["msa"] = batch["res_type"].unsqueeze(0)
        batch["profile"] = batch["msa"].float().mean(0)

        pred = boltz_model.predict_step(batch, 0, 0)
        coords = pred["coords"][0].cpu().numpy()

        n = len(template_struct.atoms)
        aligned = align_points(coords[:n], ref_coords[:n])

        struct_copy = copy.deepcopy(template_struct)
        struct_copy.atoms["coords"] = aligned

        with open(pdb_file, "a") as f:
            f.write(f"MODEL     {idx}\n")
            f.write(to_pdb(struct_copy))
            f.write("ENDMDL\n\n")

        print(f"[state {state}] aligned + appended MODEL {idx} → {pdb_file}")

        
        

def np_kabsch(a, b, return_v=False):
    '''Get alignment matrix for two sets of coordinates using numpy
    
    Args:
        a: First set of coordinates
        b: Second set of coordinates
        return_v: If True, return U matrix from SVD. If False, return rotation matrix
        
    Returns:
        Rotation matrix (or U matrix if return_v=True) to align coordinates
    '''
    # Calculate covariance matrix
    ab = np.swapaxes(a, -1, -2) @ b
    
    # Singular value decomposition
    u, s, vh = np.linalg.svd(ab, full_matrices=False)
    
    # Handle reflection case
    flip = np.linalg.det(u @ vh) < 0
    if flip:
        u[...,-1] = -u[...,-1]
    
    return u if return_v else (u @ vh)


def align_points(a, b):
    a_centroid = a.mean(axis=0)
    b_centroid = b.mean(axis=0)

    a_centered = a - a_centroid
    b_centered = b - b_centroid

    R = np_kabsch(a_centered, b_centered)
    a_aligned = a_centered @ R + b_centroid
    return a_aligned


def plot_losses(loss_log, outdir):
    total_loss = np.array([sum(x_i[-1] for _,_,x_i in loss_dict) 
                           for loss_dict in loss_log])

    plt.figure(figsize=(6, 4))
    plt.plot(total_loss)
    plt.title("Total Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()
    

def visualize(args):
    traj_path = os.path.join(args.design_dir, "visualization_info.pt")
    traj = torch.load(traj_path)

    pseudo_traj = traj["pseudo_logit_traj"]
    loss_log = traj["loss_log"]

    out_vis_dir = os.path.join(args.design_dir, "trajectory_vis")
    os.makedirs(out_vis_dir, exist_ok=True)

    # plot_losses(loss_log, out_vis_dir)

    boltz_model = init_boltz()

    designer = rebuild_designer(args.config, length_override=args.length)

    out0 = os.path.join(out_vis_dir, "vis_state0.pdb")
    out1 = os.path.join(out_vis_dir, "vis_state1.pdb")

    # already_done = max(num_frames(out0), num_frames(out1))
    # start_idx = already_done
    start_idx = 0

    indices = range(start_idx, len(pseudo_traj))

    print("Refolding frames:", indices)
    
    for idx in indices:
        get_trajectory_structures(
            boltz_model,
            designer,
            pseudo_traj[idx],
            out_vis_dir,
            idx,
        )


    # Determine redesign output path based on polymer type
    redesign_dir = "nampnn" if designer.polymer_type in ("rna", "dna") else "lmpnn"
    final0 = f"{args.design_dir}/{redesign_dir}/boltz_regen/{redesign_dir}_seq1_state0_sample0.pdb"
    final1 = f"{args.design_dir}/{redesign_dir}/boltz_regen/{redesign_dir}_seq1_state1_sample0.pdb"


    final_idx = len(pseudo_traj)

    def extract_atoms(path):
        with open(path, "r") as f:
            return "".join([line for line in f if line.startswith(("ATOM", "HETATM"))])

    final0_str = extract_atoms(final0)
    final1_str = extract_atoms(final1)

    with open(out0, "a") as f:
        f.write(f"MODEL     {final_idx}\n")
        f.write(final0_str)
        f.write("ENDMDL\n\n")

    with open(out1, "a") as f:
        f.write(f"MODEL     {final_idx}\n")
        f.write(final1_str)
        f.write("ENDMDL\n\n")

    print(f"Appended final structures as MODEL {final_idx}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--design_dir", required=True)
    parser.add_argument("--config", required=True, help="Path to YAML design config used for this run")
    parser.add_argument("--length", type=int, default=None, help="Override design length from config")
    parser.add_argument("--frames", nargs="*", default=["0", "last"])

    args = parser.parse_args()
    visualize(args)