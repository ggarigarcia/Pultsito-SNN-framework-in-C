#!/usr/bin/env python3
"""
Preprocess GIFTI files + FreeSurfer labels → binary files for C SNN framework.

Usage:
    ./preprocess_gifti.py <gifti_file> <label1> <label2> ... <out_prefix>

Example:
    ./preprocess_gifti.py sub-01_bold.func.gii \\
        lh.V1.label lh.V2.label rh.V1.label rh.V2.label \\
        test/datasets/gifti/parcellation

Outputs:
    <out_prefix>.parc     → parcellation binary (vertex indices per parcel)
    <out_prefix>.bold     → parcel-averaged BOLD (float32 [n_parcels × n_timepoints])
"""

import sys
import struct
import numpy as np

def read_label(path):
    """Read FreeSurfer .label file, return array of vertex indices."""
    with open(path) as f:
        lines = f.readlines()
    # header: first line starts with '#!ascii', second line is count
    # remaining: vertex_index  x  y  z  value
    verts = [int(line.split()[0]) for line in lines[2:]]
    return np.array(verts, dtype=np.uint64)

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    gifti_path = sys.argv[1]
    label_paths = sys.argv[2:-1]
    out_prefix = sys.argv[-1]

    print(f"Reading GIFTI: {gifti_path}")
    import nibabel as nb
    gifti = nb.load(gifti_path).agg_data()  # (n_vertices, n_timepoints)
    n_vertices, n_timepoints = gifti.shape
    print(f"  shape: {gifti.shape}  dtype: {gifti.dtype}")

    # Read labels
    parcel_names = []
    parcel_vertices = []
    for lp in label_paths:
        name = lp.rsplit("/", 1)[-1].replace(".label", "")
        verts = read_label(lp)
        parcel_names.append(name)
        parcel_vertices.append(verts)
        print(f"  label {name}: {len(verts)} vertices")

    n_parcels = len(parcel_vertices)
    all_vertices = np.concatenate(parcel_vertices).astype(np.uint64)
    n_vert_counts = np.array([len(v) for v in parcel_vertices], dtype=np.uint64)

    # Compute per-parcel averaged BOLD
    bold = np.zeros((n_parcels, n_timepoints), dtype=np.float32)
    for i, verts in enumerate(parcel_vertices):
        bold[i, :] = gifti[verts, :].mean(axis=0)

    # --- Write parcellation binary ---
    parc_path = f"{out_prefix}.parc"
    with open(parc_path, "wb") as f:
        f.write(struct.pack("QQQ", n_parcels, len(all_vertices), n_timepoints))
        f.write(n_vert_counts.tobytes())
        f.write(all_vertices.tobytes())
    print(f"  wrote {parc_path}")

    # --- Write parcel BOLD binary ---
    bold_path = f"{out_prefix}.bold"
    with open(bold_path, "wb") as f:
        f.write(bold.tobytes())
    print(f"  wrote {bold_path}  ({bold.nbytes} bytes)")

    print("Done.")

if __name__ == "__main__":
    main()
