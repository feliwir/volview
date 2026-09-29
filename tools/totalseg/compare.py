"""Compare two TotalSegmentator label maps (e.g. reference Python vs. volview's Rust port).

Usage: python compare.py <reference.nii.gz> <candidate.nii.gz> [labels.json]
"""
import json
import sys

import nibabel as nib
import numpy as np


def main():
    ref_img, cand_img = nib.load(sys.argv[1]), nib.load(sys.argv[2])
    names = {}
    if len(sys.argv) > 3:
        names = {int(k): v for k, v in json.load(open(sys.argv[3]))["labels"].items()}
    assert ref_img.shape == cand_img.shape, (ref_img.shape, cand_img.shape)
    affine_diff = np.abs(ref_img.affine - cand_img.affine).max()
    print(f"shape {ref_img.shape}, max affine difference {affine_diff:.2e}")

    ref = np.asarray(ref_img.dataobj).astype(np.uint8)
    cand = np.asarray(cand_img.dataobj).astype(np.uint8)
    print(f"voxel agreement {(ref == cand).mean():.5f} (foreground: "
          f"{(ref == cand)[(ref > 0) | (cand > 0)].mean():.5f})")

    rows = []
    for label in sorted(set(np.unique(ref)) | set(np.unique(cand)) - {0}):
        a, b = ref == label, cand == label
        dice = 2 * (a & b).sum() / (a.sum() + b.sum())
        rows.append((dice, int(label), int(a.sum()), int(b.sum())))
    dices = np.array([r[0] for r in rows])
    print(f"{len(rows)} labels, Dice mean {dices.mean():.4f}, median {np.median(dices):.4f}, min {dices.min():.4f}")
    print("lowest:")
    for dice, label, na, nb in sorted(rows)[:10]:
        print(f"  {label:3d} {names.get(label, ''):32s} dice {dice:.4f}  voxels ref {na:9d} cand {nb:9d}")


if __name__ == "__main__":
    main()
