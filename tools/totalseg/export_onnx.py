"""Export the TotalSegmentator "total" nnU-Net models (tasks 291-295) and a set of
finer subtasks to ONNX.

Each exported graph takes raw Hounsfield units, shape [N, 1, *patch] float32, with
axes (S, A, R) of a RAS-canonical volume at the model's spacing, and returns logits
[N, C, *patch]. The model's CT normalisation (clip + z-score) is baked in.

Writes <out_dir>/totalseg_<task>.onnx and <out_dir>/totalseg.json describing the
pipeline parameters and the label mapping into one combined label set: the "total"
labels (1-117) followed by the subtasks' new classes.

Usage: python export_onnx.py <out_dir> [--check] [--force]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from totalsegmentator.config import setup_nnunet
from totalsegmentator.map_tasks_config import TASK_CONFIGS
from totalsegmentator.map_to_binary import (class_map, class_map_5_parts, class_map_parts_headneck_muscles,
                                            map_taskid_to_partname_ct, map_taskid_to_partname_headneck_muscles)
from totalsegmentator.nnunet import get_output_folder  # noqa: F401  (registers custom trainers)

TASKS = [291, 292, 293, 294, 295]
TRAINER = "nnUNetTrainerNoMirroring"
STEP_SIZE = 0.8  # TotalSegmentator uses 0.8 for the "total" task
SPACING_MM = 1.5
SUBTASK_STEP_SIZE = 0.5  # and 0.5 for every other task
# TotalSegmentator replaces each task's crop_addon by 20 mm when the crop comes from
# its own rough segmentation (python_api.py), which is the case for all tasks here.
CROP_MARGIN_MM = 20.0

# Free subtasks, in merge order: where two claim a voxel the later one wins. Each
# "crop" lists the classes of "crop_source" (a task run before it) whose bounding
# box, widened by CROP_MARGIN_MM, the task runs on; None means the whole volume.
# Classes in "drop" are not merged ("head" is the whole head, skin included).
SUBTASKS = [
    # Only used to crop abdominal_muscles, as TotalSegmentator does.
    {"name": "body", "task_ids": [300], "trainer": "nnUNetTrainer", "model": "3d_fullres",
     "crop": None, "merge": False},
    {"name": "lung_vessels"},
    {"name": "abdominal_muscles", "crop_source": "body"},
    {"name": "headneck_muscles"},
    {"name": "craniofacial_structures", "drop": ["head"]},
    {"name": "headneck_bones_vessels"},
    {"name": "head_muscles"},
    {"name": "head_glands_cavities"},
]


class Normalized(torch.nn.Module):
    def __init__(self, net, lo, hi, mean, std):
        super().__init__()
        self.net = net
        self.lo, self.hi, self.mean, self.std = lo, hi, mean, std

    def forward(self, hu):
        x = (hu.clamp(self.lo, self.hi) - self.mean) / max(self.std, 1e-8)
        return self.net(x)


def load(task, trainer=TRAINER, model="3d_fullres"):
    predictor = nnUNetPredictor(tile_step_size=STEP_SIZE, use_gaussian=True, use_mirroring=False,
                                perform_everything_on_device=False, device=torch.device("cpu"))
    folder = get_output_folder(task, trainer, "nnUNetPlans", model)
    predictor.initialize_from_trained_model_folder(folder, use_folds=[0], checkpoint_name="checkpoint_final.pth")
    net = predictor.network
    net.load_state_dict(predictor.list_of_parameters[0])
    net.eval()
    cfg = predictor.configuration_manager
    props = predictor.plans_manager.foreground_intensity_properties_per_channel["0"]
    assert cfg.normalization_schemes == ["CTNormalization"], cfg.normalization_schemes
    norm = dict(lo=float(props["percentile_00_5"]), hi=float(props["percentile_99_5"]),
                mean=float(props["mean"]), std=float(props["std"]))
    return net, list(cfg.patch_size), [float(v) for v in cfg.spacing], norm


def export(task, out_dir, n_classes, check, force, **load_args):
    """Exports one model; returns its manifest entry (without label_map) and spacing (z, y, x)."""
    net, patch, spacing, norm = load(task, **load_args)
    wrapped = Normalized(net, **norm).eval()
    path = out_dir / f"totalseg_{task}.onnx"
    dummy = torch.full((1, 1, *patch), -1024.0)
    with torch.no_grad():
        out = wrapped(dummy)
    assert out.shape[1] == n_classes, (out.shape, n_classes)
    if force or not path.is_file():
        torch.onnx.export(wrapped, (dummy,), str(path), input_names=["hu"], output_names=["logits"],
                          dynamic_axes={"hu": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    print(f"{task}: {path.name} ({path.stat().st_size / 1e6:.0f} MB), {n_classes} classes, "
          f"spacing {spacing}, patch {patch}, norm {norm}", flush=True)

    if check:
        import onnxruntime as ort
        rng = np.random.default_rng(task)
        x = rng.uniform(-1024, 1500, (1, 1, *patch)).astype(np.float32)
        with torch.no_grad():
            ref = wrapped(torch.from_numpy(x)).numpy()
        sess = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        got = sess.run(None, {"hu": x})[0]
        agree = (ref.argmax(1) == got.argmax(1)).mean()
        print(f"  check: max |diff| {np.abs(ref - got).max():.2e}, argmax agreement {agree:.6f}", flush=True)

    entry = {
        "task": task,
        "file": path.name,
        "patch_size": patch,
        "num_classes": n_classes,
        "pad_hu": norm["mean"],  # nnU-Net pads the normalised image with 0
        "normalization": norm,
    }
    return entry, spacing


def subtask_parts(name, task_ids):
    """Per model: class index -> class name."""
    if len(task_ids) == 1:
        return [class_map[name]]
    assert name == "headneck_muscles", name
    return [class_map_parts_headneck_muscles[map_taskid_to_partname_headneck_muscles[t]] for t in task_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--check", action="store_true", help="compare ONNX Runtime output against PyTorch")
    ap.add_argument("--force", action="store_true", help="re-export models whose ONNX file exists")
    args = ap.parse_args()
    setup_nnunet()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = class_map["total"]
    total_inv = {name: idx for idx, name in total.items()}
    meta = {
        "spacing_mm": SPACING_MM,
        "step_size": STEP_SIZE,
        "gaussian_sigma_scale": 1.0 / 8,
        "input_axes": "SAR (RAS-canonical volume, transposed to z,y,x)",
        "labels": {str(k): v for k, v in total.items()},
        "models": [],
    }

    for task in TASKS:
        part = class_map_5_parts[map_taskid_to_partname_ct[task]]
        n_classes = max(part) + 1
        entry, spacing = export(task, args.out_dir, n_classes, args.check, args.force)
        assert spacing == [SPACING_MM] * 3, spacing
        # local class index -> combined "total" label (0 = background / unused)
        entry["label_map"] = [total_inv.get(part.get(i), 0) for i in range(n_classes)]
        meta["models"].append(entry)

    # Combined labels: a subtask class named like an existing label shares its value.
    combined = dict(total_inv)
    outputs = {"total": total_inv}  # task -> class name -> value in its output labels
    meta["subtasks"] = []
    for spec in SUBTASKS:
        name = spec["name"]
        cfg = TASK_CONFIGS[name]
        task_ids = spec.get("task_ids", cfg.get("task_id"))
        task_ids = task_ids if isinstance(task_ids, list) else [task_ids]
        merge = spec.get("merge", True)
        drop = set(spec.get("drop", []))
        crop = spec.get("crop", cfg.get("crop"))
        source = spec.get("crop_source", "total")

        names = {}  # class name -> value in this task's output labels
        for cls in class_map[name].values():
            if cls in drop:
                continue
            if merge:
                if cls not in combined:
                    combined[cls] = max(combined.values()) + 1
                names[cls] = combined[cls]
            else:
                names[cls] = len(names) + 1
        outputs[name] = names

        sub = {
            "name": name,
            "merge": merge,
            "step_size": SUBTASK_STEP_SIZE,
            "crop": None if crop is None else {
                "source": source,
                "labels": [outputs[source][c] for c in crop],
                "margin_mm": CROP_MARGIN_MM,
            },
            "labels": {str(v): k for k, v in names.items()},
            # Labels this task takes over from "total" inside its crop.
            "replaces": sorted(v for k, v in names.items() if merge and k in total_inv),
            "models": [],
        }
        spacings = []
        for tid, part in zip(task_ids, subtask_parts(name, task_ids)):
            n_classes = max(part) + 1
            entry, spacing = export(tid, args.out_dir, n_classes, args.check, args.force,
                                    trainer=spec.get("trainer") or cfg["trainer"],
                                    model=spec.get("model", cfg.get("model", "3d_fullres")))
            entry["label_map"] = [names.get(part.get(i), 0) for i in range(n_classes)]
            sub["models"].append(entry)
            spacings.append(spacing)
        assert all(s == spacings[0] for s in spacings), spacings
        resample = cfg.get("resample")
        if resample is not None and spec.get("task_ids") is None:
            # TASK_CONFIGS gives (x, y, z), nnU-Net plans (z, y, x).
            resample = resample if isinstance(resample, list) else [resample] * 3
            assert np.allclose(resample[::-1], spacings[0]), (name, resample, spacings[0])
        sub["spacing_mm"] = spacings[0][::-1]  # canonical R, A, S
        meta["subtasks"].append(sub)

    assert max(combined.values()) < 250, "combined labels collide with volview's derived labels (250+)"
    meta["labels"] = {str(v): k for k, v in sorted(combined.items(), key=lambda kv: kv[1])}
    (args.out_dir / "totalseg.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
