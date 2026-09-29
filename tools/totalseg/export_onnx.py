"""Export the TotalSegmentator "total" nnU-Net models (tasks 291-295) to ONNX.

Each exported graph takes raw Hounsfield units, shape [N, 1, 128, 128, 128] float32,
with axes (S, A, R) of a RAS-canonical 1.5 mm volume, and returns logits
[N, C, 128, 128, 128]. The model's CT normalisation (clip + z-score) is baked in.

Writes <out_dir>/totalseg_<task>.onnx and <out_dir>/totalseg.json describing the
pipeline parameters and the label mapping into the combined "total" label set.

Usage: python export_onnx.py <out_dir> [--check]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from totalsegmentator.config import setup_nnunet
from totalsegmentator.map_to_binary import class_map, class_map_5_parts, map_taskid_to_partname_ct
from totalsegmentator.nnunet import get_output_folder  # noqa: F401  (registers custom trainers)

TASKS = [291, 292, 293, 294, 295]
TRAINER = "nnUNetTrainerNoMirroring"
STEP_SIZE = 0.8  # TotalSegmentator uses 0.8 for the "total" task
SPACING_MM = 1.5


class Normalized(torch.nn.Module):
    def __init__(self, net, lo, hi, mean, std):
        super().__init__()
        self.net = net
        self.lo, self.hi, self.mean, self.std = lo, hi, mean, std

    def forward(self, hu):
        x = (hu.clamp(self.lo, self.hi) - self.mean) / max(self.std, 1e-8)
        return self.net(x)


def load(task):
    predictor = nnUNetPredictor(tile_step_size=STEP_SIZE, use_gaussian=True, use_mirroring=False,
                                perform_everything_on_device=False, device=torch.device("cpu"))
    folder = get_output_folder(task, TRAINER, "nnUNetPlans", "3d_fullres")
    predictor.initialize_from_trained_model_folder(folder, use_folds=[0], checkpoint_name="checkpoint_final.pth")
    net = predictor.network
    net.load_state_dict(predictor.list_of_parameters[0])
    net.eval()
    cfg = predictor.configuration_manager
    props = predictor.plans_manager.foreground_intensity_properties_per_channel["0"]
    assert cfg.normalization_schemes == ["CTNormalization"], cfg.normalization_schemes
    assert list(cfg.spacing) == [SPACING_MM] * 3, cfg.spacing
    norm = dict(lo=float(props["percentile_00_5"]), hi=float(props["percentile_99_5"]),
                mean=float(props["mean"]), std=float(props["std"]))
    return net, list(cfg.patch_size), norm, predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--check", action="store_true", help="compare ONNX Runtime output against PyTorch")
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
        net, patch, norm, _ = load(task)
        part = class_map_5_parts[map_taskid_to_partname_ct[task]]
        n_classes = max(part) + 1
        wrapped = Normalized(net, **norm).eval()
        path = args.out_dir / f"totalseg_{task}.onnx"
        dummy = torch.full((1, 1, *patch), -1024.0)
        with torch.no_grad():
            out = wrapped(dummy)
        assert out.shape[1] == n_classes, (out.shape, n_classes)
        torch.onnx.export(wrapped, (dummy,), str(path), input_names=["hu"], output_names=["logits"],
                          dynamic_axes={"hu": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
        print(f"{task}: {path.name} ({path.stat().st_size / 1e6:.0f} MB), {n_classes} classes, norm {norm}", flush=True)

        meta["models"].append({
            "task": task,
            "file": path.name,
            "patch_size": patch,
            "num_classes": n_classes,
            "pad_hu": norm["mean"],  # nnU-Net pads the normalised image with 0
            "normalization": norm,
            # local class index -> combined "total" label (0 = background / unused)
            "label_map": [total_inv.get(part.get(i), 0) for i in range(n_classes)],
        })

        if args.check:
            import onnxruntime as ort
            rng = np.random.default_rng(task)
            x = rng.uniform(-1024, 1500, (1, 1, *patch)).astype(np.float32)
            with torch.no_grad():
                ref = wrapped(torch.from_numpy(x)).numpy()
            sess = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            got = sess.run(None, {"hu": x})[0]
            agree = (ref.argmax(1) == got.argmax(1)).mean()
            print(f"  check: max |diff| {np.abs(ref - got).max():.2e}, argmax agreement {agree:.6f}", flush=True)

    (args.out_dir / "totalseg.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
