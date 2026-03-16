"""
Conservative Phase 4 tuning for Stage 13.
"""
import argparse
import csv
import importlib.util
import json
import math
import shutil
from pathlib import Path


def resolve_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


ROOT = resolve_root()


def load_local_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE_UTILS = load_local_module(ROOT / "13.0_phase_utils.py", "stage13_phase_utils")


DEFAULT_OUTPUT_ROOT = ROOT / "output/stage13/13.2_phase4_tune"
DEFAULT_PHASE3_META = ROOT / "output/stage13/13.1_phase3_baseline/phase3_model/train/explanation_meta.json"
DEFAULT_PHASE3_EXPL = ROOT / "output/stage13/13.1_phase3_baseline/explanation_outputs/explanation_summary.json"


def safe_float(value, default=-1e18):
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out):
        return float(default)
    return out


def selection_key(row):
    return (
        safe_float(row.get("val_rec_auc")),
        safe_float(row.get("val_c_index")),
        safe_float(row.get("val_loc_acc")),
        -safe_float(row.get("top1_path_fraction"), default=1e18),
    )


def parse_top1(summary):
    freq = summary.get("explanation_top1_path_frequency", [])
    if not freq:
        return "", None
    top = freq[0]
    path_names = top.get("path_names", [])
    return " -> ".join(str(x) for x in path_names), top.get("fraction")


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_phase4(output_root, config, shared_args):
    args = [
        "--manifest-csv",
        shared_args.manifest_csv,
        "--stage11-pack",
        shared_args.stage11_pack,
        "--labels-csv",
        shared_args.labels_csv,
        "--phase3-model-path",
        shared_args.phase3_model_path,
        "--output-root",
        output_root,
        "--epochs",
        shared_args.epochs,
        "--weight-decay",
        shared_args.weight_decay,
        "--val-ratio",
        shared_args.val_ratio,
        "--device",
        shared_args.device,
        "--seed",
        config["seed"],
        "--lr",
        config["lr"],
        "--freeze-prefixes",
        config["freeze_prefixes"],
        "--loss-weight-expl-rec",
        config["loss_weight_expl_rec"],
        "--loss-weight-expl-loc",
        config["loss_weight_expl_loc"],
        "--loss-weight-edge-prior",
        config["loss_weight_edge_prior"],
    ]
    PHASE_UTILS.run_python_script("13.2_phase4_rna_finetune.py", args)
    phase4_summary = PHASE_UTILS.read_json(Path(output_root) / "phase4_summary.json")
    top1_path, top1_path_fraction = parse_top1(phase4_summary)
    row = {
        "run_root": str(output_root),
        "stage": config["stage"],
        "run_id": config["run_id"],
        "freeze_prefixes": config["freeze_prefixes"],
        "lr": float(config["lr"]),
        "loss_weight_expl_rec": float(config["loss_weight_expl_rec"]),
        "loss_weight_expl_loc": float(config["loss_weight_expl_loc"]),
        "loss_weight_edge_prior": float(config["loss_weight_edge_prior"]),
        "seed": int(config["seed"]),
        "val_c_index": phase4_summary.get("val_metrics", {}).get("val_c_index"),
        "val_rec_auc": phase4_summary.get("val_metrics", {}).get("val_rec_auc"),
        "val_loc_acc": phase4_summary.get("val_metrics", {}).get("val_loc_acc"),
        "val_expl_rec_auc": phase4_summary.get("val_explanation_metrics", {}).get("expl_rec_auc"),
        "val_expl_loc_acc": phase4_summary.get("val_explanation_metrics", {}).get("expl_loc_acc"),
        "top1_path": top1_path,
        "top1_path_fraction": top1_path_fraction,
    }
    return row


def choose_best(rows):
    if not rows:
        raise RuntimeError("no rows to choose from")
    return sorted(rows, key=selection_key, reverse=True)[0]


def copy_best_artifacts(best_run_root, output_root):
    best_run_root = Path(best_run_root)
    output_root = Path(output_root)
    model_src = best_run_root / "phase4_model"
    expl_src = best_run_root / "explanation_outputs"
    model_dst = output_root / "phase4_model_best"
    expl_dst = output_root / "explanation_outputs_best"
    shutil.copytree(model_src, model_dst, dirs_exist_ok=True)
    shutil.copytree(expl_src, expl_dst, dirs_exist_ok=True)
    shutil.copy2(best_run_root / "phase4_summary.json", output_root / "phase4_summary_best.json")
    return {
        "model_dst": str(model_dst),
        "expl_dst": str(expl_dst),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune Phase 4 conservatively within the current Stage 13 setup",
        allow_abbrev=False,
    )
    parser.add_argument("--manifest-csv", type=str, default=str(ROOT / "output/patient_manifest.csv"))
    parser.add_argument("--stage11-pack", type=str, default=str(ROOT / "output/stage11/11.2_graph_reasoning/graph_reasoning_pack.npz"))
    parser.add_argument("--labels-csv", type=str, default=str(ROOT / "output/labels_time_zero.csv"))
    parser.add_argument("--phase3-model-path", type=str, default=str(ROOT / "output/stage13/13.1_phase3_baseline/phase3_model/model/explanation_guided_model.pt"))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    stage_a_configs = []
    run_index = 1
    for freeze_prefixes in ["pool,base_trunk", "pool", ""]:
        for lr in [1e-4, 3e-4]:
            stage_a_configs.append(
                {
                    "stage": "A",
                    "run_id": f"A{run_index:02d}",
                    "freeze_prefixes": freeze_prefixes,
                    "lr": lr,
                    "loss_weight_expl_rec": 0.5,
                    "loss_weight_expl_loc": 0.5,
                    "loss_weight_edge_prior": 0.05,
                    "seed": 2024,
                }
            )
            run_index += 1

    stage_a_rows = []
    for config in stage_a_configs:
        run_root = output_root / "stage_a" / config["run_id"]
        stage_a_rows.append(run_phase4(run_root, config, args))
        all_rows.append(stage_a_rows[-1])
    best_stage_a = choose_best(stage_a_rows)

    stage_b_configs = []
    run_index = 1
    for loss_weight_edge_prior in [0.05, 0.02, 0.0]:
        for loss_weight_expl_loc in [0.5, 0.25]:
            stage_b_configs.append(
                {
                    "stage": "B",
                    "run_id": f"B{run_index:02d}",
                    "freeze_prefixes": best_stage_a["freeze_prefixes"],
                    "lr": best_stage_a["lr"],
                    "loss_weight_expl_rec": 0.5,
                    "loss_weight_expl_loc": loss_weight_expl_loc,
                    "loss_weight_edge_prior": loss_weight_edge_prior,
                    "seed": 2024,
                }
            )
            run_index += 1

    stage_b_rows = []
    for config in stage_b_configs:
        run_root = output_root / "stage_b" / config["run_id"]
        stage_b_rows.append(run_phase4(run_root, config, args))
        all_rows.append(stage_b_rows[-1])
    best_stage_b = choose_best(stage_b_rows)

    stage_c_rows = []
    for seed in [2024, 2025, 2026]:
        config = {
            "stage": "C",
            "run_id": f"C_seed{seed}",
            "freeze_prefixes": best_stage_b["freeze_prefixes"],
            "lr": best_stage_b["lr"],
            "loss_weight_expl_rec": best_stage_b["loss_weight_expl_rec"],
            "loss_weight_expl_loc": best_stage_b["loss_weight_expl_loc"],
            "loss_weight_edge_prior": best_stage_b["loss_weight_edge_prior"],
            "seed": seed,
        }
        run_root = output_root / "stage_c" / config["run_id"]
        stage_c_rows.append(run_phase4(run_root, config, args))
        all_rows.append(stage_c_rows[-1])

    best_stage_c = choose_best(stage_c_rows)
    copied = copy_best_artifacts(best_stage_c["run_root"], output_root)

    final_refit_root = output_root / "final_refit"
    PHASE_UTILS.run_python_script(
        "13.2_phase4_rna_finetune.py",
        [
            "--manifest-csv",
            args.manifest_csv,
            "--stage11-pack",
            args.stage11_pack,
            "--labels-csv",
            args.labels_csv,
            "--phase3-model-path",
            args.phase3_model_path,
            "--output-root",
            final_refit_root,
            "--epochs",
            args.epochs,
            "--weight-decay",
            args.weight_decay,
            "--val-ratio",
            0.0,
            "--device",
            args.device,
            "--seed",
            2024,
            "--lr",
            best_stage_b["lr"],
            "--freeze-prefixes",
            best_stage_b["freeze_prefixes"],
            "--loss-weight-expl-rec",
            best_stage_b["loss_weight_expl_rec"],
            "--loss-weight-expl-loc",
            best_stage_b["loss_weight_expl_loc"],
            "--loss-weight-edge-prior",
            best_stage_b["loss_weight_edge_prior"],
        ],
    )

    PHASE_UTILS.run_python_script(
        "13.3_compare_phases.py",
        [
            "--phase3-meta",
            DEFAULT_PHASE3_META,
            "--phase3-expl",
            DEFAULT_PHASE3_EXPL,
            "--phase4-meta",
            Path(copied["model_dst"]) / "train/explanation_meta.json",
            "--phase4-expl",
            Path(copied["expl_dst"]) / "explanation_summary.json",
            "--output-root",
            output_root / "compare_best",
        ],
    )

    stage_c_metric_mean = {
        "val_rec_auc_mean": sum(safe_float(row["val_rec_auc"], default=0.0) for row in stage_c_rows) / len(stage_c_rows),
        "val_c_index_mean": sum(safe_float(row["val_c_index"], default=0.0) for row in stage_c_rows) / len(stage_c_rows),
        "val_loc_acc_mean": sum(safe_float(row["val_loc_acc"], default=0.0) for row in stage_c_rows) / len(stage_c_rows),
    }

    fieldnames = [
        "stage",
        "run_id",
        "freeze_prefixes",
        "lr",
        "loss_weight_expl_rec",
        "loss_weight_expl_loc",
        "loss_weight_edge_prior",
        "seed",
        "val_c_index",
        "val_rec_auc",
        "val_loc_acc",
        "val_expl_rec_auc",
        "val_expl_loc_acc",
        "top1_path",
        "top1_path_fraction",
        "run_root",
    ]
    write_csv(output_root / "tuning_runs.csv", fieldnames, all_rows)

    summary = {
        "stage_a_best": best_stage_a,
        "stage_b_best": best_stage_b,
        "stage_c_best_seed": best_stage_c,
        "stage_c_metric_mean": stage_c_metric_mean,
        "best_artifacts": copied,
        "final_refit_root": str(final_refit_root),
    }
    PHASE_UTILS.write_json(output_root / "best_config.json", {
        "freeze_prefixes": best_stage_b["freeze_prefixes"],
        "lr": best_stage_b["lr"],
        "loss_weight_expl_rec": best_stage_b["loss_weight_expl_rec"],
        "loss_weight_expl_loc": best_stage_b["loss_weight_expl_loc"],
        "loss_weight_edge_prior": best_stage_b["loss_weight_edge_prior"],
    })
    PHASE_UTILS.write_json(output_root / "tuning_summary.json", summary)
    print(f"wrote: {output_root / 'tuning_runs.csv'}")
    print(f"wrote: {output_root / 'best_config.json'}")
    print(f"wrote: {output_root / 'tuning_summary.json'}")
    print("complete")


if __name__ == "__main__":
    main()
