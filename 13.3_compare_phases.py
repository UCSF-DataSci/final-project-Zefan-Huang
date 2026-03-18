import argparse
import csv
import importlib.util
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


DEFAULT_PHASE3_META = ROOT / "output/stage13/13.1_phase3_baseline/phase3_model/train/explanation_meta.json"
DEFAULT_PHASE3_EXPL = ROOT / "output/stage13/13.1_phase3_baseline/explanation_outputs/explanation_summary.json"
DEFAULT_PHASE4_META = ROOT / "output/stage13/13.2_phase4_rna_finetune/phase4_model/train/explanation_meta.json"
DEFAULT_PHASE4_EXPL = ROOT / "output/stage13/13.2_phase4_rna_finetune/explanation_outputs/explanation_summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "output/stage13/13.3_compare_phases"


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_path_label(summary):
    freq = summary.get("top1_path_frequency", [])
    if not freq:
        return ""
    path_names = freq[0].get("path_names", [])
    return " -> ".join(str(x) for x in path_names)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 13.3 compare phase 3 and phase 4 results",
        allow_abbrev=False,
    )
    parser.add_argument("--phase3-meta", type=str, default=str(DEFAULT_PHASE3_META))
    parser.add_argument("--phase3-expl", type=str, default=str(DEFAULT_PHASE3_EXPL))
    parser.add_argument("--phase4-meta", type=str, default=str(DEFAULT_PHASE4_META))
    parser.add_argument("--phase4-expl", type=str, default=str(DEFAULT_PHASE4_EXPL))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    phase3_meta = PHASE_UTILS.read_json(args.phase3_meta)
    phase4_meta = PHASE_UTILS.read_json(args.phase4_meta)
    phase3_expl = PHASE_UTILS.read_json(args.phase3_expl)
    phase4_expl = PHASE_UTILS.read_json(args.phase4_expl)

    metric_rows = [
        {
            "metric": "val_c_index",
            "phase3": phase3_meta.get("val_metrics", {}).get("val_c_index"),
            "phase4": phase4_meta.get("val_metrics", {}).get("val_c_index"),
        },
        {
            "metric": "val_rec_auc",
            "phase3": phase3_meta.get("val_metrics", {}).get("val_rec_auc"),
            "phase4": phase4_meta.get("val_metrics", {}).get("val_rec_auc"),
        },
        {
            "metric": "val_loc_acc",
            "phase3": phase3_meta.get("val_metrics", {}).get("val_loc_acc"),
            "phase4": phase4_meta.get("val_metrics", {}).get("val_loc_acc"),
        },
        {
            "metric": "val_expl_rec_auc",
            "phase3": phase3_meta.get("val_explanation_metrics", {}).get("expl_rec_auc"),
            "phase4": phase4_meta.get("val_explanation_metrics", {}).get("expl_rec_auc"),
        },
        {
            "metric": "val_expl_loc_acc",
            "phase3": phase3_meta.get("val_explanation_metrics", {}).get("expl_loc_acc"),
            "phase4": phase4_meta.get("val_explanation_metrics", {}).get("expl_loc_acc"),
        },
        {
            "metric": "best_epoch",
            "phase3": phase3_meta.get("best_epoch"),
            "phase4": phase4_meta.get("best_epoch"),
        },
    ]
    explanation_rows = [
        {
            "metric": "top1_path",
            "phase3": top_path_label(phase3_expl),
            "phase4": top_path_label(phase4_expl),
        },
        {
            "metric": "top1_path_fraction",
            "phase3": (phase3_expl.get("top1_path_frequency", [{}])[0] or {}).get("fraction"),
            "phase4": (phase4_expl.get("top1_path_frequency", [{}])[0] or {}).get("fraction"),
        },
        {
            "metric": "organ_sus_min",
            "phase3": phase3_expl.get("ranges", {}).get("organ_susceptibility_min"),
            "phase4": phase4_expl.get("ranges", {}).get("organ_susceptibility_min"),
        },
        {
            "metric": "organ_sus_max",
            "phase3": phase3_expl.get("ranges", {}).get("organ_susceptibility_max"),
            "phase4": phase4_expl.get("ranges", {}).get("organ_susceptibility_max"),
        },
        {
            "metric": "edge_prob_min",
            "phase3": phase3_expl.get("ranges", {}).get("edge_diffusion_prob_min"),
            "phase4": phase4_expl.get("ranges", {}).get("edge_diffusion_prob_min"),
        },
        {
            "metric": "edge_prob_max",
            "phase3": phase3_expl.get("ranges", {}).get("edge_diffusion_prob_max"),
            "phase4": phase4_expl.get("ranges", {}).get("edge_diffusion_prob_max"),
        },
    ]

    write_csv(output_root / "phase_metric_comparison.csv", ["metric", "phase3", "phase4"], metric_rows)
    write_csv(output_root / "phase_explanation_comparison.csv", ["metric", "phase3", "phase4"], explanation_rows)

    summary = {
        "phase3_meta_path": str(args.phase3_meta),
        "phase4_meta_path": str(args.phase4_meta),
        "phase3_explanation_summary_path": str(args.phase3_expl),
        "phase4_explanation_summary_path": str(args.phase4_expl),
        "metric_rows": metric_rows,
        "explanation_rows": explanation_rows,
    }
    PHASE_UTILS.write_json(output_root / "phase_comparison.json", summary)
    print(f"wrote: {output_root / 'phase_metric_comparison.csv'}")
    print(f"wrote: {output_root / 'phase_explanation_comparison.csv'}")
    print(f"wrote: {output_root / 'phase_comparison.json'}")
    print("complete")


if __name__ == "__main__":
    main()
