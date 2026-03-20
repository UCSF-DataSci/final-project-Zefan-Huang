import argparse
import importlib.util
from pathlib import Path

## all the samples

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
STAGE10_MOD = load_local_module(ROOT / "10.1_multimodal_fusion.py", "stage10_multimodal_fusion")
STAGE11_1_MOD = load_local_module(ROOT / "11.1_graph_construction.py", "stage11_graph_construction")
STAGE11_2_MOD = load_local_module(ROOT / "11.2_graph_reasoning.py", "stage11_graph_reasoning")
TRAIN_MOD = load_local_module(ROOT / "12.2_explanation_training.py", "stage12_explanation_training")


DEFAULT_STAGE9_PACK = ROOT / "output/stage9/9.1_organ_tokenization/organ_tokenization_pack.npz"
DEFAULT_STAGE9_MODULE = ROOT / "9.2_organ_query.py"
DEFAULT_LABELS_CSV = ROOT / "output/labels_time_zero.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "output/stage13/13.1_phase3_baseline"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 13.1 baseline training on 211 cases without RNA",
        allow_abbrev=False,
    )
    parser.add_argument("--stage9-pack", type=str, default=str(DEFAULT_STAGE9_PACK))
    parser.add_argument("--stage9-module", type=str, default=str(DEFAULT_STAGE9_MODULE))
    parser.add_argument("--labels-csv", type=str, default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="auto")
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    prepared_dir = output_root / "prepared"
    stage10_dir = output_root / "stage10_no_rna"
    stage11_graph_dir = output_root / "stage11_graph_construction"
    stage11_reason_dir = output_root / "stage11_graph_reasoning"
    phase3_train_dir = output_root / "phase3_model"
    explanation_dir = output_root / "explanation_outputs"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    stage9_pack = PHASE_UTILS.load_npz(args.stage9_pack)
    no_rna_pack = PHASE_UTILS.disable_rna_modalities(stage9_pack)
    stage9_no_rna_path = prepared_dir / "organ_tokenization_pack_no_rna.npz"
    PHASE_UTILS.save_npz(stage9_no_rna_path, no_rna_pack)

    print(f"[phase3] wrote no-RNA stage9 pack: {stage9_no_rna_path}")
    stage10_result = STAGE10_MOD.run_stage10_fusion(
        stage9_pack_path=stage9_no_rna_path,
        stage9_module_path=Path(args.stage9_module),
        output_root=stage10_dir,
        device=args.device,
        seed=args.seed,
    )
    stage11_graph_result = STAGE11_1_MOD.run_stage11_graph_construction(
        stage10_npz_path=stage10_result["npz_path"],
        output_root=stage11_graph_dir,
    )
    stage11_reason_result = STAGE11_2_MOD.run_stage11_graph_reasoning(
        stage11_pack_path=stage11_graph_result["pack_path"],
        output_root=stage11_reason_dir,
        device=args.device,
        seed=args.seed,
    )

    PHASE_UTILS.run_python_script(
        "12.2_explanation_training.py",
        [
            "--stage11-pack",
            stage11_reason_result["pack_path"],
            "--labels-csv",
            args.labels_csv,
            "--output-root",
            phase3_train_dir,
            "--epochs",
            args.epochs,
            "--lr",
            args.lr,
            "--weight-decay",
            args.weight_decay,
            "--val-ratio",
            args.val_ratio,
            "--seed",
            args.seed,
            "--device",
            args.device,
        ],
    )
    PHASE_UTILS.run_python_script(
        "12.2_explanation_outputs.py",
        [
            "--graph-pack",
            phase3_train_dir / "pred/graph_reasoning_pack.npz",
            "--primary-pack",
            phase3_train_dir / "pred/primary_output_pack.npz",
            "--output-root",
            explanation_dir,
        ],
    )

    phase3_meta = PHASE_UTILS.read_json(phase3_train_dir / "train/explanation_meta.json")
    explanation_summary = PHASE_UTILS.read_json(explanation_dir / "explanation_summary.json")
    summary = {
        "phase": "phase3_baseline_211_no_rna",
        "stage9_no_rna_path": str(stage9_no_rna_path),
        "stage10_dir": str(stage10_dir),
        "stage11_graph_dir": str(stage11_graph_dir),
        "stage11_reason_dir": str(stage11_reason_dir),
        "phase3_model_dir": str(phase3_train_dir),
        "explanation_dir": str(explanation_dir),
        "patient_count": int(len(no_rna_pack["patient_ids"])),
        "rna_disabled_for_all": True,
        "immune_disabled_for_all": True,
        "val_metrics": phase3_meta.get("val_metrics"),
        "val_explanation_metrics": phase3_meta.get("val_explanation_metrics"),
        "explanation_top1_path_frequency": explanation_summary.get("top1_path_frequency", []),
    }
    PHASE_UTILS.write_json(output_root / "phase3_summary.json", summary)
    print(f"wrote: {output_root / 'phase3_summary.json'}")
    print("complete")


if __name__ == "__main__":
    main()
