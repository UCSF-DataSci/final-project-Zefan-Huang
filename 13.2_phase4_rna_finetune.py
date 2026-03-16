"""
Stage 13.2 Phase 4 multimodal fine-tuning on the 130-case RNA subset.
"""
import argparse
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


DEFAULT_MANIFEST_CSV = ROOT / "output/patient_manifest.csv"
DEFAULT_STAGE11_PACK = ROOT / "output/stage11/11.2_graph_reasoning/graph_reasoning_pack.npz"
DEFAULT_LABELS_CSV = ROOT / "output/labels_time_zero.csv"
DEFAULT_PHASE3_MODEL = ROOT / "output/stage13/13.1_phase3_baseline/phase3_model/model/explanation_guided_model.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "output/stage13/13.2_phase4_rna_finetune"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 13.2 RNA-subset fine-tuning from Phase 3 baseline",
        allow_abbrev=False,
    )
    parser.add_argument("--manifest-csv", type=str, default=str(DEFAULT_MANIFEST_CSV))
    parser.add_argument("--stage11-pack", type=str, default=str(DEFAULT_STAGE11_PACK))
    parser.add_argument("--labels-csv", type=str, default=str(DEFAULT_LABELS_CSV))
    parser.add_argument("--phase3-model-path", type=str, default=str(DEFAULT_PHASE3_MODEL))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--freeze-prefixes", type=str, default="pool,base_trunk")
    parser.add_argument("--loss-weight-expl-rec", type=float, default=0.5)
    parser.add_argument("--loss-weight-expl-loc", type=float, default=0.5)
    parser.add_argument("--loss-weight-edge-prior", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="auto")
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    prepared_dir = output_root / "prepared"
    phase4_train_dir = output_root / "phase4_model"
    explanation_dir = output_root / "explanation_outputs"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    rna_patient_ids = PHASE_UTILS.select_patient_ids_by_flag(args.manifest_csv, "has_rnaseq")
    if not rna_patient_ids:
        raise RuntimeError("no RNA patients found in manifest")

    full_stage11_pack = PHASE_UTILS.load_npz(args.stage11_pack)
    subset_pack = PHASE_UTILS.subset_pack_by_patient_ids(full_stage11_pack, rna_patient_ids)
    subset_stage11_path = prepared_dir / "stage11_rna_subset_pack.npz"
    PHASE_UTILS.save_npz(subset_stage11_path, subset_pack)
    print(f"[phase4] wrote RNA subset stage11 pack: {subset_stage11_path}")

    PHASE_UTILS.run_python_script(
        "12.2_explanation_training.py",
        [
            "--stage11-pack",
            subset_stage11_path,
            "--labels-csv",
            args.labels_csv,
            "--output-root",
            phase4_train_dir,
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
            "--init-model-path",
            args.phase3_model_path,
            "--freeze-prefixes",
            args.freeze_prefixes,
            "--loss-weight-expl-rec",
            args.loss_weight_expl_rec,
            "--loss-weight-expl-loc",
            args.loss_weight_expl_loc,
            "--loss-weight-edge-prior",
            args.loss_weight_edge_prior,
        ],
    )
    PHASE_UTILS.run_python_script(
        "12.2_explanation_outputs.py",
        [
            "--graph-pack",
            phase4_train_dir / "pred/graph_reasoning_pack.npz",
            "--primary-pack",
            phase4_train_dir / "pred/primary_output_pack.npz",
            "--output-root",
            explanation_dir,
        ],
    )

    phase4_meta = PHASE_UTILS.read_json(phase4_train_dir / "train/explanation_meta.json")
    explanation_summary = PHASE_UTILS.read_json(explanation_dir / "explanation_summary.json")
    summary = {
        "phase": "phase4_rna_finetune_130",
        "subset_stage11_path": str(subset_stage11_path),
        "phase4_model_dir": str(phase4_train_dir),
        "explanation_dir": str(explanation_dir),
        "rna_patient_count": int(len(subset_pack["patient_ids"])),
        "init_model_path": str(args.phase3_model_path),
        "freeze_prefixes": [x.strip() for x in str(args.freeze_prefixes).split(",") if x.strip()],
        "loss_weight_expl_rec": float(args.loss_weight_expl_rec),
        "loss_weight_expl_loc": float(args.loss_weight_expl_loc),
        "loss_weight_edge_prior": float(args.loss_weight_edge_prior),
        "val_metrics": phase4_meta.get("val_metrics"),
        "val_explanation_metrics": phase4_meta.get("val_explanation_metrics"),
        "explanation_top1_path_frequency": explanation_summary.get("top1_path_frequency", []),
    }
    PHASE_UTILS.write_json(output_root / "phase4_summary.json", summary)
    print(f"wrote: {output_root / 'phase4_summary.json'}")
    print("complete")


if __name__ == "__main__":
    main()
