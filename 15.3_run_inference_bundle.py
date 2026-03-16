"""
Stage 15.3 unified entrypoint: case inputs -> system outputs deliverable bundle.
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path


def resolve_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


ROOT = resolve_root()
DEFAULT_OUTPUT_ROOT = ROOT / "output/stage15/15.3_run_inference_bundle"
DEFAULT_CASE_INPUT_ROOT = DEFAULT_OUTPUT_ROOT / "case_inputs"
DEFAULT_SYSTEM_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "system_outputs"
PRIMARY_PREDICTION_CANDIDATES = [
    ROOT / "output/stage13/13.2_phase4_tune/phase4_model_best/pred/patient_primary_predictions.csv",
    ROOT / "output/stage12/12.2_explanation_training/pred/patient_primary_predictions.csv",
    ROOT / "output/stage12/12.1_primary_outputs_refit_all_seed2024/pred/patient_primary_predictions.csv",
]
EXPLANATION_ROOT_CANDIDATES = [
    ROOT / "output/stage13/13.2_phase4_tune/explanation_outputs_best",
    ROOT / "output/stage12/12.2_explanation_outputs_joint",
    ROOT / "output/stage12/12.2_explanation_outputs",
]


def load_local_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE_UTILS = load_local_module(ROOT / "13.0_phase_utils.py", "stage15_phase_utils")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Stage 15.1 and Stage 15.2 as a single deliverable-bundle entrypoint",
        allow_abbrev=False,
    )
    parser.add_argument("--patient-id", type=str, default="")
    parser.add_argument("--ct-path", type=str, default="")
    parser.add_argument("--pet-path", type=str, default="")
    parser.add_argument("--tumor-seg-path", type=str, default="")
    parser.add_argument("--aim-path", type=str, default="")
    parser.add_argument("--clinical-csv", type=str, default="")
    parser.add_argument("--clinical-json", type=str, default="")
    parser.add_argument("--clinical-row-id", type=str, default="")
    parser.add_argument("--clinical-id-column", type=str, default="")
    parser.add_argument("--rna-path", type=str, default="")
    parser.add_argument("--disable-internal-lookup", action="store_true")
    parser.add_argument("--force-external-inference", action="store_true")
    parser.add_argument("--model-strategy", type=str, default="auto", choices=["auto", "phase3", "phase4"])
    parser.add_argument("--rna-transform", type=str, default="raw", choices=["raw", "log1p", "zscore"])
    parser.add_argument("--organ-seg-run-tag", type=str, default="search_base24")
    parser.add_argument("--organ-seg-model-path", type=str, default="")
    parser.add_argument("--allow-legacy-model-fallback", action="store_true")
    parser.add_argument("--phase3-model-path", type=str, default=str(ROOT / "output/stage13/13.1_phase3_baseline/phase3_model/model/explanation_guided_model.pt"))
    parser.add_argument("--phase4-model-path", type=str, default=str(ROOT / "output/stage13/13.2_phase4_tune/phase4_model_best/model/explanation_guided_model.pt"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--explanation-root", type=str, default="")
    parser.add_argument("--primary-predictions-csv", type=str, default="")
    parser.add_argument("--attention-npz", type=str, default=str(ROOT / "output/stage10/10.1_multimodal_fusion/fused_organ_tokens.npz"))
    parser.add_argument("--attention-summary-json", type=str, default=str(ROOT / "output/stage10/10.1_multimodal_fusion/fusion_summary.json"))
    parser.add_argument("--visualization-root", type=str, default=str(ROOT / "output/stage13/13.4_visualize_diffusion"))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    args, _unknown = parser.parse_known_args(argv)
    return args


def apply_overrides(args, overrides):
    for key, value in overrides.items():
        setattr(args, str(key).replace("-", "_"), value)
    return args


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_first_existing(raw_path, candidates, label):
    if normalize_text(raw_path):
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no default {label} found")


def patient_exists_in_csv(csv_path, patient_id):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if normalize_text(row.get("patient_id")) == patient_id:
                return True
    return False


def patient_exists_in_explanation_root(explanation_root, patient_id):
    manifest_path = Path(explanation_root) / "patient_explanation_manifest.csv"
    if not manifest_path.exists():
        return False
    return patient_exists_in_csv(manifest_path, patient_id)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_bundle_index_html(patient_id, case_input_rel, report_rel):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Inference Bundle: {patient_id}</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      background: linear-gradient(180deg, #f4efe5 0%, #ede4d2 100%);
      color: #2c2318;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .panel {{
      background: rgba(255, 253, 248, 0.96);
      border: 1px solid #d7ccb5;
      border-radius: 22px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 10px 28px rgba(77, 59, 31, 0.08);
    }}
    a {{
      color: #20406d;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>Inference Bundle: {patient_id}</h1>
    <p>This entrypoint ran Stage 15.1 case-input packaging and Stage 15.2 system-output export for one patient.</p>
  </div>
  <div class="panel">
    <h2>Outputs</h2>
    <ul>
      <li><a href="{case_input_rel}">Case inputs summary</a></li>
      <li><a href="{report_rel}">System outputs report</a></li>
    </ul>
  </div>
</body>
</html>
"""


def print_usage_hint():
    print("patient_id is required")
    print("CLI usage: python 15.3_run_inference_bundle.py --patient-id R01-003")
    print("Python Console usage: run_inference_bundle(patient_id='R01-003')")


def main(argv=None, **overrides):
    args = apply_overrides(parse_args(argv), overrides)
    patient_id = normalize_text(args.patient_id)
    if not patient_id:
        raise RuntimeError(
            "patient_id is required. In Python Console, call "
            "run_inference_bundle(patient_id='R01-003')."
        )

    output_root = ensure_output_dir(args.output_root)
    case_input_root = ensure_output_dir(Path(output_root) / "case_inputs")
    system_output_root = ensure_output_dir(Path(output_root) / "system_outputs")

    if args.force_external_inference:
        PHASE_UTILS.run_python_script(
            "15.4_external_case_inference.py",
            [
                "--patient-id",
                patient_id,
                "--ct-path",
                args.ct_path,
                "--pet-path",
                args.pet_path,
                "--tumor-seg-path",
                args.tumor_seg_path,
                "--aim-path",
                args.aim_path,
                "--clinical-csv",
                args.clinical_csv,
                "--clinical-json",
                args.clinical_json,
                "--clinical-row-id",
                args.clinical_row_id,
                "--clinical-id-column",
                args.clinical_id_column,
                "--rna-path",
                args.rna_path,
                "--output-root",
                output_root,
                "--model-strategy",
                args.model_strategy,
                "--rna-transform",
                args.rna_transform,
                "--organ-seg-run-tag",
                args.organ_seg_run_tag,
                "--organ-seg-model-path",
                args.organ_seg_model_path,
                "--phase3-model-path",
                args.phase3_model_path,
                "--phase4-model-path",
                args.phase4_model_path,
                "--device",
                args.device,
                *(["--disable-internal-lookup"] if args.disable_internal_lookup else []),
                *(["--force-external-inference"] if args.force_external_inference else []),
                *(["--allow-legacy-model-fallback"] if args.allow_legacy_model_fallback else []),
            ],
        )
        return {
            "patient_id": patient_id,
            "output_root": str(output_root),
            "bundle_summary_json": str(output_root / "bundle_summary.json"),
            "index_html": str(output_root / "index.html"),
            "mode": "external_inference",
        }

    try:
        explanation_root = resolve_first_existing(args.explanation_root, EXPLANATION_ROOT_CANDIDATES, "explanation root")
        primary_csv = resolve_first_existing(args.primary_predictions_csv, PRIMARY_PREDICTION_CANDIDATES, "primary prediction csv")
    except FileNotFoundError:
        explanation_root = None
        primary_csv = None

    PHASE_UTILS.run_python_script(
        "15.1_case_inputs.py",
        [
            "--patient-id",
            patient_id,
            "--ct-path",
            args.ct_path,
            "--pet-path",
            args.pet_path,
            "--tumor-seg-path",
            args.tumor_seg_path,
            "--aim-path",
            args.aim_path,
            "--clinical-csv",
            args.clinical_csv,
            "--clinical-json",
            args.clinical_json,
            "--clinical-row-id",
            args.clinical_row_id,
            "--clinical-id-column",
            args.clinical_id_column,
            "--rna-path",
            args.rna_path,
            "--output-root",
            case_input_root,
            *(["--disable-internal-lookup"] if args.disable_internal_lookup else []),
        ],
    )

    case_summary_path = case_input_root / patient_id / "case_input_summary.json"
    case_summary = PHASE_UTILS.read_json(case_summary_path)
    if not bool(case_summary.get("ready_for_current_pipeline")):
        missing_required = case_summary.get("missing_required_inputs", [])
        raise RuntimeError(
            "case inputs are not ready for the current pipeline; missing required inputs: "
            + ",".join(str(x) for x in missing_required)
        )

    has_existing_outputs = (
        explanation_root is not None
        and primary_csv is not None
        and patient_exists_in_explanation_root(explanation_root, patient_id)
        and patient_exists_in_csv(primary_csv, patient_id)
    )
    if not has_existing_outputs:
        PHASE_UTILS.run_python_script(
            "15.4_external_case_inference.py",
            [
                "--patient-id",
                patient_id,
                "--ct-path",
                args.ct_path,
                "--pet-path",
                args.pet_path,
                "--tumor-seg-path",
                args.tumor_seg_path,
                "--aim-path",
                args.aim_path,
                "--clinical-csv",
                args.clinical_csv,
                "--clinical-json",
                args.clinical_json,
                "--clinical-row-id",
                args.clinical_row_id,
                "--clinical-id-column",
                args.clinical_id_column,
                "--rna-path",
                args.rna_path,
                "--output-root",
                output_root,
                "--model-strategy",
                args.model_strategy,
                "--rna-transform",
                args.rna_transform,
                "--organ-seg-run-tag",
                args.organ_seg_run_tag,
                "--organ-seg-model-path",
                args.organ_seg_model_path,
                "--phase3-model-path",
                args.phase3_model_path,
                "--phase4-model-path",
                args.phase4_model_path,
                "--device",
                args.device,
                *(["--disable-internal-lookup"] if args.disable_internal_lookup else []),
                *(["--allow-legacy-model-fallback"] if args.allow_legacy_model_fallback else []),
            ],
        )
        return {
            "patient_id": patient_id,
            "output_root": str(output_root),
            "bundle_summary_json": str(output_root / "bundle_summary.json"),
            "index_html": str(output_root / "index.html"),
            "mode": "external_inference",
        }

    PHASE_UTILS.run_python_script(
        "15.2_system_outputs.py",
        [
            "--explanation-root",
            explanation_root,
            "--primary-predictions-csv",
            primary_csv,
            "--attention-npz",
            args.attention_npz,
            "--attention-summary-json",
            args.attention_summary_json,
            "--visualization-root",
            args.visualization_root,
            "--case-input-root",
            case_input_root,
            "--output-root",
            system_output_root,
            "--patient-ids",
            patient_id,
        ],
    )

    report_rel = Path("system_outputs") / "cases" / patient_id / "report.html"
    case_input_rel = Path("case_inputs") / patient_id / "case_input_summary.json"
    bundle_summary = {
        "patient_id": patient_id,
        "output_root": str(output_root),
        "case_input_summary": str(case_summary_path),
        "system_output_report": str(system_output_root / "cases" / patient_id / "report.html"),
        "system_output_manifest": str(system_output_root / "system_output_manifest.csv"),
        "explanation_root": str(explanation_root),
        "primary_predictions_csv": str(primary_csv),
    }
    write_json(output_root / "bundle_summary.json", bundle_summary)
    (output_root / "index.html").write_text(
        build_bundle_index_html(patient_id, str(case_input_rel.as_posix()), str(report_rel.as_posix())),
        encoding="utf-8",
    )

    print(f"wrote: {output_root / 'bundle_summary.json'}")
    print(f"wrote: {output_root / 'index.html'}")
    print("complete")
    return {
        "patient_id": patient_id,
        "output_root": str(output_root),
        "bundle_summary_json": str(output_root / "bundle_summary.json"),
        "index_html": str(output_root / "index.html"),
        "mode": "existing_outputs_bundle",
    }


def run_inference_bundle(
    *,
    patient_id,
    ct_path="",
    pet_path="",
    tumor_seg_path="",
    aim_path="",
    clinical_csv="",
    clinical_json="",
    clinical_row_id="",
    clinical_id_column="",
    rna_path="",
    disable_internal_lookup=False,
    force_external_inference=False,
    model_strategy="auto",
    rna_transform="raw",
    organ_seg_run_tag="search_base24",
    organ_seg_model_path="",
    allow_legacy_model_fallback=False,
    phase3_model_path=str(ROOT / "output/stage13/13.1_phase3_baseline/phase3_model/model/explanation_guided_model.pt"),
    phase4_model_path=str(ROOT / "output/stage13/13.2_phase4_tune/phase4_model_best/model/explanation_guided_model.pt"),
    device="auto",
    explanation_root="",
    primary_predictions_csv="",
    attention_npz=str(ROOT / "output/stage10/10.1_multimodal_fusion/fused_organ_tokens.npz"),
    attention_summary_json=str(ROOT / "output/stage10/10.1_multimodal_fusion/fusion_summary.json"),
    visualization_root=str(ROOT / "output/stage13/13.4_visualize_diffusion"),
    output_root=str(DEFAULT_OUTPUT_ROOT),
):
    return main(
        patient_id=patient_id,
        ct_path=ct_path,
        pet_path=pet_path,
        tumor_seg_path=tumor_seg_path,
        aim_path=aim_path,
        clinical_csv=clinical_csv,
        clinical_json=clinical_json,
        clinical_row_id=clinical_row_id,
        clinical_id_column=clinical_id_column,
        rna_path=rna_path,
        disable_internal_lookup=disable_internal_lookup,
        force_external_inference=force_external_inference,
        model_strategy=model_strategy,
        rna_transform=rna_transform,
        organ_seg_run_tag=organ_seg_run_tag,
        organ_seg_model_path=organ_seg_model_path,
        allow_legacy_model_fallback=allow_legacy_model_fallback,
        phase3_model_path=phase3_model_path,
        phase4_model_path=phase4_model_path,
        device=device,
        explanation_root=explanation_root,
        primary_predictions_csv=primary_predictions_csv,
        attention_npz=attention_npz,
        attention_summary_json=attention_summary_json,
        visualization_root=visualization_root,
        output_root=output_root,
    )


def cli_main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not list(argv):
        print_usage_hint()
        return None
    return main(argv=argv)


if __name__ == "__main__":
    cli_main()
