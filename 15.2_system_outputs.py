"""
Stage 15.2 system-output export for inference and deliverables.
"""
import argparse
import csv
import html
import json
import shutil
from pathlib import Path

try:
    import numpy as np
except Exception:
    np = None


def resolve_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


ROOT = resolve_root()
DEFAULT_OUTPUT_ROOT = ROOT / "output/stage15/15.2_system_outputs"
DEFAULT_CASE_INPUT_ROOT = ROOT / "output/stage15/15.1_case_inputs"
DEFAULT_VIS_ROOT = ROOT / "output/stage13/13.4_visualize_diffusion"
DEFAULT_STAGE10_NPZ = ROOT / "output/stage10/10.1_multimodal_fusion/fused_organ_tokens.npz"
DEFAULT_STAGE10_SUMMARY = ROOT / "output/stage10/10.1_multimodal_fusion/fusion_summary.json"

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

EVIDENCE_GROUPS = {
    "t_tumor": "tumor_segmentation",
    "t_sem": "semantics",
    "g_rna": "rna",
    "g_ehr": "clinical",
    "t_imm": "immune",
}
DISPLAY_NAME_MAP = {
    "LymphNodeMediastinum": "LN Mediastinum",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export Stage 15.2 primary and explanation deliverables",
        allow_abbrev=False,
    )
    parser.add_argument("--explanation-root", type=str, default="")
    parser.add_argument("--primary-predictions-csv", type=str, default="")
    parser.add_argument("--attention-npz", type=str, default=str(DEFAULT_STAGE10_NPZ))
    parser.add_argument("--attention-summary-json", type=str, default=str(DEFAULT_STAGE10_SUMMARY))
    parser.add_argument("--visualization-root", type=str, default=str(DEFAULT_VIS_ROOT))
    parser.add_argument("--case-input-root", type=str, default=str(DEFAULT_CASE_INPUT_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--patient-ids", nargs="*", default=[])
    parser.add_argument("--max-patients", type=int, default=0)
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


def read_csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def load_json_optional(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def display_name(name):
    return DISPLAY_NAME_MAP.get(str(name), str(name))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def load_attention_payload(npz_path):
    if np is None:
        return None
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=True) as z:
        return {
            "patient_ids": [str(x) for x in z["patient_ids"].tolist()],
            "organ_node_names": [str(x) for x in z["organ_node_names"].tolist()],
            "evidence_token_names": [str(x) for x in z["evidence_token_names"].tolist()],
            "attn_weights": np.asarray(z["attn_weights"], dtype=np.float32),
        }


def classify_evidence_token(token_name):
    if token_name in EVIDENCE_GROUPS:
        return EVIDENCE_GROUPS[token_name]
    if token_name.startswith("t_img_"):
        return "imaging"
    return "other"


def build_attention_rows(attention_payload, patient_id):
    if not attention_payload:
        return []
    try:
        patient_idx = attention_payload["patient_ids"].index(patient_id)
    except ValueError:
        return []

    organ_names = attention_payload["organ_node_names"]
    token_names = attention_payload["evidence_token_names"]
    weights = attention_payload["attn_weights"][patient_idx]
    rows = []
    for organ_idx, organ_name in enumerate(organ_names):
        organ_weights = weights[organ_idx]
        ranked_idx = np.argsort(-organ_weights)
        rank_lookup = {int(token_idx): rank + 1 for rank, token_idx in enumerate(ranked_idx.tolist())}
        for token_idx, token_name in enumerate(token_names):
            rows.append(
                {
                    "patient_id": patient_id,
                    "organ_name": organ_name,
                    "evidence_token_name": token_name,
                    "evidence_group": classify_evidence_token(token_name),
                    "attention_weight": float(organ_weights[token_idx]),
                    "rank_within_organ": int(rank_lookup[token_idx]),
                }
            )
    rows.sort(key=lambda row: (row["organ_name"], row["rank_within_organ"]))
    return rows


def build_attention_summary(attention_rows):
    grouped = {}
    for row in attention_rows:
        organ_name = row["organ_name"]
        grouped.setdefault(organ_name, []).append(row)
    summary = {}
    for organ_name, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["attention_weight"], reverse=True)
        group_sum = {}
        for row in rows:
            group_name = row["evidence_group"]
            group_sum[group_name] = group_sum.get(group_name, 0.0) + float(row["attention_weight"])
        group_ranked = [
            {"evidence_group": name, "attention_weight_sum": value}
            for name, value in sorted(group_sum.items(), key=lambda item: item[1], reverse=True)
        ]
        summary[organ_name] = {
            "top_tokens": [
                {
                    "evidence_token_name": row["evidence_token_name"],
                    "evidence_group": row["evidence_group"],
                    "attention_weight": float(row["attention_weight"]),
                    "rank_within_organ": int(row["rank_within_organ"]),
                }
                for row in rows[:5]
            ],
            "group_attention": group_ranked,
        }
    return summary


def group_rows_by_patient(rows, key="patient_id"):
    out = {}
    for row in rows:
        patient_id = normalize_text(row.get(key))
        out.setdefault(patient_id, []).append(row)
    return out


def build_edge_matrix_rows(patient_id, edge_rows):
    organ_names = []
    for row in edge_rows:
        src_name = normalize_text(row.get("src_name"))
        dst_name = normalize_text(row.get("dst_name"))
        if src_name not in organ_names:
            organ_names.append(src_name)
        if dst_name not in organ_names:
            organ_names.append(dst_name)
    matrix = {src_name: {dst_name: "" for dst_name in organ_names} for src_name in organ_names}
    for row in edge_rows:
        matrix[row["src_name"]][row["dst_name"]] = safe_float(row["edge_diffusion_prob"])
    out = []
    for src_name in organ_names:
        row = {"patient_id": patient_id, "src_name": src_name}
        for dst_name in organ_names:
            value = matrix[src_name][dst_name]
            row[dst_name] = "" if value == "" else float(value)
        out.append(row)
    return organ_names, out


def parse_primary_csv_row(row):
    out = {
        "patient_id": normalize_text(row.get("patient_id")),
        "split": normalize_text(row.get("split")),
        "os_label_known": normalize_text(row.get("os_label_known")),
        "time_os_days": normalize_text(row.get("time_os_days")),
        "event_os": normalize_text(row.get("event_os")),
        "rec_label_known": normalize_text(row.get("rec_label_known")),
        "time_rec_days": normalize_text(row.get("time_rec_days")),
        "event_rec": normalize_text(row.get("event_rec")),
        "rec_location_known": normalize_text(row.get("rec_location_known")),
        "rec_location_target": normalize_text(row.get("rec_location_target")),
        "recurrence_probability": safe_float(row.get("recurrence_probability")),
    }
    hazard_json = normalize_text(row.get("hazard_prob_json"))
    survival_json = normalize_text(row.get("survival_curve_json"))
    out["hazard_probability"] = json.loads(hazard_json) if hazard_json else []
    out["survival_curve"] = json.loads(survival_json) if survival_json else []
    loc_probs = {}
    for key, value in row.items():
        if key.startswith("rec_location_prob__"):
            loc_probs[key.split("__", 1)[1]] = safe_float(value)
    out["recurrence_location_probability"] = loc_probs
    if loc_probs:
        out["predicted_recurrence_location"] = max(loc_probs.items(), key=lambda item: item[1])[0]
    else:
        out["predicted_recurrence_location"] = ""
    if out["survival_curve"]:
        out["survival_probability_last_bin"] = float(out["survival_curve"][-1])
        out["survival_mode"] = "discrete"
    else:
        out["survival_probability_last_bin"] = None
        out["survival_mode"] = "cox"
    return out


def build_case_report_html(
    *,
    patient_id,
    primary_outputs,
    explanation_payload,
    attention_summary,
    top_edges,
    top_paths,
    availability_summary,
    stage10_note,
    svg_name,
):
    recurrence_probs = explanation_payload["primary_outputs"].get("recurrence_location_probability", {})
    prob_lines = "".join(
        f"<li>{html.escape(name)}: {safe_float(value):.4f}</li>"
        for name, value in recurrence_probs.items()
    )
    organ_lines = "".join(
        f"<tr><td>{html.escape(display_name(row['organ_name']))}</td><td>{int(row['rank'])}</td><td>{safe_float(row['susceptibility']):.4f}</td></tr>"
        for row in explanation_payload["organ_susceptibility_ranked"]
    )
    edge_lines = "".join(
        f"<tr><td>{html.escape(display_name(row['src_name']))}</td><td>{html.escape(display_name(row['dst_name']))}</td>"
        f"<td>{html.escape(str(row['edge_type']))}</td><td>{safe_float(row['edge_diffusion_prob']):.4f}</td></tr>"
        for row in top_edges[:10]
    )
    path_lines = "".join(
        f"<tr><td>{html.escape(' -> '.join(display_name(x) for x in row.get('path_names', [])))}</td>"
        f"<td>{int(row.get('num_hops', 0))}</td><td>{safe_float(row.get('score_prob', 0.0)):.4f}</td></tr>"
        for row in top_paths
    )

    attention_blocks = []
    for organ_name, payload in sorted(attention_summary.items()):
        token_lines = "".join(
            f"<li>{html.escape(row['evidence_token_name'])} ({html.escape(row['evidence_group'])}): {safe_float(row['attention_weight']):.4f}</li>"
            for row in payload.get("top_tokens", [])[:5]
        )
        group_lines = "".join(
            f"<li>{html.escape(row['evidence_group'])}: {safe_float(row['attention_weight_sum']):.4f}</li>"
            for row in payload.get("group_attention", [])[:5]
        )
        attention_blocks.append(
            "<div class=\"card\">"
            f"<h4>{html.escape(display_name(organ_name))}</h4>"
            "<div class=\"cols\">"
            f"<div><strong>Top tokens</strong><ul>{token_lines}</ul></div>"
            f"<div><strong>Grouped attention</strong><ul>{group_lines}</ul></div>"
            "</div></div>"
        )

    availability_lines = ""
    if availability_summary:
        availability_lines = "".join(
            f"<li>{html.escape(key)}: {html.escape(str(value))}</li>"
            for key, value in availability_summary.get("availability", {}).items()
        )

    svg_block = ""
    if svg_name:
        svg_block = f'<div class="panel"><h2>Directional Diffusion</h2><img src="{html.escape(svg_name)}" alt="Primary diffusion"/></div>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>System Outputs: {html.escape(patient_id)}</title>
  <style>
    body {{
      margin: 0;
      padding: 28px;
      background: linear-gradient(180deg, #f4efe5 0%, #ede4d2 100%);
      color: #2c2318;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    h1, h2, h3, h4 {{ margin: 0 0 12px 0; }}
    p, li {{ line-height: 1.45; color: #594e42; }}
    .panel {{
      background: rgba(255, 253, 248, 0.96);
      border: 1px solid #d7ccb5;
      border-radius: 22px;
      padding: 22px;
      margin-bottom: 20px;
      box-shadow: 0 10px 28px rgba(77, 59, 31, 0.08);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #fffdf8;
      border: 1px solid #e5dcc8;
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 14px;
    }}
    .cols {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid #e6dcc8;
      vertical-align: top;
    }}
    th {{
      background: #f3ecdd;
    }}
    img {{
      max-width: 100%;
      border-radius: 18px;
      border: 1px solid #d7ccb5;
      background: #fffdf8;
    }}
    code {{
      background: #f3ecdd;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>System Outputs: {html.escape(patient_id)}</h1>
    <p>This deliverable bundles primary predictions, latent diffusion explanation, cross-attention interpretation, and highest-contributing edge summaries.</p>
    <p>{html.escape(stage10_note)}</p>
  </div>
  <div class="panel">
    <h2>Primary Outputs</h2>
    <div class="grid">
      <div class="card">
        <h3>Survival</h3>
        <ul>
          <li>survival_mode: {html.escape(str(primary_outputs.get('survival_mode', '')))}</li>
          <li>survival_probability_last_bin: {html.escape(str(primary_outputs.get('survival_probability_last_bin')))}</li>
          <li>time_os_days: {html.escape(str(primary_outputs.get('time_os_days', '')))}</li>
          <li>event_os: {html.escape(str(primary_outputs.get('event_os', '')))}</li>
        </ul>
      </div>
      <div class="card">
        <h3>Recurrence</h3>
        <ul>
          <li>recurrence_probability: {safe_float(primary_outputs.get('recurrence_probability')):.4f}</li>
          <li>predicted_recurrence_location: {html.escape(str(primary_outputs.get('predicted_recurrence_location', '')))}</li>
        </ul>
        <ul>{prob_lines}</ul>
      </div>
    </div>
  </div>
  <div class="panel">
    <h2>Explanation Outputs</h2>
    <div class="card">
      <p>{html.escape(str(explanation_payload.get('explanation_semantics', '')))}</p>
    </div>
    <div class="grid">
      <div class="card">
        <h3>Organ Susceptibility</h3>
        <table>
          <thead><tr><th>Organ</th><th>Rank</th><th>Susceptibility</th></tr></thead>
          <tbody>{organ_lines}</tbody>
        </table>
      </div>
      <div class="card">
        <h3>Top Paths From Primary</h3>
        <table>
          <thead><tr><th>Path</th><th>Hops</th><th>Score</th></tr></thead>
          <tbody>{path_lines}</tbody>
        </table>
      </div>
    </div>
  </div>
  {svg_block}
  <div class="panel">
    <h2>Highest-Contributing Edges</h2>
    <table>
      <thead><tr><th>Source</th><th>Target</th><th>Edge Type</th><th>Probability</th></tr></thead>
      <tbody>{edge_lines}</tbody>
    </table>
  </div>
  <div class="panel">
    <h2>Cross-Attention Interpretation</h2>
    {''.join(attention_blocks) if attention_blocks else '<p>No cross-attention weights available for this patient.</p>'}
  </div>
  <div class="panel">
    <h2>Case Input Availability</h2>
    <ul>{availability_lines or '<li>No Stage 15.1 case-input bundle linked.</li>'}</ul>
  </div>
</body>
</html>
"""


def build_index_html(output_root, rows, cohort_svg_rel):
    table_rows = []
    for row in rows:
        report_rel = normalize_text(row.get("report_html"))
        report_link = html.escape(report_rel) if report_rel else ""
        report_label = html.escape(normalize_text(row.get("patient_id")))
        table_rows.append(
            "<tr>"
            f"<td><a href=\"{report_link}\">{report_label}</a></td>"
            f"<td>{safe_float(row.get('recurrence_probability')):.4f}</td>"
            f"<td>{html.escape(normalize_text(row.get('predicted_recurrence_location')))}</td>"
            f"<td>{html.escape(display_name(normalize_text(row.get('top_susceptibility_organ'))))}</td>"
            f"<td>{html.escape(display_name(normalize_text(row.get('top_edge_dst'))))}</td>"
            f"<td>{html.escape(normalize_text(row.get('top_path')))}</td>"
            f"<td>{int(safe_float(row.get('has_cross_attention'), 0.0))}</td>"
            "</tr>"
        )

    svg_block = ""
    if cohort_svg_rel:
        svg_block = f'<div class="panel"><h2>Cohort Diffusion View</h2><img src="{html.escape(cohort_svg_rel)}" alt="Cohort diffusion"/></div>'

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Stage 15.2 System Outputs</title>
  <style>
    body {{
      margin: 0;
      padding: 30px;
      background: linear-gradient(180deg, #f4efe5 0%, #ede4d2 100%);
      color: #2c2318;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    .panel {{
      background: rgba(255, 253, 248, 0.96);
      border: 1px solid #d7ccb5;
      border-radius: 22px;
      padding: 22px;
      margin-bottom: 20px;
      box-shadow: 0 10px 28px rgba(77, 59, 31, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid #e6dcc8;
    }}
    th {{ background: #f3ecdd; }}
    a {{ color: #20406d; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    img {{
      max-width: 100%;
      border-radius: 18px;
      border: 1px solid #d7ccb5;
      background: #fffdf8;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>Stage 15.2 System Outputs</h1>
    <p>This index links each patient-level deliverable bundle. Primary outputs and latent diffusion explanations are bundled together with available cross-attention and edge interpretation materials.</p>
  </div>
  {svg_block}
  <div class="panel">
    <h2>Patient Deliverables</h2>
    <table>
      <thead>
        <tr>
          <th>Patient</th>
          <th>Rec Prob</th>
          <th>Pred Loc</th>
          <th>Top Sus Org</th>
          <th>Top Edge Dest</th>
          <th>Top Path</th>
          <th>Has Cross-Attn</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    (Path(output_root) / "index.html").write_text(page, encoding="utf-8")


def maybe_copy_file(src_path, dst_path):
    src = Path(src_path)
    if not src.exists():
        return False
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main(argv=None, **overrides):
    args = apply_overrides(parse_args(argv), overrides)
    explanation_root = resolve_first_existing(args.explanation_root, EXPLANATION_ROOT_CANDIDATES, "explanation root")
    primary_csv = resolve_first_existing(args.primary_predictions_csv, PRIMARY_PREDICTION_CANDIDATES, "primary prediction csv")
    output_root = ensure_output_dir(args.output_root)
    case_input_root = Path(args.case_input_root)
    visualization_root = Path(args.visualization_root)

    primary_rows = read_csv_rows(primary_csv)
    primary_by_patient = {normalize_text(row["patient_id"]): row for row in primary_rows}
    patient_explanations = json.loads((explanation_root / "patient_explanations.json").read_text(encoding="utf-8"))
    explanation_by_patient = {normalize_text(row["patient_id"]): row for row in patient_explanations}
    manifest_rows = read_csv_rows(explanation_root / "patient_explanation_manifest.csv")
    manifest_rows.sort(key=lambda row: (-safe_float(row.get("recurrence_probability")), normalize_text(row.get("patient_id"))))
    if args.patient_ids:
        selected = {normalize_text(x) for x in args.patient_ids if normalize_text(x)}
        manifest_rows = [row for row in manifest_rows if normalize_text(row.get("patient_id")) in selected]
    elif int(args.max_patients) > 0:
        manifest_rows = manifest_rows[: int(args.max_patients)]
    if not manifest_rows:
        raise RuntimeError(
            "no matching patients found in explanation outputs. "
            "Stage 15.2 currently packages existing model outputs; raw external-case inference is not implemented here."
        )

    attention_payload = load_attention_payload(args.attention_npz)
    attention_meta = load_json_optional(args.attention_summary_json) or {}
    attention_note = (
        "Cross-attention weights are exported from the available Stage 10 fusion run. "
        "They indicate token reliance per organ query. "
    )
    if attention_meta:
        attention_note += (
            f"export_attention_weights={attention_meta.get('export_attention_weights')} "
            f"random_init_only={attention_meta.get('random_init_only')}"
        )

    organ_rows = read_csv_rows(explanation_root / "organ_susceptibility.csv")
    edge_rows = read_csv_rows(explanation_root / "edge_diffusion_long.csv")
    organ_by_patient = group_rows_by_patient(organ_rows)
    edge_by_patient = group_rows_by_patient(edge_rows)

    manifest_out_rows = []
    cases_dir = ensure_output_dir(output_root / "cases")

    for manifest_row in manifest_rows:
        patient_id = normalize_text(manifest_row.get("patient_id"))
        if patient_id not in explanation_by_patient:
            continue
        if patient_id not in primary_by_patient:
            continue

        case_dir = ensure_output_dir(cases_dir / patient_id)
        primary_payload = parse_primary_csv_row(primary_by_patient[patient_id])
        explanation_payload = explanation_by_patient[patient_id]
        susceptibility_rows = organ_by_patient.get(patient_id, [])
        patient_edge_rows = edge_by_patient.get(patient_id, [])
        top_paths = explanation_payload.get("top_paths", [])
        top_edges = explanation_payload.get("top_edges", [])
        attention_rows = build_attention_rows(attention_payload, patient_id)
        attention_summary = build_attention_summary(attention_rows)
        organ_names, edge_matrix_rows = build_edge_matrix_rows(patient_id, patient_edge_rows)

        primary_json = case_dir / "primary_outputs.json"
        explanation_json = case_dir / "explanation_outputs.json"
        susceptibility_csv = case_dir / "organ_susceptibility.csv"
        edge_long_csv = case_dir / "edge_tendency_long.csv"
        edge_matrix_csv = case_dir / "edge_tendency_matrix.csv"
        top_edges_csv = case_dir / "top_edges.csv"
        top_paths_json = case_dir / "topk_diffusion_paths.json"
        attn_csv = case_dir / "cross_attention_weights.csv"
        attn_summary_json = case_dir / "cross_attention_summary.json"

        write_json(primary_json, primary_payload)
        write_json(explanation_json, explanation_payload)
        write_csv(
            susceptibility_csv,
            ["patient_id", "organ_index", "organ_name", "susceptibility"],
            susceptibility_rows,
        )
        write_csv(
            edge_long_csv,
            ["patient_id", "src_index", "src_name", "dst_index", "dst_name", "edge_type", "is_prior_edge", "edge_diffusion_prob"],
            patient_edge_rows,
        )
        write_csv(
            edge_matrix_csv,
            ["patient_id", "src_name"] + organ_names,
            edge_matrix_rows,
        )
        write_csv(
            top_edges_csv,
            ["src_index", "src_name", "dst_index", "dst_name", "edge_type", "is_prior_edge", "edge_diffusion_prob", "rank"],
            top_edges,
        )
        write_json(top_paths_json, top_paths)
        if attention_rows:
            write_csv(
                attn_csv,
                ["patient_id", "organ_name", "evidence_token_name", "evidence_group", "attention_weight", "rank_within_organ"],
                attention_rows,
            )
            write_json(attn_summary_json, attention_summary)

        case_input_summary = load_json_optional(case_input_root / patient_id / "case_input_summary.json")
        svg_name = ""
        patient_svg = visualization_root / "patients" / f"{patient_id}_primary_diffusion.svg"
        if maybe_copy_file(patient_svg, case_dir / "primary_diffusion.svg"):
            svg_name = "primary_diffusion.svg"

        report_path = case_dir / "report.html"
        report_path.write_text(
            build_case_report_html(
                patient_id=patient_id,
                primary_outputs=primary_payload,
                explanation_payload=explanation_payload,
                attention_summary=attention_summary,
                top_edges=top_edges,
                top_paths=top_paths,
                availability_summary=case_input_summary,
                stage10_note=attention_note,
                svg_name=svg_name,
            ),
            encoding="utf-8",
        )

        top_sus_org = ""
        if explanation_payload.get("organ_susceptibility_ranked"):
            top_sus_org = explanation_payload["organ_susceptibility_ranked"][0].get("organ_name", "")
        top_edge_dst = ""
        if top_edges:
            top_edge_dst = normalize_text(top_edges[0].get("dst_name"))
        top_path_text = ""
        if top_paths:
            top_path_text = " -> ".join(display_name(x) for x in top_paths[0].get("path_names", []))

        manifest_out_rows.append(
            {
                "patient_id": patient_id,
                "recurrence_probability": safe_float(primary_payload.get("recurrence_probability")),
                "predicted_recurrence_location": normalize_text(primary_payload.get("predicted_recurrence_location")),
                "top_susceptibility_organ": display_name(top_sus_org),
                "top_edge_dst": display_name(top_edge_dst),
                "top_path": top_path_text,
                "has_cross_attention": int(bool(attention_rows)),
                "case_dir": str(case_dir),
                "report_html": str((Path("cases") / patient_id / "report.html").as_posix()),
            }
        )
    if not manifest_out_rows:
        raise RuntimeError(
            "no patient deliverables were generated after primary/explanation alignment. "
            "Check whether the requested patient_id exists in both the primary prediction CSV and explanation outputs."
        )

    write_csv(
        output_root / "system_output_manifest.csv",
        [
            "patient_id",
            "recurrence_probability",
            "predicted_recurrence_location",
            "top_susceptibility_organ",
            "top_edge_dst",
            "top_path",
            "has_cross_attention",
            "case_dir",
            "report_html",
        ],
        manifest_out_rows,
    )

    cohort_svg_rel = ""
    cohort_svg_src = visualization_root / "cohort_primary_diffusion.svg"
    if maybe_copy_file(cohort_svg_src, output_root / "cohort_primary_diffusion.svg"):
        cohort_svg_rel = "cohort_primary_diffusion.svg"

    build_index_html(output_root, manifest_out_rows, cohort_svg_rel)
    write_json(
        output_root / "system_output_summary.json",
        {
            "explanation_root": str(explanation_root),
            "primary_predictions_csv": str(primary_csv),
            "attention_npz": str(Path(args.attention_npz)),
            "visualization_root": str(visualization_root),
            "case_input_root": str(case_input_root),
            "patient_count": len(manifest_out_rows),
            "has_attention_payload": bool(attention_payload is not None),
            "cohort_svg_rel": cohort_svg_rel,
        },
    )

    print(f"wrote: {output_root / 'system_output_manifest.csv'}")
    print(f"wrote: {output_root / 'system_output_summary.json'}")
    print(f"wrote: {output_root / 'index.html'}")
    print("complete")
    return {
        "output_root": str(output_root),
        "manifest_csv": str(output_root / "system_output_manifest.csv"),
        "summary_json": str(output_root / "system_output_summary.json"),
        "index_html": str(output_root / "index.html"),
        "patient_count": len(manifest_out_rows),
    }


def run_system_outputs(
    *,
    explanation_root="",
    primary_predictions_csv="",
    attention_npz=str(DEFAULT_STAGE10_NPZ),
    attention_summary_json=str(DEFAULT_STAGE10_SUMMARY),
    visualization_root=str(DEFAULT_VIS_ROOT),
    case_input_root=str(DEFAULT_CASE_INPUT_ROOT),
    output_root=str(DEFAULT_OUTPUT_ROOT),
    patient_ids=None,
    max_patients=0,
):
    return main(
        explanation_root=explanation_root,
        primary_predictions_csv=primary_predictions_csv,
        attention_npz=attention_npz,
        attention_summary_json=attention_summary_json,
        visualization_root=visualization_root,
        case_input_root=case_input_root,
        output_root=output_root,
        patient_ids=[] if patient_ids is None else list(patient_ids),
        max_patients=max_patients,
    )


if __name__ == "__main__":
    main()
