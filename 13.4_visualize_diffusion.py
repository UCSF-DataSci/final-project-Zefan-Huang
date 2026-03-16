"""
Render latent diffusion explanations as directional SVG graphs and an HTML dashboard.
"""
import argparse
import csv
import html
import json
import math
from pathlib import Path


def resolve_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


ROOT = resolve_root()
DEFAULT_STAGE13_EXPL = ROOT / "output/stage13/13.2_phase4_tune/explanation_outputs_best"
DEFAULT_STAGE12_EXPL = ROOT / "output/stage12/12.2_explanation_outputs_joint"
DEFAULT_OUTPUT_ROOT = ROOT / "output/stage13/13.4_visualize_diffusion"

DISPLAY_NAME_MAP = {
    "LymphNodeMediastinum": "LN Mediastinum",
}

KNOWN_POSITIONS = {
    "Primary": (220.0, 280.0),
    "Lung": (460.0, 130.0),
    "Bone": (470.0, 415.0),
    "Liver": (700.0, 360.0),
    "LymphNodeMediastinum": (360.0, 245.0),
    "Brain": (710.0, 100.0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render directional diffusion visualizations from explanation outputs",
        allow_abbrev=False,
    )
    parser.add_argument("--explanation-root", type=str, default="")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--patient-ids", nargs="*", default=[])
    parser.add_argument("--max-patients", type=int, default=0)
    args, _unknown = parser.parse_known_args()
    return args


def resolve_explanation_root(raw_path):
    if str(raw_path).strip():
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"explanation root not found: {path}")
        return path
    if DEFAULT_STAGE13_EXPL.exists():
        return DEFAULT_STAGE13_EXPL
    if DEFAULT_STAGE12_EXPL.exists():
        return DEFAULT_STAGE12_EXPL
    raise FileNotFoundError("no default explanation root found")


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_patient_explanations(path):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(item["patient_id"]): item for item in items}


def group_susceptibility(rows):
    grouped = {}
    organ_order = []
    for row in rows:
        patient_id = str(row["patient_id"])
        organ_name = str(row["organ_name"])
        grouped.setdefault(patient_id, {})[organ_name] = float(row["susceptibility"])
        if organ_name not in organ_order:
            organ_order.append(organ_name)
    return grouped, organ_order


def group_edges(rows):
    grouped = {}
    for row in rows:
        patient_id = str(row["patient_id"])
        grouped.setdefault(patient_id, []).append(
            {
                "src_name": str(row["src_name"]),
                "dst_name": str(row["dst_name"]),
                "edge_type": str(row["edge_type"]),
                "is_prior_edge": int(row["is_prior_edge"]),
                "edge_diffusion_prob": float(row["edge_diffusion_prob"]),
            }
        )
    for patient_id in grouped:
        grouped[patient_id].sort(key=lambda item: item["edge_diffusion_prob"], reverse=True)
    return grouped


def display_name(name):
    return DISPLAY_NAME_MAP.get(str(name), str(name))


def wrap_label(name):
    label = display_name(name)
    if len(label) <= 12 or " " in label:
        return label.split()
    midpoint = len(label) // 2
    return [label[:midpoint], label[midpoint:]]


def get_positions(organ_names):
    positions = {}
    unknown = [name for name in organ_names if name not in KNOWN_POSITIONS]
    for name in organ_names:
        if name in KNOWN_POSITIONS:
            positions[name] = KNOWN_POSITIONS[name]
    if unknown:
        center_x = 500.0
        center_y = 260.0
        radius = 230.0
        for idx, name in enumerate(unknown):
            angle = (2.0 * math.pi * idx) / max(len(unknown), 1)
            positions[name] = (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )
    return positions


def clamp(value, low, high):
    return max(low, min(high, value))


def blend_color(start_rgb, end_rgb, t):
    t = clamp(float(t), 0.0, 1.0)
    out = []
    for a, b in zip(start_rgb, end_rgb):
        out.append(int(round(a + (b - a) * t)))
    return "#" + "".join(f"{x:02x}" for x in out)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def shorten_segment(start_xy, end_xy, start_pad, end_pad):
    x1, y1 = start_xy
    x2, y2 = end_xy
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
        return x1, y1, x2, y2, 1.0, 0.0
    ux = dx / dist
    uy = dy / dist
    return (
        x1 + ux * start_pad,
        y1 + uy * start_pad,
        x2 - ux * end_pad,
        y2 - uy * end_pad,
        ux,
        uy,
    )


def format_prob(value):
    return f"{float(value):.2f}"


def extract_primary_outgoing(edges):
    outgoing = [row for row in edges if row["src_name"] == "Primary" and row["dst_name"] != row["src_name"]]
    return sorted(outgoing, key=lambda item: item["edge_diffusion_prob"], reverse=True)


def extract_top_paths(patient_entry, limit=3):
    return list(patient_entry.get("top_paths", []))[:limit]


def summarize_outgoing_text(edges, limit=3):
    parts = []
    for edge in edges[:limit]:
        parts.append(f"Primary -> {display_name(edge['dst_name'])} ({format_prob(edge['edge_diffusion_prob'])})")
    return " | ".join(parts)


def render_svg(
    *,
    title,
    subtitle,
    organ_names,
    susceptibility_by_organ,
    primary_outgoing_edges,
    side_lines,
    output_path,
):
    width = 1180
    height = 620
    graph_right = 820
    positions = get_positions(organ_names)
    radii = {}
    for organ_name in organ_names:
        score = clamp(safe_float(susceptibility_by_organ.get(organ_name, 0.0)), 0.0, 1.0)
        radii[organ_name] = 24.0 + 18.0 * score

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append("<defs>")
    svg.append('<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">')
    svg.append('<stop offset="0%" stop-color="#f6f4ed"/>')
    svg.append('<stop offset="100%" stop-color="#efe6d3"/>')
    svg.append("</linearGradient>")
    svg.append(
        '<marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" '
        'markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
    )
    svg.append('<path d="M 0 0 L 10 5 L 0 10 z" fill="#20406d"/>')
    svg.append("</marker>")
    svg.append("</defs>")
    svg.append('<rect x="0" y="0" width="1180" height="620" fill="url(#bg)"/>')
    svg.append('<rect x="26" y="24" width="790" height="572" rx="26" fill="#fffdf8" stroke="#d7ccb5" stroke-width="2"/>')
    svg.append('<rect x="840" y="24" width="314" height="572" rx="26" fill="#fffdf8" stroke="#d7ccb5" stroke-width="2"/>')
    svg.append(f'<text x="54" y="68" font-family="Helvetica, Arial, sans-serif" font-size="28" font-weight="700" fill="#2c2318">{html.escape(title)}</text>')
    svg.append(f'<text x="54" y="96" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#6f6558">{html.escape(subtitle)}</text>')
    svg.append('<text x="54" y="126" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#7c715f">Arrow width and opacity encode predicted diffusion probability from Primary.</text>')
    svg.append('<text x="866" y="62" font-family="Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#2c2318">Reading Panel</text>')

    for edge in primary_outgoing_edges:
        src_name = edge["src_name"]
        dst_name = edge["dst_name"]
        if src_name not in positions or dst_name not in positions:
            continue
        prob = clamp(edge["edge_diffusion_prob"], 0.0, 1.0)
        x1, y1 = positions[src_name]
        x2, y2 = positions[dst_name]
        sx, sy, ex, ey, ux, uy = shorten_segment(
            (x1, y1),
            (x2, y2),
            radii[src_name] + 8.0,
            radii[dst_name] + 12.0,
        )
        stroke = blend_color((154, 180, 213), (32, 64, 109), prob)
        stroke_width = 1.8 + 9.0 * prob
        opacity = 0.24 + 0.72 * prob
        svg.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.2f}" stroke-linecap="round" opacity="{opacity:.3f}" marker-end="url(#arrow)"/>'
        )
        mx = (sx + ex) / 2.0 - uy * 13.0
        my = (sy + ey) / 2.0 + ux * 13.0
        svg.append(
            f'<rect x="{mx - 20:.1f}" y="{my - 11:.1f}" width="40" height="20" rx="10" fill="#fffdf8" opacity="0.92"/>'
        )
        svg.append(
            f'<text x="{mx:.1f}" y="{my + 4:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="12" font-weight="700" fill="#183153">{format_prob(prob)}</text>'
        )

    for organ_name in organ_names:
        x, y = positions[organ_name]
        score = clamp(safe_float(susceptibility_by_organ.get(organ_name, 0.0)), 0.0, 1.0)
        fill = blend_color((232, 240, 231), (175, 57, 48), score)
        stroke = "#5b4e3c" if organ_name == "Primary" else "#8b7f6c"
        stroke_width = 4 if organ_name == "Primary" else 2
        radius = radii[organ_name]
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
        lines = wrap_label(organ_name)
        for idx, line in enumerate(lines):
            y_text = y - 2 + idx * 16 - ((len(lines) - 1) * 8)
            svg.append(
                f'<text x="{x:.1f}" y="{y_text:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
                f'font-size="13" font-weight="700" fill="#1f1a14">{html.escape(line)}</text>'
            )
        svg.append(
            f'<text x="{x:.1f}" y="{y + radius + 18:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="12" fill="#6f6558">sus {format_prob(score)}</text>'
        )

    svg.append('<rect x="864" y="88" width="266" height="40" rx="14" fill="#efe6d3"/>')
    svg.append('<circle cx="890" cy="108" r="9" fill="#b03831"/>')
    svg.append('<text x="910" y="113" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#2c2318">Node fill = organ susceptibility</text>')
    svg.append('<line x1="874" y1="145" x2="930" y2="145" stroke="#20406d" stroke-width="7" marker-end="url(#arrow)"/>')
    svg.append('<text x="946" y="150" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#2c2318">Arrow = predicted direction from Primary</text>')
    svg.append('<text x="866" y="190" font-family="Helvetica, Arial, sans-serif" font-size="16" font-weight="700" fill="#2c2318">Key Numbers</text>')

    y = 220
    for line in side_lines:
        svg.append(
            f'<text x="866" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#3e3429">{html.escape(line)}</text>'
        )
        y += 24

    svg.append("</svg>")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(svg), encoding="utf-8")


def build_patient_record(manifest_row, patient_entry, patient_edges, patient_sus):
    primary_outgoing = extract_primary_outgoing(patient_edges)
    recurrence_probability = safe_float(manifest_row.get("recurrence_probability"))
    top_paths = extract_top_paths(patient_entry, limit=3)
    top_path_text = []
    for path_row in top_paths:
        path_names = [display_name(name) for name in path_row.get("path_names", [])]
        top_path_text.append(f"{' -> '.join(path_names)} ({format_prob(path_row.get('score_prob', 0.0))})")
    return {
        "patient_id": str(manifest_row["patient_id"]),
        "recurrence_probability": recurrence_probability,
        "predicted_recurrence_location": str(manifest_row.get("predicted_recurrence_location", "")),
        "top1_destination": primary_outgoing[0]["dst_name"] if primary_outgoing else "",
        "top1_destination_prob": primary_outgoing[0]["edge_diffusion_prob"] if primary_outgoing else None,
        "top3_directions": summarize_outgoing_text(primary_outgoing, limit=3),
        "top_paths_text": " | ".join(top_path_text),
        "primary_outgoing": primary_outgoing,
        "susceptibility_by_organ": dict(patient_sus),
        "patient_entry": patient_entry,
    }


def render_patient_svgs(output_root, organ_names, patient_records):
    patient_dir = Path(output_root) / "patients"
    rows = []
    for record in patient_records:
        patient_id = record["patient_id"]
        primary_outputs = record["patient_entry"].get("primary_outputs", {})
        recurrence_probability = safe_float(primary_outputs.get("recurrence_probability", record["recurrence_probability"]))
        predicted_location = str(primary_outputs.get("predicted_recurrence_location", record["predicted_recurrence_location"]))
        top_paths = extract_top_paths(record["patient_entry"], limit=3)
        side_lines = [
            f"Patient: {patient_id}",
            f"Recurrence probability: {format_prob(recurrence_probability)}",
            f"Predicted recurrence location: {predicted_location}",
            "",
            "Primary outgoing ranking:",
        ]
        for edge in record["primary_outgoing"][:5]:
            side_lines.append(f"{display_name(edge['dst_name'])}: {format_prob(edge['edge_diffusion_prob'])}")
        if top_paths:
            side_lines.append("")
            side_lines.append("Top paths:")
            for path_row in top_paths[:3]:
                path_names = " -> ".join(display_name(name) for name in path_row.get("path_names", []))
                side_lines.append(f"{path_names}: {format_prob(path_row.get('score_prob', 0.0))}")

        svg_path = patient_dir / f"{patient_id}_primary_diffusion.svg"
        render_svg(
            title=f"Primary Diffusion Map: {patient_id}",
            subtitle="Latent diffusion explanation derived from the trained recurrence/OS model",
            organ_names=organ_names,
            susceptibility_by_organ=record["susceptibility_by_organ"],
            primary_outgoing_edges=record["primary_outgoing"],
            side_lines=side_lines,
            output_path=svg_path,
        )
        rows.append(
            {
                "patient_id": patient_id,
                "recurrence_probability": recurrence_probability,
                "predicted_recurrence_location": predicted_location,
                "top1_destination": display_name(record["top1_destination"]),
                "top1_destination_prob": (
                    "" if record["top1_destination_prob"] is None else format_prob(record["top1_destination_prob"])
                ),
                "top3_directions": record["top3_directions"],
                "top_paths": record["top_paths_text"],
                "svg_path": str(svg_path),
            }
        )
    write_csv(
        Path(output_root) / "patient_direction_summary.csv",
        [
            "patient_id",
            "recurrence_probability",
            "predicted_recurrence_location",
            "top1_destination",
            "top1_destination_prob",
            "top3_directions",
            "top_paths",
            "svg_path",
        ],
        rows,
    )
    return rows


def render_cohort_svg(output_root, organ_names, patient_records):
    if not patient_records:
        return None
    susceptibility_mean = {organ_name: 0.0 for organ_name in organ_names}
    outgoing_sum = {organ_name: 0.0 for organ_name in organ_names if organ_name != "Primary"}
    top1_counts = {}
    recurrence_prob_sum = 0.0
    for record in patient_records:
        recurrence_prob_sum += safe_float(record["recurrence_probability"])
        for organ_name in organ_names:
            susceptibility_mean[organ_name] += safe_float(record["susceptibility_by_organ"].get(organ_name, 0.0))
        for edge in record["primary_outgoing"]:
            outgoing_sum[edge["dst_name"]] = outgoing_sum.get(edge["dst_name"], 0.0) + safe_float(edge["edge_diffusion_prob"])
        if record["top1_destination"]:
            top1_counts[record["top1_destination"]] = top1_counts.get(record["top1_destination"], 0) + 1

    count = float(len(patient_records))
    for organ_name in susceptibility_mean:
        susceptibility_mean[organ_name] /= count
    outgoing_edges = []
    for dst_name, total_prob in outgoing_sum.items():
        outgoing_edges.append(
            {
                "src_name": "Primary",
                "dst_name": dst_name,
                "edge_type": "mean",
                "is_prior_edge": 1,
                "edge_diffusion_prob": total_prob / count,
            }
        )
    outgoing_edges.sort(key=lambda item: item["edge_diffusion_prob"], reverse=True)
    top1_sorted = sorted(top1_counts.items(), key=lambda item: (-item[1], item[0]))
    side_lines = [
        f"Patients rendered: {len(patient_records)}",
        f"Mean recurrence probability: {format_prob(recurrence_prob_sum / count)}",
        "",
        "Mean Primary outgoing ranking:",
    ]
    for edge in outgoing_edges[:5]:
        side_lines.append(f"{display_name(edge['dst_name'])}: {format_prob(edge['edge_diffusion_prob'])}")
    if top1_sorted:
        side_lines.append("")
        side_lines.append("Top-1 destination frequency:")
        for dst_name, item_count in top1_sorted[:5]:
            side_lines.append(f"{display_name(dst_name)}: {item_count}/{len(patient_records)}")

    svg_path = Path(output_root) / "cohort_primary_diffusion.svg"
    render_svg(
        title="Cohort Primary Diffusion Map",
        subtitle="Mean directional diffusion from Primary across the selected patients",
        organ_names=organ_names,
        susceptibility_by_organ=susceptibility_mean,
        primary_outgoing_edges=outgoing_edges,
        side_lines=side_lines,
        output_path=svg_path,
    )

    rows = []
    for edge in outgoing_edges:
        rows.append(
            {
                "src_name": edge["src_name"],
                "dst_name": edge["dst_name"],
                "mean_edge_diffusion_prob": edge["edge_diffusion_prob"],
            }
        )
    write_csv(
        Path(output_root) / "cohort_primary_direction_summary.csv",
        ["src_name", "dst_name", "mean_edge_diffusion_prob"],
        rows,
    )
    return svg_path


def render_dashboard(output_root, explanation_root, selected_rows, cohort_svg_path):
    output_root = Path(output_root)
    html_path = output_root / "index.html"
    cards = []
    for row in selected_rows:
        svg_rel = Path(row["svg_path"]).relative_to(output_root)
        cards.append(
            "<tr>"
            f"<td><a href=\"{html.escape(str(svg_rel))}\">{html.escape(row['patient_id'])}</a></td>"
            f"<td>{format_prob(row['recurrence_probability'])}</td>"
            f"<td>{html.escape(row['predicted_recurrence_location'])}</td>"
            f"<td>{html.escape(row['top1_destination'])}</td>"
            f"<td>{html.escape(row['top1_destination_prob'])}</td>"
            f"<td>{html.escape(row['top3_directions'])}</td>"
            f"<td>{html.escape(row['top_paths'])}</td>"
            "</tr>"
        )

    cohort_rel = Path(cohort_svg_path).relative_to(output_root) if cohort_svg_path else None
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Directional Diffusion Dashboard</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      background: linear-gradient(180deg, #f4efe5 0%, #ede4d2 100%);
      color: #2c2318;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
    }}
    p {{
      margin: 0 0 14px 0;
      line-height: 1.45;
      color: #5f5447;
    }}
    .panel {{
      background: rgba(255, 253, 248, 0.96);
      border: 1px solid #d7ccb5;
      border-radius: 22px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 10px 30px rgba(77, 59, 31, 0.08);
    }}
    img {{
      max-width: 100%;
      border-radius: 18px;
      border: 1px solid #d7ccb5;
      background: #fffdf8;
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
      vertical-align: top;
    }}
    th {{
      background: #f3ecdd;
      font-weight: 700;
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
    <h1>Directional Diffusion Dashboard</h1>
    <p>Source explanation root: {html.escape(str(explanation_root))}</p>
    <p>This view is a model-induced latent diffusion map for explaining OS and recurrence outputs. It is directional, but it is not organ-level ground-truth supervision.</p>
  </div>
  <div class="panel">
    <h2>Cohort View</h2>
    <p>The cohort graph averages Primary outgoing diffusion probabilities and organ susceptibility across the selected patients.</p>
    {"<img src=\"" + html.escape(str(cohort_rel)) + "\" alt=\"Cohort diffusion graph\"/>" if cohort_rel else ""}
  </div>
  <div class="panel">
    <h2>Patient Views</h2>
    <p>Open any patient SVG to inspect the predicted outgoing directions from Primary.</p>
    <table>
      <thead>
        <tr>
          <th>Patient</th>
          <th>Rec Prob</th>
          <th>Pred Loc</th>
          <th>Top-1 Dest</th>
          <th>Top-1 Prob</th>
          <th>Top-3 Directions</th>
          <th>Top Paths</th>
        </tr>
      </thead>
      <tbody>
        {"".join(cards)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")
    return html_path


def main():
    args = parse_args()
    explanation_root = resolve_explanation_root(args.explanation_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv_rows(explanation_root / "patient_explanation_manifest.csv")
    susceptibility_rows = read_csv_rows(explanation_root / "organ_susceptibility.csv")
    edge_rows = read_csv_rows(explanation_root / "edge_diffusion_long.csv")
    patient_explanations = load_patient_explanations(explanation_root / "patient_explanations.json")
    susceptibility_by_patient, organ_names = group_susceptibility(susceptibility_rows)
    edges_by_patient = group_edges(edge_rows)

    manifest_rows.sort(key=lambda row: (-safe_float(row.get("recurrence_probability")), str(row.get("patient_id"))))
    requested_ids = [str(x) for x in args.patient_ids if str(x).strip()]
    if requested_ids:
        requested_set = set(requested_ids)
        manifest_rows = [row for row in manifest_rows if str(row["patient_id"]) in requested_set]
    elif int(args.max_patients) > 0:
        manifest_rows = manifest_rows[: int(args.max_patients)]

    patient_records = []
    for row in manifest_rows:
        patient_id = str(row["patient_id"])
        if patient_id not in susceptibility_by_patient:
            continue
        if patient_id not in edges_by_patient:
            continue
        if patient_id not in patient_explanations:
            continue
        patient_records.append(
            build_patient_record(
                manifest_row=row,
                patient_entry=patient_explanations[patient_id],
                patient_edges=edges_by_patient[patient_id],
                patient_sus=susceptibility_by_patient[patient_id],
            )
        )

    if not patient_records:
        raise RuntimeError("no patients available after filtering")

    selected_rows = render_patient_svgs(output_root, organ_names, patient_records)
    cohort_svg_path = render_cohort_svg(output_root, organ_names, patient_records)
    html_path = render_dashboard(output_root, explanation_root, selected_rows, cohort_svg_path)

    summary = {
        "explanation_root": str(explanation_root),
        "output_root": str(output_root),
        "patient_count_rendered": len(selected_rows),
        "cohort_svg_path": str(cohort_svg_path) if cohort_svg_path else "",
        "dashboard_html": str(html_path),
    }
    (output_root / "visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote: {output_root / 'patient_direction_summary.csv'}")
    print(f"wrote: {output_root / 'cohort_primary_direction_summary.csv'}")
    print(f"wrote: {output_root / 'cohort_primary_diffusion.svg'}")
    print(f"wrote: {html_path}")
    print(f"wrote: {output_root / 'visualization_summary.json'}")
    print("complete")


if __name__ == "__main__":
    main()
