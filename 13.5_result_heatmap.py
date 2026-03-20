import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage

# reuslt heatmap graph
## not looking well, cuz i'm only using query

DEFAULT_STAGE12_CSV = Path("output/stage12/12.2_explanation_outputs_joint/organ_susceptibility.csv")
DEFAULT_STAGE13_CSV = Path("output/stage13/13.2_phase4_tune/explanation_outputs_best/organ_susceptibility.csv")
DEFAULT_OUTPUT_ROOT = Path("output/stage13/13.5_result_heatmap")
KNOWN_ORGAN_ORDER = [
    "Primary",
    "Lung",
    "Bone",
    "Liver",
    "LymphNodeMediastinum",
    "Brain",
]
DISPLAY_NAME_MAP = {
    "LymphNodeMediastinum": "LN Mediastinum",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render paired Organ x Patient heatmaps from Stage 12 and Stage 13 explanation outputs",
        allow_abbrev=False,
    )
    parser.add_argument("--stage12-csv", type=str, default=str(DEFAULT_STAGE12_CSV))
    parser.add_argument("--stage13-csv", type=str, default=str(DEFAULT_STAGE13_CSV))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    args, _unknown = parser.parse_known_args(argv)
    return args


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_required_path(path_value, label):
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_rows(rows):
    grouped = {}
    organ_names = set()
    for row in rows:
        patient_id = str(row["patient_id"]).strip()
        organ_name = str(row["organ_name"]).strip()
        value = float(row["susceptibility"])
        grouped.setdefault(patient_id, {})[organ_name] = value
        organ_names.add(organ_name)
    return grouped, organ_names


def resolve_organ_order(stage12_organs, stage13_organs):
    present = set(stage12_organs) | set(stage13_organs)
    ordered = [name for name in KNOWN_ORGAN_ORDER if name in present]
    extras = sorted(name for name in present if name not in ordered)
    return ordered + extras


def build_matrix(grouped, patient_ids, organ_names):
    matrix = np.zeros((len(organ_names), len(patient_ids)), dtype=np.float32)
    for organ_idx, organ_name in enumerate(organ_names):
        for patient_idx, patient_id in enumerate(patient_ids):
            patient_map = grouped.get(patient_id)
            if patient_map is None or organ_name not in patient_map:
                raise RuntimeError(f"missing susceptibility for patient={patient_id} organ={organ_name}")
            matrix[organ_idx, patient_idx] = float(patient_map[organ_name])
    return matrix


def write_matrix_csv(path, matrix, organ_names, patient_ids):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["organ_name"] + list(patient_ids))
        for organ_idx, organ_name in enumerate(organ_names):
            writer.writerow([organ_name] + [f"{float(value):.6f}" for value in matrix[organ_idx]])


def write_patient_order_csv(path, patient_ids):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "patient_id"])
        for rank, patient_id in enumerate(patient_ids, start=1):
            writer.writerow([rank, patient_id])


def compute_patient_order(stage12_matrix, stage13_matrix, patient_ids):
    if len(patient_ids) <= 1:
        return list(patient_ids)
    features = np.concatenate([stage12_matrix.T, stage13_matrix.T], axis=1)
    model = linkage(features, method="ward", optimal_ordering=True)
    ordered_idx = leaves_list(model).tolist()
    return [patient_ids[idx] for idx in ordered_idx]


def display_name(name):
    return DISPLAY_NAME_MAP.get(str(name), str(name))


def clamp(value, low, high):
    return max(low, min(high, value))


def blend_hex(start_hex, end_hex, t):
    start_hex = start_hex.lstrip("#")
    end_hex = end_hex.lstrip("#")
    t = clamp(float(t), 0.0, 1.0)
    start = [int(start_hex[idx:idx + 2], 16) for idx in range(0, 6, 2)]
    end = [int(end_hex[idx:idx + 2], 16) for idx in range(0, 6, 2)]
    out = []
    for a, b in zip(start, end):
        out.append(int(round(a + (b - a) * t)))
    return "#" + "".join(f"{value:02x}" for value in out)


def color_for_value(value):
    value = clamp(float(value), 0.0, 1.0)
    if value <= 0.5:
        return blend_hex("#f8f6ef", "#f0b45d", value / 0.5 if value > 0.0 else 0.0)
    return blend_hex("#f0b45d", "#8f1d1d", (value - 0.5) / 0.5)


def render_heatmap_svg(
    *,
    output_path,
    stage12_matrix,
    stage13_matrix,
    organ_names,
    patient_ids,
):
    patient_count = len(patient_ids)
    organ_count = len(organ_names)
    left_margin = 120
    right_margin = 90
    top_margin = 120
    bottom_margin = 86
    panel_gap = 92
    organ_label_width = 128
    cell_width = 8 if patient_count <= 140 else 6
    cell_height = 42
    panel_width = organ_label_width + patient_count * cell_width
    panel_height = organ_count * cell_height
    colorbar_width = 26
    width = left_margin + panel_width * 2 + panel_gap + right_margin + colorbar_width + 56
    height = top_margin + panel_height + bottom_margin
    panel1_x = left_margin
    panel2_x = left_margin + panel_width + panel_gap
    panel_y = top_margin
    colorbar_x = panel2_x + panel_width + 38
    colorbar_y = panel_y
    colorbar_height = panel_height

    def draw_panel(svg, panel_x, title, subtitle, matrix):
        svg.append(
            f'<rect x="{panel_x - 18}" y="{panel_y - 28}" width="{panel_width + 36}" '
            f'height="{panel_height + 58}" rx="22" fill="#fffdf8" stroke="#d7ccb5" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{panel_x}" y="{panel_y - 38}" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="24" font-weight="700" fill="#2c2318">{title}</text>'
        )
        svg.append(
            f'<text x="{panel_x}" y="{panel_y - 12}" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="13" fill="#6b5b45">{subtitle}</text>'
        )
        for organ_idx, organ_name in enumerate(organ_names):
            y = panel_y + organ_idx * cell_height
            svg.append(
                f'<text x="{panel_x + organ_label_width - 10}" y="{y + cell_height * 0.62:.1f}" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="14" text-anchor="end" '
                f'fill="#2c2318">{display_name(organ_name)}</text>'
            )
            for patient_idx in range(patient_count):
                x = panel_x + organ_label_width + patient_idx * cell_width
                value = float(matrix[organ_idx, patient_idx])
                svg.append(
                    f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" '
                    f'fill="{color_for_value(value)}" stroke="none"/>'
                )
        for organ_idx in range(organ_count + 1):
            y = panel_y + organ_idx * cell_height
            svg.append(
                f'<line x1="{panel_x + organ_label_width}" y1="{y}" '
                f'x2="{panel_x + organ_label_width + patient_count * cell_width}" y2="{y}" '
                f'stroke="#efe6d8" stroke-width="1"/>'
            )
        for patient_idx in range(patient_count + 1):
            x = panel_x + organ_label_width + patient_idx * cell_width
            stroke = "#e7dccd" if patient_idx % 10 == 0 else "#f5efe5"
            width_value = 1.2 if patient_idx % 10 == 0 else 0.6
            svg.append(
                f'<line x1="{x}" y1="{panel_y}" x2="{x}" y2="{panel_y + panel_height}" '
                f'stroke="{stroke}" stroke-width="{width_value}"/>'
            )
            if patient_idx < patient_count and patient_idx % 10 == 0:
                svg.append(
                    f'<text x="{x + 2}" y="{panel_y + panel_height + 18}" '
                    f'font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#8b7a63">'
                    f'{patient_idx + 1}</text>'
                )
        svg.append(
            f'<text x="{panel_x + organ_label_width}" y="{panel_y + panel_height + 38}" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#6b5b45">'
            f'Patients (shared clustering order, n={patient_count})</text>'
        )

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append("<defs>")
    svg.append('<linearGradient id="pagebg" x1="0%" y1="0%" x2="100%" y2="100%">')
    svg.append('<stop offset="0%" stop-color="#f6f3ec"/>')
    svg.append('<stop offset="100%" stop-color="#ede4d3"/>')
    svg.append("</linearGradient>")
    svg.append('<linearGradient id="cbar" x1="0%" y1="100%" x2="0%" y2="0%">')
    svg.append('<stop offset="0%" stop-color="#f8f6ef"/>')
    svg.append('<stop offset="50%" stop-color="#f0b45d"/>')
    svg.append('<stop offset="100%" stop-color="#8f1d1d"/>')
    svg.append("</linearGradient>")
    svg.append("</defs>")
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="url(#pagebg)"/>')
    svg.append(
        '<text x="56" y="56" font-family="Helvetica, Arial, sans-serif" font-size="34" '
        'font-weight="700" fill="#2c2318">Organ Susceptibility Heatmap</text>'
    )
    svg.append(
        '<text x="56" y="86" font-family="Helvetica, Arial, sans-serif" font-size="15" '
        'fill="#6b5b45">Stage 12 joint explanation vs Stage 13 tuned final explanation, shared patient clustering</text>'
    )
    draw_panel(
        svg,
        panel1_x,
        "Stage 12 Joint Explanation",
        "Intermediate joint explanation output",
        stage12_matrix,
    )
    draw_panel(
        svg,
        panel2_x,
        "Stage 13 Tuned Final",
        "Best tuned final explanation output",
        stage13_matrix,
    )
    svg.append(
        f'<rect x="{colorbar_x}" y="{colorbar_y}" width="{colorbar_width}" height="{colorbar_height}" '
        f'rx="10" fill="url(#cbar)" stroke="#c3b49f" stroke-width="1.5"/>'
    )
    for frac, label in [(0.0, "0.0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")]:
        y = colorbar_y + colorbar_height - frac * colorbar_height
        svg.append(
            f'<line x1="{colorbar_x + colorbar_width + 4}" y1="{y}" x2="{colorbar_x + colorbar_width + 12}" y2="{y}" '
            f'stroke="#7b6a56" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{colorbar_x + colorbar_width + 18}" y="{y + 4}" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#6b5b45">{label}</text>'
        )
    svg.append(
        f'<text x="{colorbar_x - 8}" y="{colorbar_y - 14}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="12" fill="#6b5b45">Susceptibility</text>'
    )
    svg.append(
        f'<text x="56" y="{height - 24}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#6b5b45">'
        f'Rows are fixed organ nodes; columns are shared patients ordered by Ward clustering on concatenated Stage 12 and Stage 13 patient features.</text>'
    )
    svg.append("</svg>")
    Path(output_path).write_text("\n".join(svg), encoding="utf-8")


def convert_svg(output_svg):
    output_svg = Path(output_svg)
    sips = shutil.which("sips")
    converted = {}
    if not sips:
        return converted
    png_path = output_svg.with_suffix(".png")
    pdf_path = output_svg.with_suffix(".pdf")
    png_run = subprocess.run(
        [sips, "-s", "format", "png", str(output_svg), "--out", str(png_path)],
        capture_output=True,
        text=True,
    )
    if png_run.returncode == 0 and png_path.exists():
        converted["png_path"] = str(png_path)
    pdf_run = subprocess.run(
        [sips, "-s", "format", "pdf", str(output_svg), "--out", str(pdf_path)],
        capture_output=True,
        text=True,
    )
    if pdf_run.returncode == 0 and pdf_path.exists():
        converted["pdf_path"] = str(pdf_path)
    return converted


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    stage12_csv = resolve_required_path(args.stage12_csv, "stage12 organ susceptibility csv")
    stage13_csv = resolve_required_path(args.stage13_csv, "stage13 organ susceptibility csv")
    output_root = ensure_output_dir(args.output_root)

    stage12_rows = read_csv_rows(stage12_csv)
    stage13_rows = read_csv_rows(stage13_csv)
    stage12_grouped, stage12_organs = group_rows(stage12_rows)
    stage13_grouped, stage13_organs = group_rows(stage13_rows)
    shared_patient_ids = sorted(set(stage12_grouped) & set(stage13_grouped))
    if not shared_patient_ids:
        raise RuntimeError("no shared patient_ids between stage12 and stage13 organ susceptibility tables")
    organ_names = resolve_organ_order(stage12_organs, stage13_organs)
    stage12_matrix = build_matrix(stage12_grouped, shared_patient_ids, organ_names)
    stage13_matrix = build_matrix(stage13_grouped, shared_patient_ids, organ_names)
    clustered_patient_ids = compute_patient_order(stage12_matrix, stage13_matrix, shared_patient_ids)
    stage12_matrix = build_matrix(stage12_grouped, clustered_patient_ids, organ_names)
    stage13_matrix = build_matrix(stage13_grouped, clustered_patient_ids, organ_names)

    svg_path = output_root / "organ_patient_heatmap_compare.svg"
    render_heatmap_svg(
        output_path=svg_path,
        stage12_matrix=stage12_matrix,
        stage13_matrix=stage13_matrix,
        organ_names=organ_names,
        patient_ids=clustered_patient_ids,
    )
    converted = convert_svg(svg_path)

    stage12_matrix_csv = output_root / "stage12_heatmap_matrix.csv"
    stage13_matrix_csv = output_root / "stage13_heatmap_matrix.csv"
    patient_order_csv = output_root / "patient_order.csv"
    write_matrix_csv(stage12_matrix_csv, stage12_matrix, organ_names, clustered_patient_ids)
    write_matrix_csv(stage13_matrix_csv, stage13_matrix, organ_names, clustered_patient_ids)
    write_patient_order_csv(patient_order_csv, clustered_patient_ids)

    summary = {
        "figure_type": "paired_organ_patient_heatmap",
        "stage12_csv": str(stage12_csv),
        "stage13_csv": str(stage13_csv),
        "shared_patient_count": len(clustered_patient_ids),
        "organ_count": len(organ_names),
        "organ_order": organ_names,
        "patient_order_source": "ward_linkage_on_concatenated_stage12_stage13_patient_features",
        "value_scale": {"min": 0.0, "max": 1.0},
        "stage12_range": {
            "min": float(stage12_matrix.min()),
            "max": float(stage12_matrix.max()),
            "mean": float(stage12_matrix.mean()),
        },
        "stage13_range": {
            "min": float(stage13_matrix.min()),
            "max": float(stage13_matrix.max()),
            "mean": float(stage13_matrix.mean()),
        },
        "stage12_mean_by_organ": {
            organ_names[idx]: float(stage12_matrix[idx].mean()) for idx in range(len(organ_names))
        },
        "stage13_mean_by_organ": {
            organ_names[idx]: float(stage13_matrix[idx].mean()) for idx in range(len(organ_names))
        },
        "svg_path": str(svg_path),
        "stage12_matrix_csv": str(stage12_matrix_csv),
        "stage13_matrix_csv": str(stage13_matrix_csv),
        "patient_order_csv": str(patient_order_csv),
    }
    summary.update(converted)
    summary_path = output_root / "heatmap_summary.json"
    write_json(summary_path, summary)

    print(f"wrote: {svg_path}")
    if "png_path" in converted:
        print(f"wrote: {converted['png_path']}")
    if "pdf_path" in converted:
        print(f"wrote: {converted['pdf_path']}")
    print(f"wrote: {stage12_matrix_csv}")
    print(f"wrote: {stage13_matrix_csv}")
    print(f"wrote: {patient_order_csv}")
    print(f"wrote: {summary_path}")
    return {
        "svg_path": str(svg_path),
        "summary_path": str(summary_path),
        "shared_patient_count": len(clustered_patient_ids),
    }


if __name__ == "__main__":
    main()
