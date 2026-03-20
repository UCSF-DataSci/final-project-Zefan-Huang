import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

##

def resolve_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


ROOT = resolve_root()


def load_local_module(path, module_name):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_npz(path):
    with np.load(path, allow_pickle=True) as z:
        out = {}
        for key in z.files:
            value = z[key]
            if getattr(value, "dtype", None) is not None and value.dtype.kind in {"U", "O"}:
                out[key] = value.astype(str)
            else:
                out[key] = value
    return out


def save_npz(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_manifest_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_patient_ids_by_flag(manifest_csv, field_name, truthy=("1", "true", "True", "TRUE")):
    patient_ids = []
    for row in read_manifest_rows(manifest_csv):
        value = str(row.get(field_name, "")).strip()
        if value in truthy:
            patient_id = str(row.get("patient_id", "")).strip()
            if patient_id:
                patient_ids.append(patient_id)
    return patient_ids


def subset_pack_by_patient_ids(pack, keep_patient_ids):
    keep_set = {str(x) for x in keep_patient_ids}
    patient_ids = [str(x) for x in pack["patient_ids"].tolist()]
    indices = [idx for idx, patient_id in enumerate(patient_ids) if patient_id in keep_set]
    if not indices:
        raise RuntimeError("no matching patient_ids for subset")
    out = {}
    for key, value in pack.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(patient_ids):
            out[key] = value[np.asarray(indices, dtype=np.int64)]
        else:
            out[key] = value
    return out


def disable_rna_modalities(stage9_pack):
    out = {}
    for key, value in stage9_pack.items():
        out[key] = value.copy() if isinstance(value, np.ndarray) else value
    if "g_rna" in out:
        out["g_rna"] = np.zeros_like(out["g_rna"], dtype=np.float32)
    if "g_rna_missing" in out:
        out["g_rna_missing"] = np.ones_like(out["g_rna_missing"], dtype=np.uint8)
    if "t_imm" in out:
        out["t_imm"] = np.zeros_like(out["t_imm"], dtype=np.float32)
    if "t_imm_missing" in out:
        out["t_imm_missing"] = np.ones_like(out["t_imm_missing"], dtype=np.uint8)
    return out


def run_python_script(script_name, args):
    cmd = [sys.executable, str(ROOT / script_name)] + [str(x) for x in args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    return {
        "cmd": cmd,
        "returncode": int(result.returncode),
    }
