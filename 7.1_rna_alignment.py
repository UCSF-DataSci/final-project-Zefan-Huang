"""
Compatibility wrapper for Stage 7.1 RNA alignment.

The implementation now lives in prepare_clean/rna_alignment.py so that
data alignment and cleaning scripts live together. This wrapper preserves
the original numbered entrypoint and re-exports the implementation module's
public symbols.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "prepare_clean" / "rna_alignment.py"
SPEC = spec_from_file_location("prepare_clean_rna_alignment", str(MODULE_PATH))
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"failed to load module from {MODULE_PATH}")

MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

for NAME in dir(MODULE):
    if NAME.startswith("_"):
        continue
    globals()[NAME] = getattr(MODULE, NAME)


if __name__ == "__main__":
    MODULE.main()
