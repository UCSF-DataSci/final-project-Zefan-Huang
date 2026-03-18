
import runpy
from pathlib import Path


def main():

    seg_script = Path(__file__).with_name("6.1_seg_model.py")
    if not seg_script.exists():
        raise SystemExit(f"missing script: {seg_script}")
    runpy.run_path(str(seg_script), run_name="__main__")


if __name__ == "__main__":
    main()
