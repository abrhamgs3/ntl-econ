import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    runpy.run_module("scripts.data_processing.process_land_use_data", run_name="__main__")
