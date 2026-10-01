"""Run the original paired resampling calculation through a descriptive entry point."""
from pathlib import Path
import runpy

script = Path(__file__).resolve().with_name("reproduce_paper_bootstrap.py")
runpy.run_path(str(script), run_name="__main__")
