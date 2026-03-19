from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
SCRIPTS_DIR = ROOT_DIR / "scripts"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
MANUSCRIPT_DIR = ROOT_DIR / "manuscript"

TABULAR_DIR = DATA_DIR / "tabular"
RASTER_DIR = DATA_DIR / "raster"
CLIPPED_RASTER_DIR = RASTER_DIR / "clipped"
SHAPEFILES_DIR = DATA_DIR / "shapefiles"

REGIONAL_RESULTS_DIR = OUTPUTS_DIR / "results" / "regional"
NATIONAL_RESULTS_DIR = OUTPUTS_DIR / "results" / "national"
REGIONAL_FIGURES_DIR = OUTPUTS_DIR / "figures" / "regional"
NATIONAL_FIGURES_DIR = OUTPUTS_DIR / "figures" / "national"
