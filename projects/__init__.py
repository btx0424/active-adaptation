import importlib
import active_adaptation
from pathlib import Path


for project_dir in Path(__file__).parent.iterdir():
    if project_dir.name.startswith("_"): # ignore hidden directories, e.g. __pycache__
        continue
    if active_adaptation.is_main_process():
        print(f"Importing project {project_dir}")
    importlib.import_module(f".{project_dir.name}", package=__package__)

