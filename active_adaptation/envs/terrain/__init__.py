import importlib
import os
import glob
from pathlib import Path

import active_adaptation

# Get all Python files in current directory
current_dir = Path(__file__).parent
terrain_files = glob.glob(os.path.join(current_dir, "*.py"))

# Import TERRAINS from each file
TERRAINS_ISAAC = {}
TERRAINS_MUJOCO = {}

if active_adaptation.get_backend() == "isaac":
    from active_adaptation.envs.terrain.wrapper import TerrainImporterCfg, TerrainGenerator, TerrainImporter
    for file in terrain_files:
        if file == __file__:  # Skip __init__.py
            continue
        
        # Get module name without .py extension
        module_name = os.path.splitext(os.path.basename(file))[0]
        
        # Import the module
        print(f"Importing terrains from {file}")
        module = importlib.import_module(f".{module_name}", package=__package__)
        
        # Get TERRAINS dict if it exists
        if hasattr(module, "TERRAINS"):
            TERRAINS_ISAAC.update(module.TERRAINS)


    for key, terrain in TERRAINS_ISAAC.items():
        assert isinstance(terrain, TerrainImporterCfg), f"Terrain {key} is not a TerrainImporterCfg"
        terrain: TerrainImporterCfg
        terrain.class_type = TerrainImporter
        if terrain.terrain_type == "generator":
            terrain.terrain_generator.class_type = TerrainGenerator
else:
    from active_adaptation.envs.mujoco import MjTerrainCfg
    path = Path(active_adaptation.__path__[0]) / "assets_mjcf" / "plane.xml"
    TERRAINS_MUJOCO["plane"] = MjTerrainCfg(mjcf_path=str(path))

