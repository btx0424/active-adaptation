import os
import importlib


dir_path = os.path.dirname(os.path.realpath(__file__))
for file in os.listdir(dir_path):
    if file.startswith("_"):
        continue
    if file.endswith(".py") and file != "__init__.py": # file module
        importlib.import_module(f".{file[:-3]}", __package__)
    elif os.path.isdir(os.path.join(dir_path, file)): # dir module
        importlib.import_module(f".{file}", __package__)

