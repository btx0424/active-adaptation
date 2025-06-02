import os
import importlib

dir_path = os.path.dirname(os.path.realpath(__file__))
for file in os.listdir(dir_path):
    if file.endswith(".py") and file != "__init__.py":
        importlib.import_module(f".{file[:-3]}", __package__)