from setuptools import find_packages, setup

setup(
    name="active_adaptation",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    package_dir={
        "active_adaptation": "active_adaptation",
        "active_adaptation_projects": "projects",
    },
    packages=[
        "active_adaptation",
        "active_adaptation_projects",
        "scripts",
    ],
    version="0.1.2",
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "moviepy",
        "imageio",
        "einops",
        "av", # for moviepy
        "pandas",
        "termcolor",
        "pygame", # for game controller
    ],
)
