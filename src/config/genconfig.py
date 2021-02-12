import os
import json
from pathlib import Path
import argparse

# this must be run from src!

root_path = Path(os.getcwd()) 

if root_path.stem != "src":
    exit("not in src directory!")

config_path = root_path / "config"

root_path = str(root_path.absolute())
with open(config_path / 'config.py', 'w') as fp:
    data = {"root": root_path}

    write_string = f"project_info = {data}"
    fp.write(write_string)
