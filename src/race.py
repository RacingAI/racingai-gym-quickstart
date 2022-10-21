import simulator
import argparse
import shutil
import os
from os import getenv
import subprocess
import importlib
import sys
import tempfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", action="append", dest="repos", default=[])
    parser.add_argument("--pull-location",
                        default=getenv("RACINGAI_PULL_LOCATION", os.path.join(tempfile.gettempdir(), 'racing-ai')))
    args = parser.parse_args()
    assert len(args.repos) > 0, "Must provide at least one github repository"
    # Clone repositories
    if not os.path.isdir(args.pull_location):
        assert not os.path.exists(args.pull_location), \
            f"'{args.pull_location}' exists and is a file"
        os.mkdir(args.pull_location)
    sys.path.append(args.pull_location)
    driver_modules = []
    original_directory = os.getcwd()
    os.chdir(args.pull_location)
    for index, repo in enumerate(args.repos):
        if os.path.exists(str(index)):
            shutil.rmtree(str(index))
        subprocess.call(["git", "clone", repo, str(index)])
        spec = importlib.util.find_spec(os.path.join(f"{str(index)}.driver"))
        driver_modules.append(spec.loader.load_module())
    os.chdir(original_directory)
    print("STARTING RACE")
    simulator.race([module.Driver() for module in driver_modules])
