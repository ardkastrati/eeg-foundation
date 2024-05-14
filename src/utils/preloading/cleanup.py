import os
import shutil
from socket import gethostname
import sys
import yaml


if __name__ == "__main__":
    print(f"Teardown called on {gethostname()}", file=sys.stderr)
    print(f"But don't teardown at the moment", file=sys.stderr)
    exit(-1)

    # Load main config file
    main_config_file = "/home/maxihuber/eeg-foundation/configs/experiment/maxim.yaml"
    with open(main_config_file, "r") as file:
        config = yaml.safe_load(file)
        STORDIR = config["data"]["STORDIR"]

    print(f"Teardown called on {gethostname()}", file=sys.stderr)
    if os.path.exists(STORDIR):
        print(f"Removing all files and subdirectories in: {STORDIR}", file=sys.stderr)
        shutil.rmtree(STORDIR)
        print(f"Successful removal of all files in {STORDIR}", file=sys.stderr)
