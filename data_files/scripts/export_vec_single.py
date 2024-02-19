"""
This script exports a single vector from a hf repository.
"""

import sys
from loguru import logger
from ANN.abstract_neural_network import AbstractNN

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Invalid number of arguments.")
        sys.exit(1)
    elif sys.argv[1] == "help":
        print("This script exports a single vector from a hf repository.")
        print("Usage: python export_vec_single.py -r <repo_name> -j <json_output_loc>")
        sys.exit(0)
    elif sys.argv[1] != "-r":
        print("Invalid argument.")
        sys.exit(1)
    elif sys.argv[3] != "-j":
        print("Invalid argument.")
        sys.exit(1)
    else:
        repo_name = sys.argv[2]
        json_output_loc = sys.argv[4]
    
    try:
        ann = AbstractNN.from_huggingface(repo_name)
        ann.export_vector(json_output_loc)
    except Exception as emsg: # pylint: disable=broad-except
        logger.error(emsg)
        sys.exit(1)