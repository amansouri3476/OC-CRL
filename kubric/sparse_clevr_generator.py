from typing import List, Optional

import os
import json
from pathlib import Path
from local_launcher import LocalLauncher

import logging
logging.basicConfig(level="INFO")

def generate_sparse_clevr_data(worker_script
                                , max_workers
                                , num_samples
                                , start_id
                                , global_start_id
                                , force_regen
                                , properties_list
                                , fixed_properties_list
                                , subdirectory
                                , number_of_objects
                                , CLEVR_OBJECTS):

    cwd = os.getcwd()

    # get the launcher. The launcher submits a command that generates the data
    
    # script = "singularity exec --nv kubruntu_latest.sif python examples/sparse_clevr_worker.py"

    # to write to directories other than home, you should mount the output dir in home, to your destination
    script = f"singularity exec --nv kubruntu_latest.sif python {worker_script}"
    launcher: LocalLauncher = LocalLauncher(work_dir=cwd, script=script)
    
    # get generator
    max_workers = max_workers
    num_samples = num_samples
    generate_data(launcher
                , max_workers
                , num_samples
                , start_id
                , global_start_id
                , force_regen
                , properties_list
                , fixed_properties_list
                , subdirectory
                , number_of_objects
                , CLEVR_OBJECTS)

import concurrent.futures
def generate_data(launcher
                , max_workers
                , num_samples
                , start_id
                , global_start_id
                , force_regen
                , properties_list
                , fixed_properties_list
                , subdirectory
                , number_of_objects
                , CLEVR_OBJECTS):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # we should use a thread that submits a python command for generating one scene and sample
        for idx in range(num_samples):
            abs_idx = idx + start_id + global_start_id
            config = {} # sample_filename
            if abs_idx < 10:
                filename_str = f"0000{abs_idx}"
            elif abs_idx < 100:
                filename_str = f"000{abs_idx}"
            elif abs_idx < 1000:
                filename_str = f"00{abs_idx}"
            elif abs_idx < 10000:
                filename_str = f"0{abs_idx}"
            else:
                filename_str = f"{abs_idx}"

            if not force_regen and os.path.isfile(f"output/{filename_str}.pickle"):
                logging.info(f"Sample {abs_idx} already exists. Skipping ...")
                continue
            else:
                if force_regen:
                    logging.info(f"Sample {abs_idx} already exists BUT FORCED TO BE REGENERATED.")
                config["--sample_filename"] = filename_str
                config["--properties_list"] = properties_list
                config["--fixed_properties_list"] = fixed_properties_list
                config["--subdirectory"] = subdirectory
                config["--number_of_objects"] = number_of_objects
                config["--CLEVR_OBJECTS"] = CLEVR_OBJECTS
                executor.submit(launcher.run_worker, abs_idx, config)


import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_script", type=str, default="examples/sparse_clevr_worker.py", help="The script to use for generating pairs")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to generate")
    parser.add_argument("--start_id", type=int, default=10, help="The first index according to the whole dataset generated within a batch of jobs")
    parser.add_argument("--global_start_id", type=int, default=0, help="The first index according to the whole dataset (created async)")
    parser.add_argument("--force_regen", type=bool, default=False, help="Whether or not to force regenerating the sample if it exists")
    parser.add_argument("--properties_list", type=str, nargs='+', default="['x','y','c']", help="List of properties that should be changing across t,t+1")
    parser.add_argument("--fixed_properties_list", type=str, default="['l','p']", help="List of properties that should be kept fixed across t,t+1")
    parser.add_argument("--subdirectory", type=str, default="xyc/sphere", help="For each folder corresponding to some number of objects, this field specifies the subfolder for the data to be saved")
    parser.add_argument("--number_of_objects", type=int, default=2, help="Number of objects to be spawned")
    parser.add_argument("--CLEVR_OBJECTS", type=str, default="['sphere']", help="List of available CLEVR objects to be used")
    args = parser.parse_args()

    generate_sparse_clevr_data(args.worker_script
                                , args.max_workers
                                , args.num_samples
                                , args.start_id
                                , args.global_start_id
                                , args.force_regen
                                , args.properties_list
                                , args.fixed_properties_list
                                , args.subdirectory
                                , args.number_of_objects
                                , args.CLEVR_OBJECTS)

if __name__ == "__main__":
    main()
