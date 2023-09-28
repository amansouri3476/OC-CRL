import subprocess
import logging
logging.basicConfig(level="INFO")
import os


def submit_launcher(script
                    , worker_script
                    , max_workers
                    , num_samples
                    , num_samples_per_job
                    , global_start_id
                    , force_regen
                    , properties_list
                    , fixed_properties_list
                    , subdirectory
                    , number_of_objects
                    , CLEVR_OBJECTS
                    ):

    n_batch = num_samples // num_samples_per_job
    for batch in range(n_batch):
        start_id = batch * num_samples_per_job
        args = [f"{worker_script}"
                , f"{max_workers}"
                , f"{num_samples_per_job}"
                , f"{start_id}"
                , f"{global_start_id}"
                , f"{force_regen}"
                , f"{properties_list}"
                , f"{fixed_properties_list}"
                , f"{subdirectory}"
                , f"{number_of_objects}"
                , f"{CLEVR_OBJECTS}"
                ]
        args = " ".join(args)
        cmd = f"sbatch {script} {args}"
        logging.info(f"Submitted run.sh to generate samples {batch * num_samples_per_job + global_start_id} to {global_start_id + (batch+1) * num_samples_per_job - 1} ")
        result_code = subprocess.run(cmd.split(" "), cwd=os.getcwd())


import argparse
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher_path", type=str, default="/home/user/kubric/run.sh", help="The path to the launcher that submits each job")
    parser.add_argument("--worker_script", type=str, default="examples/sparse_clevr_worker.py", help="The script to use for generating pairs")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to generate")
    parser.add_argument("--num_samples_per_job", type=int, default=50, help="Number of samples to generate per each submitted job")
    parser.add_argument("--global_start_id", type=int, default=0, help="Startind index of the large batch acquired after all jobs are done.")
    parser.add_argument("--force_regen", type=bool, default=False, help="Whether or not to force regenerating the sample if it exists.")
    parser.add_argument("--properties_list", type=str, nargs='+', default="['x','y','c']", help="List of properties that should be changing across t,t+1")
    parser.add_argument("--fixed_properties_list", type=str, default="['l','p']", help="List of properties that should be kept fixed across t,t+1")
    parser.add_argument("--subdirectory", type=str, default="xyc/sphere", help="For each folder corresponding to some number of objects, this field specifies the subfolder for the data to be saved")
    parser.add_argument("--number_of_objects", type=int, default=2, help="Number of objects to be spawned")
    parser.add_argument("--CLEVR_OBJECTS", type=str, default="['sphere']", help="List of available CLEVR objects to be used")
    args = parser.parse_args()
    assert args.num_samples_per_job < args.num_samples, "The number of samples to be generated per job should be smaller than the total number of samples."
    submit_launcher(args.launcher_path
                    , args.worker_script
                    , args.max_workers
                    , args.num_samples
                    , args.num_samples_per_job
                    , args.global_start_id
                    , args.force_regen
                    , args.properties_list
                    , args.fixed_properties_list
                    , args.subdirectory
                    , args.number_of_objects
                    , args.CLEVR_OBJECTS
                    )

if __name__ == "__main__":
    main()

# How to use this file?
# python3 submit_scripts.py --num_samples 1000 --num_samples_per_job 20 --global_start_id 20000 --force_regen True --worker_script examples/sparse_clevr_worker.py
# python3 submit_scripts.py --num_samples 100 --num_samples_per_job 50 --global_start_id 0 --force_regen True --worker_script examples/sparse_clevr_worker.py --properties_list "x" "y" "c" "p" --fixed_properties_list "['l']" --subdirectory "xycp" --number_of_objects 2 --CLEVR_OBJECTS "['cube']"