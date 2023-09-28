import subprocess
import os
import concurrent.futures

import logging
logging.basicConfig(level="INFO")


class LocalLauncher():
    def __init__(self, work_dir, script):
        self.work_dir = work_dir
        self.script = script

    def run_worker(self, idx, config):

        args = " ".join([f"{key}={value}" for key, value in config.items()])
        cmd = f"{self.script} {args}"
        logging.info(f"[Sample {idx}] Executing command: {cmd}")
        result_code = subprocess.run(cmd.split(" "), cwd=self.work_dir)