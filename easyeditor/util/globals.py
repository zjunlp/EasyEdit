from pathlib import Path

import logging
import os

import yaml


def get_handler(path, log_name):
    log_file_path = os.path.join(path, log_name)
    try:
        if not os.path.exists(path):
            print("We are creating the logger files")
            os.makedirs(path)
    except:
        pass
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    return file_handler, stream_handler


# def get_run_dir(dir_name):
#
#     alg_dir = RESULTS_DIR / dir_name
#     if alg_dir.exists():
#         id_list = [
#             int(str(x).split("_")[-1])
#             for x in alg_dir.iterdir()
#             if str(x).split("_")[-1].isnumeric()
#         ]
#         run_id = 0 if not id_list else max(id_list) + 1
#     else:
#         run_id = 0
#     run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
#     run_dir.mkdir(parents=True, exist_ok=True)
#     print(f"Results will be stored at {run_dir}")
#
#     return run_dir
