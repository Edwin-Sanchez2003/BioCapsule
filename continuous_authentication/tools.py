"""
    BioCapsule
    Tools

    This module contains useful functions that can help
    in various situations. They are mainly for simplifying
    small tasks that may be repreated often and could be
    better represented as an abstraction.
"""

import os
import gzip
import json

import numpy as np


# jsonify data dictionary, then gzip to make it more compressed
# after compression, write to the file
def write_to_json_gz(file_path: str, data: dict, comp_lvl: int = 6) -> None:
    json_data = json.dumps(data, cls=DataEncoder)
    encoded_data = json_data.encode("utf-8")
    with open(file_path, "wb") as file:
        file.write(gzip.compress(data=encoded_data, compresslevel=comp_lvl))
    print("Finished writing file!")


# loads a json file into a python dictionary
def load_json_gz_file(file_path: str) -> dict:
    with gzip.open(file_path, "rb") as file:
        return json.load(file)


# writes a dictionary to a json file
def write_to_json(file_path: str, data: dict) -> None:
    with open(file_path, "w") as file:
        file.write(json.dumps(data, cls=DataEncoder))


# loads a json file into a python dictionary
def load_json_file(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return json.load(file)


# json encoder class to deal with numpy
# numbers that aren't json serializeable
class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(DataEncoder, self).default(obj)


# check if an item exists in a list of items
def is_in_list(item, arr: list) -> bool:
    for val in arr:
        if val == item:
            return True
    return False


# gets the files in a directory and adds back their paths
def get_files_with_paths(input_dir: str) -> "list[str]":
    # get results files in the results_dir
    file_names = os.listdir(input_dir)

    # add back dir to get paths
    file_paths = []
    for file_name in file_names:
        file_paths.append(os.path.join(input_dir, file_name))
    return file_paths
