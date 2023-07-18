import json
import gzip


def main():
    compress_data("./m432_01_r07_i1_0.json")


def compress_data(file_name:str, comp_lvl:int=6):
    # open JSON file
    data = load_json_file(file_path=file_name)
    # Convert to JSON
    json_data = json.dumps(data, indent=2)
    # Convert to bytes
    encoded = json_data.encode('utf-8')
    # Compress
    compressed = gzip.compress(encoded, compresslevel=comp_lvl)
    
    # write to compressed file
    new_file_name = f"{file_name}.gz"
    with open(new_file_name, "wb") as file:
        file.write(compressed)


# writes a dictionary to a json file
def write_to_json(file_path:str, data:dict)-> None:
    with open(file_path, "w") as file:
        file.write(json.dumps(data))


# loads a json file into a python dictionary
def load_json_file(file_path:str)-> dict:
    with open(file_path, "r") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
