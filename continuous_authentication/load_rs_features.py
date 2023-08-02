"""
    BioCapsule
    Load RS Features

    This module loads the features of reference subjects for us
    to use during the the biocapsule generation process.
"""

import tools

def main():
    pass


# generator function to yield out dicts
# containing feature vectors and the reference subject
# file name from lfw
def yield_reference_subject(
        file_path:str, 
        feature_extraction_model:str
    )-> "tuple[str, list[float]]":
    # load in extracted data from the reference subjects
    data = tools.load_json_file(file_path=file_path)

    # yield a feature vector and the file name
    for ref_subj in data:
        yield (ref_subj["file_name"], ref_subj["features"][feature_extraction_model])
# end yield_reference_subject
    



if __name__ == "__main__":
    main()
