"""
Generate Feature Vectors from a Dataset
    BioCapsule
    Edwin Sanchez

    Generates feature vectors from a set of images or videos.
    Stores the feature vectors in a JSON file with other metadata.
    This file in particular appends feature vectors generated from
    a different model into an already existing json file.

    Bottleneck: GPU Size/Memory Constraints
"""

# Imports
import os
import time

import numpy as np
import cv2

from extract_MOBIO_data import load_json_gz_file
from extract_MOBIO_data import write_to_json_gz

import face as fr


# load the model used for feature extraction
#model = fr.ArcFace(gpu=0)
model = fr.FaceNet(gpu=0)


def main():
    """
    - load a list of directories to search through
    - get all of the .gz files in a directory
        - open each .gz file
        - iterate through the frame data samples
            - take the mtcnn data, format as an array, 
              pass to feature extractor model
            - store features back into json
            - save json file again
    """

    # the directories to search through for compressed files
    base_path = "./MOBIO_extracted/one_sec_intervals/"
    dirs_to_search = [
        #"but/",
        #"idiap/",
        #"lia/",
        #"uman/",
        #"unis/",
        "uoulu"
    ] # end dirs_to_search

    # get the file paths of all of the files
    file_paths = []
    for data_dir in dirs_to_search:
        search_path = os.path.join(base_path, data_dir)
        found_paths = os.listdir(search_path)
        for path in found_paths:
            full_path = os.path.join(search_path, path)
            file_names = os.listdir(full_path)
            for file_name in file_names:
                file_path  = os.path.join(full_path, file_name)
                file_paths.append(file_path)
    num_files = len(file_paths)

    # loop over files, extracting data
    for i, file_path in enumerate(file_paths):
        tic = time.perf_counter()
        # open .gz file
        data = load_json_gz_file(file_path=file_path)

        # loop over frame_data samples
        frame_data = data["frame_data"]

        # loop over the samples, load the preprocessed frames
        # perform feature extraction on the frame and
        # add back to data
        for sample in frame_data:
            # load the stored frame
            pre_proc_frame = np.array(sample["preprocessing_tensors"]["mtcnn"]) 
            
            # convert to supported integer type
            pre_proc_frame = pre_proc_frame.astype(float)
            
            # shuffle dimensions around
            pre_proc_frame = np.rollaxis(pre_proc_frame, 0, 3)
            flipped_frame = cv2.flip(pre_proc_frame, 1) # flips along vertical axis
            
            #print(pre_proc_frame.shape)
            #cv2.imshow("img", pre_proc_frame)
            #cv2.waitKey(0)

            # run through model (make sure to not re-preprocess)
            feature_vector:np.ndarray = model.extract(
                face_img=pre_proc_frame, # pass in the frame to extract from
                align=False # already did preprocessing step, don't do again
            ) # end model.extract

            flipped_vector = model.extract(
                face_img=flipped_frame,
                align=False
            )

            # add new feature vector to data
            # since sample is a ptr, just add to existing dict
            sample["feature_vectors"]["facenet"] = feature_vector.tolist()
            sample["feature_vectors"]["facenet_flip"] = flipped_vector.tolist()

        # after all samples gotten their appended
        # feature vectors, write back to compressed json file
        write_to_json_gz(file_path=file_path, data=data)
        toc = time.perf_counter()
        print(f"Finished {i+1} of {num_files}: Took {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    main()
