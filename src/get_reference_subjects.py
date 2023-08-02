"""
    BioCapsule
    Get Reference Subjects

    This module contains functions used to generate feature vectors 
    for reference subjects for running tests. While this code will
    work for any files organized in the same structure, this was made 
    to be used with the LFW dataset.
"""

import os
import random
import copy
import json

from PIL import Image
import numpy as np

import face


def main():
    generate_rs_features(images_dir="./images/lfw/")


# gets features using biocapsule
# these feature can then be used for reference subjects
def generate_rs_features(images_dir:str, num_faces_to_select:int=150, selection_seed:int=42):
    # get a list of all reference subjects
    rs_list = os.listdir(images_dir)

    # randomize selection of people from the dataset
    # use a seed for reproducible results
    rand_gen = random.Random(selection_seed)
    rand_gen.shuffle(rs_list)

    # slice off only the subject count we need to generate
    rs_list = rs_list[:num_faces_to_select]

    # load images to get features from
    folder_paths = []
    for folder_name in rs_list:
        folder_paths.append(os.path.join(images_dir, folder_name))
    
    # loop over subjects, getting the first file, 
    # and loading that image into a list
    file_names = []
    images = []
    for folder_path in folder_paths:
        img_name = os.listdir(folder_path)[0] # get first image for the subject
        img_path = os.path.join(folder_path, img_name) # get the full path

        # load the image into an object
        img = np.array(Image.open(fp=img_path))
        print(img.shape)
        images.append(img)
        file_names.append(img_name)
    # end loop getting images

    # load facenet model
    facenet = face.FaceNet(gpu=0)

    # get mtcnn outputs
    data = []
    for image, file_name in zip(images, file_names):
        # preprocess
        cropped_img, face_count = facenet.preprocess(face_img=image)
        data.append({
            "file_name": file_name,
            "face_count": face_count,
            "mtcnn": copy.deepcopy(cropped_img),
            "features": {}
        })  
    
    # run through facenet
    for img_dict in data:
        # extract features
        features = facenet.extract(face_img=img_dict["mtcnn"], align=False)
        img_dict["features"]["facenet"] = copy.deepcopy(features)
    facenet = None # de allocate facenet model
   
    # load arcface model
    arcface = face.ArcFace(gpu=0)

    # run through arcface
    for img_dict in data:
        features = arcface.extract(face_img=img_dict["mtcnn"], align=False)
        img_dict["features"]["arcface"] = copy.deepcopy(features)

    # store into json file
    write_to_json(file_path="./rs_features.json", data=data)
# end generate_rs_features
    

# writes a dictionary to a json file
def write_to_json(file_path:str, data:dict)-> None:
    with open(file_path, "w") as file:
        file.write(json.dumps(data, cls=DataEncoder))


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



if __name__ == "__main__":
    main()
