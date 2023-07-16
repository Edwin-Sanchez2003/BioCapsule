# Generate Feature Vectors
This file documents how feature vectors are generated using the `generate_feature_vectors.py` script. The program is written to work with the MOBIO dataset. The goal is to extract feature vectors from the videos or images in the MOBIO dataset and store them for easy use when performing later testing.

Overview:
* File Params
* JSON format

## JSON Format
This is the file format for the stored feature vectors. Each JSON file contains one video's worth of features - a feature vector generated from each frame. The following information is stored in each JSON file:

### Meta Data
This information is stored once in each JSON file, rather than stored in with each frame's feature.
* **location:** *(string)* Which place the video was taken.
* **total_frames:** *(int)* The total number of rames collected from the video.
* **video_name:** *(string)* The name of the video the features are extracted from.
* **subject_ID:** *(string)* The ID of the subject in the video.
* **session_ID:** *(int)* The ID of the session the video belongs to. A number from 1 to 12, inclusive.
* **phase:** *(int)* The phase of the test the video belongs to (either phase 1 or phase 2).
* **device:** *(string)* The device the video was recorded on. Either *phone/mobile* or *laptop*.
* **pre_proc_model:** *(string)* The model used during preprocessing.
* **feat_ext_model:** *(string)* The model used to perform the feature extraction.
* **feat_vect_length:** *(int)* The length of the resulting feature vectors - based on the selected model for feature_extraction.
* **feature_vectors:** *(array of frame data)* An array of data collected from each frame. Details on what data is collected for each frame is below.

### Per Frame Data
The data below is collected for each frame of the video:

* **num_faces_detected:** *(int)* The number of faces detected in  the video. In most cases this will be one, but some videos have no faces.
* **frame_num:** *(int)* The index of the frame from the video that the feature vector came from.
* **time_stamp:** *(float)* The time during the video from which the frame was taken, in miliseconds.
* **feature_vector:** The resulting feature vector (of the person) derived from the frame, after preprocessing and feature extraction.

## Config File Format
This file is a JSON file passed in as a parameter to the program and specifies certain properties of the program and the extraction process. JSON files are named after the subjectID, sessionID, and device that the video came from. The format is specified below:

* **input_path:** *(string)* The directory containing a list of videos OR the path to a single video.
* **output_dir:** *(string)* The directory to store the file. File names are generated based on subjectID, sessionID, and the device that the video came from.
* **json_template_path:** The path to the JSON_DATA_TEMPLATE.json file, which is used as a starting point for the json file to be stored.
* **pre_proc_model:** *(string)* The model to use for preprocessing. *Limited to options set in code.
* **feat_ext_model:** *(string)* The model to use for feature extraction. *Limited to options set in code.
* **max_num_processes:** *(int)* The number of processes to run to maximize CPU utilization. Defaults to 1.
* **apply_partial_occlusion:** *(boolean)* Whether or not to generate black squares randomly on the input frames to make detection harder. *Later need to add parameters here for settings on partial occlusion.* 
* **ignore_pre_sets**: *(boolean)* Tells the program whether or not it should ignore the presets when generating the feature vector JSON files.
* **pre_sets:** *(dict)* Describes other details that will be stored in the output json file. Details below.

### Pre Sets
The pre-sets section in the config file specify important details about the videos that will be stored with the generated JSON file. This includes:

* **location**
* **subject_ID**
* **session_ID**
* **phase**
* **device**

*Descriptions of these pre-sets are the same as described in the JSON format Section of this document.*

These will need to be specified for the program to run. This is to document important details about the video, specific to the MOBIO dataset.

However, if you wish to ingore presets, you can do so with the **ignore_presets** parameter in the config file. These details will be excluded from the generated JSON files.