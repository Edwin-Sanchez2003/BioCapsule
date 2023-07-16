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
* **device:** *(string)* The device the video was recorded on. Either *phone* or *laptop*.
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