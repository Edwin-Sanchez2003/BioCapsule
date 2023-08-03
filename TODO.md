BioCapsule

# TODO List
* ~~Re-write code to be easier to understand~~
* ~~Account for No-Face/Multi-Face detection for MTCNN~~
    * store the number of NoFace/Multi Face in final report
* ~~Run FaceNet (use_bc=False, model_type="facenet", platform="single") testing, but with threshold set based on what gets us to a FAR of 0.1% and 1.0%. Check if Facenet performance is better.~~
    * it got worse...
* ~~Add code to use different window sizes~~
* Get Results for different window sizes.
    * lock in a **few important starting params** & mess with the window size & averaging method.
    * main focus: bc/no bc, so lock all other variables in place (arface & single platform)
* ~~Re-Run the experiment with all using a single reference subject and all using a different reference subject, to verify bc performance~~


# TODO:
- ~~Re-Run extraction for FaceNet!!!~~
- ~~Get flipped features for arcface as well!!!~~
- ~~run single test to make sure that performance is consistent for facenet & arcface~~
- ~~Updated code to use flipped features for arcface as well~~
- Get updated results for my table!!!
* ~~Store probabilities from each sesssion's test to use for later~~
    * What data to store to get as much results later???
    - ~~threshold per subject (pos)~~
    - ~~classification probabilities -> per session~~
- start making figures & charts
- start running tests for windowing
- start running other tests as needed
- get timed results to see the extra time cost
- write the tuning threshold target value & the far/frr for validation per tuned classifier!!!
- write data extraction code for YouTube Faces, run all tests on theirs for more comparisons!!!

- test in more difficult scenarios? Test live? -> don't try to do too much

## Figures:
- MMOC Comparison Table


## ~~Update No Detections Logic!!!~~ Finished
* For training, if no or multi faces -> remove sample; don't use it if its bad!!!! (pos or neg)
* For testing -> if its the person themselves
    - if **pos subject test**, but bad detection, automatically a false negative w/ 0.000 probability
        - penalize our results due to failure of MTCNN
    - if **neg subject test**, but bad detection, automatically true negative, w/ 0.000 probability

## What was the big error for facenet?
- couple extraction errors
- threshold tuning with flipped frames helped
- **when loading in models for training, make sure that the preprecessed frames are loaded  in consistently. If the model loads the first time and reshapes the image, but doesn't the second time, then we're going to run into issues in performance!!! re-write the code so that extraction and testing will consistently load images for feature extraction**