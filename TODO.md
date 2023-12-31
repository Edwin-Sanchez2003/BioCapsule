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
    * to time & window size
        * minimize time, maximize window. Time interval is more important so push that to its limit before using the window size.
        * run the test without windowing and find the best time interval with no windowing
        * then see how much shorter we can get the authentication before the system declines
        * authentication
        * simulate a user leaving the screen and getting replaced by an attacker.
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
    - get figure of one pos subject being auth w/diff windows
    - get figure of one neg subject being auth w/diff windows
    - x-axis time, y-axis the probability from the classifier
    - put the threshold that was set for the user as a dotted horizontal line
- start running other tests as needed
- get timed results to see the extra time cost
- write the tuning threshold target value & the far/frr for validation per tuned classifier!!!
- write data extraction code for YouTube Faces, run all tests on theirs for more comparisons!!!
- re-run the experiment for multi-rs, but this time store the rs & fuse w/the current subj's rs (this will get more accurate results for the Evil-Maid type attack. This will also result in more realistic testing and results that aren't counter-intuitive)

- if time, re-run experiment to get results with a more precise threshold

- test in more difficult scenarios? Test live? -> don't try to do too much
- ^^^ Still super important though for really seeing how this system would work in real life on a laptop. Less so for mobile, though.

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
