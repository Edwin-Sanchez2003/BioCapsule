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
* Store probabilities from each sesssion's test to use for later

## ~~Update No Detections Logic!!!~~ Finished
* For training, if no or multi faces -> remove sample; don't use it if its bad!!!! (pos or neg)
* For testing -> if its the person themselves
    - if **pos subject test**, but bad detection, automatically a false negative w/ 0.000 probability
        - penalize our results due to failure of MTCNN
    - if **neg subject test**, but bad detection, automatically true negative, w/ 0.000 probability