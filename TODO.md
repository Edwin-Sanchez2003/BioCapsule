BioCapsule

# TODO List
* ~~Re-write code to be easier to understand~~
* ~~Account for No-Face/Multi-Face detection for MTCNN~~
    * store the number of NoFace/Multi Face in final report
* Run FaceNet (use_bc=False, model_type="facenet", platform="single") testing, but with threshold set based on what gets us to a FAR of 0.1% and 1.0%. Check if Facenet performance is better.
* Get Results for different window sizes.
    * pick a **few important rows???** & mess with the window size & averaging method.
    * main focus: bc/no bc, so lock all other variables in place (arface & single platform)
* Re-Run the experiment with all using a single reference subject and all using a different reference subject, to verify bc performance
 