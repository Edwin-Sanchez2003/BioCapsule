# Questions

## Set 1
* What is different about AuthN-AuthZ that helps with Authorization? I understand it implements Heir. RS Role-Based Access Control, but it seems like it basically does the same job.
* How is STPC any different from having the client compute the BC & sending to Auth. Server? They achieve the same thing, but having the client compute the BC is arguably easier. The only thing is the user has to have the RS locally.
*The client doesn't have to do the work to compute the BC (but still retains privacy)*

* Make sure you can use any sort of signature extraction function - as long as it can't be used to get the biometrics.
* Why Iris extraction method used?
* Double-check bc steps in code (walk through with Tyler & Kai)

## Set 2
* **face.py:** what is going on with the `feature` variable??? its shaped like this: (img_cnt, 513). Why?
  * Answered my own question: *img_cnt* -> creates an array for each image. *513* -> 512 values to contain the features for each image. Ans extra number for something else... classification??? Edit: Nope, its the *id* of the subject. a new id for each person in the dataset.