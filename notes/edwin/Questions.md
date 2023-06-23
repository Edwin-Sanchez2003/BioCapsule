# Questions
* What is different about AuthN-AuthZ that helps with Authorization? I understand it implements Heir. RS Role-Based Access Control, but it seems like it basically does the same job.
* How is STPC any different from having the client compute the BC & sending to Auth. Server? They achieve the same thing, but having the client compute the BC is arguably easier. The only thing is the user has to have the RS locally.
*The client doesn't have to do the work to compute the BC (but still retains privacy)*

* Make sure you can use any sort of signature extraction function - as long as it can't be used to get the biometrics.
* Why Iris extraction method used?
* Double-check bc steps in code (walk through with Tyler & Kai)