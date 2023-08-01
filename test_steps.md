# Testing Steps - High Level

## Generate Data
Generate data from the original file formats, extract important data and use for testing purposes.

## Test Code
Train classifiers for each person using the generated data, then store the probabilities so that we can do further result gathering.

1. Load dataset into a standard format abstract into a function that returns the data in the format we want.
2. Loop over each subject, focusing on this subject
    a. get this subject's train & validation data as positive samples
    b. get train & validation data for every other subject as negative samples
    c. train a classifier
    d. do threshold tuning
    e. Perform the test!
        i. Get the performance per subject, per session
        ii. Get probability of yes or no authentication from classifier
        iii. get the following scores?:
            1. TP
            2. FP
            3. TN
            4. FN
        iv. Store values into easy access json


## Evaluation Code 
Read in metrics, get FAR/FRR from output, play with threshold, etc. Idea is to get as much information from the results of a single test as possible.

1. Get average performance across each subject (avg of session tests)
2. get average performance across the entire dataset for a single pos subject (avg for a single subject's classifier)
3. get global performance across the entire test - sum up all tp, fp, tn, fn