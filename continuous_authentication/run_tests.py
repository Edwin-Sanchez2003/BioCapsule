"""
    BioCapsule
    Run Tests

    This file is used to run testing for Continuous Authentication
    performance of the BioCapsule (BC) system with Face Authentication.
"""

from load_mobio_data import load_MOBIO_dataset, SessionData, SubjectData

# Params #
EXTRACTED_MOBIO_DIR = ""
TIME_INTERVAL = 10
FEATURE_EXTRACTION_MODEL = "arcface"
TRAINING_PLATFORM = "single"
USE_BC = False
MULTI_RS = False


def main():
    # load data into a simple to use format
    subjects = load_MOBIO_dataset(
        extracted_MOBIO_dir=EXTRACTED_MOBIO_DIR,
        time_interval=TIME_INTERVAL,
        feature_extraction_model=FEATURE_EXTRACTION_MODEL,
        use_bc=USE_BC,
        multi_rs=MULTI_RS
    ) # end load_MOBIO_dataset function call

    # perform tests per user
    print("Performing Tests...")
    for i in range(len(subjects)):
        print(f"Test with Positive subject as subject {subjects[i].get_subject_id()}")
        single_user_test()



    # collect information from test
    # store the probability classifications so we can extract more data later
    # make sure to document the parameters
    # used for each test


# performs a single test with a given index for which user to use
def single_user_test(
        subjects:"list[SubjectData]",
        subject_index:int,
        training_platform:str,
        window_size:int=1,
    )-> dict:
    subject = subjects[subject_index]

    # get positive train, val, and test samples and labels


    # loop over every subject again
    for i, test_subject in enumerate(subjects):
        # get test, val, and train samples into single arrays
        pass


    # end for testing on each subject
# end singel_user_test function


# takes a list of a list of samples, a list of list of labels,
# and puts them all into a single list for each using list.extend()
def combine_samples_and_labels(
        samples:"list[list[list[float]]]", 
        labels:"list[list[int]]"
    )-> "tuple[list[list[float]], list[int]]":
    out_samples = []
    out_labels = []

    for sample_list in samples:
        out_samples.extend(sample_list)
    
    for label_list in labels:
        out_labels.extend(label_list)

    return (out_samples, out_labels)
# end combine_samples_and_labels


if __name__ == "__main__":
    main()
