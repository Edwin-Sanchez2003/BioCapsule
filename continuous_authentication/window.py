"""
    BioCapsule
    Window Code

    This module contains code related to how the test
    uses the window size to affect authentication decisions.
"""

import statistics

def main():
    test_window_avg()


def test_window_avg():
    arr = [0.1, 0.5, 0.2, 0.3, 0.7]
    window_size = 3
    avg_preds = apply_window_to_probabilities(
        window_size=window_size,
        preds=arr,
        avg_fn=simple_average
    ) # end apply_window_to_probabilities

    print(f"Window size: {window_size}")
    print(f"Arr: {arr}")
    print(f"Avg: {avg_preds}")


# steps through predictions, using a specific window size
# and averaging algorithm to get a new probability for a 
# given list of probabilities. Used as a wrapper function
# to be passed a function for averaging the results.
def apply_window_to_probabilities(
        window_size:int,
        preds:"list[float]",
        avg_fn
    ) -> "list[float]":
    """
    This function applies a window to the predicted probabilities
    straight from a classifier. It takes each probability, factors
    in the previously stored probabilities (using the window size
    to determine how far back to go) and then uses those scores to
    generate a new probabilty.

    Inputs:
    -------
    window_size : int
        How many predictions to look at when determining whether
        or not to authenticate a user. When set to 1, it will only
        look at the currently sampled classification probability
        to determine user authenticity. When greater than one,
        the function will use the current sampled classification 
        probability along with previous classification probabilities.

    preds : list[float]
        The predictions for a given user over time. The time interval
        is determined by a parameter when loading the data, this function
        does not account for time interval. The given list of probabilities
        should be the "yes" class probabilities, as in the probability that
        this is the "correct" user. In many cases, this will be class 1.

    avg_fn : function
        A function that takes a prediction, the predictions up until
        the current prediction, the index of the current prediction,
        and the window size. This will generate the new prediction value 
        for a single predicted probability value.
    """
    new_preds = avg_fn(preds=preds, window_size=window_size)
    # new predicitons MUST be of the same length as the old ones
    assert len(new_preds) == len(preds)

    return new_preds
# end apply window to probabilities


# a simple averaging function for probabilities
# takes a window size and averages across that window
# using the current time frame and previous times
# starts from the last prediction, sum
# backwards until we have hit our window size
def simple_average(preds:"list[float]", window_size:int)-> float:
    new_preds = []
    for i in range(len(preds)):
        # slice to the current time, including the current time
        current_preds = preds[:i+1]
        # slice only the window size, if applicable
        if len(current_preds) >= window_size:
            slice_point = len(current_preds)-window_size
            current_preds = current_preds[slice_point:]
        new_preds.append(statistics.mean(current_preds))
    return new_preds
# end simple_average


# a weighted average that scales based on how
# recent the the frame is
# weighted average = (sum(vars*weights)) / sum(weights))
def time_weighted_average(preds:"list[float]", window_size:int)-> float:
    # get the weights for each frame of the window
    weights = []
    weights_sum = 0
    for i in range(window_size):
        # oldest frame gets a weight proportional to
        # 1/window_size -> smallest weight
        # newest frame gets a weight proportional to
        # window_size/window_size -> full weight
        weight = (i+1) / window_size
        weights.append(weight)
        weight_sum += weight
    # end weight generation

    # loop over predictions
    new_preds = []
    for i in range(len(preds)):
        # slice to the current time, including the current time
        current_preds = preds[:i+1]
        # slice only the window size, if applicable
        if len(current_preds) >= window_size:
            slice_point = len(current_preds)-window_size
            current_preds = current_preds[slice_point:]
        # apply weights to all values in the current predictions
        weighted_pred_sum = 0
        for weight_index, current_pred in enumerate(current_preds):
            #  oldest probability is the first index,
            # newest probability is the final index
            weighted_pred_sum += current_pred*weights[weight_index]
        # end for loop weighting probabilities
        # for weighted average, sum the weighted current predictions
        # and divide by the sum of the weights
        new_preds.append(weighted_pred_sum/weights_sum)
    # end loop over predictions to generate new weighted values for
# end time_weighted_average


if __name__ == "__main__":
    main()
