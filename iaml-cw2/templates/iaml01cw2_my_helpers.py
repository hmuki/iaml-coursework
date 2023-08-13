##########################################################
#  Python module template for helper functions of your own (IAML Level 10)
#  Note that:
#  - Those helper functions of your own for Questions 1, 2, and 3 should be defined in this file.
#  - You can decide function names by yourself.
#  - You do not need to include this header in your submission
##########################################################
import numpy as np

def get_sample_num(Xtrn, X):
    for i in range(len(Xtrn)):
        if (np.array_equal(Xtrn[i], X)):
            return i
        else:
            continue
    return -1