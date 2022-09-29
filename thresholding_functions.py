import numpy
import math

def kapur(search_agent,histogram):
    cumulativehistogram = histogram.cumsum()

    final_entropy = 0
    for i in range(len(search_agent)-1):
        # Thresholds
        th1 = search_agent[i] + 1
        th2 = search_agent[i + 1]

        # Cumulative histogram
        hc_val =  cumulativehistogram[th2] -  cumulativehistogram[th1 - 1]

        # Normalized histogram
        h_val = histogram[th1:th2 + 1] / hc_val if hc_val > 0 else 1

        # entropy
        entropy = -(h_val * numpy.log(h_val + (h_val <= 0))).sum()

        # Updating total entropy
        final_entropy += entropy

    return final_entropy

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "kapur": ["kapur", 0, 255,2],
        "otsu": ["otsu", 0, 25,2],
    }
    return param.get(a, "nothing")