import numpy
import math

def kapur(search_agent, image,noofthresholds):
    histogram, _ = numpy.histogram(image, bins=range(256), density=True)
    cumulativehistogram = histogram.cumsum()

    final_entropy = 0
    for i in range(noofthresholds):
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

