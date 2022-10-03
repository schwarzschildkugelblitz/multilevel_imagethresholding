import numpy
import math

def kapur(search_agents,histogram):

    search_agent = [0]
    search_agent= numpy.concatenate((search_agent, search_agents), axis=None)
    search_agents = [255]
    search_agent= numpy.concatenate((search_agent, search_agents), axis=None)
    numpy.sort(search_agent)
    cumulativehistogram = histogram.cumsum()

    final_entropy = 0
    for i in range(len(search_agent)-1):
        # Thresholds
        th1 = int(search_agent[i] + 1)
        th2 = int(search_agent[i + 1])

        # Cumulative histogram
        hc_val =  cumulativehistogram[th2] -  cumulativehistogram[th1 - 1]

        # Normalized histogram
        h_val = histogram[th1:th2 + 1] / hc_val if hc_val > 0 else 1

        # entropy
        entropy = -(h_val * numpy.log(h_val + (h_val <= 0))).sum()

        # Updating total entropy
        final_entropy += entropy

    return final_entropy

def otsu(search_agent,histogram):

    # Cumulative histograms
    c_hist = numpy.cumsum(histogram)
    cdf = numpy.cumsum(numpy.arange(len(histogram)) * histogram)


    e_thresholds = [0]
    e_thresholds.extend(search_agent)
    e_thresholds.extend([len(hist) - 1])
    variance =0 
    for i in range(len(e_thresholds) - 1):
        t1 = e_thresholds[i] + 1
        t2 = e_thresholds[i + 1]

        # Cumulative histogram
        weight = c_hist[t2] - c_hist[t1 - 1]

        # Region CDF
        r_cdf = cdf[t2] - cdf[t1 - 1]

        # Region mean
        r_mean = r_cdf / weight if weight != 0 else 0

        variance += weight * r_mean ** 2

    return variance

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "kapur": ["kapur", 0, 254,2],
        "otsu": ["otsu", 0, 255,2],
    }
    return param.get(a, "nothing")