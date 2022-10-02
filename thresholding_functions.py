import numpy
import math

def kapur(search_agents,histogram):

    search_agent = [0]
    search_agent= numpy.concatenate((search_agent, search_agents), axis=None)
    search_agents = [254]
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
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    # Cumulative histograms
    c_hist = np.cumsum(hist)
    cdf = np.cumsum(np.arange(len(hist)) * hist)
    thr_combinations = combinations(range(255), nthrs)

    max_var = 0
    opt_thresholds = None

    # Extending histograms for convenience
    c_hist = np.append(c_hist, [0])
    cdf = np.append(cdf, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # Computing variance for the current combination of thresholds
        regions_var = _get_variance(hist, c_hist, cdf, e_thresholds)

        if regions_var > max_var:
            max_var = regions_var
            opt_thresholds = thresholds

    return opt_thresholds
    variance = 0

    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

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