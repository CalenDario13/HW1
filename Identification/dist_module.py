import numpy as np
import math
from scipy.spatial import distance


# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
    minima = np.minimum(x, y)
    hist_intersect = np.sum(minima)
    # res = 1 - hist_intersect
    
    # Normalization
    a = hist_intersect / np.sum(x)
    b = hist_intersect / np.sum(y)
    norm_hist = 0.5 * (a + b)
    res = 1 - norm_hist
    return res
    
  

# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
    
    # distance.euclidean(x,y)
    diff = y - x
    power = diff**2
    summ = np.sum(power)
    res = math.sqrt(summ)
    return res


# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
    
    diff = x - y
    power = diff**2
    summ = x + y
    div = power / summ
    res = np.sum(div)
    return res

    
def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert False, 'unknown distance: %s'%dist_name
  




