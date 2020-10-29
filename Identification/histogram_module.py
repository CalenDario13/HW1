import numpy as np
from numpy import histogram as hist
from collections import defaultdict
import time
#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module
import time



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    bins = np.linspace(start=0, stop=256, num=num_bins + 1)
    interval_values = [x for x in zip(bins[0:-1], bins[1:])]
    print(interval_values)
    def change_values(x):
        for idx, interval in enumerate(interval_values):
            if x < interval[1] and x >= interval[0]:
                return idx

    image=list(map(change_values, img_gray.reshape(-1)))
    hists=np.zeros(num_bins,dtype=int)
    for i in image:
        hists[i] += 1
    hists = hists / np.sum(hists)

    return hists, bins



#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    s = time.time()
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    R,G,B=np.dsplit(img_color_double, img_color_double.shape[-1])
    bins = np.linspace(start=0, stop=256, num=num_bins + 1)
    interval_values=[x for x in zip(bins[0:-1], bins[1:])]
    def change_values(x):
        for idx,interval in enumerate(interval_values):
            if x < interval[1] and x >= interval[0]:
                return idx

    r = list(map(change_values, R.reshape(-1)))
    g = list(map(change_values, G.reshape(-1)))
    b = list(map(change_values, B.reshape(-1)))
    hists = np.zeros((num_bins, num_bins, num_bins))
    # Loop for each pixel i in the image
    for coordinate in zip(r,g,b):
        hists[coordinate] += 1


    #Normalize the histogram such that its integral (sum) is equal 1
    hists= hists/np.sum(hists)

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    print("time:", time.time() - s)
    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch {}'
    assert img_color_double.dtype == 'float', 'incorrect image type'
    R, G, B = np.dsplit(img_color_double, img_color_double.shape[-1])
    G = G.reshape(128, 128)
    R = R.reshape(128, 128)

    bins = np.linspace(start=0, stop=256, num=num_bins + 1)
    interval_values = [x for x in zip(bins[0:-1], bins[1:])]

    def change_values(x):
        for idx, interval in enumerate(interval_values):
            if x < interval[1] and x >= interval[0]:
                return idx
    r = list(map(change_values, R.reshape(-1)))
    g = list(map(change_values,G.reshape(-1)))



    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    for coordinate in zip(r,g):
        hists[coordinate] += 1
    hists = hists / np.sum(hists)
    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists




#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    #Define a 2D histogram  with "num_bins^2" number of entries

    bins = np.linspace(start=-6, stop=6.01, num=num_bins)
    interval_values = [x for x in zip(bins[0:-1], bins[1:])]

    [imgDx, imgDy] = gauss_module.gaussderiv(img_gray, 3.0)

    imgDx[imgDx > 6] = 6
    imgDx[imgDx < -6] = -6

    imgDy[imgDy > 6] = 6
    imgDy[imgDy < -6] = -6
    print("dxdy_hist_interval",interval_values)
    def change_values(x):
        for idx, interval in enumerate(interval_values):
            if x < interval[1] and x >= interval[0]:
                return idx

    imgDx = list(map(change_values, imgDx.reshape(-1)))
    imgDy = list(map(change_values, imgDy.reshape(-1)))
    hists = np.zeros((num_bins, num_bins))


    for coordinate in zip(imgDx, imgDy):
        hists[coordinate] += 1
    hists = hists / np.sum(hists)
    hists = hists.reshape(hists.size)
    return hists



def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name

