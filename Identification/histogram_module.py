import numpy as np


#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    
    # Prepare the variables:
    space = np.linspace(0, 255, num_bins + 1)
    to_array = img_gray.reshape(img_gray.size)
    
    # Do the binning:         
    binning = np.digitize(to_array, space)
    img_binned = space[binning]
    
    # Create the histogram.
    bins, hists = np.unique(img_binned, return_counts=True)

    # Add the bins that count 0:
    missing_bins = np.setdiff1d(space, bins)
    missing_hists = np.zeros(missing_bins.shape)
    
    bins = np.concatenate((bins, missing_bins))
    hists = np.concatenate((hists, missing_hists))
        
    idx = bins.argsort()
    bins = bins[idx]
    hists = hists[idx]
    hists = hists[1:]
    
    # Normalization:
    hists = (1/np.sum(hists)) * hists
    
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
    
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'
    
    # Iniitilize vars:
    flat = img_color_double.reshape(-1, 3)
    bin_size = 255 / num_bins

    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    
    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        
        # Get the pixels:
        R = flat[i][0]
        G = flat[i][2]
        B = flat[i][1]
        
        #Find the bin:
        idxr = int(R // bin_size)
        idxg = int(G // bin_size)
        idxb = int(B // bin_size)
      
        # Increase by 1 the given position in hist:
        hists[idxr, idxg, idxb] += 1
          
    #Normalize the histogram such that its integral (sum) is equal 1
    hists = 1/np.sum(hists) * hists

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    
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
    
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'
    
    flat = img_color_double.reshape(-1, 3)
    bin_size = 255 / num_bins

    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        
        # Get the pixels:
        R = flat[i][0]
        G = flat[i][1]
        
        #Find the bin:
        idxr = int(R // bin_size)
        idxg = int(G // bin_size)
        
        # Increment the histogram bin which corresponds to the R and G value of the pixel i
        hists[idxr, idxg] += 1

    #Normalize the histogram such that its integral (sum) is equal 1
    hists = 1/np.sum(hists) * hists

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
    
    # Get the Derivatives:
    sigma = 3.
    imgdx, imgdy = gauss_module.gaussderiv(img_gray, sigma)    
    imgd = np.stack([imgdx,imgdy], axis=0)
    flat = imgd.reshape(-1, 2)
    
    # Cap the values:
    flat[flat > 6] = 6
    flat[flat < -6] = -6
    
    # Get the bin_size:
    bin_size = len(range(-6, 6 + 1)) / num_bins 
    
    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    # Fill the histogram:
    for i in range(img_gray.shape[0]*img_gray.shape[1]):
        
        # Get the pixels
        x = flat[i][0]
        y = flat[i][1]
        
        #Find the bin:
        idxx = num_bins // 2 + int(x // bin_size)
        idxy = num_bins // 2 + int(y // bin_size)
        
        # Increase by 1 the given position of hist:
        
        hists[idxx, idxy] += 1
    
    # Normalize hist:
    hists = 1/np.sum(hists) * hists   
    
    #Return the histogram as a 1D vector
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

