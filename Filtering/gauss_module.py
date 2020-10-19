# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    """
    it generates the domain of the Function and its values.
    
    :type sigma: the standard deviation of the Normal distribution
    :par sigma: float
    
    """  
    
    # Set parameters 

    X  = np.arange(-3*int(sigma), 3*int(sigma) + 1)
    Gx = np.zeros(X.shape)
    m = X.shape[0] // 2
    
    # Build the array:
    for x in range(-m, m+1):
        t1 = 1/(sqrt(2 * pi) * sigma)
        t2 = exp(-x**2  / 2 * sigma**2)
        g = t1 * t2
        Gx[x + m] = g
    
    # Normalization
    w = np.sum(Gx)
    Gx = 1/w * Gx
    
    return Gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    
    #...

    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    #...
    
    return Dx, x



def gaussderiv(img, sigma):

    #...
    
    return imgDx, imgDy

