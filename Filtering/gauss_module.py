# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
#from scipy.signal import convolve2d as conv2
from math import sqrt,pi,exp
import scipy.signal



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    
    #...
    x = range(-int(3*sigma),int(3*sigma))
    Gx = [1 / (sigma * sqrt(2*pi)) * exp(-float(i)**2/(2*sigma**2)) for i in x]
    
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
  
    #Set the Parameters:
    X  = np.arange(-3*int(sigma), 3*int(sigma) + 1)
    Dx = np.zeros(X.shape)
    m = X.shape[0] // 2
    
    # Build the array:
    for x in range(-m, m+1):
        t1 = - (1 / ( (sigma**3) * sqrt(2*pi) ))
        t2 = x * exp( -x**2 / (2 * sigma**2) )
        g = t1 * t2
        Dx[x + m] = g
        
    return Dx, x



def gaussderiv(img, sigma):

    #...
    
    return imgDx, imgDy

