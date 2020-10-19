# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
from math import pi, sqrt, exp
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

def resize_img(img, Gx, axs):
    """
    resize the img matrix to apply the 1D kernel on a given axis.
    
    :type img: array
    :par img: the matrix which represents the picture
    
    :type Gx: array
    :par Gx: it is the array which contains the values of the kernel
    
    :type axs: int
    :par axs: the axis on which the adjustment must be done (0: rows, 1: cols)
    
    """  
    
    # Define variables:
    m = img.shape[0]
    n = img.shape[1]
    k = Gx.shape[0]  
    
    # Workout image:

    if axs == 0:
        
        # Check if the size is correct or must be adjusted:
        fit = m / k 
        if fit.is_integer():
            return img
        else:
            pxa = m - int(fit) * k
            
            # Do the adjustment:
            if pxa % 2 == 0:
                
                zeros = np.zeros( (m, pxa/2) )
                img = np.concatenate((zeros, img, zeros), axis=1)
                
            else:
                
                zeros_d = np.zeros( ( (m, int( (pxa - 1)/2 )) ) )
                zeros_u = np.zeros( ( m, (int( (pxa -1)/2 + 1)) ) )
                img = np.concatenate((zeros_d, img, zeros_u), axis=1)
            
            return img, fit
            
        if axs == 1:
            
            fit = n / k
            if fit.is_integer():
                return img
            else:
                pxa = n - int(fit) * k
                
                if pxa % 2 == 0:
                    
                    zeros = np.zeros( (pxa/2, n) )
                    img = np.concatenate((zeros, img, zeros), axis=0)
                    
                else:
                    
                    zeros_l = np.zeros( ( (int( (pxa - 1)/2 ), n) ) )
                    zeros_r = np.zeros( ( (int( (pxa -1)/2 + 1), n) ) )
                    img = np.concatenate((zeros_l, img, zeros_r), axis=0)


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

