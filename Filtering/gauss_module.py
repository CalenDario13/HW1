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
    
    #Set the Parameters:
    x  = np.arange(-3*int(sigma), 3*int(sigma) + 1)
    Gx = np.zeros(x.shape)
    
    # Build the array:

    for i in range(x.size):

        t1 = 1 / ( sigma * sqrt(2 * pi) )
        t2 = exp( -x[i]**2 / (2 * sigma**2) )
        g = t1 * t2
        Gx[i] = g
    
    return Gx, x

def resize_img(img, Gx, axs):
    """
    resize the img matrix to apply the 1D kernel on a given axis, 
    axs 1 is to be adjusted for convolution along rows, while axs 0 must be adjusted
    for convolution on columns.
    
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
            pxa = (k * (int(fit) + 1)) - m
            
            if pxa < 0:
                raise TypeError("The function doesn't work properly")
            
            # Do the adjustment:
            if pxa % 2 == 0:
            
                zeros = np.zeros( (int(pxa/2), n) )
                img = np.concatenate((zeros, img, zeros), axis=0)
                
            else:
                
                zeros_u = np.zeros( ( ( int( (pxa - 1)/2 ), n ) ) )
                zeros_d = np.zeros( ( ( int( (pxa -1)/2 + 1), n ) ) )
                img = np.concatenate((zeros_u, img, zeros_d), axis=0)
            
        if axs == 1:
            
            fit = n / k
            if fit.is_integer():
                return img
            else:
                pxa = (k * (int(fit) + 1)) - n
                
                if pxa < 0:
                    raise TypeError("The function doesn't work properly")
                
                if pxa % 2 == 0:
                    
                    zeros = np.zeros( (m, int(pxa/2)) )
                    img = np.concatenate((zeros, img, zeros), axis=0)
                    
                else:
                    
                    zeros_l = np.zeros( ( ( m, int( (pxa - 1)/2 ) ) ) )
                    zeros_r = np.zeros( ( ( m, int( (pxa -1)/2 + 1) ) ) )
                    img = np.concatenate((zeros_l, img, zeros_r), axis=0)
                
    return img

"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    
    """
    it implements a gaussian filter exploiting the linearity of the system,
    namely decomposing it in 2 1D filters applied to the picture.
    
    :type img: array
    :par img: the matrix which represents the picture
    
    :type sigma: float
    :par sigma: the standard deviation of a Normal distribution
    
    """  
    
    # Generate the Kernel:
    Gx, x = gauss(sigma)  
    cp = (Gx.size - 1) // 2
    
    # Adjust the size of the picture:
    
    #img = resize_img(img, Gx, axs = 1)
    #img = resize_img(img, Gx, axs = 0)
    
    # Convolve by rows
    filter1 = np.zeros( img.shape )
    for n in range(img.shape[1]):
        for m in range(img.shape[0] - cp):
            Fmn = 0
            for j in range(Gx.size):
                Fmn += Gx[j] * img[m - cp + j, n]
            filter1[m, n] = Fmn
            
    # Convolve by cols:
    smooth_img = np.zeros(img.shape)
    for m in range(img.shape[0]):
        for n in range(img.shape[1] - cp):
            Fmn = 0
            for j in range(Gx.size):
                Fmn += Gx[j] * img[m, n -cp + j]
            smooth_img[m, n] = Fmn
            
    return smooth_img

"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    """
    it generates the domain of the First Derivate of a Gaussina Function and its values.
    
    :type sigma: the standard deviation of the Normal distribution
    :par sigma: float
    
    """ 
    
    #Set the Parameters:
    x  = np.arange(-3*sigma, 3*sigma + 1)
    Dx = np.zeros(x.size)

    # Build the array:
    for i in range(x.size):
        t1 = - (1 / ( (sigma**3) * sqrt(2*pi) ))
        t2 = x[i] * exp( -x[i]**2 / (2 * sigma**2) )
        g = t1 * t2
        Dx[i] = g
    
    return Dx, x



def gaussderiv(img, sigma):

    #...
    
    return imgDx, imgDy

