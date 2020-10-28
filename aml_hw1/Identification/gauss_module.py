# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
import time


"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):

    x=np.arange(-3*sigma,3*sigma+1, 1)
    Gx=np.array([(1/((np.sqrt(2*np.pi))*sigma))*np.exp(-np.power(x_in, 2.) / (2 * np.power(sigma, 2.)))for x_in in x])

    Gx= Gx/np.sum(Gx)

    return Gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):

    smooth_img=conv2(img,gauss(sigma)[0].reshape(-1,1),mode="same")
    smooth_img=conv2(smooth_img,gauss(sigma)[0].reshape(1,-1),mode="same")
    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    x=np.arange(-3*sigma,3*sigma+1, 1)
    Dx=np.array(
        [-(1 / ((np.sqrt(2 * np.pi)) * np.power(sigma,3)))*x_in *
         np.exp(-np.power(x_in, 2.) / (2 * np.power(sigma, 2.))) for x_in in x]
    )
    Dx = Dx / np.sum(Dx)
    print("here",Dx)
    return Dx, x


def gaussderiv(img, sigma):
    smoothed_image=gaussianfilter(img,sigma)
    print(smoothed_image.shape)
    print(gaussdx(sigma)[0].reshape(1,-1).shape)
    print(gaussdx(sigma)[0].reshape(-1,1).shape)

    imgDx=conv2(smoothed_image,gaussdx(sigma)[0].reshape(1,-1),mode="same")
    imgDy=conv2(smoothed_image,gaussdx(sigma)[0].reshape(-1,1),mode="same")
    return imgDx, imgDy



print(gauss(1))
print(gaussdx(1))


