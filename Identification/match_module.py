import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type) 
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(model_images), len(query_images)))
    for j in range(len(query_images)):
        for i in range(len(model_images)):
            dist = dist_module.get_dist_by_name(query_hists[j], model_hists[i], dist_type)
            D[i, j] = dist
    
    # Find best matches:
    best_match = np.argpartition(D, 1, axis=0)[0, :].reshape(-1)
    
    return best_match, D

    
def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist

    for img_name in image_list:
        
        # Load the image:
        img = np.array(Image.open(img_name)).astype('double')
        if hist_isgray:
            img = rgb2gray(img)
        
        # Compute the given histogram and append:
        hists = histogram_module.get_hist_by_name(img, num_bins, hist_type)
        image_hist.append(hists)
       

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    num_nearest = 5  # show the top-5 neighbors
   
    [best_match, D] = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    
    
    fig, axs = plt.subplots(len(query_images), num_nearest + 1, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    
    # Find index of closest images from the query (each row is a query):
    top = np.argsort(D, axis=0)[:num_nearest, :]
    
    # Plot
    for i in range(len(query_images)):
        
        query_img = np.array(Image.open(query_images[i]))
        axs[i,0].imshow(query_img, vmin=0, vmax=255)
        axs[i,0].set_title(''.join(['Q', str(i)]))
        axs[i,0].axis('off')
        
        for j in range(top.shape[0]):
            
            idx = top[j, i]
            neighbor_img = np.array(Image.open(model_images[idx]))
            
            axs[i,j + 1].imshow(neighbor_img, vmin=0, vmax=255)
            axs[i,j + 1].set_title('{0}{1:.2f}'.format('M', round(D[idx, i], 2)))
            axs[i,j + 1].axis('off')
        
    plt.show()

  