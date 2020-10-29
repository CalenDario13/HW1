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

    for idx,query_hist in enumerate(query_hists):
        for idx2,model_hist in enumerate(model_hists):
            D[idx2,idx]=dist_module.get_dist_by_name(query_hist, model_hist, dist_type)

    best_match=np.argmin(D, axis=0)
    return best_match, D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []
    # Compute hisgoram for each image and add it at the bottom of image_hist
    for images_path in image_list:

        image=np.array(Image.open(images_path)).astype('double')
        if hist_isgray:
            image=rgb2gray(image)

        image_hist.append(histogram_module.get_hist_by_name(image, num_bins, hist_type))

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    best_match, D= find_best_match(model_images, query_images, dist_type, hist_type, num_bins)
    num_nearest = 5  # show the top-5 neighbors
    plt.figure(1, figsize=(20, 10))
    graph_index=1
    for i in range(D.shape[1]):
        plt.subplot(D.shape[1], 6, graph_index)
        plt.imshow(np.array(Image.open(query_images[i])), vmin=0, vmax=255)
        graph_index += 1
        plt.title("Q{}".format(i))
        model_scores = D[:,i].reshape(D[:,i].size)
        idx_of_the_top_5=model_scores.argsort()[:num_nearest]
        for model_image in idx_of_the_top_5:
            name = model_images[model_image]
            plt.subplot(D.shape[1], 6, graph_index)
            plt.imshow(np.array(Image.open(name)), vmin=0, vmax=255)
            plt.title("M {:.3f}".format(model_scores[model_image]))

            graph_index += 1

    plt.show()
    


