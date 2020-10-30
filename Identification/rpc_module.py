import matplotlib.pyplot as plt
import histogram_module
import dist_module
import match_module
import pandas as pd
import numpy as np



# compute and plot the recall/precision curve
#
# D - square matrix, D(i, j) = distance between model image i, and query image j
#
# note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image

def plot_rpc(D, plot_color,dist_type,hist_type,num_bins):
    
    recall = []
    precision = []
    num_queries = D.shape[1]
    num_images = D.shape[0]
    assert(num_images == num_queries), 'Distance matrix should be a square matrix'
    
    labels = np.diag([1]*num_images)

    d = D.reshape(D.size)
    l = labels.reshape(labels.size)
    sortidx = d.argsort()
    d = d[sortidx]
    l = l[sortidx]
    tp=0

    for idt in range(len(d)):
        tp += l[idt]

        precision.append(tp/(idt+1))
        recall.append(tp/num_queries)


    plt.plot([1-precision[i] for i in range(len(precision))], recall, plot_color+'-')
    
    '''
    dict_pd={"threshold":d,"recall":recall,"precission":precision,}
    df = pd.DataFrame(dict_pd)
    df["f-value"] = (2*df["recall"] * df["precission"])/(df["recall"]+df["precission"])
    df.to_csv("fscores/{}_{}_{}.csv".format(dist_type,hist_type,num_bins))
    '''
def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):
    
    assert len(plot_colors) == len(dist_types), 'number of distance types should match the requested plot colors'

    for idx in range(len(dist_types) ):

        [best_match, D] = match_module.find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)

        plot_rpc(D, plot_colors[idx],dist_types[idx], hist_type, num_bins)
    

    plt.axis([0, 1, 0, 1]);
    plt.xlabel('1 - precision');
    plt.ylabel('recall');
    
    # legend(dist_types, 'Location', 'Best')
    
    plt.legend( dist_types, loc='best')








