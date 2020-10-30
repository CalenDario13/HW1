import numpy as np
import pandas as pd
import csv
from time import time
import match_module

from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import SparkTrials
import mlflow

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import threading
import random

# Load Pictures:
global model_images, query_images

with open('model.txt') as fp:
    model_images = fp.readlines()
model_images = [x.strip() for x in model_images] 

with open('query.txt') as fp:
    query_images = fp.readlines()
query_images = [x.strip() for x in query_images] 

# Grid of Parameters
param_grid = {
    
    'hist_type' : hp.choice('hist_type', ['grayvalue', 'dxdy', 'rgb', 'rg']),
    'dist_type' : hp.choice('dist_type', ['chi2', 'intersect', 'l2']),
    'num_bins' : hp.quniform('num_bins', 5, 155, 5)
}

# Initizialize the file:
out_file = 'out_file.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
writer.writerow(['ITERATION', 'dist_type', 'hist_type', 'num_bins', 'run_time', 'score'])
of_connection.close()

# Objective Function:
    
def objective(param):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    
    # Keep track of evals
    global ITERATION, model_images, query_images
    
    ITERATION += 1
    
    # Get the parameters
    params = sample(param_grid)
  
    
    # Execute the function and get the score:
    
    start = time()
    
    [best_match, D] = match_module.find_best_match(model_images, query_images, 
                                                   params['dist_type'], params['hist_type'],int(params['num_bins']))

    num_correct = sum( best_match == range(len(query_images)) )
    score = 1.0 * num_correct / len(query_images)
    
    run_time = time() - start
    
    # Loss must be minimized
    loss = 1 - score

    # Write to the csv file ('a' means append)
    of_connection = open('out_file.csv', 'a')
    writer = csv.writer(of_connection)
    writer.writerow([ITERATION, params['dist_type'], params['hist_type'], params['num_bins'], run_time, score])
    of_connection.close()
    
    # Dictionary with information for evaluation
    return {'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'loss': loss, 
            'status': STATUS_OK}

# Optimization:

global ITERATION
ITERATION = 0
MAX_EVALS = 100

tpe_algorithm = tpe.suggest
bayes_trials = Trials()

spark_trials = SparkTrials(parallelism= 12)
with mlflow.start_run():
    best = fmin(fn = objective, space = param_grid, algo = tpe_algorithm, 
                max_evals = MAX_EVALS, trials = spark_trials, rstate = np.random.RandomState(50))
    



# Simulation:
    
global ITERATION, model_images, query_images, param_grid

with open('model.txt') as fp:
    model_images = fp.readlines()
model_images = [x.strip() for x in model_images] 

with open('query.txt') as fp:
    query_images = fp.readlines()
query_images = [x.strip() for x in query_images] 

param_grid = {
    
    'hist_type' : ['grayvalue', 'dxdy', 'rgb', 'rg'],
    'dist_type' : ['chi2', 'intersect', 'l2'],
    'num_bins' : list(np.arange(5,155, 5))
}

out_file = 'out_file.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
writer.writerow(['ITERATION', 'dist_type', 'hist_type', 'num_bins', 'score'])
of_connection.close()


with open('model.txt') as fp:
    model_images = fp.readlines()
model_images = [x.strip() for x in model_images] 

with open('query.txt') as fp:
    query_images = fp.readlines()
query_images = [x.strip() for x in query_images]  

def get_score(n):
    
    global ITERATION, model_images, query_images
    ITERATION = 0
    for _ in range(n):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        [best_match, D] = match_module.find_best_match(model_images, query_images, 
                                                       params['dist_type'], params['hist_type'],int(params['num_bins']))
    
        num_correct = sum( best_match == range(len(query_images)) )
        score = 1.0 * num_correct / len(query_images)
        of_connection = open('out_file.csv', 'a')
        
        writer = csv.writer(of_connection)
        writer.writerow([ITERATION, params['dist_type'], params['hist_type'], params['num_bins'], score])
        of_connection.close()
        
        ITERATION += 1
    
from multiprocessing import Process
  
if __name__ == '__main__':
    
    r = [50,50,50,50,50,50]
    process = []
    for i in range(len(r)):
        p = Process(target=get_score, args=(r[i],))
        p.start()
        process.append(p)
    for p in process:
        p.join()   
   

# Work on the data:
    
df = pd.read_csv('out_file.csv')
hists = df['hist_type'].unique()


# mean Score Analysis:
avg_score = df.groupby(['hist_type', 'dist_type']).mean().sort_values('score', ascending = False).reset_index()
    
sns.catplot(x='hist_type', y="score", hue="dist_type", 
            kind="bar", orient = 'v', edgecolor=".2",
            legend=False,  aspect=1.5, data=avg_score)
plt.xlabel('')
plt.ylabel('average score')
plt.title('Mean Score Analysis', size = 15, pad = 10)
plt.legend(loc='upper right')
plt.savefig('image.jpg')

# Number of bins vs score:
    
fig, axs = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey = True)
axs = axs.ravel()

hists = df['hist_type'].unique()
for i in range(len(hists)):
    sns.scatterplot(x="num_bins", y="score", hue="dist_type",
                ax = axs[i], legend = False, s = 50,
                hue_order = ['intersect', 'chi2', 'l2'],
                palette = ['green', 'orange', 'blue'],
                data = df[df['hist_type'] == hists[i]].sort_values('dist_type'))

    axs[i].set_title(' '.join(['hyst_type =', hists[i]]), size = 14)


l2_patch = mpatches.Patch(color='green', label='intersect')
chi2_patch = mpatches.Patch(color='blue', label='l2')
intersect_patch = mpatches.Patch(color='orange', label='chi2')
fig.legend(handles = [l2_patch, chi2_patch, intersect_patch], ncol = 3,
           bbox_to_anchor=(0.62, 0.96), fontsize = 'medium', frameon = False)

fig.suptitle('Number of Bins vs Score', size = 24, va = 'center', y = .97)
fig.tight_layout(pad=2)
plt.savefig('image.jpg')



    
    
