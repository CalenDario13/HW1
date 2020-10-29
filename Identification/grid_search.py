import numpy as np
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

global  ITERATION
ITERATION = 0
MAX_EVALS = 200

tpe_algorithm = tpe.suggest
bayes_trials = Trials()


spark_trials = SparkTrials(parallelism= 12)
best = fmin(fn = objective, space = param_grid, algo = tpe_algorithm, 
            max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))


    


