# Funderingar:

# - Automatiserad testning
#

print 'importing'
import sys, csv
import GPyOpt
from numpy.random import seed
from numpy import array
import numpy as np

from numpy import vstack
seed(12345)
# np.set_printoptions(precision=3)
bounds = [(0.001,0.080),(-0.122,0.122),(-0.122,0.122),(0.005,0.040),(0,1),(0.001,0.524),(0.101,0.160),(0,1)]
max_iter=1

# :param *f* the function to optimize. Should get a nxp numpy array as imput and return a nx1 numpy array.
def myf(x):
	header = 'MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency'
	print header
	print np.array_repr(x[0]).replace('\n', '').replace('\t', '')
	speed = input('Input 1/average_speed = ')
	output = np.array(float(speed))

	with open("data_new.py",'a') as f:
		np.savetxt(f, x, delimiter=",")
	# 	for item in x:
	# 		f.write("%s\n" % str(np.array_repr(item).replace('\n', '').replace('\t', '')))
	with open("readings_new.py",'a') as f:
		f.write("%s\n" % str(output))
	return output

print 'load data'
from custom import *
X = np.array([MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency])
Y = 1/average_speed
data = X
readings = Y
from default import *
X = ([MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency])
Y = 1/average_speed
data = vstack((data,X))
readings = vstack((readings,Y))
from lowstiffness import *
X = ([MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency])
Y = 1/average_speed
data = vstack((data,X))
readings = vstack((readings,Y))
from msh import *
X = ([MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency])
Y = 1/average_speed
data = vstack((data,X))
readings = vstack((readings,Y))
from msl import *
X = ([MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency])
Y = 1/average_speed
data = vstack((data,X))
readings = vstack((readings,Y))


# READ COLLECTED DATA
# import string
# all=string.maketrans('','')
# nodigs=all.translate(all, string.digits)
# row.translate(all, nodigs)

with open("data_new.py", "r") as f:
	rows = np.loadtxt(f,delimiter=',')
	print 'length of data: ' + str(len(rows))
	for X in rows:
		data = vstack((data,X))

with open('readings_new.py', "r") as f:
	rows = np.loadtxt(f,delimiter='\n')
	print rows
	for Y in rows:
		readings = vstack((readings,Y))

print 'starting script'
BOnao = GPyOpt.methods.BayesianOptimization(myf,bounds,X=data,Y=readings)
N_iter = 50
for i in range(N_iter):
	if BOnao.run_optimization(max_iter) == 0: break
	BOnao.save_report()
BOnao.plot_convergence()

''' >>> help(GPyOpt.methods.BayesianOptimization)
	run_optimization(self, max_iter=None, n_inbatch=1, acqu_optimize_method='fast_random', acqu_optimize_restarts=200, batch_method='predictive', eps=1e-08, n_procs=1, true_gradients=True, verbose=True)
	 |      Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)
	 |
	 |      :param max_iter: exploration horizon, or number of acquisitions. It nothing is provided optimizes the current acquisition.
	 |          :param n_inbatch: number of samples to collected everytime *f* is evaluated (one by default).
	 |      :param acqu_optimize_method: method to optimize the acquisition function
	 |          -'DIRECT': uses the DIRECT algorithm of Jones and Stuckmann.
	 |          -'CMA': uses the Covariance Matrix Adaptation Algorithm.
	 |              -'brute': Run local optimizers in a grid of points.
	 |              -'random': Run local optimizers started at random locations.
	 |          -'fast_brute': the same as brute but runs only one optimizer in the best location. It is used by default.
	 |          -'fast_random': the same as random but runs only one optimizer in the best location.
	 |      :param acqu_optimize_restarts: numbers of random restarts in the optimization of the acquisition function, default = 20.
	 |          :param batch_method: method to collect samples in batches
	 |          -'predictive': uses the predicted mean in the selected sample to update the acquisition function.
	 |          -'lp': used a penalization of the acquisition function to based on exclusion zones.
	 |          -'random': collects the element of the batch randomly
	 |      :param eps: minimum distance between two consecutive x's to keep running the model
	 |      :param n_procs: The number of processes used for evaluating the given function *f* (ideally nprocs=n_inbatch).
	 |      :param true_gradients: If the true gradients (can be slow) of the acquisition ar an approximation is used (True, default).
	 |      :param save_interval: number of iterations after which a file is produced with the current results.
'''
