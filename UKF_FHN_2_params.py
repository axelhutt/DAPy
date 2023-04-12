#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

class Kalman:
	def __init__(self):
		self.filename 				= './'
		self.label_state 			= self.filename+'UKF_FHN_bg.dat'
		self.label_state_false 		= self.filename+'UKF_FHN_bg_false.dat'
		self.label_obs 				= self.filename+'UKF_FHN_obs.dat'
		self.label_obs_false 		= self.filename+'UKF_FHN_obs_false.dat'
		self.label_reconstr_state 	= self.filename+'UKF_FHN_bg_reconstructed.dat'
		self.label_K 				= self.filename+'UKF_FHN_K.dat'
		
		self.num_time     = 1000  # number of observation times = number of DA cycles
		self.dt           = 0.05  # observation time step

		self.num_iteration = 1 # number of model iterations between two observation times
		self.dt_iteration  = self.dt/float(self.num_iteration) # model integration time step 

		self.t             = 0  # count the loops in the model
		self.time          = np.zeros(self.num_time)  # time points ob observations
		self.dim_obs       = 2
		self.dim_state	   = 4
		self.num_sigma 	   = 2*self.dim_state+1
		self.x	           = np.zeros(self.dim_state)
		self.xmean	       = np.zeros(self.dim_state)
		self.xamean	       = np.zeros(self.dim_state)
		self.ymean	       = np.zeros(self.dim_obs)
		self.s	           = np.zeros((self.num_sigma,self.dim_state)) # sigma points
		self.xs	           = np.zeros((self.num_sigma,self.dim_state)) # UT result
		self.sb	           = np.zeros((self.num_sigma,self.dim_state)) # sigma points
		self.ys	           = np.zeros((self.num_sigma,self.dim_obs)) # UT result
		self.W_m	           = np.zeros(self.num_sigma)
		self.W_c	           = np.zeros(self.num_sigma)
		
		self.xbt 		 = np.zeros((self.dim_state,self.num_time))    # background state estimate
		self.xat 	     = np.zeros((self.dim_state,self.num_time))    # analysis state estimate
		self.yt          = np.zeros((self.dim_obs,self.num_time))      # background equivalent in observation space
		self.Hxbt        = np.zeros((self.dim_obs,self.num_time))      # background equivalent in observation space
		self.Hxat 	     = np.zeros((self.dim_obs,self.num_time))      # analysis equivalent in observation space

		# PARAMETERS OF KALMAN FILTER
		# model error covariance matrix
		self.Q          = np.zeros((self.dim_state, self.dim_state))
		# observation error covariance matrix
		self.R          = np.zeros((self.dim_obs,self.dim_obs))
		# background error covariance matrix
		self.Pb         = np.zeros((self.dim_state, self.dim_state))
		# analysis error covariance matrix
		self.Pa         = np.zeros((self.dim_state, self.dim_state))

		# Jacobian matrix
		self.M          = np.zeros((self.dim_state, self.dim_state))
		
		# Observation operator matrix
		self.H       = np.zeros((self.dim_obs,self.dim_state))
		
		# Identity matrix
		self.Id = np.identity(self.dim_state)

	def initialize_basic(self):
		# MODEL PARAMETERS
		##### rhythm1
		# true model
		self.I_1_true     = 1.5
		self.b0_1_true    = 2.0
		self.b1_1_true    = 1.3
		self.tau_1_true   = 10.0
		# false model
		self.I_1_false     = 2.7 # different parameter
		self.b0_1_false    = 2.0
		self.b1_1_false    = 1.3
		self.tau_1_false   = 10.0
		
		### rhythm2
		# true model
		self.I_2_true     = 1.5
		self.b0_2_true    = 2.0
		self.b1_2_true    = 10.0
		self.tau_2_true   = 10.0
		# false model		
		self.I_2_false     = 1.5
		self.b0_2_false    = 2.0
		self.b1_2_false    = 10.0
		self.tau_2_false   = 9.0 # different parameter
		
		self.model_noise   = 0.8  

		alpha_  	= 0.5
		beta_	= 2.0
		#self.lambda_ 	= self.dim_state*(alpha_**2-1.0)
		self.lambda_ 	= 3-self.dim_state
		
		self.W_m[0] = self.lambda_/(self.lambda_+self.dim_state) # parameter for Unscented Transform
		self.W_c[0] = self.lambda_/(self.dim_state+self.lambda_)+1.0-alpha_**2+beta_ # parameter for Unscented Transform
		for k in range(self.dim_state*2):
			self.W_m[1+k]=1.0/(2*(self.lambda_+self.dim_state))
			self.W_c[1+k]=1.0/(2*(self.lambda_+self.dim_state))
			
		self.Q0         = 0.01
		self.Q1			= 0.01
		self.R1   		= 5.0
		self.R2   		= 5.0

		self.H[0,0]  = 1.0
		self.H[1,2]  = 1.0

	def observation_operator(self, x):
		return self.H@x

	def model_system(self, b0_1, b1_1, tau_1, I_1, b0_2, b1_2, tau_2, I_2,noise1,noise2):
		self.x = self.Id@self.x
		######################-------------
		
		#rhythm1
		xp0 = self.x[0]-(self.x[0]**3)/3.0-self.x[1]+I_1 
		xp1 = (b1_1*self.x[0]+b0_1-self.x[1])/tau_1
		#rhythm1
		xp2 = self.x[2]-(self.x[2]**3)/3.0-self.x[3]+I_2
		xp3 = (b1_2*self.x[2]+b0_2-self.x[3])/tau_2

		self.x[0] += xp0*self.dt_iteration + np.sqrt(self.dt_iteration)*noise1*np.random.normal(0,1.0)
		self.x[1] += xp1*self.dt_iteration
		self.x[2] += xp2*self.dt_iteration + np.sqrt(self.dt_iteration)*noise2*np.random.normal(0,1.0)
		self.x[3] += xp3*self.dt_iteration
		######################-------------
		
		