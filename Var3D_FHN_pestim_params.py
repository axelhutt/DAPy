#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

class Var3D:
	def __init__(self):
		self.filename 				= './'
		self.label_state 			= self.filename+'3DVar_FHN_bg.dat'
		self.label_state_false 		= self.filename+'3DVar_FHN_bg_false.dat'
		self.label_obs 				= self.filename+'3DVar_FHN_obs.dat'
		self.label_obs_false 		= self.filename+'3DVar_FHN_obs_false.dat'
		self.label_reconstr_state 	= self.filename+'3DVar_FHN_bg_reconstructed.dat'
		
		self.num_time     = 3000#1000  # number of observation times = number of DA cycles
		self.dt           = 0.05  # observation time step

		self.num_iteration = 1 # number of model iterations between two observation times
		self.dt_iteration  = self.dt/float(self.num_iteration) # model integration time step 

		self.t             = 0  # count the loops in the model
		self.time          = np.zeros(self.num_time)  # time points ob observations
		self.dim_obs       = 2
		self.num_params    = 2 # number of model parameters to be estimated 
		self.dim_model     = 4 # number of model variables
		self.dim_state	   = self.dim_model + self.num_params
		self.x	           = np.zeros(self.dim_state)

		self.xbt 		 = np.zeros((self.dim_state,self.num_time))    # background state estimate
		self.xat 	     = np.zeros((self.dim_state,self.num_time))    # analysis state estimate
		self.yt          = np.zeros((self.dim_obs,self.num_time))      # background equivalent in observation space
		self.Hxbt        = np.zeros((self.dim_obs,self.num_time))      # background equivalent in observation space
		self.Hxat 	     = np.zeros((self.dim_obs,self.num_time))      # analysis equivalent in observation space

		# model error covariance matrix
		self.B          = np.zeros((self.dim_state, self.dim_state))
		# observation error covariance matrix
		self.R          = np.zeros((self.dim_obs,self.dim_obs))
		
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
		self.I_1_false     = 2.9 # is estimated
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
		self.tau_2_false   = 9.0 # is estimated
		
		self.model_noise   = 0.8  
		
		self.B1         = 1.0
		self.B2			= 1.0
		self.B3			= 1.0
		self.B04		= 0.05
		self.B25		= 0.05
		self.R1   		= 2.0
		self.R2   		= 2.0
		
		self.H[0,0]  = 1.0
		self.H[1,2]  = 1.0

	def observation_operator(self, x):
		return self.H@x

	def model_system(self, b0_1, b1_1, tau_1, I_1, b0_2, b1_2, tau_2, I_2,noise1,noise2):
		#rhythm1
		xp0 = self.x[0]-(self.x[0]**3)/3.0-self.x[1]+self.x[4] 
		xp1 = (b1_1*self.x[0]+b0_1-self.x[1])/tau_1
		#rhythm1
		xp2 = self.x[2]-(self.x[2]**3)/3.0-self.x[3]+I_2
		xp3 = (b1_2*self.x[2]+b0_2-self.x[3])/self.x[5]
		xp4 = 0.0
		xp5 = 0.0

		self.x[0] += xp0*self.dt_iteration + np.sqrt(self.dt_iteration)*noise1*np.random.normal(0,1.0)
		self.x[1] += xp1*self.dt_iteration
		self.x[2] += xp2*self.dt_iteration + np.sqrt(self.dt_iteration)*noise2*np.random.normal(0,1.0)
		self.x[3] += xp3*self.dt_iteration
		self.x[4] += xp4*self.dt_iteration 
		self.x[5] += xp5*self.dt_iteration






		