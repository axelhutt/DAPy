#Genrate and store observations, with/without kalman filter and control application, for the system and the observer
# /!\ Choose flags for control and kalman filter
# /!\ if you want to use the terminal as command line, activate the 'read_commandline' function at the end.


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import argparse

import Var3D_FHN_pestim_params as params


class da(params.Var3D):
    def initialize(self):

        self.initialize_basic()
        # Initial parameters
        self.xbt[0,0] = -1.5
        self.xbt[1,0] = 0.0
        self.xbt[2,0] = -1.5
        self.xbt[3,0] = 0.0
        self.xbt[4,0] = self.I_1_false
        self.xbt[5,0] = self.tau_2_false

        self.x[:]        = self.xbt[:,0]
        self.xat[:,0]    = self.xbt[:,0]
        self.Hxbt[:,0]   = self.observation_operator(self.xbt[:,0])
        self.Hxat[:,0]   = self.observation_operator(self.xat[:,0])

        #fill covariance matrix
        self.B[0,0]		  = self.B1
        self.B[0,4]		  = self.B04
        self.B[1,1]		  = self.B1
        self.B[2,5]		  = self.B25
        self.B[2,2]		  = self.B2
        self.B[3,3]		  = self.B2
        self.B[4,0]		  = self.B04
        self.B[4,4]		  = self.B3
        self.B[5,5]		  = self.B3
        self.B[5,2]		  = self.B25
        self.R[0,0]     = self.R1
        self.R[1,1]     = self.R2
        

    def read_observations(self):  
        data = np.loadtxt(self.label_obs).transpose()
        self.yt = data[1:,:]
        data = np.loadtxt(self.label_state).transpose()
        self.xbt_obs = data
        data = np.loadtxt(self.label_obs_false).transpose()
        self.yt_false = data[1:,:]

    def compute_gainmatrix(self):
        HB        = self.H@self.B
        RHBHt     = self.R + HB@self.H.transpose()
        RHBHtinv  = np.linalg.inv(RHBHt)
        BHt       = self.B@self.H.transpose()
        self.K    = BHt@RHBHtinv
        
    def compute_prediction(self):
        
        self.model_system(self.b0_1_false, self.b1_1_false, self.tau_1_false, self.I_1_false,\
                          self.b0_2_false, self.b1_2_false, self.tau_2_false, self.I_2_false,\
                          0.0,0.0)        
        for k in range(1,self.num_iteration):
            self.model_system(self.b0_1_false, self.b1_1_false, self.tau_1_false, self.I_1_false,\
                              self.b0_2_false, self.b1_2_false, self.tau_2_false, self.I_2_false,\
                              0.0,0.0)

    def compute_analysis(self,i):
        self.xat[:,i] = self.xbt[:,i]+self.K@(self.yt[:,i]-self.Hxbt[:,i])
    
    def cycle_step(self,i):

        self.compute_prediction()
        self.xbt[:,i]  = self.x
        self.Hxbt[:,i] = self.observation_operator(self.xbt[:,i])

        self.compute_gainmatrix()       #update Kalmanmatrix (K)
        self.compute_analysis(i)
        self.Hxat[:,i] = self.observation_operator(self.xat[:,i])
        
        str = '%d '%i
        for k in range(self.dim_state):
          str += ' %f '%self.xbt[k,i]
        str += '\n'
        self.fout.write(str)
        
        #print("i=%d Hxbt[0]=%f Hxat[0]=%f    Hxbt[1]=%f Hxat[1]=%f"%(\
        #        i,\
        #        self.Hxbt[0,i],self.Hxat[0,i],\
        #        self.Hxbt[1,i],self.Hxat[1,i]))
        self.time[i]     = self.dt*i

        ### important since analysis is initial state of next cycle
        self.x = self.xat[:,i]


    def iterate_da(self):
      self.fout = open(self.label_reconstr_state,"w+")  
      for i in range(0,self.num_time):
            self.cycle_step(i)
      self.fout.close()


    def plot(self):
        fign=plt.figure()
        ax = fign.add_subplot(221)
        plt.plot(self.time,self.yt[0,:],'k',\
            self.time,self.Hxbt[0,:],'b',\
            self.time,self.Hxat[0,:],'r',\
            self.time,self.yt_false[0,:],'g')
        ax.set(ylabel='observation')
        ax.set(xlabel='time')
        plt.title('observation 1')

        ax = fign.add_subplot(223)
        plt.plot(self.time,self.yt[1,:],'k',\
            self.time,self.Hxbt[1,:],'b',\
            self.time,self.Hxat[1,:],'r',\
            self.time,self.yt_false[1,:],'g')
        ax.set(ylabel='observation')
        ax.set(xlabel='time')
        plt.title('observation 2')
        
        ax = fign.add_subplot(222)
        plt.plot(self.time,self.xbt[4,:],'b',\
            self.time,self.xat[4,:],'r')
        ax.set(ylabel='parameter')
        ax.set(xlabel='time')
        plt.title('parameter 1')
      
        ax = fign.add_subplot(224)
        plt.plot(self.time,self.xbt[5,:],'b',\
            self.time,self.xat[5,:],'r')
        ax.set(ylabel='parameter')
        ax.set(xlabel='time')
        plt.title('parameter 2')
        
        fign.subplots_adjust(hspace=0.8)
        plt.show()
        plt.close()



##
DA= da()
DA.initialize()  
DA.read_observations()
DA.iterate_da() 
DA.plot()



