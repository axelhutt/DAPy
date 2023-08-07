
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import argparse

import ETKF_FHN_5_params as params


class forecast(params.Kalman):
    def initialize(self):

        self.initialize_basic()
        # Initial parameters
        self.xbtens = np.zeros((self.num_time,self.dim_state,self.L))
        self.read_files()
        
        self.num_leadtimes  = 6#4
        self.leadtimes      = np.zeros(self.num_leadtimes,dtype=int)
        self.leadtimes[0]   = 5
        self.leadtimes[1]   = 10
        self.leadtimes[2]   = 50
        self.leadtimes[3]   = 100
        self.leadtimes[4]   = 200
        self.leadtimes[5]   = 400
      
        self.max_leadtime   = int(np.max(self.leadtimes))
        self.yensmean       = np.zeros((self.num_leadtimes,self.num_time,self.dim_obs))
        self.yensvar        = np.zeros((self.num_leadtimes,self.num_time,self.dim_obs))
        self.bias           = np.zeros((self.num_leadtimes,self.dim_obs))
        self.rmse           = np.zeros((self.num_leadtimes,self.dim_obs))
        self.ensspread      = np.zeros((self.num_leadtimes,self.dim_obs))
      
      
    def read_files(self):  
        data = np.loadtxt(self.label_obs).transpose()
        self.yt = data[1:,:]
        data = np.loadtxt(self.label_state).transpose()
        self.xbt_true = data[1:,:]
        for l in range(self.L):
          label = self.label_reconstr_state_ens+'%d.dat'%l
          data = np.loadtxt(label)
          self.xbtens[:,:,l]=data[:,1:]
        data = np.loadtxt(self.label_obs_false).transpose()
        self.yt_false = data[1:,:]

    def integrate_model_da(self,model_flag):
        for k in range(self.num_iteration):
          if model_flag == 1: ## true model
            if i<self.num_time_switch:
              self.model_system(self.b0_1_true, self.b1_1_true, self.tau_1_true, self.I_1_true,\
                              self.b0_2_true, self.b1_2_true, self.tau_2_true, self.I_2_true,\
                              0.0,0.0,0.0)
            else:
              self.model_system(self.b0_1_s_true, self.b1_1_s_true, self.tau_1_s_true, self.I_1_s_true,\
                              self.b0_2_s_true, self.b1_2_s_true, self.tau_2_s_true, self.I_2_s_true,\
                              self.model_noise,self.model_noise,0.0)
          else:
              self.model_system(self.b0_1_false, self.b1_1_false, self.tau_1_false, self.I_1_false,\
                                  self.b0_2_false, self.b1_2_false, self.tau_2_false, self.I_2_false,\
                                  0.0,0.0,self.Q2)

    def Compute_forecasts(self):
      
      yfc          = np.zeros((self.num_leadtimes,self.num_time,self.dim_obs))
      
      for l in range(self.L):## loop over ensemble
        
        print("ens#%d"%l)
        
        for i in range(0,self.num_time): # loop over initial conditions
        
          self.x=self.xbtens[i,:,l].copy()
          
          counter_leadtimes = 0
          for lt in range(self.max_leadtime+1): # temporal forecasts 
            
            self.integrate_model_da(2) ########################## choice of model: true or false model ?
            if lt==self.leadtimes[counter_leadtimes]:
              
              yfc[counter_leadtimes,i,:]=self.observation_operator(self.x)
              
              self.yensmean[counter_leadtimes,i,:]+=\
                      yfc[counter_leadtimes,i,:]/float(self.L)
              self.yensvar[counter_leadtimes,i,:]+=\
                      yfc[counter_leadtimes,i,:]**2/float(self.L)
              
              counter_leadtimes += 1
      
      for counter_leadtimes in range(self.num_leadtimes):
        for i in range(self.num_time):     
          self.yensvar[counter_leadtimes,i,:] = (self.num_time/(self.num_time-1))*\
            (self.yensvar[counter_leadtimes,i,:]-self.yensmean[counter_leadtimes,i,:]**2)
          
    def Compute_statistics(self):
      
      for counter_leadtimes in range(self.num_leadtimes):
        
        for i in range(self.num_time):
          
          # spread
          self.ensspread[counter_leadtimes,:] += self.yensvar[counter_leadtimes,i,:]/float(self.num_time)
          
          # bias
          self.bias[counter_leadtimes,:] += (self.yt[:,i]-self.yensmean[counter_leadtimes,i,:])/float(self.num_time)
          
          self.rmse[counter_leadtimes,:] += (self.yt[:,i]-self.yensmean[counter_leadtimes,i,:])**2/float(self.num_time)
          
          
        # rmse
        self.rmse[counter_leadtimes,:] = \
                    np.sqrt(self.rmse[counter_leadtimes,:]-self.bias[counter_leadtimes,:]**2)
        

    def plot(self):
        
        fign=plt.figure(1)
        
        ax = fign.add_subplot(421)
        counter = 0
        LT = self.leadtimes[counter]
        plt.plot(range(self.num_time),self.yt[0,:],'k',\
            range(LT,self.num_time+LT),self.yensmean[counter,:,0],'b')
        ax.set(ylabel='data')
        ax.set(xlabel='time')
        label = 'observation/forecast 1 LT=%d'%LT
        plt.title(label)

        ax = fign.add_subplot(423)
        counter = 2
        LT = self.leadtimes[counter]
        plt.plot(range(self.num_time),self.yt[0,:],'k',\
            range(LT,self.num_time+LT),self.yensmean[counter,:,0],'b')
        ax.set(ylabel='data')
        ax.set(xlabel='time')
        label = 'LT=%d'%LT
        plt.title(label)

      
        ax = fign.add_subplot(425)
        counter = 4
        LT = self.leadtimes[counter]
        plt.plot(range(self.num_time),self.yt[0,:],'k',\
            range(LT,self.num_time+LT),self.yensmean[counter,:,0],'b')
        ax.set(ylabel='data')
        ax.set(xlabel='time')
        label = 'LT=%d'%LT
        plt.title(label)
      
        ax = fign.add_subplot(427)
        counter = 5
        LT = self.leadtimes[counter]
        plt.plot(range(self.num_time),self.yt[0,:],'k',\
            range(LT,self.num_time+LT),self.yensmean[counter,:,0],'b')
        ax.set(ylabel='data')
        ax.set(xlabel='time')
        label = 'LT=%d'%LT
        plt.title(label)

        ax = fign.add_subplot(422)
        counter = 0
        LT = self.leadtimes[counter]
        plt.plot(range(self.num_time),self.yt[1,:],'k',\
            range(LT,self.num_time+LT),self.yensmean[counter,:,1],'b')
        ax.set(ylabel='data')
        ax.set(xlabel='time')
        label = 'observation/forecast 2 LT=%d'%LT
        plt.title(label)

        ax = fign.add_subplot(424)
        counter = 2
        LT = self.leadtimes[counter]
        plt.plot(range(self.num_time),self.yt[1,:],'k',\
            range(LT,self.num_time+LT),self.yensmean[counter,:,1],'b')
        ax.set(ylabel='data')
        ax.set(xlabel='time')
        label = 'LT=%d'%LT
        plt.title(label)
      
      
        ax = fign.add_subplot(426)
        counter = 4
        LT = self.leadtimes[counter]
        plt.plot(range(self.num_time),self.yt[1,:],'k',\
            range(LT,self.num_time+LT),self.yensmean[counter,:,1],'b')
        ax.set(ylabel='data')
        ax.set(xlabel='time')
        label = 'LT=%d'%LT
        plt.title(label)
      
        ax = fign.add_subplot(428)
        counter = 5
        LT = self.leadtimes[counter]
        plt.plot(range(self.num_time),self.yt[1,:],'k',\
            range(LT,self.num_time+LT),self.yensmean[counter,:,1],'b')
        ax.set(ylabel='data')
        ax.set(xlabel='time')
        label = 'LT=%d'%LT
        plt.title(label)
      
        fign.subplots_adjust(hspace=0.8)
        plt.show()
        
        
        
        fign=plt.figure(2)
      
        ax = fign.add_subplot(311)
        plt.plot(self.leadtimes,self.bias[:,0],'g',\
            self.leadtimes,self.bias[:,1],'r')
        ax.set(ylabel='data')
        ax.set(xlabel='lead time')
        label = 'bias'
        plt.title(label)
      
        ax = fign.add_subplot(312)
        plt.plot(self.leadtimes,self.rmse[:,0],'g',\
            self.leadtimes,self.rmse[:,1],'r')
        ax.set(ylabel='data')
        ax.set(xlabel='lead time')
        label = 'root mean square error'
        plt.title(label)

        ax = fign.add_subplot(313)
        plt.plot(self.leadtimes,self.ensspread[:,0],'g',\
            self.leadtimes,self.ensspread[:,1],'r')
        ax.set(ylabel='data')
        ax.set(xlabel='lead time')
        label = 'ensemble spread'
        plt.title(label)
      
      
        fign.subplots_adjust(hspace=0.8)
        plt.show()
        plt.close()


FC= forecast()
FC.initialize()  
FC.Compute_forecasts() 
FC.Compute_statistics()
FC.plot()



