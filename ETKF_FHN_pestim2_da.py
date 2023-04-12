
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import argparse

import ETKF_FHN_pestim2_params as params


class da(params.Kalman):
    def initialize(self):

        self.initialize_basic()
        # Initial parameters
        self.xbt[0,0] = -1.5
        self.xbt[1,0] = 0.0
        self.xbt[2,0] = -1.5
        self.xbt[3,0] = 0.0
        self.xbt[4,0] = 20.0#self.I_1_false
        
        for l in range(self.L):
          self.xbens[:,l]                 = self.xbt[:,0]
          self.xbens[0:self.dim_model,l] += self.initial_ensnoise*np.random.normal(0.0,1.0,self.dim_model)
          self.xaens[:,l]                 = self.xbens[:,l]
          
        self.xat[:,0]    = self.xbt[:,0]
        self.xamean      = self.xat[:,0]
        self.Hxbt[:,0]   = self.observation_operator(self.xbt[:,0])
        self.Hxat[:,0]   = self.observation_operator(self.xat[:,0])

        self.R[0,0]     = self.R1
        self.R[1,1]     = self.R2
        self.Rinv       = np.linalg.inv(self.R)

        self.yvar = np.zeros((self.dim_obs,self.dim_obs))+0.1
        self.yvar_old = 0.0

    def read_observations(self):  
        data = np.loadtxt(self.label_obs).transpose()
        self.yt = data[1:,:]
        data = np.loadtxt(self.label_state).transpose()
        self.xbt_true = data[1:,:]
        data = np.loadtxt(self.label_obs_false).transpose()
        self.yt_false = data[1:,:]

    def integrate_model(self,model_flag):
        for k in range(self.num_iteration):
            if model_flag == 1: ## true model
                self.model_system(self.b0_1_true, self.b1_1_true, self.tau_1_true, self.I_1_true,\
                                  self.b0_2_true, self.b1_2_true, self.tau_2_true, self.I_2_true,\
                                  0.0,0.0,0.0)             
            else:
                self.model_system(self.b0_1_false, self.b1_1_false, self.tau_1_false, self.I_1_false,\
                                  self.b0_2_false, self.b1_2_false, self.tau_2_false, self.I_2_false,\
                                  0.0,0.0,0.0)
 
    def Compute_prediction(self):
      for l in range(self.L):
        self.x=self.xaens[:,l].copy()
        self.integrate_model(2) ########################## choice of model: true or false model ?
        self.xbens[:,l]=self.x.copy()
        self.ybens[:,l]=self.observation_operator(self.xbens[:,l])
      self.xbmean = np.mean(self.xbens,axis=1)
      self.ybmean = np.mean(self.ybens,axis=1)
      
      tile = np.tile( np.reshape(self.xbmean,(self.dim_state,1)) ,(1,self.L) )
      ### add here the additive covariance inflation
      self.X = self.xbens-tile
      self.X[0:self.dim_model,:]+=\
                    np.random.normal(0.0,1.0,(self.dim_model,self.L))*self.addcovinfl
      self.X[self.dim_model,:]+=np.random.normal(0.0,1.0,self.L)*np.sqrt(self.Q2)
      
      tile = np.tile( np.reshape(self.ybmean,(self.dim_obs,1)) ,(1,self.L) )
      self.Y = self.ybens-tile
      
    def Compute_analysis(self,i):
      
      YtRinvY=self.Y.transpose()@(self.Rinv@self.Y)
      IYtRinvY=self.Id*(self.L-1.0)+YtRinvY
      [U, s, Ut] = np.linalg.svd(IYtRinvY) ## 
      S=np.diag(s)
      Sinv=np.linalg.inv(S)
      self.Pens_a=U@(Sinv@Ut)
      
      ## mean ensemble in ensemble space bar{w}_a
      YtRinv=self.Y.transpose()@self.Rinv
      PaYtRinv=self.Pens_a@YtRinv
      wam=PaYtRinv@(self.yt[:,i]-self.ybmean)
      Xbwam=self.X@wam
      self.xamean=self.xbmean+Xbwam    
      
      # analysis ensemble
      sqrtVSVt=U@(np.sqrt(Sinv)@Ut)
      W = sqrtVSVt*np.sqrt(self.L-1.0)
      tile=np.tile(np.reshape(wam,(self.L,1)),(1,self.L))
      wa = W+tile
      
      tilexbm=np.tile(np.reshape(self.xbmean,(self.dim_state,1)),(1,self.L))
      Xwa=self.X@wa*self.multensinfl
      self.xaens =  tilexbm + Xwa

    def cycle_step(self,i):

        self.Compute_prediction()
        self.xbt[:,i]=self.xbmean.copy()
        self.Hxbt[:,i]=self.ybmean.copy()
        self.Compute_analysis(i)
        self.xat[:,i]=self.xamean.copy()
        # Compute analysis mean in observation space.
        # It should be transformed by an Unscented Transform for nonlinear observation operators.
        # For simplicity, here we assume a linear observation operator and simply
        # apply H since we just need Hxat for the visualization.
        self.Hxat[:,i]=self.observation_operator(self.xat[:,i])
        
        str = '%d '%i
        for k in range(self.dim_state):
          str += ' %f '%self.xbt[k,i]
        str += '\n'
        self.fout.write(str)
      
        self.time[i]     = self.dt*i

        ### important since analysis is initial state of next cycle
        self.x = self.xat[:,i]


    def iterate_da(self):
      self.fout = open(self.label_reconstr_state,"w+")   
      for i in range(1,self.num_time):
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
        plt.plot(self.time,self.parameter_function_I1(self.xbt[4,:]),'b',\
            self.time,self.parameter_function_I1(self.xat[4,:]),'r')
        ax.set(ylabel='parameter')
        ax.set(xlabel='time')
        plt.title('parameter 1')
      
        fign.subplots_adjust(hspace=0.8)
        plt.show()
        plt.close()



##
DA= da()
DA.initialize()  
DA.read_observations()
DA.iterate_da() 
DA.plot()



