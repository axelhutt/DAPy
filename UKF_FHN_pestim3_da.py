
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import argparse

import UKF_FHN_pestim3_params as params


class da(params.Kalman):
    def initialize(self):

        self.initialize_basic()
        # Initial parameters
        self.xbt[0,0] = -1.5
        self.xbt[1,0] = 0.0
        self.xbt[2,0] = -1.5
        self.xbt[3,0] = 0.0
        self.xbt[4,0] = 10.0#self.I_1_false
        self.xbt[5,0] = 10.0#self.tau_2_false

        self.x[:]        = self.xbt[:,0]
        self.xat[:,0]    = self.xbt[:,0]
        self.xamean      = self.xat[:,0]
        self.Hxbt[:,0]   = self.observation_operator(self.xbt[:,0])
        self.Hxat[:,0]   = self.observation_operator(self.xat[:,0])

        self.Pa = self.Id

        #fill covariance matrix
        self.Q[0,0]		= self.Q0
        self.Q[0,4]		= self.Q2
        self.Q[1,1]		= self.Q0
        self.Q[2,2]		= self.Q1
        self.Q[2,5]		= self.Q2
        self.Q[3,3]		= self.Q1
        self.Q[4,0]		= self.Q2
        self.Q[4,4]		= self.Q2
        self.Q[5,2]		= self.Q2
        self.Q[5,5]		= self.Q2
        self.R[0,0]    = self.R1
        self.R[1,1]    = self.R2
        

    def read_observations(self):  
        data = np.loadtxt(self.label_obs).transpose()
        self.yt = data[1:,:]
        data = np.loadtxt(self.label_state).transpose()
        self.xbt_obs = data
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
                                  0.0,0.0,self.Q2)

    def Compute_prediction(self):
      
      # compute A for Unscented Transform
      [U, s, Ut] = np.linalg.svd(self.Pa)
      S=np.diag(np.sqrt(s))
      A = U@S
      
      # compute sigma points
      self.s[0,:]= self.xamean.copy()
      for k in range(self.dim_state):
        self.s[1+k,:]                   = self.s[0,:] + np.sqrt(self.dim_state+self.lambda_)*A[k,:]
        self.s[1+self.dim_state+k,:]    = self.s[0,:] - np.sqrt(self.dim_state+self.lambda_)*A[k,:]
      
        #prediction of sigma points and computation of bg mean and bg covariance
      self.xmean = 0.0*self.x
      for k in range(self.num_sigma):
        self.x=self.s[k,:].copy()
        self.integrate_model(2) ########################## choice of model: true or false model ?
        self.xs[k,:]=self.x.copy()
        self.xmean += self.W_m[k]*self.xs[k,:]
      self.Pb = 0.0*self.Id
      for k in range(self.num_sigma):
        self.Pb+=self.W_c[k]*np.outer(self.xs[k,:]-self.xmean[:],self.xs[k,:]-self.xmean[:])
      self.Pb += self.Q
      

    def Compute_analysis(self,i):

      [U, s, Ut] = np.linalg.svd(self.Pb)
      S=np.diag(np.sqrt(s))
      A = U@S

      # compute sigma points
      self.sb[0,:]= self.xmean[:]
      for k in range(self.dim_state):
        self.sb[1+k,:]                  = self.sb[0,:] + np.sqrt(self.dim_state+self.lambda_)*A[k,:]
        self.sb[1+self.dim_state+k,:]   = self.sb[0,:] - np.sqrt(self.dim_state+self.lambda_)*A[k,:]
      
      #prediction of sigma points and computation of bg mean and bg covariance
      self.ymean[:]=0.0
      for k in range(self.num_sigma):
        self.ys[k,:]=self.observation_operator(self.sb[k,:])
        self.ymean[:]+=self.W_m[k]*self.ys[k,:]
        
      C_yy = np.identity(self.dim_obs)*0.0
      for k in range(self.num_sigma):
        C_yy += self.W_c[k]*np.outer(self.ys[k,:]-self.ymean[:],self.ys[k,:]-self.ymean[:])
      C_yy += self.R
      C_sy = 0.0*np.outer(self.sb[0,:]-self.xmean[:],self.ys[0,:]-self.ymean[:])
      for k in range(self.num_sigma):
        C_sy += self.W_c[k]*np.outer(self.sb[k,:]-self.xmean[:],self.ys[k,:]-self.ymean[:])
      C_yy_inv = np.linalg.inv(C_yy)
      self.K = C_sy@C_yy_inv
      
      innovation = self.yt[:,i]-self.ymean[:]
      self.xamean[:] = self.xmean[:]+self.K@innovation 
      self.Pa       = self.Pb - self.K@C_yy@(self.K.transpose())
      
      

    def cycle_step(self,i):

        self.Compute_prediction()
        self.xbt[:,i]=self.xmean.copy()
        self.Compute_analysis(i)
        self.xat[:,i]=self.xamean.copy()
        self.Hxbt[:,i]=self.ymean.copy()
        # Compute analysis mean in observation space.
        # It should be transformed by an Unscented Transform for nonlinear observation operators.
        # For simplicity, here we assume a linear observation operator and simply
        # apply H since we just need Hxat for the visualization.
        self.Hxat[:,i]=self.observation_operator(self.xat[:,i])
        
        str = '%d '%i
        for k in range(self.dim_model):
          str += ' %f '%self.xbt[k,i]
        str += ' %f '%self.parameter_function_I1(self.xbt[self.dim_model,i])
        str += ' %f '%self.parameter_function_tau2(self.xbt[self.dim_model+1,i])
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
        ax = fign.add_subplot(421)
        plt.plot(self.time,self.yt[0,:],'k',\
            self.time,self.Hxbt[0,:],'b',\
            self.time,self.Hxat[0,:],'r',\
            self.time,self.yt_false[0,:],'g')
        ax.set(ylabel='observation')
        ax.set(xlabel='time')
        plt.title('observation 1')
       
        ax = fign.add_subplot(423)
        plt.plot(self.time,self.yt[1,:],'k',\
            self.time,self.Hxbt[1,:],'b',\
            self.time,self.Hxat[1,:],'r',\
            self.time,self.yt_false[1,:],'g')
        ax.set(ylabel='observation')
        ax.set(xlabel='time')
        plt.title('observation 2')
        
        ax = fign.add_subplot(425)
        plt.plot(self.time,self.yt[2,:],'k',\
            self.time,self.Hxbt[2,:],'b',\
            self.time,self.Hxat[2,:],'r',\
            self.time,self.yt_false[2,:],'g')
        ax.set(ylabel='observation')
        ax.set(xlabel='time')
        plt.title('observation 1')
      
        ax = fign.add_subplot(427)
        plt.plot(self.time,self.yt[3,:],'k',\
            self.time,self.Hxbt[3,:],'b',\
            self.time,self.Hxat[3,:],'r',\
            self.time,self.yt_false[3,:],'g')
        ax.set(ylabel='observation')
        ax.set(xlabel='time')
        plt.title('observation 3')
      
        ax = fign.add_subplot(422)
        plt.plot(self.time,self.parameter_function_I1(self.xbt[4,:]),'b',\
            self.time,self.parameter_function_I1(self.xat[4,:]),'r')
        ax.set(ylabel='parameter')
        ax.set(xlabel='time')
        plt.title('parameter 1')
        
        ax = fign.add_subplot(424)
        plt.plot(self.time,self.parameter_function_tau2(self.xbt[5,:]),'b',\
            self.time,self.parameter_function_tau2(self.xat[5,:]),'r')
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



  