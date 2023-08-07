#genrate and store observations, without control, for the system and the observer

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import Var3D_FHN_params as params

class Observations(params.Var3D):
    def initialize(self):
        
        self.initialize_basic()
        #model system
        self.xbt[0,0]    = -1.5
        self.xbt[1,0]    = 0.0
        self.xbt[2,0]    = -1.5
        self.xbt[3,0]    = 0.0
        self.x[:]       = self.xbt[:,0]+0.0
        
        self.yt[:,0]    = self.observation_operator(self.xbt[:,0])
        self.time[0]    = 0.0

    def integrate_model(self,model_flag):
        for k in range(self.num_iteration):
            if model_flag == 1: ## true model
                self.model_system(self.b0_1_true, self.b1_1_true, self.tau_1_true, self.I_1_true,\
                                  self.b0_2_true, self.b1_2_true, self.tau_2_true, self.I_2_true,\
                                  self.model_noise,self.model_noise)             
            else:
                self.model_system(self.b0_1_false, self.b1_1_false, self.tau_1_false, self.I_1_false,\
                                  self.b0_2_false, self.b1_2_false, self.tau_2_false, self.I_2_false,\
                                  self.model_noise,self.model_noise)

    def generate_observations(self):
        
        ## true observations
        f   = open(self.label_obs,"w+")
        fs  = open(self.label_state, "w+")
        str = '0 '
        for d in range(self.dim_state):
            str += ' %f'%self.xbt[d,0]
        str += '\n'
        fs.write(str)
        str = '0 '
        for k in range(self.dim_obs):
            str += '  %f'%self.yt[k,0]
        str += '\n'
        f.write(str)

        for i in range(1,self.num_time):
            self.integrate_model(1)
            self.xbt[:,i] = self.x
            self.yt[:,i]  = self.observation_operator(self.xbt[:,i])  
            self.time[i]  = self.dt*i

            #storage
            str = '%d  '%i
            for d in range(self.dim_state):
                str += '%f  '%self.xbt[d,i]
            str += '\n'
            fs.write(str)

            str = '%d  '%i
            for k in range(self.dim_obs):
                str += '%f  '%self.yt[k,i]
            str += '\n'
            f.write(str)
        fs.close()
        f.close()

        ## false model observations
        f                   = open(self.label_obs_false,"w+")
        f_state             = open(self.label_state_false,"w+")
        self.xbt_false      = np.zeros((self.dim_state,self.num_time))
        self.yt_false       = np.zeros((self.dim_obs,self.num_time))
        self.xbt_false[0,0] = -1.5
        self.xbt_false[1,0] = 0.0
        self.xbt_false[2,0] = -1.5
        self.xbt_false[3,0] = 0.0
        self.x[:]           = self.xbt_false[:,0]
        self.yt_false[:,0]  = self.observation_operator(self.xbt_false[:,0])
        self.time[0]        = 0.0
        
        str = '0 '
        for k in range(self.dim_state):
            str += '  %f'%self.xbt_false[k,0]
        str += '\n'
        f_state.write(str)
        str = '0 '
        for k in range(self.dim_obs):
            str += '  %f'%self.yt_false[k,0]
        str += '\n'
        f.write(str)
        
        for i in range(1,self.num_time):
            self.integrate_model(2)
            self.xbt_false[:,i] = self.x
            self.yt_false[:,i]  = self.observation_operator(self.xbt_false[:,i])  
            
            str = '%d  '%i
            for k in range(self.dim_state):
                str += '%f  '%self.xbt_false[k,i]
            str += '\n'
            f_state.write(str)
            str = '%d  '%i
            for k in range(self.dim_obs):
                str += '%f  '%self.yt_false[k,i]
            str += '\n'
            f.write(str)
        f.close()
        f_state.close()


    def plot(self):
        fign=plt.figure()
        for k in range(self.dim_obs):
            plt.subplot(self.dim_obs,1,k+1)
            plt.plot(self.time,self.yt[k,:],'k',\
                     self.time,self.yt_false[k,:],'g')
            label = 'observations %d'%(k+1)
            plt.ylabel(label)
            plt.xlabel('time ')

        fign.subplots_adjust(hspace=0.8)
        plt.show()
        plt.close()

##System
observation = Observations()
observation.initialize()
observation.generate_observations()
observation.plot()

