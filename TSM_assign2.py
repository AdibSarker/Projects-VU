# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:41:00 2020

@author: Group4
"""

#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


#%% Loading data
# In this cell we read the data.
df_data = pd.read_table("sv.dat")
df_data.rename(columns={df_data.columns[0]: "Returns" }, inplace = True)
n = len(df_data["Returns"])


#%% Plot returns (Part a)
# Here we demean the returns en divide by 100 to get the plots (14.5) of the book 
df_data["logRet"] = (df_data["Returns"]-np.mean(df_data["Returns"]))/100


#plt.plot(df_data["logRet"])
df_data["logRet"].plot(figsize=(10,5))
plt.title("Returns")
plt.show()

#%% Transforming model
# Here we transform the model for question c

df_data["linReturns"] = np.log(df_data["logRet"]**2)

plt.figure(figsize=(10,5))
plt.scatter(df_data.index,df_data["linReturns"])
plt.title("Linear returns")
plt.show()

x = np.min(df_data["linReturns"])

#%%Initializing state space matrices
# In this cell we define some variables which needs to be used in the log likelihood

logmean = -1.27
logvar = np.pi**2/2

Z = 1
d = 1
c = logmean
H = logvar
R = 1
Q = 1

#%% Maximizer function
# Here we define the log likelihood of question c. We maximize the log likelihood with L-BFGS-B.
# we optimize omega, sigma_eta and phi
def maximize_function(data):
    
    def Log_likelihood(theta,data):
        observations=n
        d=theta[0]
        R=theta[1]
        T=theta[2]
        v_t=np.zeros(observations)
        a_t=np.zeros(observations+1)
        F_t=np.zeros(observations)
        P_t=np.zeros(observations+1)
        K_t=np.zeros(observations)
        a_t[0]=d/(1-T)
        P_t[0]=R**2/(1-T**2)
        for i in range(observations):
            v_t[i]= data[i] - c - Z*a_t[i]
            F_t[i]= Z*P_t[i]*Z + H
            K_t[i]= T*P_t[i]*Z*F_t[i]**-1
            a_t[i+1]= d + T*a_t[i] + K_t[i]*v_t[i]     
            P_t[i+1]= T*P_t[i]*T + R*Q*R - K_t[i]*F_t[i]*K_t[i]
        l= -(np.log(F_t)+(v_t**2/F_t)) 
        
        return -np.sum(l)
    
    def Optimizer(data, initials, function):
        bnds = ((None,None),(0.001,None),(-0.9999,0.9999))
        result = minimize(function, initials, args=(data), \
                          options ={'disp': False, 'maxiter':50}, method='L-BFGS-B',bounds=bnds)
        return result
    
    d = -0.2
    R=np.var(data)
    T=0.4
    theta=np.array([d,R,T])
    
    result=Optimizer(data, theta, Log_likelihood)
    return result.x, -result.fun, result.success


xx = maximize_function(df_data['linReturns'])

print(xx)

#%%
# Here we obtain the optimized parameters. These paramters are save in opt_theta. 
opt_theta = xx[0]

opt_omega = opt_theta[0]
opt_sig_eta = opt_theta[1]
opt_phi = opt_theta[2]
T = opt_phi

opt_zeta = opt_omega/(1-opt_phi)

sig = np.exp(0.5*opt_zeta)

#%% Calculating state using optimized parameters
# This log-likelihood is used to get the values for a_t, v_t, F_t, P_t, K_t. These values are needed to calculate the smooted state
def Log_likelihood2(theta,data):
        H=logvar
        observations=len(data)
        d=theta[0]
        R=theta[1]
        T=theta[2]
        v_t=np.zeros(observations)
        a_t=np.zeros(observations+1)
        F_t=np.zeros(observations)
        P_t=np.zeros(observations+1)
        K_t=np.zeros(observations)
        a_t[0]=d/(1-T)
        P_t[0]=R**2/(1-T**2)
        for i in range(observations):
            v_t[i]=data[i] -logmean - Z*a_t[i]
            F_t[i]=Z*P_t[i]*Z + H
            K_t[i]=T*P_t[i]*Z*F_t[i]**-1
            a_t[i+1]= d + T*a_t[i] + K_t[i]*v_t[i]     
            P_t[i+1]=T*P_t[i]*T + R*Q*R - K_t[i]*F_t[i]*K_t[i]    
        return v_t, F_t, K_t, a_t, P_t 
    
v_t, F_t, K_t, a_t, P_t  = Log_likelihood2(opt_theta,df_data["linReturns"])

plt.figure(figsize=(10,5))
plt.scatter(df_data.index,df_data["linReturns"])
plt.plot(df_data.index,a_t[1:],color='k')
plt.title("Linear returns")
plt.show()

#%% Smoothing
# In this cell we calculate the smoothed state alpha_hat. In order to calculate this value we need r_t, N_t, V and L
r = np.zeros(n+1)
L = np.zeros(n)
N = np.zeros(n+1)
a_hat = np.zeros(n)
V = np.zeros(n)

#Initializing smoothing cumulant
r[-1] = 0

for i in range(n):
    L[i] = T - K_t[i]*Z

#Equation 4.38 and 4.42
for i in range(n-1,-1,-1):
    r[i] = Z*F_t[i]**-1*v_t[i] + L[i]*r[i+1]
    N[i] = Z*F_t[i]**-1*Z + L[i]*N[i+1]*L[i]
    
#Equation 4.44    
for i in range(n):
    a_hat[i] = a_t[i] + P_t[i]*r[i] 
    V[i] = P_t[i] - P_t[i]*N[i]*P_t[i]
    
plt.figure(figsize=(10,5))
plt.scatter(df_data.index,df_data["linReturns"])
plt.plot(df_data.index[1:],a_hat[1:],color='k')
plt.title("Linear returns")
plt.show()


#%% Mode estimation of SV-model
# In this cell we calculate the mode. This needs to be done in a recursion. For each iteration we calculate a new value for data and H.
# We incorporate these values to calculate the value for g+
def Log_likelihood3(H,data,opt_theta):
        H=H
        observations=len(data)
        d=0
        R=opt_theta[1]
        T=opt_theta[2]
        v_t=np.zeros(observations)
        a_t=np.zeros(observations+1)
        F_t=np.zeros(observations)
        P_t=np.zeros(observations+1)
        K_t=np.zeros(observations)
        r = np.zeros(observations+1)
        L = np.zeros(observations)
        N = np.zeros(observations+1)
        a_hat = np.zeros(observations)
        V = np.zeros(observations)
        a_t[0]=d/(1-T)
        P_t[0]=R**2/(1-T**2)
        r[-1] = 0
        for i in range(observations):
            v_t[i]=data[i] - 0 - Z*a_t[i]
            F_t[i]=Z*P_t[i]*Z + H[i]
            K_t[i]=T*P_t[i]*Z*F_t[i]**-1
            a_t[i+1]= d + T*a_t[i] + K_t[i]*v_t[i]     
            P_t[i+1]=T*P_t[i]*T + R*Q*R - K_t[i]*F_t[i]*K_t[i]    
            
        for i in range(n):
            L[i] = T - K_t[i]*Z

        #Equation 4.38 and 4.42
        for i in range(n-1,-1,-1):
            r[i] = Z*F_t[i]**-1*v_t[i] + L[i]*r[i+1]
            N[i] = Z*F_t[i]**-1*Z + L[i]*N[i+1]*L[i]
    
        #Equation 4.44    
        for i in range(n):
            a_hat[i] = a_t[i] + P_t[i]*r[i] 
            V[i] = P_t[i] - P_t[i]*N[i]*P_t[i]    
        return a_hat
    
def compA(g,z):
    A = 2*np.exp(g)/(z**2)
    return A

def compZ(g,A,data):
    Z = g -0.5*A +1
    return Z
    
def modeest(parameters, data, observations):
    conv = 10 ** (-9)
    G_plus = np.zeros(observations)
    G_ini = 2*np.ones(observations)
    sig = np.exp(0.5*(opt_omega/(1-opt_phi)))
    z_t = ((data - np.mean(data))/100) / sig 
    for i in range(len(z_t)):
        if z_t[i] < 0.001 and z_t[i] > 0:
            z_t[i] += 0.001
        elif z_t[i] > -0.001 and z_t[i] < 0:
            z_t[i] -= 0.001
    i = 0
    while max(abs(G_ini - G_plus)) > conv:
        G_ini = G_plus
        A = 2*np.exp(G_ini) / (z_t**2)
        z = G_ini - 0.5 * A + 1
        alpha_hat = Log_likelihood3(A,z,parameters)
        G_plus = alpha_hat
        print(i)
        i += 1
        
    return G_plus, A, z, z_t

(g_est,A,y_plus,std_y) = modeest(opt_theta, np.array(df_data["Returns"]), n)

#%% Plotting H_t
# Here we plot the value for H
plt.figure(figsize=(10,5))
plt.plot(df_data.index,g_est,label="Mode estimation")
#plt.plot(a_hat-opt_zeta, label="Linear model approximation")
plt.title("Mode estimation H")
plt.legend()
plt.show()
