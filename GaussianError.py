#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[88]:


x = [4.6, 6.0, 2.0, 5.8] 
sigma = [2.0, 1.5, 5.0, 1.0]


# In[78]:


def prior(interval_mu):
    p=np.ones(len(interval_mu))/(np.max(interval_mu)-np.min(interval_mu))
    return p

def gaussiana(x,mu,sigma):
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*((x-mu)**2)/(sigma**2))

def L(x,mu,sigma):    
    L=np.zeros(len(mu))
    for i in range(len(x)):
        L+=np.log(gaussiana(x[i],mu,sigma[i]))
    return L

def post(mu,x,sigma):
    
    post =  np.exp(L(x,mu,sigma))/ np.trapz(np.exp(L(x,mu,sigma)), mu) 
    return  post


def maximo_sigma(x, y):
    deltax = x[1] - x[0]

    # maximo de y
    ii = np.argmax(y)

    # segunda derivada
    d = (y[ii+1] - 2*y[ii] + y[ii-1]) / (deltax**2)

    return x[ii], 1.0/np.sqrt(-d)


# In[89]:


mu=np.linspace(np.min(x),1.5*np.max(x),1000)

plt.plot(mu,post(mu,x,sigma))


