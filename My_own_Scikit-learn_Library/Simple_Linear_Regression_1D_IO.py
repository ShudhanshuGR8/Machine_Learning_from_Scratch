# -*- coding: utf-8 -*-
"""
@author: shudh
"""



#%%
import numpy as np
from sklearn.datasets import make_regression 
import matplotlib.pyplot as plt

                                                                                                                                                                        
#%%
X , t = make_regression (100, 1, shuffle = True , bias = 0.0 ,noise = 20, random_state= 7)

                                                                                                                                                                
#%%
plt.scatter(X,t)


#%%
mean_x = np.mean(np.squeeze(X)) 
mean_t = np.mean(t)
std_x  = np.std(X)
std_t = np.std(t)
                                                                                                                                                                        

#%%
d_x= X - mean_x
d_t= t - mean_t
num = np.sum(np.squeeze(d_x) * d_t)
deno = np.sum(np.squeeze(d_x) * np.squeeze(d_x))
B1= num / deno 
B0= mean_t - (B1 * mean_x)


#%%
print(B1)
print(B0)
                                                                                                                                                                       

#%%
y= B0 + B1 * X
print(y)
print(t)


#%%
#plotting line 
plt.scatter(X, t, color='#000ff0')
plt.plot(X, y, color='#00ff00', label='Linear Regression Line')#plotting the data point

