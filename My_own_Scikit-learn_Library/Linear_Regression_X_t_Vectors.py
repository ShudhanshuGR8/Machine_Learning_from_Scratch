# -*- coding: utf-8 -*-
"""
@author: shudh
"""

#%%
import numpy as np
from sklearn.datasets import make_regression 


#%%
X , t = make_regression (100,5, shuffle = True , bias = 0.0 , n_targets=3 ,noise = 10, random_state= 4) ;


#%%	
x0=np.ones((100,1), dtype=int);
new_x = np.concatenate((x0,X), axis = 1);
tr=new_x.transpose();
m3 = np. dot(tr,new_x);


#%%
i = np.linalg.inv(m3);
m4 = np. dot(i,tr);
m5=np.dot(m4,t);
transp=m5.transpose();


#%%
print("The value of W is:-");
print(m5);


#%%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
w = model.fit(new_x,t)


#%%
print("Comparing the Value of w using sklearn function:")
print(model.intercept_)
print(model.coef_)

