# -*- coding: utf-8 -*-
"""
@author: shudh
"""

#%%
from sklearn.datasets import make_regression
X,t = make_regression(100, 5, shuffle = True, bias = 0, noise = 77, random_state = 9)


#%%
X_mean = X.mean()
t_mean = t.mean()


#%%
X_std = X.std()
t_std = t.std()


#%%
print(X)
print(t)


#%%
import numpy as np
x0 = np.ones((100,1))


#%%
new_X = np.concatenate((x0, X), axis =1)
new_X.shape


#%%
X_t = np.transpose(new_X)


#%%
W_inv = np.dot(X_t,new_X)
W_1 = np.linalg.inv(W_inv)


#%%
W_2 = np.dot(X_t,t)


#%%
W = np.dot(W_1, W_2)


#%%
print(W)


#%%
#For comparison of our created model and the one used in scikit-learn library
from sklearn.linear_model import LinearRegression
model = LinearRegression()
w = model.fit(X,t)

#%%
print(model.coef_)
print(model.intercept_)
#Now we can see that our model is exactly the same - it shows the same model intercept and same model coefficients


#%%
X_vec,t_vec = make_regression(100, 3, shuffle = True, bias = 0, noise = 40, random_state = 9, n_targets = 3)


#%%
print(X_vec)
print(t_vec)


#%%
x0 = np.ones((100,1))
new_X = np.concatenate((x0, X_vec), axis =1)
X_vec_t = np.transpose(new_X)

W_inv = np.dot(X_vec_t,new_X)
W_1 = np.linalg.inv(W_inv)

W_2 = np.dot(W_1, X_vec_t)

W = np.dot(W_2, t_vec)


#%%
print(W)


#%%
#For comparison of our created model and the one used in scikit-learn library
from sklearn.linear_model import LinearRegression
#model = LinearRegression(fit_intercept=False)
model = LinearRegression()

w = model.fit(X_vec,t_vec)

print(model.intercept_)

print(model.coef_)
#Now we can see that our model is exactly the same - it shows the same model intercept and same model coefficients


#%%
from sklearn.metrics import r2_score
y_predict = model.predict(X_vec)
y_predict_1 = np.dot(X_vec, W[1:])

score = r2_score(t_vec, y_predict)
score_1 = r2_score(t_vec, y_predict_1)

print(score)
print(score_1)


#%%
noise_lt = [0, 10, 30, 50, 70, 90, 100]
r2_lt = [1.0, 0.9911277146802907, 0.9233227047988551, 0.8082131795660822, 0.6769299465749685, 0.5533416629728133, 0.4983050464146044]
import matplotlib.pyplot as plt
plt.plot(noise_lt, r2_lt, label = 'Relation between Noise and R2_score')
plt.legend()
plt.show()


#%%
import sklearn.model_selection as model_selection
X_vec,t_vec = make_regression(100, 3, shuffle = True, bias = 0, noise = 40, random_state = 9, n_targets = 3)


#%%
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_vec, t_vec, test_size=0.3)


#%%
from sklearn.linear_model import LinearRegression
model = LinearRegression()

w = model.fit(X_train,Y_train)

model.coef_

model.intercept_


#%%
y_pred_train = model.predict(X_train)

score_train = r2_score(y_pred_train, Y_train)
print(score_train)


#%%
y_pred_test = model.predict(X_test)
score_test = r2_score(y_pred_test, Y_test)
print(score_test)


#%%
noise_list = [0, 10, 30, 50, 70 ,90 ,100]
score_train_list = [1,0.9892422892838316,0.9123535974919882,0.7851886218805498,0.4866233488454797,0.2540158327162871,-0.07161444865168864]
score_test_list = [1,0.9927037387643658,0.9000725805417417,0.698401043958205,0.5189450019757639,0.2725673734949044,-0.2642270301805036]
plt.plot(noise_list, score_train_list, label = 'Training_data')
plt.plot(noise_list, score_test_list, label = 'Test_data')
plt.legend(loc="upper right")
plt.show()


#%%
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax1.plot(noise_list, score_train_list, label = 'Training_data')
ax1.legend()
ax2.plot(noise_list, score_test_list, label = 'Test_data')
ax2.legend()
