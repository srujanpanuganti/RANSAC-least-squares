import pickle
import sys
import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

#importing the pickle data file data1_new

name1 = 'data1_new.pkl'
name2 = 'data2_new.pkl'
name3 = 'data3_new.pkl'

#function to import data from pickel file
def pkl_imp(name):
    fname = name
    data_pickle1 = open(fname, 'rb')
    new_list1 = pickle.load(data_pickle1)
    data_pickle1.close()
    return new_list1

new_list1 = pkl_imp(name1)
new_list2 = pkl_imp(name2)
new_list3 = pkl_imp(name3)

data_list1 = np.asarray(new_list1)                          #converting the imported list to an array
data_list2 = np.asarray(new_list2)
data_list3 = np.asarray(new_list3)

X1 = data_list1[:, 0]                                       #Splitting the converted array coloumn by coloumns.
Y1 = data_list1[:, 1]
S1 = np.stack((X1, Y1), axis=0)                             #creating a stack
S1_t = S1.T


X2 = data_list2[:, 0]                                       #Splitting the converted array coloumn by coloumns.
Y2 = data_list2[:, 1]
S2 = np.stack((X2, Y2), axis=0)                             #creating a stack
S2_t = S2.T


X3 = data_list3[:, 0]                                       #Splitting the converted array coloumn by coloumns.
Y3 = data_list3[:, 1]
S3 = np.stack((X3, Y3), axis=0)                             #creating a stack
S3_t = S3.T


# Defining the Covariance function
def cov(x, y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)

# Covariance matrix
def cov_mat(X):
    return np.array([[cov(X[0], X[0]), cov(X[0], X[1])], \
                     [cov(X[1], X[0]), cov(X[1], X[1])]])

# Calculate covariance matrix
C1_x = cov_mat(S1)
C2_x = cov_mat(S2)
C3_x = cov_mat(S3)

#Finding the Eigen Values and corresponding Eigen Vectors
w1, v1 = LA.eig(C1_x)
w2, v2 = LA.eig(C2_x)
w3, v3 = LA.eig(C3_x)

#splitting the eigen vectors
Vd1_1 =w1[0]*v1[:, 0]
Vd1_2 =w1[1]*v1[:, 1]


Vd2_1 = w2[0]*v2[:, 0]
Vd2_2 = w2[1]*v2[:, 1]


Vd3_1 = w3[0]*v3[:, 0]
Vd3_2 = w3[1]*v3[:, 1]


#Least Squares using Vertical Distances

def ls(X1,Y1):

    S1 = np.stack((X1, Y1), axis=0)                             #creating a stack
    S1_t = S1.T

    ones = np.ones(S1_t.shape[0])                             #preparing the data

    X_P = np.stack((X1,ones),axis=0)
    X_P_t = np.transpose(X_P)

    B = np.dot(np.linalg.inv(np.dot(X_P,X_P_t)),(np.dot(X_P,Y1)))

    #obtained  slope and intersept from the above equation
    m = B[0]
    b = B[1]
    return m,b

m1,b1 = ls(X1,Y1)
m2,b2 = ls(X2,Y2)
m3,b3 = ls(X3,Y3)

#total least squares perpendicular distances


def tls(x,y):

    n = x.shape[0]

    x_tilde = x - np.mean(x)
    y_tilde = y - np.mean(y)

    #X_stack = np.stack(x_tilde,y_tilde)
    X_stack = np.stack((x_tilde, y_tilde), axis=0)
    X_t = np.transpose(X_stack)

    U = np.dot(X_stack,X_t)     #Calculating the A_T_A
    W,V = np.linalg.eig(U)      #Eigen values and vectors
    s_eig = W.argmin()      #getting the smallest eigen value
    a,b = V[:,s_eig]        #Eigen vector corresponding to the smallest eigen value

    c = (-a*np.mean(x))+(-b*np.mean(y))     #solving for non-homogeneous equation

    return a,b,c

a1,b1,c1 = tls(X1,Y1)
a2,b2,c2 = tls(X2,Y2)
a3,b3,c3 = tls(X3,Y3)


def Y_fit(a1,b1,c1,X1):
    y_fitted = (-a1/b1)*X1 + (-c1/b1)                        #Line-Equation in from slope intercept form
    return y_fitted

y_fitted1 = Y_fit(a1,b1,c1,X1)
y_fitted2 = Y_fit(a2,b2,c2,X2)
y_fitted3 = Y_fit(a3,b3,c3,X3)


#plotting the final model

plt.subplot(131)
plt.scatter(X1, Y1, marker='o',label='data points')                                #Scatter Plot
plt.plot(X1,X1*m1+b1, linestyle = 'solid',color='green',label='LS')                #Least Squares Plot
plt.plot(X1,y_fitted1,linestyle = 'solid', color = 'red',label='TLS')               #Total Least Squares Plot
plt.quiver(0,0, Vd1_1[0], Vd1_1[1],scale=9000,label='eigen vector1')                 #eigen vector1
plt.quiver(0,0, Vd1_2[0],Vd1_2[1], scale=9000,label='eigen vector2')                 #eigen vector2
plt.legend()

plt.subplot(132)
plt.scatter(X2, Y2, marker='o',label='data points')                                #Scatter Plot
plt.plot(X2,X2*m2+b2, linestyle = 'solid',color='green',label='LS')                #Least Squares Plot
plt.plot(X2,y_fitted2,linestyle = 'solid', color = 'red',label='TLS')               #Total Least Squares Plot
plt.quiver(0,0, Vd2_1[0], Vd2_1[1],scale=9000,label='eigen vector1')                 #eigen vector1
plt.quiver(0,0, Vd2_2[0], Vd2_2[1], scale=9000,label='eigen vector2')                 #eigen vector2
plt.legend()

plt.subplot(133)
plt.scatter(X3, Y3, marker='o',label='data points')                                #Scatter Plot
plt.plot(X3,X3*m3+b3, linestyle = 'solid',color='green',label='LS')                #Least Squares Plot
plt.plot(X3,y_fitted3,linestyle = 'solid', color = 'red',label='TLS')               #Total Least Squares Plot
plt.quiver(0,0, Vd3_1[0], Vd3_1[1],scale=9000,label='eigen vector1')                 #eigen vector1
plt.quiver(0,0, Vd3_2[0], Vd3_2[1], scale=9000,label='eigen vector2')                 #eigen vector2
plt.legend()
plt.show()



