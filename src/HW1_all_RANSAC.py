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



#Implementation of RANSAC Outlier technique

#function to evaluate the perpendicular distance from the selected line model to a point
def dist(m,b,point):
    x = point[0]
    y = point[1]
    dist_per = np.abs((y-m*x-b)/np.sqrt(1+m*m))                          #computing the perpendicular distance
    return dist_per



def iter(S1_t,ransac_threshold,inl_rat):
    for iteration in range(0,50):                                            #'iteration' is the current iteration step

        inliers_list = []                                                   #creating empty lists to store inliers
        outliers_list = []                                                  #creating empty lists to store outliers

        pt_1 = S1_t[np.random.randint(0,S1_t.shape[0]-1),:]                 #randomly selecting point1
        pt_2 = S1_t[np.random.randint(0,S1_t.shape[0]-1),:]                 #randomly selecting point2

        m2 = (pt_2[1]-pt_1[1])/(pt_2[0]-pt_1[0]+ sys.float_info.epsilon)    #computing the slope of line for selected points
        b2 = pt_1[1]-m2*pt_1[0]                                             #computing the y-intersept for selected points

        for i in range(S1_t.shape[0]-1):
            if dist(m2,b2,S1_t[i,:])<= ransac_threshold:             #comparing the perpendicular distances with threshold
               inliers_list.append(S1_t[i,:])
            else:
                outliers_list.append(S1_t[i,:])

        inliers_array = np.asarray(inliers_list)
        outliers_array = np.asarray(outliers_list)

        print('iteration number:',format(iteration))
        inlier_ratio = inliers_array.shape[0]/S1_t.shape[0];
        print('inlier ratio:',format(inlier_ratio))

        if inlier_ratio >= inl_rat:
            break

    return m2,b2,inliers_array,outliers_array

m1,b1, inliers_array1, outliers_array1= iter(S1_t,25,0.9)
m2,b2, inliers_array2, outliers_array2 = iter(S2_t,25,0.7)
m3,b3, inliers_array3, outliers_array3 = iter(S3_t,25,0.5)


plt.subplot(131)
#plotting the final model
plt.plot(X1,m1*X1+b1,linestyle='solid', color='green',label='line fit')              #fitting the lie based on RANSAC algorithm
plt.plot(inliers_array1[:,0],inliers_array1[:,1],'b+',label='inliers')                #plotting all the inliers
plt.plot(outliers_array1[:,0],outliers_array1[:,1],'r+',label='outliers')              #plotting all the outliers
plt.legend()

plt.subplot(132)
#plotting the final model
plt.plot(X2,m2*X2+b2,linestyle='solid', color='green',label='line fit')              #fitting the lie based on RANSAC algorithm
plt.plot(inliers_array2[:,0],inliers_array2[:,1],'b+',label='inliers')                #plotting all the inliers
plt.plot(outliers_array2[:,0],outliers_array2[:,1],'r+',label='outliers')              #plotting all the outliers
plt.legend()

plt.subplot(133)
#plotting the final model
plt.plot(X3,m3*X3+b3,linestyle='solid', color='green',label='line fit')              #fitting the lie based on RANSAC algorithm
plt.plot(inliers_array3[:,0],inliers_array3[:,1],'b+',label='inliers')                #plotting all the inliers
plt.plot(outliers_array3[:,0],outliers_array3[:,1],'r+',label='outliers')              #plotting all the outliers
plt.legend()

plt.show()

