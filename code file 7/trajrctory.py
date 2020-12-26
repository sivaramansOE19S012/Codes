"""
Created on Thu Nov  5 09:30:25 2020

@author: Sivaraman Sivaraj
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
import os,sys
# import plots as pt

for dirname, _, filenames in os.walk('../input/zig-zag-20'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings("ignore", category=RuntimeWarning) 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    
    
data  = pd.read_csv('data 3.csv', header = 0)
"""
7*1 format

0. time
1.surge_speed
2.sway_speed
3.yaw_rate
4.yaw_angle
5.total_speed
6.rudder_angle

note : pandas is only for visualization, we need to create list as of our requirment. 
"""
time = data.iloc[:,0][1:][::]
__u = data.iloc[:,1][1:][::]# raw values
__v = data.iloc[:,2][1:][::]
__r = data.iloc[:,3][1:][::]
yaw_angle = data.iloc[:,4][1:][::]
__U = data.iloc[:,5][1:][::]
__rac_ = data.iloc[:,6][1:][::]


def radian(var):
    op = np.pi *var/180
    return op

__rac = radian(__rac_) # changing the values to radian mode

def listform(var): # pandas library doesn't give list, we need to create list
    ans = []
    for i in var:
        ans.append(i)
    return ans
_u = listform(__u)
_v = listform(__v)
_r = listform(__r)
_U = listform(__U)
_rac = listform(__rac)

def delta_val(var): # to find the delta value, 
    D =[]
    for i in range(1,len(var)):
        temp = var[i]-var[0]
        D.append(temp)
    return D

u = delta_val(_u)
v = delta_val(_v)
r = delta_val(_r)
rac = delta_val(_rac) 
U = _U # fifth element total veocity may take as it is.


# pt.graph_delta_all(u,v,r,rac) # pltting the delta values of u,v,r components

"""
surgeXC = first governing equation
swayYC = second governing equation
yawRC = third governing equation

"""

def Create_Surgecomponents(u,v, yawRate, RAC, TV,L):
    """
    Parameters
    ----------
    u : surge velocity-delta
    v : sway velocity_delta
    yawRate : delta
    RAC : rudder angle change in radian
    TV : total velocity
    L : length of ship

    Returns
    -------
    Surge_Components : list of 11 surege components & plot components as P_surge

    """
    P = [list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list()]
    Surge_Components = []
    for i in range(len(u)):
        c0 = U[i]
        c1 = u[i]*TV[i]
        c2 = u[i]**2
        c3 = (u[i]**3)/TV[i]
        c4 = v[i]**2
        c5 = (yawRate[i]**2)*(L**2)
        c6 = (RAC[i]**2)*(TV[i]**2)
        c7 = (RAC[i]**2)*u[i]*TV[i]
        c8 = v[i]*yawRate[i]*L
        c9 = v[i]*RAC[i]*TV[i]
        c10 = v[i]*RAC[i]*u[i]
        cb = TV[i]**2 #bias term
        
        temp = [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,cb]
        Surge_Components.append(temp)
        P[0].append(c1)
        P[1].append(c2)
        P[2].append(c3)
        P[3].append(c4)
        P[4].append(c5)
        P[5].append(c6)
        P[6].append(c7)
        P[7].append(c8)
        P[8].append(c9)
        P[9].append(c10)
        P[10].append(cb)
       
    Surge_Components.pop()# for last element, we can't have (k+1) component  
    return Surge_Components, P

surgeXC_all, P_surge_all = Create_Surgecomponents(u, v, r, rac, U,171.8) # P - for plotting components

surgeXC_train = surgeXC_all[0:12800]
surgeXC_validation = surgeXC_all[12800:]
# print(len(surgeXC_all))

def Create_Swaycomponents(U,V,R,RAC,TV,L):
    """
    Parameters
    ----------
    u : surge velocity-delta
    v : sway velocity_delta
    yawRate : delta
    RAC : rudder angle change in radian
    TV : total velocity
    L : length of ship

    Returns
    -------
    Sway_Components : list of 15 sway components & plot components as P_sway


    """
    Sway_Components = []
    P = [list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),
         list(),list(),list(),list()]
    for i in range(len(U)):
        c0 = V[i]
        cb = TV[i]**2 #bias term
        c1 = U[i]*TV[i]
        c2 = U[i]**2
        c3 = V[i]*TV[i]
        c4 = R[i]*TV[i]*L
        c5 = RAC[i]*(TV[i]**2)
        c6 = (V[i]**3)/TV[i]
        c7 = (RAC[i]**3)*(TV[i]**2)
        c8 = (V[i]**2)*R[i]*L/TV[i]#1 by L
        c9 = (V[i]**2)*RAC[i]
        c10 = V[i]*(RAC[i]**2)*TV[i]
        c11 = RAC[i]*U[i]*TV[i]
        c12 = V[i]*U[i]
        c13 = R[i]*U[i]*L
        c14 = RAC[i]*(U[i]**2)
        
        temp=[c0,cb,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14]
        Sway_Components.append(temp)
        P[0].append(cb)
        P[1].append(c1)
        P[2].append(c2)
        P[3].append(c3)
        P[4].append(c4)
        P[5].append(c5)
        P[6].append(c6)
        P[7].append(c7)
        P[8].append(c8)
        P[9].append(c9)
        P[10].append(c10)
        P[11].append(c11)
        P[12].append(c12)
        P[13].append(c13)
        P[14].append(c14)
    Sway_Components.pop()
    return Sway_Components, P


swayYC_all, P_sway_all = Create_Swaycomponents(u, v, r, rac, U, 171.8)

swayYC_train = swayYC_all[0:12800]
swayYC_validation = swayYC_all[12800:]



def Create_Yawcomponents(U,V,R,RAC,TV,L):
    """
    Parameters
    ----------
    u : surge velocity-delta
    v : sway velocity_delta
    yawRate : delta
    RAC : rudder angle change in radian
    TV : total velocity
    L : length of ship

    Returns
    -------
    yaw_Components : list of 16 yaw components & plot components as P

    """
    Yaw_Components = []
    P = [list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),
         list(),list(),list(),list()]
    for i in range(len(U)):
        c0 = R[i]
        cb = U[i]**2#bias term
        c1 = U[i]*TV[i]
        c2 = U[i]**2
        c3 = V[i]*TV[i]
        c4 = R[i]*TV[i]*L
        c5 = RAC[i]*(TV[i]**2)
        c6 = V[i]**3 / TV[i]
        c7 = (RAC[i]**3)*(TV[i]**2)
        c8 = (V[i]**2)*R[i]*L/TV[i]# 1 by L
        c9 = (V[i]**2)*RAC[i]
        c10 = V[i]*(RAC[i]**2)*TV[i]
        c11 = RAC[i]*U[i]*TV[i]
        c12 = V[i]*U[i]
        c13 = R[i]*U[i]*L
        c14 = RAC[i]*(U[i]**2)
        
        temp=[c0,cb,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14]
        Yaw_Components.append(temp)
        P[0].append(cb)
        P[1].append(c1)
        P[2].append(c2)
        P[3].append(c3)
        P[4].append(c4)
        P[5].append(c5)
        P[6].append(c6)
        P[7].append(c7)
        P[8].append(c8)
        P[9].append(c9)
        P[10].append(c10)
        P[11].append(c11)
        P[12].append(c12)
        P[13].append(c13)
        P[14].append(c14)
        
    Yaw_Components.pop()
    return Yaw_Components,P

yawNC_all , P_yaw_all = Create_Yawcomponents(u, v, r, rac, U, 171.8)

yawNC_train = yawNC_all[0:12800]
yawNC_validation = yawNC_all[12800:]


def EqL_next(var):# left hand side of the equation
    var_next = []
    for i in range(len(var) -1):
        temp = var[i+1]
        var_next.append(temp)
    return var_next


nU = EqL_next(u)# we subtracting u(k+1) and u(k) values
nV = EqL_next(v)
nR = EqL_next(r)

def ramp_signal(ip,Lambda): #lambda = ramp value
    op = []
    for i in range(len(ip)):
        temp = ip[i]+Lambda
        op.append(temp)
    return op

nU_ramp_train = ramp_signal(nU, 0.01)[:12800]
nV_ramp_train = ramp_signal(nV, 0.01)[:12800]
nR_ramp_train = ramp_signal(nR, 0.01)[:12800]

nU_ramp_val = ramp_signal(nU, 0.01)[12800:]
nV_ramp_val = ramp_signal(nV, 0.01)[12800:]
nR_ramp_val = ramp_signal(nR, 0.01)[12800:]

np.save("D3_nU_ramp.npy",nU_ramp_val)
np.save("D3_nV_ramp.npy",nV_ramp_val)
np.save("D3_nR_ramp.npy",nR_ramp_val)

from sklearn.svm import LinearSVR
from sklearn.svm import SVR

# """
# SVM construction for Surge
# """



# linear_svm1 = LinearSVR(C= 1e04,fit_intercept = False, dual = True ,
#                         epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
#                         max_iter = 10000, random_state = None, tol = 0.000001,
#                         verbose = 0).fit(np.array(surgeXC_train), np.array(nU_ramp_train))

# print("Train set accuracy of Surge on LinearSVR method: {:.2f}".format(linear_svm1.score(np.array(surgeXC_train), np.array(nU_ramp_train))))
# u_predicted = linear_svm1.predict(surgeXC_validation)


# """
# SVM construction for Sway
# """



# linear_svm2 =  LinearSVR(C= 1e04,fit_intercept = False, dual = True ,
#                         epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
#                         max_iter = 10000, random_state = None, tol = 0.000001,
#                         verbose = 0).fit(np.array(swayYC_train), np.array(nV_ramp_train))
# print("Train set accuracy of Sway on LinearSVR method: {:.2f}".format(linear_svm2.score(swayYC_train,nV_ramp_train)))
# v_predicted = linear_svm2.predict(swayYC_validation)

# """
# SVM construction for yaw
# """


# linear_svm3 = LinearSVR(C= 1e04,fit_intercept = False, dual = True ,
#                         epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
#                         max_iter = 10000, random_state = None, tol = 0.000001,
#                         verbose = 0).fit(np.array(yawNC_train), np.array(nR_ramp_train))
# print("Train set accuracy of Yaw on LinearSVR method: {:.2f}".format(linear_svm3.score(yawNC_train,nR_ramp_train)))
# n_predicted = linear_svm3.predict(yawNC_validation)





# def graph_uvr(nU_ramp_train,nV_ramp_train,nR_ramp_train,
#               u_predicted,v_predicted,n_predicted,time_interval):
#     plt.figure(figsize=(15,12))
    
#     plt.subplot(311)
#     x311 = np.arange(0,len(nU_ramp_train))
#     plt.plot(x311,nU_ramp_train,'g',label = 'Simulation')
#     plt.plot(x311,u_predicted,'r--',label = 'SVM')
#     plt.xlabel('time in '+str(time_interval)+' seconds increment')
#     plt.ylabel('Surge Speed')
#     plt.legend(loc='best')
#     plt.title("Result Comparision")
#     #plt.axvline(x=100,color = 'r')
      
#     plt.subplot(312)
#     x312 = np.arange(0,len(nV_ramp_train))
#     plt.plot(x312,nV_ramp_train,'g',label = 'Simulation')
#     plt.plot(x312,v_predicted,'r--',label = 'SVM')
#     plt.xlabel('time in '+str(time_interval)+' seconds increment')
#     plt.ylabel('Sway Speed')
#     plt.legend(loc='best')
#     #plt.axvline(x=100,color = 'r')
    
#     plt.subplot(313)
#     x313 = np.arange(0,len(nR_ramp_train))
#     plt.plot(x313,nR_ramp_train,'g',label = 'Simulation')
#     plt.plot(x313,n_predicted,'r--',label = 'SVM')
#     plt.xlabel('time in '+str(time_interval)+' seconds increment')
#     plt.ylabel('yaw rate')
#     plt.savefig("2.5_deg_5_SVM_result_80_20.pdf", dpi=2400)
#     plt.legend(loc='best')
    
#     #plt.axvline(x=100,color = 'r')    
#     # plt.savefig('plot_1.jpg')
#     plt.show()

# graph_uvr(nU_ramp_val,nV_ramp_val,nR_ramp_val,
#               u_predicted,v_predicted,n_predicted,0.1)
    
# # def graph_Uy(yaw_angle,__U,__rac_, time_interval):
# #     plt.figure(figsize=(15,12))
# #     plt.subplot(211)
# #     x212 = np.arange(0,len(yaw_angle))
# #     plt.plot(x212,yaw_angle,'g')
# #     plt.plot(x212,__rac_,'m')
# #     plt.xlabel('time in '+str(time_interval)+' seconds increment')
# #     plt.ylabel('Yaw Angle')
# #     plt.title('Heading angle (Ψ) vs Rudder Angle(𝛿)')
# #     #plt.axvline(x=100,color = 'r')
    











