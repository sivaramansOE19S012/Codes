
"""
Created on Thu Nov  5 09:30:25 2020

@author: Sivaraman Sivaraj
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
import os,sys


for dirname, _, filenames in os.walk('../input/zig-zag-20'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings("ignore", category=RuntimeWarning) 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    
    
data  = pd.read_csv('data 1.csv', header = 0)
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
time = data.iloc[:,0][1:][::20]
__u = data.iloc[:,1][1:][::20]# raw values
__v = data.iloc[:,2][1:][::20]
__r = data.iloc[:,3][1:][::20]
yaw_angle = data.iloc[:,4][1:][::20]
__U = data.iloc[:,5][1:][::20]
__rac_ = data.iloc[:,6][1:][::20]

# pt.graph_uvr(__u,__v,__r,2) # pltting u,v,r components
# pt.graph_Uy(yaw_angle,__U,__rac_, 2) # plotting total velocity, rudder angle change and yaw angle


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
        # c0 = U[i]
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
        
        temp = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,cb]
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

surgeXC, P_surge = Create_Surgecomponents(u, v, r, rac, U,171.8) # P - for plotting components
# pt.plot_surge_components_jsr2009(P_surge)


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
        # c0 = V[i]
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
        
        temp=[cb,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14]
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


swayYC, P_sway = Create_Swaycomponents(u, v, r, rac, U, 171.8)
# pt.plot_YN_components1_jsr2009(P_sway)
# pt.plot_YN_components2_jsr2009(P_sway)

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
        # c0 = R[i]
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
        
        temp=[cb,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14]
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

yawRC , P_yaw = Create_Yawcomponents(u, v, r, rac, U, 171.8)

def EqL_next(var):# left hand side of the equation
    var_next = []
    for i in range(len(var) -1):
        temp = var[i+1]
        var_next.append(temp)
    return var_next

def term_LHS(x,xU):
    op = []
    for i in range(298):
        temp = xU[i]-x[i]
        op.append(temp)
    return op

_nU = EqL_next(u)
nU = term_LHS(u,_nU) # we subtracting u(k+1) and u(k) values

_nV = EqL_next(v)
nV = term_LHS(v,_nV)

_nR = EqL_next(r)
nR = term_LHS(r,_nR) 

def ramp_signal(ip,Lambda): #lambda = ramp value
    op = []
    for i in range(len(ip)):
        temp = ip[i]+Lambda
        op.append(temp)
    return op

nU_ramp = ramp_signal(nU, 0.01)
nV_ramp = ramp_signal(nV, 0.01)
nR_ramp = ramp_signal(nR, 0.01)

"""
Start of features description
"""
h = 2 # time step size
L = 171.8 # length of ship
Xg = -0.023 # Longitutional co-ordinate of ship center of gravity 
m = 0.00798 # mass of ship
IzG = 39.2/(10**5) # Moment of inertia of ship around center of gravity

Xau= -42/(10**5) # accelaration derivative of surge force with respect to u 

Yav = -748/(10**5) # accelaration derivative of sway force with respect to v
Yar = -9.354/(10**5) # accelaration derivative of sway force with respect to r

Nav = 4.646/(10**5) # Yaw moment derivative with respect to sway velocity
Nar = -43.8/(10**5) # Yaw moment derivative with respect to rudder angle

S = ((IzG-Nar)*(m-Yav))-(((m*Xg)-Yar)*((m*Xg)-Nav))

"""
End of features description
"""

def term1(h,L,m,Xau):
    return (L*(m-Xau))/h
CC1 = term1(h,L,m,Xau)

def surge_derivatives(CC1,var):
    t1 = var[1:]
    surget = [i for i in map(lambda x : CC1*x, t1)]
    surge = []
    for elem in surget:
        surge.append(int(elem*(10**5)))
    return surge

from sklearn.svm import LinearSVR



def CN_simulation_surge(N,ip,op):
    """
    Parameters
    ----------
    N : Condition Number
    ip : input vector
    op : output vector

    Returns
    -------
    HDV - hydrodynamic derivative values
    S - score

    """
    HDV = list() # to store hydrodynamic values
    S = list() #to store train set accuracy
    
        
    for i in range(len(N)):
        svr = LinearSVR(C= N[i],fit_intercept = False, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(np.array(ip), np.array(op))
        coef = svr.coef_
        train_score = svr.score(np.array(ip), np.array(op))# score
        surge = surge_derivatives(CC1,coef)# surge derivatives values
        surge.append(train_score)#last term represents the train score
        HDV.append(surge)
        S.append(train_score)
        print(N[i])
        
    return HDV,S


import csv
def Documentation(D,l1):
    with open(str(l1)+'.csv', 'w') as f:
        write = csv.writer(f) 
        write.writerows(D)



l1n = [1,10,100,1000,1e04,1e05,1e06,1e07,1e08,1e09,1e10,1e11,1e12,1e13,1e14,1e15,1e16]
l1 = "1-10^16_surge"
Xx1,Sx1 = CN_simulation_surge(l1n, surgeXC, nU_ramp)# Xx1 - surge derivaties, Sx1 - score
Documentation(Xx1, l1)

N = np.arange(1e04-1000,1e04+1000,1)
n = "1000-1e04+1000_surge"
Xx2,Sx2 = CN_simulation_surge(N, surgeXC, nU_ramp)# Xx2 - surge derivaties, Sx2 - score

Documentation(Xx2,n)        
    



surge_hydrodynamic_derivatives = ['X`u', 'X`uu', 'X`uuu', 'X`vv', 'X`rr',
                                  'X`ðð', 'X`ððu', 'X`vr', 'X`vð', 'X`vðu']

Actual_Value = [-184,-110,-215,-899,18,
                    -95,-190,798,93,93]







def solution_Matrix(h,m,S,L,IzG,Nav,Yav,Yar,Xg):
    M11= h*(IzG-Nar)/(S*L)
    M12= (-h)*((m*Xg)-Yar)/(S*L)
    M21= (-h)*((m*Xg)-Nav)/(S*(L**2))
    M22= h*(m-Yav)/(S*(L**2))
    return np.array([[M11,M12],[M21,M22]])

def two_one_Matrix(t2,t3):
    List = []
    for i in range(15):
        temp = np.array([[t2[i]],[t3[i]]])
        List.append(temp)
    return List

def SNsolution(M,c): #sway and yaw moment solution
    List = []
    im = np.linalg.inv(M)
    for i in range(len(c)):
        temp = im.dot(c[i])
        List.append(temp)
    return List

def separation(M):
    sway_components = []
    yaw_components = []
    for i in M:
        sway_components.append(int(i[0][0]*(10**5)))
        yaw_components.append(int(i[1][0]*(10**5)))
    return sway_components,yaw_components

    
def CN_simulation_YN(N,ip1,op1,ip2,op2):
    """
    Parameters
    ----------
    N : Condition Number
    ip : input vector
    op : output vector

    Returns
    -------
    HDV_Y- hydrodynamic derivative values of sway
    HDV_N - hydrodynamic derivative values of yaw
    S1 -sway score
    S2 - yaw score
    """
    HDV_Y = list() #sway hydrodynamic values
    HDV_N = list() # yaw hydrodynamic values
    S1,S2 = list(),list()# score of sway and yaw
    for i in range(len(N)):
        svr1 = LinearSVR(C= N[i],fit_intercept = False, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(np.array(ip1), np.array(op1))
        svr2 = LinearSVR(C= N[i],fit_intercept = False, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(np.array(ip2), np.array(op2))
        Y = svr1.coef_
        R = svr2.coef_
        train_score1 = svr1.score(np.array(ip1), np.array(op1))# score
        train_score2 = svr2.score(np.array(ip2), np.array(op2))# score
        solMatrix = solution_Matrix(h,m,S,L,IzG,Nav,Yav,Yar,Xg) # equation 24 in JSR 2009 paper
        c = two_one_Matrix(Y,R) # equation 24 contiunes
        Sway_Yaw_derivatives = SNsolution(solMatrix,c)
        sway, yaw = separation(Sway_Yaw_derivatives)
        sway.append(train_score1)
        yaw.append(train_score2)
        HDV_Y.append(sway)
        HDV_N.append(yaw)
        S1.append(train_score1)
        S2.append(train_score2)
        print(N[i])
    return HDV_Y, HDV_N, S1,S2



        
l1n = [1,10,100,1000,1e04,1e05,1e06,1e07,1e08,1e09,1e10,1e11,1e12,1e13,1e14,1e15,1e16]
y1 = "1-10^16_sway"
n1 = "1-10^16_yaw"
Yy1,Nn1,Sy1,Sn1 = CN_simulation_YN(l1n,swayYC,nV_ramp,yawRC,nR_ramp) 

Documentation(Yy1, y1)
Documentation(Nn1,n1)

N = np.arange(1e04-1000,1e04+1000,1)
y2 = "1000-1e04+1000_sway"
n2 = "1000-1e04+1000_yaw"
Yy2,Nn2,Sy2,Sn2 = CN_simulation_YN(N,swayYC,nV_ramp,yawRC,nR_ramp)

Documentation(Yy2, y2)
Documentation(Nn2,n2)



        
    
    
    





