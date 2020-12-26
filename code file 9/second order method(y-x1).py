
"""
Created on Thu Nov  5 09:30:25 2020

@author: Sivaraman Sivaraj
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
import os,sys
import plots as pt

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
time = data.iloc[:,0][1:][::]
__u = data.iloc[:,1][1:][::]# raw values
__v = data.iloc[:,2][1:][::]
__r = data.iloc[:,3][1:][::]
yaw_angle = data.iloc[:,4][1:][::]
__U = data.iloc[:,5][1:][::]
__rac_ = data.iloc[:,6][1:][::]

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

"""
Delta k values
"""
Delta_u_k = delta_val(_u) #Delta k values
Delta_v_k = delta_val(_v)
Delta_r_k = delta_val(_r)
Delta_rac_k = delta_val(_rac) 
U = _U # fifth element total veocity may take as it is.

"""
Delta (k+1) and Delta (k+2) values

"""

def Delta_k_next(var):# left hand side of the equation
    var_next = []
    for i in range(len(var) -1):
        temp = var[i+1]
        var_next.append(temp)
    return var_next


Delta_u_k1 = Delta_k_next(Delta_u_k)
Delta_v_k1 = Delta_k_next(Delta_v_k)
Delta_r_k1 = Delta_k_next(Delta_r_k)
Delta_rac_k1 = Delta_k_next(Delta_rac_k)


Delta_u_k2 = Delta_k_next(Delta_u_k1)
Delta_v_k2 = Delta_k_next(Delta_v_k1)
Delta_r_k2 = Delta_k_next(Delta_r_k1)
Delta_rac_k2 = Delta_k_next(Delta_rac_k1)

"""
Left Hand side of the equation

L.H.S = ðu(k+1)-ðu(k) = Δu(k+2)-Δu(k+1)+Δu(k) -->surge
L.H.S = ðv(k+1)-ðv(k) = Δv(k+2)-Δv(k+1)+Δv(k)  --> sway
L.H.S = ðr(k+1)-ðr(k) = Δr(k+2)-Δr(k+1)+Δr(k) --> yaw
"""


def subtraction_cons(L1,L2):
    difference = list()
    zip_object = zip(L1,L2)
    for t1,t2 in zip_object:
        difference.append(t1-t2)
    return difference


delta_u_k = subtraction_cons(Delta_u_k1,Delta_u_k)
delta_u_k1 = subtraction_cons(Delta_u_k2,Delta_u_k1)
nU = subtraction_cons(delta_u_k1,delta_u_k)# left hand side of the first Equ.

delta_v_k = subtraction_cons(Delta_v_k1,Delta_v_k)
delta_v_k1 = subtraction_cons(Delta_v_k2,Delta_v_k1)
nV = subtraction_cons(delta_v_k1,delta_v_k)# left hand side of the second Equ.

delta_r_k = subtraction_cons(Delta_r_k1,Delta_r_k)
delta_r_k1 = subtraction_cons(Delta_r_k2,Delta_r_k1)
nR = subtraction_cons(delta_r_k1,delta_r_k)# left hand side of the third Equ.

# pt.graph_delta_all(Delta_u_k1,Delta_v_k1,Delta_r_k1,Delta_rac_k1) # plotting the delta values of u,v,r components


"""
Ramp signal addition
"""

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
Input vectors to SVM
Δu(k+1)-Δu(k) --> surge
Δv(k+1)-Δv(k) --> sway
Δr(k+1)-Δr(k) --> yaw

"""

# u = subtraction_cons(Delta_u_k1,Delta_u_k) 
# v = subtraction_cons(Delta_v_k1,Delta_v_k)
# r = subtraction_cons(Delta_r_k1,Delta_r_k)
# rac = subtraction_cons(Delta_rac_k1,Delta_rac_k)
# U = _U


"""
surgeXC = first governing equation
swayYC = second governing equation
yawRC = third governing equation

"""

def Create_Surgecomponents(u1,v1,r1,u,v,r, rac,rac1, U,L):
    """
    Parameters
    ----------
    u1,u : surge velocity- Δu(k+1), Δu(k)
    v1,v : sway velocity_ Δv(k+1),Δv(k)
    r1,r : yaw rate_ Δr(k+1),Δr(k)
    rac : rudder angle change in radian- Δrac(k+1)-Δrac(k)
    U : total velocity
    L : length of ship

    Returns
    -------
    Surge_Components : list of 11 surege components & plot components as P_surge

    """
    P = [list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list()]
    Surge_Components = []
    for i in range(len(u1)):
        
        c1 = (u1[i]-u[i])*U[i]
        c2 = (u1[i]**2) -(u[i]**2)
        c3 = ((u1[i]**3)-(u[i]**3))/U[i]
        c4 = (v1[i]**2)-(v[i]**2)
        c5 = ((r1[i]**2)-(r[i]**2))*(L**2)
        c6 = ((rac1[i]**2)-(rac[i]**2))*(U[i]**2)
        c7 = ((rac1[i]**2)*u1[i]-(rac[i]**2)*u[i])*U[i]
        c8 = ((v1[i]*r1[i]) - (v1[i]*r1[i]))*L
        c9 = ((v1[i]*rac1[i])-(v[i]*rac[i]))*U[i]
        c10 = (v1[i]*rac1[i]*u1[i])-(v[i]*rac[i]*u[i])
        cb = U[i]**2 #bias term
        
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

surgeXC, P_surge = Create_Surgecomponents(Delta_u_k1,Delta_u_k,Delta_v_k1,Delta_v_k,
                                          Delta_r_k1,Delta_r_k,
                                          Delta_rac_k1,Delta_rac_k, U,171.8) # P - for plotting components
# pt.plot_surge_components_jsr2009(P_surge)
print(len(surgeXC))

def Create_Swaycomponents(u1,v1,r1,u,v,r, rac,rac1, U,L):
    """
    Parameters
    ----------
    u1,u : surge velocity- Δu(k+1), Δu(k)
    v1,v : sway velocity_ Δv(k+1),Δv(k)
    r1,r : yaw rate_ Δr(k+1),Δr(k)
    rac : rudder angle change in radian- Δrac(k+1)-Δrac(k)
    U : total velocity
    L : length of ship

    Returns
    -------
    Sway_Components : list of 15 sway components & plot components as P_sway


    """
    Sway_Components = []
    P = [list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),
         list(),list(),list(),list()]
    for i in range(len(u1)):
        
        cb = U[i]**2 #bias term
        c1 = (u1[i]-u[i])*U[i]
        c2 = (u1[i]**2)-(u[i]**2)
        c3 = (v1[i]-v[i])*U[i]
        c4 = (r1[i]-r[i])*U[i]*L
        c5 = (rac1[i]-rac[i])*(U[i]**2)
        c6 = ((v1[i]**3)-(v[i]**3))/U[i]
        c7 = ((rac1[i]**3)-(rac[i]**3))*(U[i]**2)
        c8 = (((v1[i]**2)*r1[i])-((v[i]**2)*r[i]))*L/U[i]
        c9 = (((v[i]**2)*rac1[i])-((v[i]**2)*rac[i]))
        c10 = ((v1[i]*(rac1[i]**2))-(v[i]*(rac[i]**2)))*U[i]
        c11 = ((rac1[i]*u1[i])-(rac[i]*u[i]))*U[i]
        c12 = (v1[i]*u1[i]) - (v[i]*u[i])
        c13 = (r1[i]*u1[i]*L) - (r[i]*u[i]*L)
        c14 = (rac1[i]*(u1[i]**2))- (rac[i]*(u[i]**2))
        
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


swayYC, P_sway = Create_Swaycomponents(Delta_u_k1,Delta_u_k,Delta_v_k1,Delta_v_k,
                                          Delta_r_k1,Delta_r_k,
                                          Delta_rac_k1,Delta_rac_k, U,171.8)
# pt.plot_YN_components1_jsr2009(P_sway)
# pt.plot_YN_components2_jsr2009(P_sway)

def Create_Yawcomponents(u1,v1,r1,u,v,r, rac,rac1, U,L):
    """
    Parameters
    ----------
    u1,u : surge velocity- Δu(k+1),Δu(k)
    v1,v : sway velocity_ Δv(k+1),Δv(k)
    r1,r : yaw rate_ Δr(k+1),Δr(k)
    rac : rudder angle change in radian- Δrac(k+1)-Δrac(k)
    U : total velocity
    L : length of ship

    Returns
    -------
    yaw_Components : list of 16 yaw components & plot components as P

    """
    Yaw_Components = []
    P = [list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),list(),
         list(),list(),list(),list()]
    for i in range(len(u1)):
        
        cb = U[i]**2 #bias term
        c1 = (u1[i]-u[i])*U[i]
        c2 = (u1[i]**2)-(u[i]**2)
        c3 = (v1[i]-v[i])*U[i]
        c4 = (r1[i]-r[i])*U[i]*L
        c5 = (rac1[i]-rac[i])*(U[i]**2)
        c6 = ((v1[i]**3)-(v[i]**3))/U[i]
        c7 = ((rac1[i]**3)-(rac[i]**3))*(U[i]**2)
        c8 = (((v1[i]**2)*r1[i])-((v[i]**2)*r[i]))*L/U[i]
        c9 = (((v[i]**2)*rac1[i])-((v[i]**2)*rac[i]))
        c10 = ((v1[i]*(rac1[i]**2))-(v[i]*(rac[i]**2)))*U[i]
        c11 = ((rac1[i]*u1[i])-(rac[i]*u[i]))*U[i]
        c12 = (v1[i]*u1[i]) - (v[i]*u[i])
        c13 = (r1[i]*u1[i]*L) - (r[i]*u[i]*L)
        c14 = (rac1[i]*(u1[i]**2))- (rac[i]*(u[i]**2))
        
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

yawRC , P_yaw = Create_Yawcomponents(Delta_u_k1,Delta_u_k,Delta_v_k1,Delta_v_k,
                                          Delta_r_k1,Delta_r_k,
                                          Delta_rac_k1,Delta_rac_k, U,171.8)







from sklearn.svm import LinearSVR

"""
SVM construction for Surge
"""

linear_svm1 = LinearSVR(C=1e08,fit_intercept = False, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, tol = 0.000001,
                        verbose = 0).fit(np.array(surgeXC), np.array(nU))

coefLinear1 = linear_svm1.coef_

linear_svm1_ramp =  LinearSVR(C=1e04,fit_intercept = False, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, verbose = 0).fit(np.array(surgeXC), np.array(nU_ramp))

coefLinear1_ramp = linear_svm1_ramp.coef_

print("Train set accuracy of Surge on LinearSVR method: {:.2f}".format(linear_svm1.score(np.array(surgeXC), np.array(nU))))


"""
SVM construction for Sway
"""

linear_svm2 =  LinearSVR(C=1e08,fit_intercept = False, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, verbose = 0).fit(np.array(swayYC), np.array(nV))

coefLinear2 = linear_svm2.coef_

linear_svm2_ramp =  LinearSVR(C=1e04,fit_intercept = False, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, verbose = 0).fit(np.array(swayYC), np.array(nV_ramp))

coefLinear2_ramp = linear_svm2_ramp.coef_

print("Train set accuracy of Sway on LinearSVR method: {:.2f}".format(linear_svm2.score(swayYC,nV)))


"""
SVM construction for yaw
"""

linear_svm3 =  LinearSVR(C=1e08,fit_intercept = False, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, verbose = 0).fit(np.array(yawRC), np.array(nR))

coefLinear3 = linear_svm3.coef_

linear_svm3_ramp =  LinearSVR(C=1e04,fit_intercept = False, dual = True ,
                        epsilon = 1e-6, loss = 'squared_epsilon_insensitive',
                        max_iter = 10000, random_state = None, verbose = 0).fit(np.array(yawRC), np.array(nR_ramp))

coefLinear3_ramp = linear_svm3_ramp.coef_

print("Train set accuracy of Yaw on LinearSVR method: {:.2f}".format(linear_svm3.score(yawRC,nR)))

from tabulate import tabulate
X = coefLinear1
X_ramp = coefLinear1_ramp
Y = coefLinear2
R = coefLinear3
Y_ramp = coefLinear2_ramp
R_ramp = coefLinear3_ramp

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

"""
surgre solution and derivatives
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
    

surge = surge_derivatives(CC1,X)
surge_ramp = surge_derivatives(CC1, X_ramp)

sl_no = np.arange(1,11)
surge_hydrodynamic_derivatives = ['X`u', 'X`uu', 'X`uuu', 'X`vv', 'X`rr',
                                  'X`ðð', 'X`ððu', 'X`vr', 'X`vð', 'X`vðu']

Actual_Value = [-184,-110,-215,-899,18,
                    -95,-190,798,93,93]

table1 = zip(sl_no,surge_hydrodynamic_derivatives,Actual_Value,surge, surge_ramp)
headers1 = ['sl_no','Surge_hydrodynamic_derivatives','Original','case 1(C = 10^8)','case 2(C= 10^4, ramp added)']

surge_table = tabulate(table1, headers1, tablefmt="pretty")
print(surge_table)

"""

sway and yaw solution
"""

def solution_Matrix(h,m,S,L,IzG,Nav,Yav,Yar,Xg):
    M11= h*(IzG-Nar)/(S*L)
    M12= (-h)*((m*Xg)-Yar)/(S*L)
    M21= (-h)*((m*Xg)-Nav)/(S*(L**2))
    M22= h*(m-Yav)/(S*(L**2))
    return np.array([[M11,M12],[M21,M22]])
    


solMatrix = solution_Matrix(h,m,S,L,IzG,Nav,Yav,Yar,Xg) # equation 24 in JSR 2009 paper


def two_one_Matrix(t2,t3):
    List = []
    for i in range(15):
        temp = np.array([[t2[i]],[t3[i]]])
        List.append(temp)
    return List


c = two_one_Matrix(Y,R)
c_ramp = two_one_Matrix(Y_ramp,R_ramp)

def SNsolution(M,c): #sway and yaw moment solution
    List = []
    im = np.linalg.inv(M)
    for i in range(len(c)):
        temp = im.dot(c[i])
        List.append(temp)
    return List

Sway_Yaw_derivatives = SNsolution(solMatrix,c)
Sway_Yaw_derivatives_ramp = SNsolution(solMatrix,c_ramp)

def separation(M):
    sway_components = []
    yaw_components = []
    for i in M:
        sway_components.append(int(i[0][0]*(10**5)))
        yaw_components.append(int(i[1][0]*(10**5)))
    return sway_components,yaw_components

sway, yaw = separation(Sway_Yaw_derivatives)
sway_ramp, yaw_ramp = separation(Sway_Yaw_derivatives_ramp)

sl_no = np.arange(1,16)
sway_hydrodynamic_derivatives = ['Y`o','Y`ou','Y`ouu','Y`v','Y`r',
                                  'Y`ð','Y`vvv','Y`ððð','Y`vvr','Y`vvð',
                                  'Y`vðð','Y`ðu','Y`vu','Y`ru','Y`ðuu',
                                  ]

yaw_hydrodynamic_derivatives = ['N`ou','N`ouu','N`v','N`r',
                                'N`ð','N`vvv','N`ððð','N`vvr','N`vvð',
                                'N`vðð','N`ðu','N`vu','N`ru','N`ðuu',
                                ]

sway_original = [-4,-8,-4,-1160,-499,
                  278,-8078,-90,15356,1190,
                  -4,556,-1160,-499,278]

yaw_original = [3,6,3,-264,-166,
                  -139,1636,45,-5483,-489,
                  13,-278,-264,0,-139]

table1 = zip(sl_no,yaw_hydrodynamic_derivatives,yaw_original,yaw, yaw_ramp)
headers1 = ['sl_no','yaw_hydrodynamic_derivatives','Original','case 1(C = 10^8)','case 2(C= 10^4, ramp added)' ]

yaw_table = tabulate(table1, headers1, tablefmt="pretty")


table2 = zip(sl_no,sway_hydrodynamic_derivatives,sway_original,sway,sway_ramp )
headers2 = ['sl_no','sway_hydrodynamic_derivatives','Original','case 1(C = 10^8)','case 2(C= 10^4, ramp added)']

sway_table = tabulate(table2, headers2, tablefmt="pretty")
print(sway_table)
print(yaw_table)









# def EqL_next(var):# left hand side of the equation
#     var_next = []
#     for i in range(len(var) -1):
#         temp = var[i+1]
#         var_next.append(temp)
#     return var_next

# def term_LHS(x,xU):
#     op = []
#     for i in range(298):
#         temp = xU[i]-x[i]
#         op.append(temp)
#     return op

# _nU = EqL_next(u)
# nU = term_LHS(u,_nU) # we subtracting u(k+1) and u(k) values

# _nV = EqL_next(v)
# nV = term_LHS(v,_nV)

# _nR = EqL_next(r)
# nR = term_LHS(r,_nR) 








