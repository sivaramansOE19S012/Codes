"""
Generates the Spiral maneuver for given different type of ship

Author:   Sivaraman Sivaraj, Suresh Rajendran
Date:     08 December 2020

Reference Author :  Thor I. Fossen
Date:               23th July 2001

"""

import numpy as np
import matplotlib.pylab as plt
import mariner#, straight_mariner

def euler_integration(xdot,x,h):
    """
    Integrate a system of ordinary differential equations using 
    Euler's 2nd-order method.

    x_next = euler_integration(xdot,x,h)
    
    Parameters
    ----------
    xdot : dx/dt(k) = f(x(k))
    x : x(k)
    h     - step size

    Returns
    -------
    x_next - x(k+1)

    """
    a = np.array(x)
    b = np.array(xdot)
    return a + (h*b)


def activate(ship,x,ui,Req_simulation_time,t_rudderexecute,h,maneuver="ccw"):
    """
    It performs the Sprial maneuvere of ship
    
    Input Variables
    ----------
    ship    : ship model. Compatible with the models under .../gnc/VesselModels/
    x       : initial state vector for ship model
    ui      : given rudder angle
    t_final : final simulation time

    t_rudderexecute : rudder's time control input is activated
    
    h       : sampling time
    
    maneuver : it reduces by one degree for an desired time interval

    Returns
    -------
    t               = time vector
    u,v,r,x,y,psi,U,delta_c = time series
    X_out = as matrix
    D = [advance,transfer,tactical]

    """
    N = round(Req_simulation_time/h)               #Number of samples
    xout = np.zeros((N+1,9))                       #Empty Allocation
    
    print("Simulating the Maneuver data.....")
    
    u_ship=ui
    
    for i in range(N):
        
        time = (i-1)*h
        
        
        u_ship = ui
        if time > 500 and time<=900:
            u_ship = (6*np.pi)/180
        if time > 900 and time<=1300:
            u_ship = (7*np.pi)/180
        if time > 1300 and time<=1600:
            u_ship = (8*np.pi)/180
        if time > 1600 and time<=1850:
            u_ship = (9*np.pi)/180
        if time > 1850 and time<=2050:
            u_ship = (10*np.pi)/180
        if time > 2050 and time<=2250:
            u_ship = (11*np.pi)/180
        if time > 2250 and time<=2450:
            u_ship = (12*np.pi)/180
        if time > 2450 and time<=2650:
            u_ship = (13*np.pi)/180
        if time > 2650 and time<=2850:
            u_ship = (14*np.pi)/180
        if time > 2850 and time<=3050:
            u_ship = (15*np.pi)/180
        if time > 3050 and time<=3200:
            u_ship = (16*np.pi)/180
        if time > 3200 and time<=3350:
            u_ship = (17*np.pi)/180
        if time > 3350 and time<=3500:
            u_ship = (18*np.pi)/180
        if time > 3500 and time<=3650:
            u_ship = (19*np.pi)/180
        if time > 3650 and time<=3800:
            u_ship = (20*np.pi)/180
        if time > 3800 and time<=3900:
            u_ship = (21*np.pi)/180
        if time > 3900 and time<=4000:
            u_ship = (22*np.pi)/180
        if time > 4000 and time<=4100:
            u_ship = (23*np.pi)/180
        if time > 4100 and time<=4200:
            u_ship = (24*np.pi)/180
        if time > 4200 and time<=4300:
            u_ship = (25*np.pi)/180
        if time > 4300 and time<=4400:
            u_ship = (26*np.pi)/180
        if time > 4400 and time<=4500:
            u_ship = (27*np.pi)/180
        if time > 4500 and time<=4600:
            u_ship = (28*np.pi)/180
        if time > 4600 and time<=4700:
            u_ship = (29*np.pi)/180
        if time > 4700 and time<=4800:
            u_ship = (30*np.pi)/180
        if time > 4800 and time<=4900:
            u_ship = (31*np.pi)/180
        if time > 4900 and time<=5000:
            u_ship = (32*np.pi)/180
        if time > 5000 and time<=5100:
            u_ship = (33*np.pi)/180
        if time > 5100 and time<=5200:
            u_ship = (34*np.pi)/180
        if time > 5200 and time<=5300:
            u_ship = (35*np.pi)/180
            
        
        if round(time) < t_rudderexecute: 
           u_ship = 0
           
        xdot,U =  mariner.activate(x,u_ship)
        x = euler_integration(xdot,x,h)
        
        ###########
        temp = list()
        temp.append(time)
        for j in range(6):
            temp.append(x[j])
        temp.append(U[0])
        temp.append(u_ship)
        xout[i,:] = temp     
        # print(temp)
        ############
        # print(i)
        
    #Declassification
    t     = xout[:,0]
    u     = xout[:,1] 
    v     = xout[:,2]        
    r     = xout[:,3]*180/np.pi
    x     = xout[:,4]
    y     = xout[:,5]
    psi   = xout[:,6]*180/np.pi
    U     = xout[:,7]
    delta_c = xout[:,8]*180/np.pi
      
       
    return t,u,v,r,x,y,psi,U,delta_c
            

def plot_components_xy(x,y):
    plt.figure(figsize=(15,12))
    plt.grid()
    plt.plot(x[:len(x)-2],y[:len(y)-2],'g',label="Trajectory")
    plt.scatter(250,1000,marker="*",color='r')
    plt.annotate("Loci",xy = (250,1000),xytext=(400,800),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.scatter(0,0,marker='*',color='r')
    plt.annotate("Starting Point",xy = (0,0),xytext=(-250,0),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.scatter(40,490,marker='*',color='r')
    plt.annotate("Terminal Point",xy = (40,490),xytext=(250,600),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Spiral Test - Ship Maneuvering 5-35 degress")
    plt.legend(loc="best")
    plt.show()
    
def plot_components_psi_delta_U(t,psi,delta_c,U):
    plt.figure(figsize=(15,8))
    
    plt.subplot(211)
    plt.plot(t[:len(psi)-2],psi[:len(psi)-2],"g",label="ψ")
    plt.plot(t[:len(U)-2],delta_c[:len(U)-2],'r',label = "$ \delta_c $'")
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("Time in seconds")
    plt.title("Rudder Angle $ \delta $ & Heading Angle ψ (deg)")
    
    
    plt.subplot(212)
    plt.plot(t[:len(U)-2],U[:len(U)-2], label ="Total Speed (U)")
    plt.grid()
    plt.legend(loc = "best")
    plt.xlabel("Time in seconds")
    plt.show()
    
    

         