import numpy as np
import matplotlib.pylab as plt
import mariner, straight_mariner

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
    It performs the zig-zag maneuver
    
    Input Variables
    ----------
    ship    : ship model. Compatible with the models under .../gnc/VesselModels/
    x       : initial state vector for ship model
    ui      : given rudder angle
    t_final : final simulation time

    t_rudderexecute : rudder's time control input is activated
    
    h       : sampling time
    
    maneuver : [rudder angle, heading angle]. Default 20-20 deg that is: maneuver = [20, 20] 
               rudder is changed to maneuver(1) when heading angle is larger than maneuver(2)

    Returns
    -------
    t               = time vector
    u,v,r,x,y,psi,U,delta_c = time series
    X_out = as matrix
    D = [advance,transfer,tactical]

    """
    N = round(Req_simulation_time/h)               #Number of samples
    xout = np.zeros((N+1,9))                       #Empty Allocation
    T_var1, T_var2 = 1,1                           #Terminate Variable
    print("Simulating the Maneuver data.....")
    
    u_ship=ui
    
    for i in range(N):
        
        time = (i-1)*h
        if round(float(x[5])*180/np.pi, 3)>= 90 and T_var1 == 1:
            transfer = x[4]    #transfer at 90 deg
            advance = x[3]     #advance at 90 deg
            T_var1 = 0
            
        if round(float(x[5])*180/np.pi,3) >= 180 and T_var2 == 1:
            tactical=x[4]  #% tactical diameter at 180 deg
            T_var2 = 0
           
        u_ship = ui;
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
    D = [advance,transfer,tactical]
    
    Nrudder = round(t_rudderexecute/h)
    print('Rudder execute (x-coordinate)          : ',abs(x[Nrudder]))
    print('Steady turning radius                  : ',U[Nrudder]/abs(r[Nrudder]*np.pi/180))
    print('Maximum transfer                       : ',abs(max(abs(y))))
    print('Maximum advance                        : ',abs(max(abs(x))-x[Nrudder]))   
    print('Transfer at 90 (deg) heading           : ',abs(transfer))  
    print('Advance at 90 (deg) heading            : ',abs(advance-x[Nrudder]))
    
    return t,u,v,r,x,y,psi,U,delta_c,D
            

def plot_components_xy(x,y):
    plt.figure(figsize=(15,12))
    plt.grid()
    plt.plot(x[:len(x)-2],y[:len(y)-2])
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Turning Circle Test - Ship Maneuvering")
    plt.show()
    
    
def plot_components_psi_delta_U(t,psi,delta_c,U):
    plt.figure(figsize=(15,8))
    
    plt.subplot(211)
    plt.plot(t[:len(psi)-2],psi[:len(psi)-2],"g",label="ψ")
    plt.plot(t[:len(U)-2],delta_c[:len(U)-2],'r',label = "$ \delta_c $'")
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("Time in seconds")
    plt.title("Rudder Angle $ \delta $ & Yaw Angle ψ (deg)")
    
    
    plt.subplot(212)
    plt.plot(t[:len(U)-2],U[:len(U)-2], label ="Total Speed (U)")
    plt.grid()
    plt.legend(loc = "best")
    plt.xlabel("Time in seconds")
    plt.show()
    

         