import numpy as np
import matplotlib.pylab as plt
import mariner

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

def activate(ship,x,ui,Req_simulation_time,t_rudderexecute,h,maneuver=[39,39]):
    """
    It performs the zig-zag maneuver
    
    Input Variables
    ----------
    ship    : ship model. Compatible with the models under .../gnc/VesselModels/
    x       : initial state vector for ship model
    ui      : [delta,:] where delta=0 and the other values are non-zero if any
    t_final : final simulation time

    t_rudderexecute : rudder's time control input is activated
    
    h       : sampling time
    
    maneuver : [rudder angle, heading angle]. Default 20-20 deg that is: maneuver = [20, 20] 
               rudder is changed to maneuver(1) when heading angle is larger than maneuver(2)

    Returns
    -------
    t               = time vector
    u,v,r,x,y,psi,U,delta_c = time series

    """
    
    N = round(Req_simulation_time/h)               #number of samples
    xout = np.zeros((N+1,9))
    
    print("Simulating the Maneuver data.....")
    
    u_ship=ui
    
    for i in range(N):
        time = (i-1)*h
        
        t_index = np.floor(40-(time/100))
        # if time > 500 and time < 1000:
        maneuver = [t_index,t_index]
        # elif time>1000 and time<1500:
        #     maneuver = [10,10]
        # elif  time>1500 :
        #     maneuver = [5,5]
            
        
        psi = x[5]*180/np.pi
        r   = x[2]
        
        if round(time) == t_rudderexecute:
            u_ship = maneuver[0]*np.pi/180
        
        if round(time) > t_rudderexecute:
            if psi >= maneuver[1] and r > 0:
                u_ship = -(maneuver[0]*np.pi)/180
            elif psi <= -maneuver[1] and r < 0:
                u_ship = (maneuver[0]*np.pi)/180
                
        xdot,U =  mariner.activate(x,u_ship)#feval(ship,x,u_ship)       #ship model
                
        x = euler_integration(xdot,x,h) #Euler integration
        ###########
        # xdot_q = np.squeeze(xdot).tolist()
        temp = list()
        temp.append(time)
        for j in range(6):
            temp.append(x[j])
        temp.append(U[0])
        temp.append(u_ship)
        xout[i,:] = temp     #[time,x[1:6].T,U,u_ship[0]]
        # print(temp)
        ############
        # print(i)
    # maneuver
    # time-series
    t     = xout[:,0]
    u     = xout[:,1] 
    v     = xout[:,2]        
    r     = xout[:,3]*180/np.pi
    x     = xout[:,4]
    y     = xout[:,5]
    psi   = xout[:,6]*180/np.pi
    U     = xout[:,7]
    delta_c = xout[:,8]*180/np.pi
    
    return t,u,v,r,x,y,psi,U,delta_c,xout



def plot_components_xy(x,y):
    plt.figure(figsize=(15,12))
    plt.grid()
    plt.plot(x[:len(x)-2],y[:len(y)-2])
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Zig Zag Test - Ship Maneuvering")
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
    
    

# x =np.array([0.8,0.5,0.3,100,100,40,30])
# x_dot =  np.array([-54.140214199177095, -671.6694249610749, 39.804570543970456, -6.053201520362815, 6.013032313556724, 0.30000000000000004, -0.08726646259971647])
# sds = euler_integration(x_dot,x,0.1)
# print(sds)

    
    
