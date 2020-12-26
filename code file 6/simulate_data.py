import numpy as np
import matplotlib.pyplot as plt
import spiral

"""
Generates the Spiral maneuver for given different type of ship

Author:   Sivaraman Sivaraj, Suresh Rajendran
Date:     01 December 2020

Reference Author :  Thor I. Fossen
Date:               23th July 2001

"""

Req_simulation_time = 5300 #Total simulation time (sec)
t_rudderexecute = 10       #time for rudder is executed at particular angle(sec) 
h = 0.1                     #sampling time (sec)

print("Spiral test for given ship model is about to start...")

xt = np.zeros((7,1)) #x  = [ u v r x y psi delta ]' (initial values)
ui = 5*np.pi/180 

t,u,v,r,x,y,psi,U,delta = spiral.activate('mariner',xt,ui,Req_simulation_time,t_rudderexecute,h)

psi11 = psi%90
psi1 = psi11-45

t = t.tolist()
u1 = u + 7.7175
u = u1.tolist()
v = v.tolist()
r = r.tolist()
x = x.tolist()
y = y.tolist()
psi = psi1.tolist()
U = U.tolist()
delta = delta.tolist()


spiral.plot_components_xy(x,y)
spiral.plot_components_psi_delta_U(t,psi,delta,U)

def Plot_simulated_Data1():
    plt.figure(figsize=(15,12))
        
    plt.subplot(311)
    plt.plot(t[:len(t)-2],(np.array(u[:len(u)-2])).tolist(),'k',label= "Surge Speed")
    plt.grid(b=0.1)
    plt.legend(loc="best")
    plt.title("Surge-Sway-Yaw Components")
    
    plt.subplot(312)
    plt.plot(t[:len(t)-2],v[:len(v)-2],'c',label= "Sway Speed")
    plt.grid(b=0.1)
    plt.legend(loc="best")
    
    plt.subplot(313)
    plt.plot(t[:len(t)-2],r[:len(r)-2],'g',label= "Yaw Rate")
    plt.grid(b=0.1)
    plt.legend(loc="best")
    plt.xlabel("Time in Seconds")
    plt.show()
   
    

Plot_simulated_Data1()

output_spiral = np.asarray([t,u,v,r,psi,U,delta])
np.savetxt("spira_test_5300.csv", output_spiral.T, delimiter=",")






