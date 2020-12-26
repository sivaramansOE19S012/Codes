import numpy as np
import matplotlib.pyplot as plt
import zig_zag

"""
Generates the zigzag maneuver for given different type of ship

Author:   Sivaraman Sivaraj, Suresh Rajendran
Date:     01 December 2020

Reference Author :  Thor I. Fossen
Date:               23th July 2001

"""

Req_simulation_time = 2000           #Total simulation time (sec)
t_rudderexecute = 10     #time for rudder is executed at particular angle(sec) (bump in the graph,becuase of this)
h = 0.1                  #sampling time (sec)

print("Zig Zag test for given ship model is about to start...")

xt = np.zeros((7,1)) #x  = [ u v r x y psi delta ]' (initial values)
ui = 0; 

t,u,v,r,x,y,psi,U,delta,DATA = zig_zag.activate('mariner',xt,ui,Req_simulation_time,t_rudderexecute,h,[20,20])


t_a = np.array(t)
u_a = np.array(u)
v_a = np.array(v)
r_a = np.array(r)
x_a = np.array(x)
y_a = np.array(y)
psi_a = np.array(psi)
U_a = np.array(U)
delta_a = np.array(delta)

t = t.tolist()

v = v.tolist()
r = r.tolist()
x = x.tolist()
y = y.tolist()
psi = psi.tolist()
U = U.tolist()
delta = delta.tolist()

zig_zag.plot_components_psi_delta_U(t,psi,delta,U)
zig_zag.plot_components_xy(x,y)


def Plot_simulated_Data1():
    plt.figure(figsize=(15,12))
        
    plt.subplot(311)
    plt.plot(t[:len(t)-2],7.7175+u[:len(u)-2],'k',label= "Surge Speed")
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
    plt.show()
   
    
def Plot_simulated_Data2():
    plt.figure(figsize=(15,8))
    
    plt.subplot(211)
    plt.grid(b=0.1)
    plt.plot(t[:len(t)-2],delta[:len(t)-2],'-r')
    plt.plot(t[:len(t)-2],psi[:len(t)-2],'-b')
    
    plt.subplot(212)
    plt.plot(t[:len(t)-2],U[:len(U)-2],'m',label= "Total Speed")
    plt.grid()
    plt.legend(loc="best")
    plt.show()

Plot_simulated_Data1()
# Plot_simulated_Data2()
    
du = u_a[2:]-u_a[1:len(u)-1]
dv = v_a[2:]-v_a[1:len(v)-1]
dr = r_a[2:]-r_a[1:len(r)-1]
output1=[du,dv,dr]

u = u_a + 7.7175
u = u.tolist()
output_zig_zag = np.asarray([t,u,v,r,psi,U,delta])
np.savetxt("2000_sec_20_15_10_5.csv", output_zig_zag.T, delimiter=",")






