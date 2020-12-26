import matplotlib.pyplot as plt
import numpy as np

def graph_uvr(__u,__v,__r,time_interval):
    plt.figure(figsize=(15,12))
    plt.subplot(311)
    x212 = np.arange(0,len(__u))
    plt.plot(x212,__u,'g')
    plt.xlabel('time in '+str(time_interval)+' seconds increment')
    plt.ylabel('Surge Speed')
    plt.title("Turning Circle Surge,Sway,Yaw Plot")
    #plt.axvline(x=100,color = 'r')
      
    plt.subplot(312)
    x211 = np.arange(0,len(__v))
    plt.plot(x211,__v,'m')
    plt.xlabel('time in '+str(time_interval)+' seconds increment')
    plt.ylabel('Sway Speed')
    #plt.axvline(x=100,color = 'r')
    
    plt.subplot(313)
    x313 = np.arange(0,len(__r))
    plt.plot(x313,__r,'b')
    plt.xlabel('time in '+str(time_interval)+' seconds increment')
    plt.ylabel('yaw rate')
    #plt.axvline(x=100,color = 'r')    
    # plt.savefig('plot_1.jpg')
    plt.show()
    
    
def graph_Uy(yaw_angle,__U,__rac_, time_interval):
    plt.figure(figsize=(15,12))
    plt.subplot(211)
    x212 = np.arange(0,len(yaw_angle))
    plt.plot(x212,yaw_angle,'g')
    plt.plot(x212,__rac_,'m')
    plt.xlabel('time in '+str(time_interval)+' seconds increment')
    plt.ylabel('Yaw Angle')
    plt.title('Heading angle (Ψ) vs Rudder Angle(𝛿)')
    #plt.axvline(x=100,color = 'r')
      
       
    plt.subplot(212)
    x313 = np.arange(0,len(__U))
    plt.plot(x313,__U,'b')
    plt.xlabel('time in '+str(time_interval)+' seconds increment')
    plt.ylabel('Total Speed')
    #plt.axvline(x=100,color = 'r')
    # plt.savefig('plot_2.jpg')    
    plt.show()
    
    
def graph_delta_all(u,v,r,rac):
    plt.figure(figsize=(15,7.5))
    plt.plot(u,'g',label = 'surge')
    plt.plot(v,'m', label='sway')
    plt.plot(r,'b', label='yaw')
    plt.plot(rac,'grey',label = 'rudder angle change')
    plt.xlabel('time in 2 seconds increment')
    plt.ylabel('delta value Δ')
    plt.title('Δu, Δv, Δr,Δrac  plot')
    plt.axhline(y = 0, color ="red", linestyle ="--")
    plt.legend(loc = 'best')
    plt.show()
    

def plot_surge_components_jsr2009(P):
        
    plt.figure(figsize=(15,12))
    plt.title("Surge Components plot")
    plt.subplot(311)
    plt.plot(P[0],label = 'c1')
    plt.plot(P[1],label = 'c2')
    plt.plot(P[2],label = 'c3')
    plt.plot(P[3],label = 'c4')
    plt.plot(P[5],'y',label = 'c6')
    plt.plot(P[6],label = 'c7')
    plt.plot(P[8],'pink',label = 'c9')
    plt.plot(P[9],label = 'c10')
    plt.xlabel('time in 2 seconds increment')
    plt.ylabel('Numerical Value of components except c5, c8 & cb')
    plt.legend(loc = 'best')
      
    plt.subplot(312)
    plt.plot(P[4],'g',label = 'c5')
    plt.xlabel('time in 2 seconds increment')
    plt.ylabel('Numerical Value c5')
    plt.legend(loc = 'best')
    
    plt.subplot(313)
    plt.plot(P[7],label = 'c8')
    plt.plot(P[10],label = 'c bias')
    plt.xlabel('time in 2 seconds increment')
    plt.ylabel('Numerical Value of c8 & c bias')
    plt.legend(loc = 'best')
    

def plot_YN_components1_jsr2009(P):
        
    plt.figure(figsize=(15,12))
   
    plt.subplot(311)
    plt.plot(P[2],label = 'c2')
    plt.plot(P[6],label = 'c6')
    plt.plot(P[9],label = 'c9')
    plt.plot(P[10],label = 'c10')
    plt.plot(P[12],label = 'c12')
    plt.plot(P[14],label = 'c14')
    plt.title("Sway Components Plot")
    plt.xlabel('time in 2 seconds increment')
    plt.ylabel('Numerical Values')
    plt.legend(loc = 'best')
      
    plt.subplot(312)
    plt.plot(P[1],label = 'c1')
    plt.plot(P[3],label = 'c3')
    plt.plot(P[7],label = 'c7')
    plt.plot(P[11],label = 'c11')
    plt.xlabel('time in 2 seconds increment')
    plt.ylabel('Numerical Values')
    plt.legend(loc = 'best')
       
    plt.subplot(313)
    plt.plot(P[5],label = 'c5')
    plt.plot(P[8],'m',label = 'c8')
    plt.xlabel('time in 2 seconds increment')
    plt.ylabel('Numerical Values ')
    plt.legend(loc = 'best')

def plot_YN_components2_jsr2009(P):
        
    plt.figure(figsize=(15,8))
   
    plt.subplot(211)
    plt.plot(P[0],label = 'c bias')
    plt.plot(P[13],label = 'c14')
    plt.xlabel('time in 2 seconds increment')
    plt.ylabel('Numerical Values ')
    plt.legend(loc = 'best')
    
    plt.subplot(212)
    plt.plot(P[4],'grey',label = 'c5')
    plt.xlabel('time in 2 seconds increment')
    plt.ylabel('Numerical Values ')
    plt.legend(loc = 'best')

















    
    