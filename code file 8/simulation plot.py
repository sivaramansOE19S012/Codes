import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data  = pd.read_csv('1-10^16_surge.csv', header = 0)
data1  = pd.read_csv('1-10^16_sway.csv', header = 0)
data2  = pd.read_csv('1-10^16_yaw.csv', header = 0)

a1 = data.iloc[:,0][1:]
a2 = data.iloc[:,1][1:]
a3 = data.iloc[:,2][1:]
a4 = data.iloc[:,3][1:]
a5 = data.iloc[:,4][1:]
a6 = data.iloc[:,5][1:]
a7 = data.iloc[:,6][1:]
a8 = data.iloc[:,7][1:]
a9 = data.iloc[:,8][1:]
a10 = data.iloc[:,9][1:]

b1 = data1.iloc[:,0][1:]
b2 = data1.iloc[:,1][1:]
b3 = data1.iloc[:,2][1:]
b4 = data1.iloc[:,3][1:]
b5 = data1.iloc[:,4][1:]
b6 = data1.iloc[:,5][1:]
b7 = data1.iloc[:,6][1:]
b8 = data1.iloc[:,7][1:]
b9 = data1.iloc[:,8][1:]
b10 = data1.iloc[:,9][1:]
b11 = data1.iloc[:,10][1:]
b12 = data1.iloc[:,11][1:]
b13 = data1.iloc[:,12][1:]
b14 = data1.iloc[:,13][1:]
b15 = data1.iloc[:,14][1:]

c1 = data2.iloc[:,0][1:]
c2 = data2.iloc[:,1][1:]
c3 = data2.iloc[:,2][1:]
c4 = data2.iloc[:,3][1:]
c5 = data2.iloc[:,4][1:]
c6 = data2.iloc[:,5][1:]
c7 = data2.iloc[:,6][1:]
c8 = data2.iloc[:,7][1:]
c9 = data2.iloc[:,8][1:]
c10 = data2.iloc[:,9][1:]
c11 = data2.iloc[:,10][1:]
c12 = data2.iloc[:,11][1:]
c13 = data2.iloc[:,12][1:]
c14 = data2.iloc[:,13][1:]
c15 = data2.iloc[:,14][1:]





surge_hydrodynamic_derivatives = ['X`u', 'X`uu', 'X`uuu', 'X`vv', 'X`rr',
                                  'X`ðð', 'X`ððu', 'X`vr', 'X`vð', 'X`vðu']
Actual_Value = [-184,-110,-215,-899,18,-95,-190,798,93,93]




def HDV_plot():
    plt.figure(figsize=(15,12))
    plt.subplot(331)
    plt.plot(a1)
    plt.axhline(y=-184,color = 'r')
    plt.title('X`u')
    
    plt.subplot(332)
    plt.plot(a2)
    plt.axhline(y=-110,color = 'r')
    plt.title('X`uu')
    
    plt.subplot(333)
    plt.plot(a3)
    plt.axhline(y=-215,color = 'r')
    plt.title('X`uuu')
    
    plt.subplot(334)
    plt.plot(a4)
    plt.axhline(y=-899,color = 'r')
    plt.title('X`vv')
    
    plt.subplot(335)
    plt.plot(a5)
    plt.axhline(y=18, color = 'r')
    plt.title('X`rr')
    
    plt.subplot(336)
    plt.plot(a6)
    plt.axhline(y=-95, color = 'r')
    plt.title('X`ðð')
    
    plt.subplot(337)
    plt.plot(a7)
    plt.axhline(y=-190, color = 'r')
    plt.title('X`ððu')
    
    plt.subplot(338)
    plt.plot(a8)
    plt.axhline(y= 798, color = 'r')
    plt.title('X`vr')
    
    plt.subplot(339)
    plt.plot(a9)
    plt.plot(a10)
    plt.axhline(y= 93, color = 'r')
    plt.title('X`vð and X`vðu')
    plt.suptitle('Hydrodynamic Values Comparision')
    plt.annotate("(Note : Red values are original values)",(1,50))
    plt.savefig('surge.jpg')
    plt.show()
    


HDV_plot()

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

def HDV_plot_Y1():
    plt.figure(figsize=(15,12))
    plt.subplot(331)
    plt.plot(b1)
    plt.axhline(y=-4,color = 'r')
    plt.title('Y`0')
    
    plt.subplot(332)
    plt.plot(b2)
    plt.axhline(y=-8,color = 'r')
    plt.title('Y`ou')
    
    plt.subplot(333)
    plt.plot(b3)
    plt.axhline(y=-4,color = 'r')
    plt.title('Y`ouu')
    
    plt.subplot(334)
    plt.plot(b4)
    plt.axhline(y=-1160,color = 'r')
    plt.title('Y`v')
    
    plt.subplot(335)
    plt.plot(b5)
    plt.axhline(y= -499, color = 'r')
    plt.title('Y`r')
    
    plt.subplot(336)
    plt.plot(b6)
    plt.axhline(y= 278, color = 'r')
    plt.title('Y`ð')
    
    plt.subplot(337)
    plt.plot(b7)
    plt.axhline(y=-8078, color = 'r')
    plt.title('Y`vvv')
    
    plt.subplot(338)
    plt.plot(b8)
    plt.axhline(y= -90, color = 'r')
    plt.title('Y`ððð')
    
    plt.subplot(339)
    plt.plot(b9)
    plt.axhline(y= 15356, color = 'r')
    plt.title('Y`vvr')
    
    plt.suptitle('Sway Hydrodynamic Values Comparision')
    plt.annotate("(Note : Red values are original values)",(1,5000))
    plt.savefig('sway1.jpg')
    plt.show()

HDV_plot_Y1()


def HDV_plot_Y2():
    plt.figure(figsize=(15,12))
    plt.subplot(321)
    plt.plot(b10)
    plt.axhline(y= 1190, color = 'r')
    plt.title('Y`vvð')
    
    plt.subplot(322)
    plt.plot(b11)
    plt.axhline(y= -4, color = 'r')
    plt.title('Y`vðð')
    
    plt.subplot(323)
    plt.plot(b12)
    plt.axhline(y= 556, color = 'r')
    plt.title('Y`ðu')
    
    plt.subplot(324)
    plt.plot(b13)
    plt.axhline(y= -1160, color = 'r')
    plt.title('Y`vu')
    
    plt.subplot(325)
    plt.plot(b14)
    plt.axhline(y= -499, color = 'r')
    plt.title('Y`ru')
    
    plt.subplot(326)
    plt.plot(b15)
    plt.axhline(y= 278, color = 'r')
    plt.title('Y`ðuu')
    plt.savefig("sway2.jpg")
    plt.show()
    
    
HDV_plot_Y2()


def HDV_plot_N1():
    plt.figure(figsize=(15,12))
    plt.subplot(331)
    plt.plot(c1)
    plt.axhline(y= 3,color = 'r')
    plt.title('N`0')
    
    plt.subplot(332)
    plt.plot(c2)
    plt.axhline(y= 6,color = 'r')
    plt.title('N`ou')
    
    plt.subplot(333)
    plt.plot(c3)
    plt.axhline(y= 3,color = 'r')
    plt.title('N`ouu')
    
    plt.subplot(334)
    plt.plot(c4)
    plt.axhline(y=-264,color = 'r')
    plt.title('N`v')
    
    plt.subplot(335)
    plt.plot(c5)
    plt.axhline(y= -166, color = 'r')
    plt.title('N`r')
    
    plt.subplot(336)
    plt.plot(c6)
    plt.axhline(y= -139, color = 'r')
    plt.title('N`ð')
    
    plt.subplot(337)
    plt.plot(c7)
    plt.axhline(y= 1636, color = 'r')
    plt.title('N`vvv')
    
    plt.subplot(338)
    plt.plot(c8)
    plt.axhline(y= 45, color = 'r')
    plt.title('N`ððð')
    
    plt.subplot(339)
    plt.plot(c9)
    plt.axhline(y= -5483, color = 'r')
    plt.title('N`vvr')
    
    plt.suptitle('Yaw Hydrodynamic Values Comparision')
    plt.annotate("(Note : Red values are original values)",(1,5000))
    plt.savefig('yaw1.jpg')
    plt.show()

HDV_plot_N1()

def HDV_plot_N2():
    plt.figure(figsize=(15,12))
    plt.subplot(321)
    plt.plot(c10)
    plt.axhline(y= -489, color = 'r')
    plt.title('N`vvð')
    
    plt.subplot(322)
    plt.plot(c11)
    plt.axhline(y= 13, color = 'r')
    plt.title('N`vðð')
    
    plt.subplot(323)
    plt.plot(c12)
    plt.axhline(y= -278, color = 'r')
    plt.title('N`ðu')
    
    plt.subplot(324)
    plt.plot(c13)
    plt.axhline(y= -264, color = 'r')
    plt.title('N`vu')
    
    plt.subplot(325)
    plt.plot(c14)
    plt.axhline(y= -166, color = 'r')
    plt.title('N`ru')
    
    plt.subplot(326)
    plt.plot(c15)
    plt.axhline(y= -139, color = 'r')
    plt.title('N`ðuu')
    plt.savefig("yaw2.jpg")
    plt.show()

HDV_plot_N2()














