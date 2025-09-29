import numpy as np
import matplotlib.pyplot as plt
from integration import *
from metodos import *

def ej3():
    f = lambda x,y: x**2*y**2+x*y-1.5
    g = lambda x,y: x**2+(y**2)/2-2.55
    x = np.linspace(-3,3,1000)
    X,Y = np.meshgrid(x,x)
    plt.contour(X,Y,f(X,Y),0)
    plt.contour(X,Y,g(X,Y),0)
    
    F = lambda P: np.array([f(P[0],P[1]),g(P[0],P[1])]).T
    DF = lambda P: np.array([[2*P[0]*P[1]**2+P[1],2*P[0]**2*P[1]+P[0]],[2*P[0],P[1]]])
    p0_1 = np.array([-1.5524,-0.53])
    p0_2 = np.array([-0.375,-2.1950])
    p0_3 = np.array([1.15,-1.65])
    p0_4 = np.array([1.5525,0.53])
    p0_5 = np.array([0.3749,2.1950])
    p0_6 = np.array([-1.15,1.55])
    print(newtonvec(F,DF,p0_1))
    print(newtonvec(F,DF,p0_2))
    print(newtonvec(F,DF,p0_3))
    print(newtonvec(F,DF,p0_4))
    print(newtonvec(F,DF,p0_5))
    print(newtonvec(F,DF,p0_6))

    plt.show()

ej3()