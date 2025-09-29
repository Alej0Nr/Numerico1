import numpy as np
import matplotlib.pyplot as plt
from integration import *
from metodos import *

def ej2():
    ## b)
    K = 0.016
    Ca0 = 42
    Cb0 = 28
    Cc0 = 4
    # K = (Cc0 + x)/((Ca0 - 2*x)**2*(Cb0 - x))
    # K*((Ca0 - 2*x)**2*(Cb0 - x)) - (Cc0 + x) = 0
    f = lambda x: -K + (Cc0 + x)/((Ca0 - 2*x)**2*(Cb0 - x))
    f2 = lambda x: K*((Ca0 - 2*x)**2*(Cb0 - x)) - (Cc0 + x)
    p = secante(f2,15.8,16.2,tol=1e-8,maxiter=1000)
    print(p)
    # x = np.linspace(0,100,1000)
    # plt.plot(x,f2(x))
    # plt.grid()
    # plt.show()


def ej3():
    t =[1,2]
    z = [-0.035,0.94]
    k = z[1]-z[0]-np.log(2)
    fa = lambda a: a*(np.arctan(2-a)-np.arctan(1-a))-k
    tiempos = np.linspace(-10,10,1000)
    plt.plot(tiempos,fa(tiempos))
    plt.grid()
    # plt.show()
    Dfa = lambda a: (np.arctan(2-a)-np.arctan(1-a))+a*(1/(1+(1-a)**2)-1/(1+(2-a)**2))
    a1 = newton(fa,Dfa,0,1e-8)
    a2 = newton(fa,Dfa,5,1e-8)
    plt.scatter([a1[0],a2[0]],[fa(a1[0]),fa(a2[0])])
    # plt.show()
    b1, b2 = [np.exp(z[0]-a1[0]*np.arctan(1-a1[0])),np.exp(z[0]-a2[0]*np.arctan(1-a2[0]))]
    Z = lambda x, a, b: a*np.arctan(x-a)+np.log(b*x)
    print(Z(1,a1[0],b1))
    print(Z(2,a1[0],b1))
    print(Z(1,a2[0],b2))
    print(Z(2,a2[0],b2))
    print(a1,b1,a2,b2) # todas son positivas entonces deberian servir todas

def ej4():
    L=12
    x = np.array([0,200,400,600,800,1000,1200])
    rho = np.array([4,3.95,3.89,3.8,3.6,3.41,3.3])
    Ac= np.array([100,103,106,110,120,133,149.6])

    Int = rho*Ac
    integral = simpsoncomp(x,Int)
    print(integral)

ej4()
