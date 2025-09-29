import numpy as np
import matplotlib.pyplot as plt
from integration import *
from metodos import *


def ej1():
    x = np.array([0,.25,.5,.75,1])
    y = np.array([2.01,1.47,1.3,1.54,2.1])
    # y = ae^{-x} + bx^2 +c
    f1 = lambda t: np.exp(-t)
    f2 = lambda t: t**2
    M = np.array([f1(x),f2(x),np.ones_like(x)]).T
    A = M.T@M
    y2 = M.T @ y
    a,b,c = np.linalg.solve(A,y2)
    print(f"{a = },{b = },{c = }")
    plt.scatter(x,y)
    f = lambda t: a*np.exp(-t) + b*t**2 + c
    # tiempos = np.linspace(0,1,1000)
    # plt.plot(tiempos,f(tiempos))
    # plt.grid()
    print(f"Error en x = 0.5 es {np.abs(1.3-f(0.5))}")
    # plt.show()

def ej2():
    Fint = lambda t: np.sqrt(1+(np.cos(t))**2)
    a, b = 0, np.pi/4
    print(intNCcompuesta_h(Fint,a,b,1e-8,3))



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

def ej4():
    g = lambda t: np.arctan(t-3)+np.sqrt(t) + 1.3
    Dg = lambda t: 1/(1+(t-3)**2) + 1/(2*np.sqrt(t))
    

    ### """calcular puntos fijos en t\in [1,5]"""
    tiempos = np.linspace(1,5,1000)
    plt.plot(tiempos,g(tiempos))
    plt.plot(tiempos,tiempos)
    plt.grid()
    plt.plot(tiempos,Dg(tiempos))
    plt.plot(tiempos,[1/2]*1000)
    # """posee 3 puntos fijos"""
    g2 = lambda t: g(t) - t
    fijos = [secante(g2,1.6549,1.6550,1e-8)[0],secante(g2,2.8884,1.8888,1e-8)[0],secante(g2,4.27025,4.27100,1e-8)[0]]
    plt.scatter(fijos,[g(fijos) for fijos in fijos])
    plt.show()
