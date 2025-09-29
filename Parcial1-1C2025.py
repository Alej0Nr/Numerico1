import numpy as np
import matplotlib.pyplot as plt
from integration import *
from metodos import *

def ej1():
    ti = np.array([0,2,7,8,10])
    yi = np.array([1.65,2.44,4.38,5.38,6.92])
    # f = Ce^{t/r}
    lny = np.log(yi)
    parameters = np.polyfit(ti,lny,1)
    r = 1/parameters[0]
    C = np.exp(parameters[1])
    plt.scatter(ti,yi)
    tiempos = np.linspace(0,10,1000)
    f = lambda t: C*np.exp((1/r) * t)
    plt.plot(tiempos, f(tiempos))
    plt.grid()
    plt.show()
    print(C,"* e^{",1/r,"t}")
    Error = max(yi - f(ti))
    print(f"Error maximo = {Error}")
    parameters_i = np.polyfit(ti,yi,len(ti)-1)
    print(parameters_i)
    # int_0^10 f(t) dt
    simp = simpsoncomp(ti,yi)
    print(f"c_i simpson = {simp}")
    polyint = lambda t: np.polyval(np.polyint(parameters_i),t)
    print(f"c_ii = {polyint(10)-polyint(0)}")

    print(f"c_iii simpson = {intNCcompuesta_h(f,0,10,1e-7,3)}")

def ej2():
    f = lambda t: 3*t**3-9*np.sqrt(5)*t**2+45*t-15*np.sqrt(5)
    dfdt = lambda t: 9*t**2-18*np.sqrt(5)*t+45
    tiempos = np.linspace(0,5,1000)
    plt.plot(tiempos,f(tiempos))
    plt.grid()
    plt.show()
    # xr ~ \in (2.2275, 2.2450)
    print(f"biseccion {bisec(f,2.2275,2.2450,tol=1e-9)}")
    print(f"newton {newton(f,dfdt,2.2275,tol=1e-9)}")
    print(f"secante {secante(f,2.2275, 2.2450,tol=1e-9)}")

def ej3():

    F = lambda t,T,k1,k2,k3: k1+k2*t**2/T +k3*np.sqrt(T)
    #a)
    #para t = 2 se tiene
    T = np.array([40,45,50,55,60,65,70])
    C_2 = np.array([0.51,0.54,0.56,0.59,0.61,0.63,0.65])

    f1 = lambda t,T: t**2/T 
    f2 = lambda T: np.sqrt(T)
    M = np.array([np.ones_like(T),f1(2,T),f2(T)]).T
    y = M.T@C_2
    A = M.T @ M 
    k1,k2,k3 = np.linalg.solve(A,y)
    # print(k1,k2,k3)
    print(f"a) {F(2,52,k1,k2,k3)}")

    # b)
    t = np.array([2,4,6,8,10,12])
    C_45 = np.array([0.54,0.62,0.75,0.94,1.18,1.47])
    M = np.array([np.ones_like(t),f1(t,45),[f2(45)]*len(t)]).T
    A = M.T @ M
    y = M.T @ C_45
    k1,k2,k3 = np.linalg.solve(A,y)
    Fb = lambda t: F(t,45,k1,k2,k3)- 0.65
    print(f"b) {bisec(Fb,4,6,1e-10)}")





ej3()