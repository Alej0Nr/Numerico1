import numpy as np
import matplotlib.pyplot as plt
from integration import *
from metodos import *


def ej2():
    f = lambda X: np.array([8*10**(-6)*X[0]**3+400*X[0]**2*X[1]-X[0]*X[2]/500+6,np.exp(X[0]/50)+np.exp(10**6 *X[1])-X[2]/10,10**12*X[1]**2-X[0]*X[2]/250 -4]).T
    Df = lambda X: np.array([[24*10**(-6)*X[0]**2+800*X[0]*X[1]-X[2]/500,400*X[0]**2,-X[0]/500],[1/50 *np.exp(X[0]/50),10**6*np.exp(10**6 *X[1]),-1/10],[-X[2]/250,2*10**12*X[1],-X[0]/250]])
    p0 = np.array([-50,-0.001,10])
    print(newtonvec(f,Df,p0,1e-13))
    # print(f(p0))
    # print(Df(p0))


def ej3():
    ti = np.array([0.0,0.30,1.0,1.15,1.98,2.17,3.03,3.13])
    vi = np.array([.48,.43,.20,.17,-.19,-.22,-.47,-.45])

    # v(t) = A w cos(wt+phi)
    # Aw = 0.5
    vi2 = np.arccos(2*vi)

    w, phi = np.polyfit(ti,vi2,1)
    tiempos = np.linspace(0,3.13,1000)
    v = lambda t:  0.5*np.cos(w*t+phi)
    plt.scatter(ti,vi)
    plt.plot(tiempos,v(tiempos)) 
    plt.grid()
    # plt.show()
    print(f"a) {np.abs(-0.19-v(1.98))}")

    print(trapcomp(ti,vi)+0.17)
    # usando la ec integral calculo p(t)
    p = lambda t: 0.5*(np.cos(w*t+phi))
    psol = lambda t: 0.17+0.5/w *(np.sin(w*t+phi)-np.sin(phi))
    print(psol(3.13))
    print(0.17+intNCcompuesta_h(p,0,3.13,1e-14,3)[0],f"L = {intNCcompuesta_h(p,0,3.13,1e-14,3)[2]}")



ej3()