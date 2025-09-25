from integration import *
import numpy as np
import matplotlib.pyplot as plt


def ej2():
    ## f1
    a,b=[0,5]
    f = lambda t: np.sin(np.pi*t)
    F = lambda t: - np.cos(np.pi*t)/np.pi
    error=[]

    ## f2
    g = lambda t: 1/(1+t**2)
    a2,b2 = [-5,5]
    G = lambda t: np.arctan(t)
    error2=[]

    ## errores y plot
    tiempo1=np.linspace(a,b,1000)
    tiempo2=np.linspace(a2,b2,1000)
    for n in range(2,14):
        error.append(float(np.abs((F(b)-F(a))-integralNC(f,a,b,n))))
        error2.append(float(np.abs((G(b2)-F(a2))-integralNC(g,a2,b2,n))))
        plt.subplot(211)
        plt.plot(tiempo1,np.polyval(np.polyfit(np.linspace(a,b,n),f(np.linspace(a,b,n)),n),tiempo1))
        plt.subplot(212)
        plt.plot(tiempo2,np.polyval(np.polyfit(np.linspace(a2,b2,n),g(np.linspace(a2,b2,n)),n),tiempo2),label=f"{n}")
    print(error)
    print(error2)
    plt.subplot(211)
    plt.plot(tiempo1,f(tiempo1),color="maroon",label=r"f(x)")
    plt.grid()
    plt.subplot(212)
    plt.plot(tiempo2,g(tiempo2),color="maroon",label=r"f(x)")
    plt.grid()
    plt.legend()
    plt.show()

def ej3():
    f = lambda t: np.sin(np.pi*t)
    F = lambda t: - np.cos(np.pi*t)/np.pi
    a,b = [0,5]
    f2 = lambda t: 1/(1+t**2)
    F2 = lambda t: np.arctan(t)
    a2,b2=[-5,5]
    f3 = lambda t: np.abs(t)**(3/2)
    F3 = lambda t: (2/5) * t**(5/2)

    # h = (b-a)/L
    # DataS = ["L","trapecio", "Error Trapecio","E_L/2 | E_L"]
    # DataT =  ["L","Simpson", "Error Simpson","E_L/2 | E_L"]
    # 
    def impresora(f,F,a,b,k):
        DataS=[]
        DataT=[]
        integral=np.abs((F(b)-F(a)))
        for L in [(2**n)*5 for n in range(1,14)]:
            trap = intNCcompuesta(f, a, b, L, 2)
            simp = intNCcompuesta(f, a, b, L, 3)
            errorT = float(integral - trap)
            errorS = float(integral - simp)
            # trap_2 = intNCcompuesta(f, a, b, int(L/2), 2)
            # simp_2 = intNCcompuesta(f, a, b, int(L/2), 3)
            # errorT_2 = float(np.abs((F(b)-F(a))-trap_2))
            # errorS_2 = float(np.abs((F(b)-F(a))-simp_2))
            
            DataT.append([L,trap,errorT]) #,errorT_2/errorT
            DataS.append([L,simp,errorS])#,errorS_2/errorS
        np.savetxt(f"Ej3_f{k}_trapecio.txt", DataT, delimiter=',')
        np.savetxt(f"Ej3_f{k}_simpson.txt", DataS, delimiter=',')

    impresora(f,F,a,b,1)
    print(np.loadtxt("Ej3_f1_trapecio.txt",delimiter=','))
    impresora(f2,F2,a2,b2,2)
    impresora(f3,F3,a,b,3)


