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
    # print(np.loadtxt("Ej3_f1_trapecio.txt",delimiter=','))
    impresora(f2,F2,a2,b2,2)
    impresora(f3,F3,a,b,3)


def ej5():
    t_i = [0,1,4,6,8,12,16,20]
    c_i = [12,22,32,45,58,75,70,48]  
    Q   = 0.3
    c_iQ = [Q*k for k in c_i]
    trap = trapcomp(t_i,c_iQ)
    print(trap)

def ej7():
    f = lambda t: t*(t-1)*(t-2)+2
    dfdt = lambda t: 3*(t**2)-6*t+2
    dint = lambda t: f(t)*np.sqrt(1+(dfdt(t)**2)) 
    a,b =[0,3]
    CotaError = 1e-3
    #int = 2*pi * int_a^b f*sqrt(1+f'^2)
    # f' = 3t^2-6t+2
    k = 1
    integral2 = 0
    Error = np.inf
    while Error >= CotaError:
        integral = 2*np.pi*intNCcompuesta(dint,a,b,10*2**k,3)
        Error = np.abs(integral-integral2)
        integral2 = integral
        k+=1
    print(f"El Área vale {integral} y {k = }")
    

def ej8():
    f = lambda t: 1+t+np.cos(t)
    dfdt = lambda t: 1-np.sin(t)
    dint = lambda t: f(t)*np.sqrt(1+(dfdt(t)**2)) 
    a,b =[0,4]
    CotaError = 1e-13
    #int = 2*pi * int_a^b f*sqrt(1+f'^2)
    k = 1
    integral2 = 0
    Error = np.inf
    while Error >= CotaError:
        integral = 2*np.pi*intNCcompuesta(dint,a,b,10*2**k,3)
        Error = np.abs(integral-integral2)
        integral2 = integral
        k+=1
        if k>1000:
            raise ValueError(f"Se supero el límite de iteraciones,{integral = }, {k = }, {Error = }")
            return 
    return f"El Área vale {integral}, {k = } y {Error = }"
    

    print(np.loadtxt("Ej3_f1_trapecio.txt",delimiter=','))
    impresora(f2,F2,a2,b2,2)
    impresora(f3,F3,a,b,3)


