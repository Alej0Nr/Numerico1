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


def ej9():
    datos = np.loadtxt("DatosTP3\Energias_renovables.txt")
    tiempo = [t[0] for t in datos ]
    ET = [E[1]+E[2]+E[3]+E[4] for E in datos]
    D = [100*ET[i]/datos[i][5] for i in range(len(datos))]
    plt.plot(tiempo,ET, color="orange", label="Potencia Electrica")
    plt.plot(tiempo,D, color="red", label="Demanda")

    Dmax = [tiempo[D.index(max(D))],max(D)]
    Dmin = [tiempo[D.index(min(D))],min(D)]
    ETmax = [tiempo[ET.index(max(ET))],max(ET)]
    print(f"La minima demanda es {Dmin[1]} y se da en t = {Dmin[0]}")
    print(f"La maxima demanda es {Dmax[1]} y se da en t = {Dmax[0]}")
    print(f"La maxima energia renovable es {ETmax[1]} y se da en t = {ETmax[0]}")
    plt.vlines(Dmin[0],ymin=0,ymax=Dmin[1],linestyle='--',color="black")
    plt.hlines(Dmin[1],xmin=0,xmax=Dmin[0],linestyle='--',color="black")
    plt.vlines(Dmax[0],ymin=0,ymax=Dmax[1],linestyle='--',color="black")
    plt.vlines(ETmax[0],ymin=0,ymax=ETmax[1],linestyle='--',color="black")
    plt.hlines(Dmax[1],xmin=0,xmax=Dmax[0],linestyle='--',color="black")
    plt.hlines(ETmax[1],xmin=0,xmax=ETmax[0],linestyle='--',color="black")
    plt.scatter(Dmax[0],Dmax[1],color="green",label = "Demanda maxima")
    plt.scatter(ETmax[0],ETmax[1],color="forestgreen",label = "Energia maxima")
    plt.scatter(Dmin[0],Dmin[1],color="cyan",label = "Demanda minima")
    plt.legend()
    plt.grid()
    plt.show()

    ETP = simpsoncomp(tiempo,ET)
    print(f"Se genero un total de {ETP} MWh de energía renovable")
    DT = simpsoncomp(tiempo,D)
    print(f"representa un %{ETP*100/DT} de la demanda")


    Fmax = [tiempo[[E[2] for E in datos].index(max([E[2] for E in datos]))],max([E[2] for E in datos])]
    print(f"La mayor potencia fotovoltaica es {Fmax[1]} y se dio en t = {Fmax[0]}")
    EET = simpsoncomp(tiempo,[E[1] for E in datos])
    print(f"Se genero un total de {EET} MWh a partir de la energia eólica, representa un %{100*EET/ETP} de la energia producida")
    print(f"")
    plt.plot(tiempo,[E[1] for E in datos],label="Eólica")
    plt.plot(tiempo,[E[2] for E in datos],label="Fotovoltaica")
    plt.plot(tiempo,[E[3] for E in datos],label="Bioenergías")
    plt.plot(tiempo,[E[4] for E in datos],label="Hidráulica")
    plt.vlines(Fmax[0],ymin=0,ymax=Fmax[1],linestyle='--',color="black")
    plt.hlines(Fmax[1],xmin=0,xmax=Fmax[0],linestyle='--',color="black")
    plt.scatter(Fmax[0],Fmax[1],label=r"$E_f max$")
    plt.grid()
    plt.legend()
    plt.show()
    
ej9()