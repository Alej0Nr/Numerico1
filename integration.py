import numpy as np

def pesosNC(n):
    # Calcula los pesos de la fórmula de Newton-Cotes de n puntos
    x = np.linspace(0, 1, n)
    A = np.ones((n, n))
    for i in range(1, n):
        A[i, :] = A[i-1, :] * x
    b = 1 / np.arange(1, n+1)
    w = np.linalg.solve(A, b)
    return w

def integralNC(f,a,b,n):
    w = pesosNC(n)
    x = np.linspace(a,b,n)
    y = f(x)
    Q = (b - a)*np.sum(y * w)
    return Q

def NCcompuesta(f,a,b,L,n):
    " function Q = NCcompuesta(f,a,b,L,n)"
    " aproxima la integral de f sobre [a,b]"
    " utilizando la formula de Newton-Cotes compuesta"
    " de n puntos, subdividiendo en L subintervalos"
    y = np.linspace(a,b,L+1)
    Q = 0
    for i in range(L):
        Q = Q + integralNC(f,y[i],y[i+1],n)
    return Q

def intNCcompuesta(f, a, b, L, n):
    z = np.linspace(a, b, L + 1)
    h = (b - a) / L
    w = pesosNC(n)
    Q = 0
    for i in range(L):
        x = np.linspace(z[i], z[i+1], n)
        y = f(x)
        Q += h * np.sum(y * w)
    return Q

def trapcomp(x,y):
    L=np.size(x)-1
    deltax=np.diff(x)
    Q=0
    for i in range(0,L):
        Q+=0.5*deltax[i]*(y[i]+y[i+1])
    return Q

def simpsoncomp(x,y):
    L=np.size(x)-1
    if L%2:
        raise ValueError("dame una cantidad impar")
        Q=np.nan
        return Q
    h=(x[-1]-x[0])/(L/2)
    Q=h/6 * (y[0] + 4*sum(y[1:-1:2]) + 2*sum(y[2:-2:2]) + y[-1])
    return Q

def AreaRevolucion(f,df,a,b,CotaError):
    inte = lambda t: f(t)*np.sqrt(1+(df(t))**2)
    k = 1
    integral2 = 0
    Error = np.inf
    while Error >= CotaError:
        integral = 2*np.pi*intNCcompuesta(inte,a,b,10*2**k,3)
        Error = np.abs(integral-integral2)
        integral2 = integral
        k+=1
    return integral

def intNCcompuesta_h(f,a,b,Cota_Error,n):
    integrales = [intNCcompuesta(f,a,b,10,n),intNCcompuesta(f,a,b,20,n)]
    i=2
    while np.abs(integrales[-2]-integrales[-1])<Cota_Error:
        integrales.append(intNCcompuesta(f,a,b,10*2**i,n))
        i+=1
    h = np.abs((b-a)/(10*2**(i)))
    return integrales[-1],h,10*2**(i)


    