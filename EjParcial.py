import numpy as np
import matplotlib.pyplot as plt

# N=2000
# I0 = 1
# i = lambda t: (N*I0)/(I0+(N-i0)*np.exp(-p*t))
datos = np.loadtxt("infectados.txt")
dia = datos[:,0]
infectados = datos[:,1]
N = 2000
suceptibles = [2000-i for i in infectados]
I1 = [np.log(1/i - 1/N) for i in infectados]

parameters = np.polyfit(dia,I1,1)
p = -parameters[0]
I0 = N/(1+N*np.exp(parameters[1]))

i = lambda t: (N*I0)/(I0+(N-I0)*np.exp(-p*t))
plt.scatter(dia,infectados)
tiempos = np.linspace(0,200,5000)
plt.plot(tiempos,i(tiempos))
print(f"{i(181) = }")
print(f"Error cuuadratico = {np.linalg.norm(infectados-i(dia))}")
plt.show()

