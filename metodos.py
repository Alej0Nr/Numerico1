import numpy as np

def bisec(f, a, b, tol = 1e-6, max_iter = 1000):
  """
  función para resolver una ecuación de la forma
  f(x) = 0 mediante el método de bisección,
  donde x es una raiz dentro del intervalo [a,b].
  Se resuelve hasta la tolerancia tol, con un 
  maximo nro de iteraciones dado por maxiter.

  f(a)f(b) debe ser < 0
  """

  if (f(a) * f(b) > 0):
    print("f(a) y f(b) tienen el mismo signo!")

  contador = 1
  I = (b-a)/2
  m = (a+b)/2

  while (I > tol and contador < max_iter):
    if (f(a) * f(m) < 0):
      b = m
    elif (f(m) * f(b) < 0):
      a = m
    else:
      break

    contador += 1
    I = I/2
    m = a + I

  if (contador == max_iter):
    print("Se alcanzó el numero maximo de iteraciones")

  return m, contador

def newton(f, df, x0, tol=1e-6, max_iter=100):
    """
    Metodo de Newton para encontrar la raiz de f(x) = 0.

    Parametros:
    - f: funcion f(x)
    - df: derivada f'(x)
    - x0: aproximacion inicial
    - tol: tolerancia para la convergencia
    - max_iter: numero maximo de iteraciones

    Retorna:
    - xn: aproximacion de la raiz
    - iteraciones: numero de iteraciones realizadas
    """
    xn = x0
    for i in range(max_iter):
        fxn = f(xn)
        dfxn = df(xn)

        if (dfxn == 0):  # Evitar division por cero
          raise ValueError("Derivada cero. El metodo de Newton falla.")
        else:
          delta = - fxn / dfxn  # Paso de Newton

          if (abs(delta) < tol):  # Criterio de convergencia
            return xn + delta, i + 1
          else:
            xn = xn + delta

    raise ValueError("El metodo de Newton no converge despues de {} iteraciones".format(max_iter))



def puntofijo(g, p0, tol, maxiter):
  """
  funcion para resolver la ecuacion de la forma
  x = g(x) mediante el metodo de punto fijo.
  """
  p = g(p0)
  contador = 1

  while ((abs(p-p0) > tol) and (contador <= maxiter)):
      p0 = p
      p = g(p0)
      contador += 1


  if (abs(p-p0) > tol):
      print('Se alcanzo el numero maximo de iteraciones')
  else:
      print('La solucion y el contador estan almacenados en las variable p y contador')
      return p , contador



def secante(f, p0, p1, tol, maxiter=1000):
  """
  Metodo de la secante para ecuaciones no lineales.

  Args:
    f: funcion escalar. Ecuacion a resolver f(x) = 0.
    p0, p1: aproximaciones iniciales.
    tol: tolerancia para el error absoluto.
    maxiter: maximo numero de iteraciones permitido.

  Returns:
    p: aproximacion de la raiz.
    residuo: valor de la funcion en la aproximacion de la raiz.
  """
  contador = 1
  f0 = f(p0)
  f1 = f(p1)
  p = p1 - f1 * (p1 - p0) / (f1 - f0)

  while (abs(p - p1) > tol) and (contador <= maxiter):
    p0 = p1
    f0 = f1
    p1 = p
    f1 = f(p)
    p = p1 - f1 * (p1 - p0) / (f1 - f0)
    contador += 1

  if (contador > maxiter):
    print("Se supero el maximo de iteraciones")

  residuo = f(p)
  return p, residuo,contador

def newtonvec(F, Fprima, p0, tol=1e-6, maxiter=100):
  """
  Metodo de Newton para sistemas de ecuaciones.

  Args:
    F: funcion que recibe un vector columna y devuelve un vector columna.
    Fprima: función que recibe un vector columna y devuelve la matriz Jacobiana de F.
    p0: aproximacion inicial (vector columna).
    tol: tolerancia para el error absoluto.
    maxiter: maximo numero de iteraciones permitido.

  Returns:
    p: Solucion aproximada.
    residuo: Residuo de la solucion.
  """
  contador = 1
  deltap = np.linalg.solve(Fprima(p0), -F(p0))
  p = p0 + deltap

  while (np.max(abs(deltap)) > tol) and (contador <= maxiter):
    deltap = np.linalg.solve(Fprima(p), -F(p))
    p = p + deltap
    contador += 1

  if (contador > maxiter):
    print("Se supero el maximo de iteraciones")

  residuo = F(p)
  return p, residuo, contador