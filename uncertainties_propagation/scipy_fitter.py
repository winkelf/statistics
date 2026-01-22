###################################################################################
#  Script for basic scipy fitter usage. For now, only in Argentinean Spanish ***  #
#  Fits a set of points with two different funcions                               #
###################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Setear seed para obtener siempre los mismos resultados
np.random.seed(1)

# Genero los puntos que voy a ajustar
## Valores en x
x = [i for i in range(1,10)]

## Valores en y (son puntos fluctuando gaussianamente al rededor de y=0.5x + 1, con sigma=0.5)
y = []
for i in x:
    g = np.random.normal(0.5*i + 1, 0.5)
    y.append(g)

## Mismos errores en y para todos los puntos
yerr = [ 0.5 for i in range(len(x)) ]

# Arranca el ploteo
fig, ax = plt.subplots()

ax.errorbar(x, y, yerr, fmt='o', linewidth=2, capsize=6)
ax.set(xlim=(0, 10), xticks=np.arange(0, 10),
       ylim=(0, 10), yticks=np.arange(0, 10))

# Funciones con las que vamos a ajustar. Los argumentos tienen que ser siempre de la forma: (x, param1, param2, ...)
def funcionLineal(x, a0, a1):
    return(a0 + a1*x)

def funcionCuadratica(x, a0, a1, a2):
    return(a0 + a1*x + a2*x**2)

# Ajuste a los puntos con funcion lineal, llamamos a curve_fit de scipy.optimize
# popt es un array con el valor de los parametros ajustados
# pcov es la matriz de covarianza de los parametros 
popt,pcov = curve_fit(funcionLineal,x,y,sigma = yerr)
a_0,a_1   = popt[0], popt[1]
y_a       = a_0 + a_1*0.5

# Para plotear la funcion lineal ajustada:
x_fit = [ i for i in range(0,11) ]
y_fit = [ funcionLineal(i, a_0, a_1) for i in x_fit  ]

# Ajuste a los puntos con funcion cuadratica, llamamos a curve_fit de scipy.optimize
popt1,pcov1   = curve_fit(funcionCuadratica,x,y,sigma = yerr)
p_0,p_1,p_2   = popt1[0], popt1[1], popt1[2]

# Para plotear la funcion cuadratica ajustada:
y_par_fit = [ funcionCuadratica(i, p_0, p_1, p_2) for i in x_fit  ]

# Mas ploteo
ax.plot(x_fit, y_fit, '', linewidth=2.0, label='Ajuste lineal')
ax.plot(x_fit, y_par_fit,'', linewidth=2.0, label='Ajuste cuadratico')

print("Matrices de covarianza:")
print(pcov)
print(pcov1)

fig.suptitle('Ajuste a puntos y = 0.5x + 1')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.legend()

plt.grid()
plt.show()




