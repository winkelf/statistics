####################################################################################
#  Script for basic matplotlib plotting. For now, only in Argentinean Spanish ***  #
####################################################################################

import numpy as np
from array import array
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

# Parametros
mu       = 0
sigma    = 1
nEntries = 1000
L = []
for i in range(nEntries):
    g = np.random.normal(mu, sigma)
    L.append(g)

# Vector con entradas del histograma
L = np.array(L)

# Histograma
hist, bins = np.histogram(L,density = True, bins = 50)
ancho_bin = bins[1] - bins[0]
err = np.sqrt(hist/(nEntries*ancho_bin))

# Eje X para funcion gaussiana
x_plot = np.linspace(min(bins)-0.1,max(bins)+0.1,1000)

# Ploteo
fig, ax = plt.subplots()
ax.bar(bins[:-1]+(bins[1]-bins[0])/2, hist, width = 0.1, align = 'center', color = 'cornflowerblue', yerr = err, ecolor = 'b',capsize = 2)
ax.plot(x_plot, norm.pdf(x_plot,loc = 0, scale = sigma), color = 'darkblue', label = 'Normal')
fig.suptitle('Distribución N(0,1)')
ax.set_xlabel('X~N(0,1)')
ax.set_ylabel('Frecuencia')
plt.show()


    






