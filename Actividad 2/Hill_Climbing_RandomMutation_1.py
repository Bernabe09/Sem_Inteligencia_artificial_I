import matplotlib.pyplot as plt
import numpy as np
from Plot_Surf import *
from Plot_Contour import *
from math import e

f = lambda x,y: x*(e**(-(x**2)-(y**2)))

xl = np.array([-1, -1])
xu = np.array([1, 1])

G = 300
f_plot = np.zeros(G)
mu, sigma = 0, 0.2 # ES

x = xl + (xu-xl) * np.random.uniform(-2,2,2)
f_plot = np.zeros(G)

for i in range(G):
    y = x.copy()
    j = np.random.randint(2)
    y[j] = xl[j] + (xu[j] - xl[j]) * np.random.random()

    if f(y[0], y[1]) < f(x[0], x[1]):
        x = y

    #display.display(plt.gcf())
    #display.clear_output(wait=True)
    
    f_plot[i] = f(x[0], x[1])

plot_contour(f, x, y, xl, xu)
plot_surf(f,x,xl,xu)
print("Mínimo global en x=", x[0], " y=", x[1], " f(x,y)=", f(x[0],x[1]))

plt.plot(range(G), f_plot)
plt.title("Convergencia")