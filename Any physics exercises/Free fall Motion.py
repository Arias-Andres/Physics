## import libreries ########
import sympy as smp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

##### Define symbols ###### 

t, g, m = smp.symbols('t g m', positive=True)
F, a = smp.symbols('F a', real=True)

####  That is Free Fall, the aceleration and force thea have a unique component y ###

Eq = smp.Eq(F, m*a)

#### y position is a time function ####

y = smp.Function('y', real=True)
y = y(t)

#### Temporal Derivates of y #####

d1_y = smp.diff(y, t)
d2_y = smp.diff(d1_y, t)

#### Gravity in y axis ###

Fy = -m*g

Eq = Eq.subs([[F, Fy], [a, d2_y]])
print (Eq)

### First we will determine the exact solution using dsolve function #####

y_explicit = smp.dsolve(Eq, y).args[1]

print (y_explicit)

d1_y_explicit = smp.diff(y_explicit, t)
print (d1_y_explicit)

### Initial Conditions ####


t0 = 0
y0 = 500
vy0 = 0
S0 = {y.subs(t, t0):y0, d1_y.subs(t, t0):vy0}


y_explicit = smp.dsolve(Eq, y, ics=S0).args[1]
print(y_explicit)


d1_y_explicit = smp.diff(y_explicit, t)
print (d1_y_explicit)


y_exact = smp.lambdify((t, g), y_explicit, modules='numpy')
d1_y_exact = smp.lambdify((t, g), d1_y_explicit, modules='numpy')


t0 = 0
tf = 10
t_size = 1001
t = np.linspace(t0, tf, t_size)

g = 9.81


y_sol_exact = y_exact(t, g)
d1_y_sol_exact = d1_y_exact(t, g)

plt.figure(figsize=(8, 8))

plt.plot(t, y_sol_exact, lw=3, c='blue', label='position')
plt.plot(t, d1_y_sol_exact, lw=3, c='red', label='speed')

plt.xlabel('time (s)')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(y_sol_exact, d1_y_sol_exact, lw=3)
plt.xlabel('height (m)')
plt.ylabel('speed (m/s)')
plt.grid()
plt.show()