__author__ = 'maryhallow'

import numpy as np
import cosmolopy as cp
from scipy import integrate
import matplotlib.pyplot as plt


# [parameters]

h = 0.704
pc = 3.0856776e18      # [cm]
t_h = 3.09e17/h        # [sec]
D_h = 9.26e27/h        # [cm]
Omega_lam = 0.7274     # 0.7274
Omega_b_0 = 0.0456
Omega_M_0 = 0.2726     # 2726
Omega_k = 1-Omega_lam-Omega_M_0


E = lambda x: 1/np.sqrt(Omega_M_0*np.power(1+x,3)+Omega_lam+Omega_k*np.power(1+x,2))

def D_m(z):

    if(Omega_k>0):
        return  D_h/np.sqrt(Omega_k)*np.sinh(np.sqrt(Omega_k)*D_c(z)/D_h)
    else:
        return D_c(z)

D_c = lambda z: D_h*integrate.quad(E, 0, z)[0]

D_A = lambda z: D_m(z)/(1+z)


test = np.linspace(0,50,500)
D_test = []
for i in range(0,len(test)):
    D_test = np.append(D_test,D_A(test[i])/pc/1e6)
plt.plot(test,D_test)
plt.show()

redshift = 8.0
print(cp.distance.angular_diameter_distance(redshift, **cp.fidcosmo))
print(D_A(redshift)/pc/1e6)
print(cp.fidcosmo)


