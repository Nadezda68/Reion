import yt
import glob
import numpy as np
from scipy                import integrate

# Constants
cm_in_pc = 3.0857e18
sun_luminosity = 3.828e33
arcsec_in_rad = 206265
c = 2.9927e10

# Cosmo params
Omega_lam = 0.7274
Omega_M_0 = 0.2726
Omega_k = 0.0
h = 0.704

# Functions to compute Angular diameter distance D_A [Mpc]
E   = lambda x: 1/np.sqrt(Omega_M_0*np.power(1+x,3)+Omega_lam+Omega_k*np.power(1+x,2))
D_m = lambda x: D_c(x)
D_c = lambda x: (9.26e27/h)*integrate.quad(E, 0, x)[0]
D_A = lambda x: D_m(x)/(1+x)/cm_in_pc/1e6  # Angular distance [Mpc]
N_sim = 0

def f():

    redshifts = np.zeros((28,5))
    Angle = np.zeros((28,5))
    S_angle = np.zeros((28,5))
    D_phys = np.zeros((28,5))

    for sim,ii in zip(['1','2','4','5','6'],[0,1,2,3,4]):
        files = sorted(glob.glob('/home/kaurov/scratch/rei/random10/OUT_0000'+sim+'_010_010_random/rei0000'+sim+'_010_010_random_a0*/rei0000'+sim+'_010_010_random_a0.*.art'))

        for simulation in range(0, 28):

            pf = yt.load(files[simulation])
            redshifts[simulation,ii] = pf.current_redshift
            theta_arcsec = (1e3 * pf.domain_right_edge.in_units('kpc')[2]) / (D_A(pf.current_redshift) * 1e6) * arcsec_in_rad
            S_angle[simulation,ii] = theta_arcsec * theta_arcsec
            Angle[simulation,ii] = theta_arcsec
            D_phys[simulation,ii] = 1e3 * pf.domain_right_edge.in_units('kpc')[2]

    np.savetxt('processed_huge/redshifts.dat', redshifts, fmt='%1.5e')
    np.savetxt('processed_huge/S_angle.dat', S_angle, fmt='%1.5e')
    np.savetxt('processed_huge/Angle.dat', Angle, fmt='%1.5e')
    np.savetxt('processed_huge/D_phys.dat', D_phys, fmt='%1.5e')

    print(pf.domain_right_edge.in_units('kpc')[2],pf.domain_right_edge[2])
    print(pf.domain_left_edge.in_units('kpc')[2],pf.domain_left_edge[2])


f()


