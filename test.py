import yt
import numpy as np
import glob
import cosmolopy as cp
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
from scipy import integrate


def gal():

    muf_list = glob.glob("./drt/muv.bin*") # it is the list of all available files

    lam_list = np.zeros(len(muf_list))
    lookup = np.zeros([len(muf_list), 188, 22])
    print(len(muf_list))

    for i in range(len(muf_list)):
        f = open(muf_list[i])
        header = f.readline()
        f.close()
        d1 = header.split()[0]
        d2 = header.split()[1]
        lam = float(header.split()[2])
        lam_list[i] = lam
        data = np.genfromtxt(muf_list[i], skip_header=1)
        lookup[i, :, :] = data[1:,1:]

    dx = data[0, 1:]
    dy = data[1:, 0]
    print(d1,d2)


    files = glob.glob("./rei05B_a0*/rei05B_a*.art")
    iii = 0
    snap_file = files[iii]
    pf = yt.load(snap_file)


    " 1 ---> 7.20807692e+22 "
    data = pf.sphere([1.52643699e+2/7.20807692,  1.08564619e+2/7.20807692,  9.16425327e+1/7.20807692], (10.0, "kpc"))
    print(data)

    x = np.array(data[('STAR', 'POSITION_X')]) - 1.52643699e+2/7.20807692
    y = np.array(data[('STAR', 'POSITION_Y')]) - 1.08564619e+2/7.20807692
    z = np.array(data[('STAR', 'POSITION_Z')]) - 9.16425327e+1/7.20807692
    m = data[('STAR', 'MASS')]
    im = data[('STAR', 'INITIAL_MASS')]
    t = np.log10(data[('STAR', 'age')].in_units('yr'))
    met = data[('STAR', 'METALLICITY_SNIa')] + data[('STAR', 'METALLICITY_SNII')]
    species = data[('STAR', 'SPECIES')]

    plt.hist2d(x,y,100)
    plt.show()

    redshift = 8.0
    DA = cp.distance.angular_diameter_distance(redshift, **cp.fidcosmo) # [Mpc]
    print(str(cp.fidcosmo))
    secinrad = 206265.0
    #nbins #[3 arcsec -> 100 bin], theta(rad) = X [fixed] / D_a;
    theta_arcsec = 10.0 * 1e3 / DA / 1e6 * secinrad
    print(theta_arcsec)
    nbins =  theta_arcsec * 100/3  # theta_arcsec * 100 bins / 3 arcsec
    image = np.zeros([nbins, nbins, 3, len(lam_list)])

    for i in range(0,2):

        interp = interp2d(dx, dy, lookup[i, :, :])
        temp = m.copy()
        for j in range(len(m)):
            temp[j] *= interp(met[j], t[j])[0]
        xedges = np.linspace(-data.radius, data.radius, nbins+1)
        yedges = np.linspace(-data.radius, data.radius, nbins+1)
        H, X, Y = np.histogram2d(x, y, bins=(xedges, yedges), weights = temp)
        print(X)
        image[:, :, 0, i] = H
        H, X, Y = np.histogram2d(y, z, bins=(xedges, yedges), weights = temp)
        image[:, :, 1, i] = H
        H, X, Y = np.histogram2d(x, z, bins=(xedges, yedges), weights = temp)
        image[:, :, 2, i] = H


    plt.imshow(np.log10(image[:,:,0,0]), interpolation='nearest')
    plt.show()



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

D_A = lambda z: D_m(z)/(1+z) # [ in case Omega_k = 0 ]

'''
test = np.linspace(0,5,200)
D_test = []
for i in range(0,len(test)):
    D_test = np.append(D_test,D_A(test[i])/D_h)
plt.plot(test,D_test)
plt.show()

redshift = 8.0
print(cp.distance.angular_diameter_distance(redshift, **cp.fidcosmo))
print(D_A(redshift)/pc/1e6)
'''
gal()