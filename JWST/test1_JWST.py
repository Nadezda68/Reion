import numpy              as np
import matplotlib.pyplot  as plt
from   astropy.io           import fits

wavelengths,transmission = np.loadtxt('data/filter_F150W.dat',skiprows=1).T
plt.plot(wavelengths*1e2,transmission)
wavelengths,transmission = np.loadtxt('data/filter_F200W.dat',skiprows=1).T
plt.plot(wavelengths*1e2,transmission)

PSF150 = fits.open('data/PSF3_F150W_TYP.fits')[0].data
PSF200 = fits.open('data/PSF3_F200W_TYP.fits')[0].data

print(np.shape(PSF150),np.shape(PSF200))
plt.show()
