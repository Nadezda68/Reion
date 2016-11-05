import numpy              as np
import matplotlib.pyplot  as plt
from   astropy.io           import fits

plt.figure(1)
wavelengths,transmission = np.loadtxt('data/filter_F150W.dat',skiprows=1).T
plt.plot(wavelengths*1e2,transmission)
wavelengths,transmission = np.loadtxt('data/filter_F200W.dat',skiprows=1).T
plt.plot(wavelengths*1e2,transmission)

PSF150 = fits.open('data/PSF_NIRCam_F150W.fits')[0].data
PSF200 = fits.open('data/PSF_NIRCam_F200W.fits')[0].data

print(np.shape(PSF150),np.shape(PSF200))

ramps  = np.zeros((63,63,4))
names  = ['5e-7_1','5e-7_2','5e-7_5','5e-7_10']
source = np.array([5e-7,10e-7,25e-7,50e-7])

flux   = np.zeros(4)
pixels = np.zeros(4)
N      = np.shape(ramps)[0]
print(N*N)
print(N)
for i in range(4):
    ramps[:,:,i] = fits.open('data/F150W/' + names[i] + '.fits')[0].data
    for j in range(N):
        for k in range(N):
            if(ramps[j,k,i]>=0.085):
                flux[i] += ramps[j,k,i]
                pixels[i] += 1


print(flux)
print(pixels)
print(pixels/N/N)
print(flux/source)

plt.figure(2)
plt.imshow(ramps[:,:,3],interpolation='nearest',vmax=1)
coef = 30542561


noise_no_background  = fits.open('data/F150W/noise_no_bg_exp_t_3e6_sec.fits')[0].data
noise_low_background = fits.open('data/F150W/noise_low_background.fits')[0].data
print(np.max(noise_no_background),np.min(noise_no_background),np.mean(noise_no_background))
print(np.max(noise_low_background),np.min(noise_low_background),np.mean(noise_low_background))

fluxes=np.array([95.84,130.49,145.29,150.74,155.16,158.04,160.80,163.92,166.65,168.52,169.83,170.90,171.84,172.61])
apertures = np.linspace(0.05,0.75,len(fluxes))

plt.figure(3)
plt.plot(apertures,fluxes,'r^')
plt.show()

plt.figure(2)
plt.imshow(ramps[:,:,3],interpolation='nearest',vmax=1)
coef = 30542561


noise_no_background  = fits.open('data/F150W/noise_no_bg_exp_t_3e6_sec.fits')[0].data
noise_low_background = fits.open('data/F150W/noise_low_background.fits')[0].data
print(np.max(noise_no_background),np.min(noise_no_background),np.mean(noise_no_background))
print(np.max(noise_low_background),np.min(noise_low_background),np.mean(noise_low_background))
