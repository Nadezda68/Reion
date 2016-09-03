import numpy              as np
import matplotlib.pyplot  as plt
from scipy.interpolate    import interp1d
from scipy                import integrate
from scipy                import signal
from astropy.io           import fits
from scipy                import stats



# 13200 9700
nbins_min = 1500
pixels_with_noise = fits.open('/home/maryhallow/Desktop/python/Reionizatoin/Reion/hlsp_hlf_hst_wfc3-60mas_goodss'
                              '_f160w_v1.0_sci.fits')[0].data[13200:13200+nbins_min,9700:9700+nbins_min]
max = np.max(pixels_with_noise)
min = np.min(pixels_with_noise)

data_array = np.reshape(pixels_with_noise,(nbins_min*nbins_min,))
#data_array = stats.norm.rvs(size=5000)
pdf = stats.gaussian_kde(data_array)
print(np.max(data_array))

x = np.concatenate([np.linspace(np.min(data_array),0.03,500),np.linspace(0.03,1,1000)[1:]])
y = pdf.evaluate(x)
plt.figure(1)
plt.plot(x,y)

#plt.imshow(pixels_with_noise[::-1,:],interpolation='nearest',vmax=max/99.5,cmap=plt.cm.gray)
plt.figure(2)
y2 = np.zeros_like(x)
for i in range(0,len(x)):
    for j in range(1,i+1):
        y2[i] += (x[j]-x[j-1])*y[j]

plt.plot(x,y2)

plt.show()
