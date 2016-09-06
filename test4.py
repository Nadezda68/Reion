import yt
import glob
import numpy              as np
import matplotlib.pyplot  as plt

from matplotlib           import ticker
from scipy.interpolate    import interp1d
from scipy                import integrate
from scipy                import signal
from scipy                import stats
from astropy.io           import fits


def init():

    global info,lums,lam_list,lamb,max_flux,z_array,nbins_array,D_A_array,indices,nbins_min,pixels_with_noise,noise_std,noise_mean

    muf_list = glob.glob("/home/kaurov/kicp/code05A/drt/muv.bin*")
    lums  = glob.glob("./output/lum_*")
    info  = glob.glob("./output/info_*")

    lam_list = np.zeros(len(muf_list)-1)

    for i in range(len(muf_list)-1):

        f = open(muf_list[i])
        header = f.readline()
        f.close()
        lam_list[i] = float(header.split()[2])

    lamb = lam_list[np.argsort(lam_list)]

    max_flux    = []
    z_array     = []
    nbins_array = []
    D_A_array   = []
    
    for i in range(0,len(lums)):

        nbins,redshift,D_A,theta = np.loadtxt(info[i],skiprows=1)
        nbins = int(nbins)

        D_A_array.append(D_A)
        z_array.append(redshift)
        nbins_array.append(nbins)

    z_array = np.array(z_array)
    indices = np.argsort(z_array)[::-1]
    z_array = z_array[indices]

    D_A_array = np.array(D_A_array)
    D_A_array = D_A_array[indices]

    nbins_array = np.array(nbins_array)
    nbins_array = nbins_array[indices]
    nbins_min   = np.min(nbins_array)

    pixels_with_noise = stats.norm.rvs(-0.00020178,0.00239519,nbins_min*nbins_min) # deep field
    noise_std         = np.std(pixels_with_noise)
    noise_mean        = np.mean(pixels_with_noise)
    #pixels_with_noise = stats.norm.rvs(-5.49671285e-05,1.07018043e-02,nbins_min*nbins_min) # less deep field
    pixels_with_noise = np.reshape(pixels_with_noise,(nbins_min,nbins_min))

E   = lambda x: 1/np.sqrt(Omega_M_0*np.power(1+x,3)+Omega_lam+Omega_k*np.power(1+x,2))
D_m = lambda z: D_c(z) # Omega_k = 0
D_c = lambda z: (9.26e27/h)*integrate.quad(E, 0, z)[0]
D_A = lambda z: D_m(z)/(1+z)/3.0857e18/1e6

def compute():

    global Omega_lam, Omega_M_0, Omega_k, h, input_data

    Omega_lam = 0.7274
    Omega_M_0 = 0.2726
    Omega_k   = 0.0
    h         = 0.704

    plot_index = 0

    for i in indices:

        input_data = np.load(lums[i])[:,:,:]
        print(lums[i])
        luminosity(plot_index)
        if(plot_index % 3 ==0):
            spectrum(plot_index)
        max_flux.append(filter_flux(plot_index))
        print(z_array[plot_index],nbins_array[plot_index],max_flux[-1])
   
        plot_index += 1

    plt.figure(0)
    plt.subplot(2,2,1)
    plt.title('Spectrum')

    plt.figure(0)
    plt.subplot(2,2,2)
    plt.title('Observed flux')
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.plot(z_array,max_flux,'b^',label='flux')
    plt.plot(z_array,np.ones(len(z_array))*max_noise,label='max noise = '+str(round(max_noise,5)))
    plt.plot(z_array,np.ones(len(z_array))*noise_mean,label='mean noise = '+str(round(noise_mean,5)))
    plt.plot(z_array,np.ones(len(z_array))*noise_std,label='std noise = '+str(round(noise_std,5)))
    plt.legend(prop={'size':10},loc=4)
    plt.xlabel('z')

    plt.subplot(2,2,3)
    plt.title('Angular distance')
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.plot(z_array,D_A_array,'b^')
    plt.xlabel('z')
    plt.ylabel('$D_{L}$ [Mpc]')

    plt.subplot(2,2,4)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.title('N pixels in image')
    plt.plot(z_array,nbins_array,'b^')
    plt.xlabel('z')
    plt.ylabel('N')

    plt.figure(2)
    plt.subplot(5,5,25)
    plt.imshow(signal.fftconvolve(pixels_with_noise, PSF, mode='same'), interpolation='nearest',cmap=plt.cm.gray)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.title('Noise')


    plt.figure(0)
    plt.savefig('figure0.pdf')
    
    plt.figure(1)
    plt.savefig('figure1.pdf')

    plt.figure(2)
    plt.savefig('figure2.pdf')

    plt.figure(3)
    plt.savefig('figure3.pdf')


def luminosity(i):

    nu = 2.99792458e10/(lamb/1e8)

    total_lum = np.zeros_like(input_data[:,:,0])

    for j in range(0,len(total_lum[:,0])):
        for k in range(0,len(total_lum[:,0])):
            total_lum[j,k] = integrate.trapz(input_data[j,k,::-1], nu[::-1])

    plt.figure(1)
    plt.subplot(5,5,i+1)
    plt.imshow(np.log10(total_lum), interpolation='nearest')
    plt.yticks([])
    plt.xticks([])
    plt.title(str(round(z_array[i],2)))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)


def spectrum(i):

    lum_nu = np.zeros_like(input_data[0,0,:])

    for j in range(0,len(lamb)):
        lum_nu[j] = np.max(input_data[:,:,j])

    plt.figure(0)
    plt.ylim([26,29])
    plt.subplot(2,2,1)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.plot(lamb,np.log10(lum_nu),label='z=' + str(round(z_array[i],2)))
    plt.xlabel('$\\lambda$ [Angstrom]')
    plt.ylabel('$log10(L_{\\nu})$ [erg/s/Hz]')
    plt.legend(prop={'size':3},loc=3)

def filter_init():

    global F_filter

    temp_filters = open('data/FILTER.RES.latest', 'r')
    filters = []
    filters_names = []
    first = True

    for line in temp_filters:

        if line[0] == ' ':
            if not first:
                filters.append(np.array(temp))

            first = False
            filters_names.append(line.split())
            temp = []

        else:
            temp.append(np.array(line.split()).astype('float'))

    filters.append(np.array(temp))
    filters = np.array(filters)

    for ifilt in range(len(filters_names)):
        if filters_names[ifilt][1] == 'hst/wfc3/IR/f160w.dat':
            filter_b = np.array([filters[ifilt][:,1],filters[ifilt][:,2]])
            filter_b = np.transpose(filter_b)

    F_filter = interp1d(filter_b[:,0], filter_b[:,1],fill_value=0.0,bounds_error=False)

def filter_flux(i):

    global max_noise,pixels_with_noise,PSF

    nu = 2.99792458e10/(lamb/1e8)
    total_flux = np.zeros_like(input_data[:,:,0])
    lum_dist = D_A(z_array[i]) * (1 + z_array[i]) * (1 + z_array[i])
    if(nbins_array[i]%2==0):
        b = int((nbins_array[i] - nbins_min)/2)
        a = b

    else:
        b = int((nbins_array[i] - nbins_min)/2)
        a = b+1
 
    for j in range(0,len(total_flux[:,0])):
        for k in range(0,len(total_flux[:,0])):
            total_flux[j,k] = integrate.trapz(input_data[j,k,::-1] * F_filter(lamb[::-1]*(1+z_array[i])), nu[::-1]) / \
                             integrate.trapz(F_filter(lamb[::-1]*(1+z_array[i])), nu[::-1]) * (1+z_array[i]) / (4 * np.pi * np.power(lum_dist*3.0857e18*1e6,2))

    zero_point = 25.94
    coeff = 10 ** (0.4 * (zero_point + 48.6) )
    flux_with_noise = total_flux[a:len(total_flux)-b,a:len(total_flux)-b]*coeff + pixels_with_noise
    PSF = fits.open('psf_wfc3ir_f160w.fits')[0].data
    flux_with_noise_psf = signal.fftconvolve(flux_with_noise, PSF, mode='same')

    max_noise  = np.max(pixels_with_noise)

    plt.figure(2)
    plt.subplot(5,5,i+1)
    plt.imshow(flux_with_noise_psf, interpolation='nearest',cmap=plt.cm.gray)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.title( str(round(z_array[i],2)))

    plt.figure(3)
    plt.subplot(5,5,i+1)
    flux_std_psf = signal.fftconvolve(total_flux[a:len(total_flux)-b,a:len(total_flux)-b]*coeff/noise_std,PSF, mode='same')
    plt.imshow(flux_std_psf, interpolation='nearest',vmin=0, vmax=5)
    plt.yticks([])
    plt.xticks([])

    if(i==22):
        plt.colorbar(ticks=[0,1,2,3,4,5])

    plt.title( str(round(z_array[i],2)))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    return np.max(total_flux*coeff)

init()
filter_init()
compute()
