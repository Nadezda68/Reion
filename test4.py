import yt
import glob
import numpy              as np
import matplotlib.pyplot  as plt
from scipy.interpolate    import interp1d
from scipy                import integrate
from scipy                import signal
from astropy.io           import fits


def init():

    global files,lums,lam_list,lamb

    #muf_list = glob.glob("/home/kaurov/kicp/code05A/drt/muv.bin*")
    #files = glob.glob("/home/kaurov/kicp/code05A/OUT/rei05B_a0*/rei05B_a*.art")
    muf_list = glob.glob("./drt/muv.bin*")
    files = glob.glob("./rei05B_a0*/rei05B_a*.art")
    lums  = glob.glob("./output/lum_*")
    sorted(files)
    sorted(lums)
    lam_list = np.zeros(len(muf_list))
    print(len(lam_list))

    for i in range(len(muf_list)-1):

        f = open(muf_list[i])
        header = f.readline()
        f.close()
        lam_list[i] = float(header.split()[2])

    lamb = lam_list[np.argsort(lam_list)]

def D_m(z):
    if(Omega_k>0):
        return  (9.26e27/h)/np.sqrt(Omega_k)*np.sinh(np.sqrt(Omega_k)*D_c(z)/(9.26e27/h))
    else:
        return D_c(z)

E = lambda x: 1/np.sqrt(Omega_M_0*np.power(1+x,3)+Omega_lam+Omega_k*np.power(1+x,2))
D_c = lambda z: (9.26e27/h)*integrate.quad(E, 0, z)[0]
D_A = lambda z: D_m(z)/(1+z)/3.0857e18/1e6

def compute():

    global redshift, Omega_lam, Omega_M_0, Omega_k, h, input_data,nbins

    max_flux    = []
    z_array     = []
    nbins_array = []
    lum_d_array = []


    for i in range(0,len(lums)):

        print(files[i])
        print(lums[i])

        pf = yt.load(files[i])
        redshift = pf.current_redshift
        Omega_lam = pf.omega_lambda
        Omega_M_0 = pf.omega_matter
        Omega_k   = 1 - Omega_lam - Omega_M_0
        h         = pf.hubble_constant
        input_data = np.load(lums[i])
        nbins = np.shape(input_data)[0]
        print(np.shape(input_data))

        luminosity(i)
        spectrum(i)
        max_flux.append(filter_flux(i)[0])
        lum_d_array.append(filter_flux(i)[1])
        z_array.append(redshift)
        nbins_array.append(nbins)


    plt.figure(0)
    plt.subplot(2,2,1)
    plt.title('Spectrum')

    plt.figure(0)
    plt.subplot(2,2,2)
    plt.title('Observed flux')
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.plot(z_array,max_flux,label='flux')
    plt.plot(z_array,np.ones(len(z_array))*max_noise,label='max noise')
    plt.plot(z_array,np.ones(len(z_array))*mean_noise,label='mean noise')
    plt.legend(prop={'size':10},loc=3)
    plt.xlabel('z')

    plt.subplot(2,2,3)
    plt.title('Luminosity distance')
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.plot(z_array,lum_d_array)
    plt.legend(prop={'size':10},loc=3)
    plt.xlabel('z')
    plt.ylabel('$D_{L}$ [Mpc]')

    plt.subplot(2,2,4)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.title('N pixels in image')
    plt.plot(z_array,nbins_array)
    plt.xlabel('z')
    plt.ylabel('N')


    plt.show()


def luminosity(i):

    nu = 2.99792458e10/(lamb/1e8)

    total_lum = np.zeros_like(input_data[:,:,0])

    for j in range(0,len(total_lum[:,0])):
        for k in range(0,len(total_lum[:,0])):
            total_lum[j,k] = integrate.trapz(input_data[j,k,::-1], nu[::-1])

    plt.figure(1)
    plt.subplot(5,4,i+1)
    plt.imshow(np.log10(total_lum), interpolation='nearest')
    plt.yticks([])
    plt.xticks([])
    plt.title('z=' + str(round(redshift,2)) + ',nbins=' + str(nbins))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.colorbar()


def spectrum(i):

    lum_nu = np.zeros_like(input_data[0,0,:])

    for j in range(0,len(lamb)):
        lum_nu[j] = np.max(input_data[:,:,j])

    plt.figure(0)
    plt.subplot(2,2,1)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.plot(lamb,np.log10(lum_nu),label='z=' + str(round(redshift,2)) + ',nbins=' + str(nbins))
    plt.xlabel('$\\lambda$ [Angstrom]')
    plt.ylabel('$log10(L_{\\nu})$ [erg/s/Hz]')
    plt.legend(prop={'size':10},loc=3)

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

    global max_noise,mean_noise

    nu = 2.99792458e10/(lamb/1e8)
    total_flux = np.zeros_like(input_data[:,:,0])
    lum_dist = D_A(redshift) * (1 + redshift) * (1 + redshift)
    print(D_A(redshift), lum_dist)

    for j in range(0,len(total_flux[:,0])):
        for k in range(0,len(total_flux[:,0])):
            total_flux[j,k] = integrate.trapz(input_data[j,k,::-1] * F_filter(lamb[::-1]*(1+redshift)), nu[::-1]) / \
                             integrate.trapz(F_filter(lamb[::-1]*(1+redshift)), nu[::-1]) * (1+redshift) / (4 * np.pi * np.power(lum_dist*3.0857e18*1e6,2))

    pixels_with_noise = fits.open('hlsp_hlf_hst_wfc3-60mas_goodss_f160w_v1.0_sci.fits')[0].data[25000-15258:25000-15258+nbins,10326:10326+nbins]
    zero_point = 25.94
    coeff = 10 ** (0.4 * (zero_point + 48.6) )
    flux_with_noise = total_flux*coeff + pixels_with_noise
    PSF = fits.open('psf_wfc3ir_f160w.fits')[0].data
    flux_with_noise_psf = signal.fftconvolve(flux_with_noise, PSF, mode='same')

    max_noise  = np.max(pixels_with_noise)
    mean_noise = np.mean(pixels_with_noise)

    plt.figure(2)
    plt.subplot(5,4,i+1)
    plt.imshow(flux_with_noise_psf, interpolation='nearest',cmap=plt.cm.gray)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.title('z=' + str(round(redshift,2)) + ',nbins=' + str(nbins))
    plt.colorbar()

    return np.max(flux_with_noise_psf),lum_dist

init()
filter_init()
compute()