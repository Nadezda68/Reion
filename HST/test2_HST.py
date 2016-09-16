import glob
import numpy              as np
import matplotlib.pyplot  as plt

from scipy.interpolate    import interp1d
from scipy                import integrate
from scipy                import signal
from scipy                import stats
from astropy.io           import fits


def init():

    global info,lums,lam_list,lamb,z,indices,nbins_min,nbins,PSF,noise,noise_std

    muf_list = glob.glob("/home/kaurov/kicp/code05A/drt/muv.bin*")
    info  = glob.glob("./output/info_*")
    lums  = np.vstack((sorted(glob.glob("./output/lum_125_*")),
                       sorted(glob.glob("./output/lum_140_*")),
                       sorted(glob.glob("./output/lum_160_*"))))

    lam_list = np.zeros(len(muf_list)-1)
    for i in range(len(muf_list)-1):
        f = open(muf_list[i])
        header = f.readline()
        f.close()
        lam_list[i] = float(header.split()[2])
    lamb = lam_list[np.argsort(lam_list)]

    z = []
    nbins = []
    for i in range(0,len(info)):
        N,redshift,D_A,theta = np.loadtxt(info[i],skiprows=1)
        z.append(redshift)
        N = int(N)
        nbins.append(N)

    z = np.array(z)
    indices = np.argsort(z)[::-1]
    z = z[indices]

    nbins = np.array(nbins)
    nbins = nbins[indices]
    nbins_min   = np.min(nbins)

    PSF = np.dstack((fits.open('data/PSFSTD_WFC3IR_F125W.fits')[0].data[1,:,:],
                     fits.open('data/PSFSTD_WFC3IR_F140W.fits')[0].data[1,:,:],
                     fits.open('data/PSFSTD_WFC3IR_F160W.fits')[0].data[1,:,:]))

    noise = np.vstack((stats.norm.rvs(-0.00015609,0.00275845,nbins_min*nbins_min),
                       stats.norm.rvs(-3.24841830e-05,3.26572605e-03,nbins_min*nbins_min),
                       stats.norm.rvs(-0.00020178,0.00239519,nbins_min*nbins_min)))

    zero_point = np.array([26.23,26.45,25.94])
    coeff = 10 ** ( 0.4 * (zero_point + 48.6) )

    noise[0,:] = noise[0,:] * 1e23 * 1e9 / coeff[0]
    noise[1,:] = noise[1,:] * 1e23 * 1e9 / coeff[1]
    noise[2,:] = noise[2,:] * 1e23 * 1e9 / coeff[2]

    noise_std = np.array([np.std(noise[0,:]),
                          np.std(noise[1,:]),
                          np.std(noise[2,:])])

    noise = np.dstack((np.reshape(noise[0,:],(nbins_min,nbins_min)),
                       np.reshape(noise[1,:],(nbins_min,nbins_min)),
                       np.reshape(noise[2,:],(nbins_min,nbins_min))))

E   = lambda x: 1/np.sqrt(Omega_M_0*np.power(1+x,3)+Omega_lam+Omega_k*np.power(1+x,2))
D_m = lambda z: D_c(z)
D_c = lambda z: (9.26e27/h)*integrate.quad(E, 0, z)[0]
D_A = lambda z: D_m(z)/(1+z)/3.0857e18/1e6

def compute():

    global Omega_lam, Omega_M_0, Omega_k, h, input_data, filter_flux_sum,lamb_filter

    Omega_lam    = 0.7274
    Omega_M_0    = 0.2726
    Omega_k      = 0.0
    h            = 0.704
    filter_index = 0

    filter_flux_sum = np.zeros((nbins_min,nbins_min,len(z)))

    for filter_name in ['125','140','160']:

        for i in range(0,len(z)):

            input_data = np.load(lums[filter_index,i])[:,:,:]
            print(np.shape(input_data))
            lamb_filter = filter_init(filter_name,z[i])
            filter_flux(i,filter_index,filter_name)
            print(z[i],nbins[i])

        plt.figure(1+2*filter_index)
        plt.savefig('fig_f'+filter_name+'w_1.pdf')

        plt.figure(2+2*filter_index)
        plt.savefig('fig_f'+filter_name+'w_2.pdf')

        filter_index += 1

    for i in range(0,len(z)):

        if i<10:
            np.savetxt('./output_processed/data_total_0' + str(i) +'.dat', filter_flux_sum[:,:,i],fmt='%1.5e')
        else:
            np.savetxt('./output_processed/data_total_' + str(i) +'.dat', filter_flux_sum[:,:,i],fmt='%1.5e')

        plt.figure(0)
        plt.subplot(4,5,i+1)
        plt.imshow(filter_flux_sum[:,:,i], interpolation='nearest',vmin=0, vmax=5)
        plt.yticks([])
        plt.xticks([])
        plt.title( str(round(z[i],2)))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.contour(filter_flux_sum[:,:,i], levels=np.array([2,5]), lw=0.4,colors='k')

    plt.figure(0)
    plt.savefig('fig_total.pdf')


def filter_init(name,z):

    global F_filter

    filter_b = np.loadtxt('data/filter_f' + name + 'w.dat')
    F_filter = interp1d(filter_b[:,0], filter_b[:,1],fill_value=0.0,bounds_error=False)
    a,b = np.min(filter_b[:,0]),np.max(filter_b[:,0])
    lamb_filter = filter_bandwidth(a,b,lamb*(1+z))/(1+z)

    return lamb_filter

def filter_bandwidth(a,b,x):

    global lambdas

    lambdas = []

    for i in range(0,len(x)):
        if (a<=x[i] and x[i]<=b):
            if(F_filter(x[i])>=1e-3):
                lambdas.append(x[i])

    return lambdas

def filter_flux(i,l,name):

    nu = 2.99792458e10/(lamb_filter/1e8)
    total_flux = np.zeros_like(input_data[:,:,0])
    lum_dist = D_A(z[i]) * (1 + z[i]) * (1 + z[i])

    a = int((nbins[i] - nbins_min)/2)
 
    for j in range(0,len(total_flux[:,0])):
        for k in range(0,len(total_flux[:,0])):
            total_flux[j,k] = 1e23 * 1e9 * integrate.trapz(input_data[j,k,::-1] * F_filter(lamb_filter[::-1]*(1+z[i])), nu[::-1]) / \
                             integrate.trapz(F_filter(lamb_filter[::-1]*(1+z[i])), nu[::-1]) * (1+z[i]) / (4 * np.pi * np.power(lum_dist*3.0857e18*1e6,2))

    flux_with_noise = total_flux[a:a+nbins_min,a:a+nbins_min] + noise[:,:,l]
    flux_with_noise_psf = signal.fftconvolve(flux_with_noise, PSF[:,:,l], mode='same')

    plt.figure(1+2*l)
    plt.subplot(4,5,i+1)
    plt.imshow(flux_with_noise_psf, interpolation='nearest',cmap=plt.cm.gray)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.title( str(round(z[i],2)))

    filter_flux_sum[:,:,i] += flux_with_noise_psf/noise_std[l]

    if(i<10):
        np.savetxt('./output_processed/data_f' + name + 'w_0'+ str(i) +'.dat', flux_with_noise_psf,fmt='%1.5e')
    else:
        np.savetxt('./output_processed/data_f' + name + 'w_'+ str(i) +'.dat', flux_with_noise_psf,fmt='%1.5e')

    plt.figure(2+2*l)
    plt.subplot(4,5,i+1)
    flux_std_psf = signal.fftconvolve(total_flux[a:a+nbins_min,a:a+nbins_min]/noise_std[l],PSF[:,:,l], mode='same')
    plt.imshow(flux_std_psf, interpolation='nearest',vmin=0, vmax=5)
    plt.yticks([])
    plt.xticks([])
    plt.title( str(round(z[i],2)))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.contour(flux_std_psf, levels=np.array([2,5]),lw=0.4, colors='k')

init()
compute()
