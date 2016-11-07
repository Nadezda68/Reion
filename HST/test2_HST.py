import glob
import numpy              as np
import matplotlib.pyplot  as plt
import argparse

from scipy.interpolate    import interp1d
from scipy                import integrate
from scipy                import signal
from scipy                import stats
from astropy.io           import fits

parser = argparse.ArgumentParser(description='Process integers.')
parser.add_argument('floats', metavar='N', type=int, nargs='+', help='a float for the accumulator')
args = parser.parse_args()
external_param = np.array(args.floats)

cm_in_pc            = 3.0857e18
sun_luminosity      = 3.828e33  # [erg/sec]
arcsec_in_rad       = 206265
speed_of_light      = 2.99792458e10  # [cm/sec]

Omega_lam           = 0.7274
Omega_M_0           = 0.2726
Omega_k             = 0.0
h                   = 0.704

E   = lambda x: 1/np.sqrt(Omega_M_0*np.power(1+x,3)+Omega_lam+Omega_k*np.power(1+x,2))
D_m = lambda x: D_c(x)
D_c = lambda x: (9.26e27/h)*integrate.quad(E, 0, x)[0]
D_A = lambda x: D_m(x)/(1+x)/cm_in_pc/1e6  # Angular distance [Mpc]


def init_lum_tables():

    global lam_list, lamb

    muf_list = sorted(glob.glob("data/drt/muv.bin*"))
    lam_list = np.zeros(len(muf_list))

    for i in range(len(muf_list)):

        f           = open(muf_list[i])
        header      = f.readline()
        lam_list[i] = float(header.split()[2])

        f.close()

    lamb = lam_list[np.argsort(lam_list)]


def read_simulation_data():

    global info, lums, z, nbins, nbins_min, z, indices

    info = glob.glob("./output/info_rei000" + str(N_sim_1) + str(N_sim_2) + "_*")
    lums = np.vstack((sorted(glob.glob("./output/lum_125_rei000" + str(N_sim_1) + str(N_sim_2) + "_*")),
                      sorted(glob.glob("./output/lum_140_rei000" + str(N_sim_1) + str(N_sim_2) + "_*")),
                      sorted(glob.glob("./output/lum_160_rei000" + str(N_sim_1) + str(N_sim_2) + "_*"))))

    z     = []
    nbins = []

    for i in range(0,len(info)):
        N, redshift, D_A, theta = np.loadtxt(info[i], skiprows=1)
        z.append(redshift)
        nbins.append(int(N))

    z       = np.array(z)
    indices = np.argsort(z)[::-1]
    z       = z[indices]

    nbins     = np.array(nbins)
    nbins     = nbins[indices]
    nbins_min = np.min(nbins)

    print(nbins_min)


def init_noise():

    global noise, noise_std

    noise = np.vstack((stats.norm.rvs(-0.00015609,0.00275845,nbins_min*nbins_min),
                       stats.norm.rvs(-3.24841830e-05,3.26572605e-03,nbins_min*nbins_min),
                       stats.norm.rvs(-0.00020178,0.00239519,nbins_min*nbins_min)))

    zero_point = np.array([26.23,26.45,25.94])
    coeff = 10 ** (0.4 * (zero_point + 48.6))

    noise[0,:] = noise[0,:] * 1e23 * 1e9 / coeff[0]
    noise[1,:] = noise[1,:] * 1e23 * 1e9 / coeff[1]
    noise[2,:] = noise[2,:] * 1e23 * 1e9 / coeff[2]

    noise_std = np.array([np.std(noise[0,:]),
                          np.std(noise[1,:]),
                          np.std(noise[2,:])])

    noise = np.dstack((np.reshape(noise[0,:],(nbins_min,nbins_min)),
                       np.reshape(noise[1,:],(nbins_min,nbins_min)),
                       np.reshape(noise[2,:],(nbins_min,nbins_min))))


def init_PSF():

    global PSF

    PSF = np.dstack((fits.open('data/PSFSTD_WFC3IR_F125W.fits')[0].data[1,:,:],
                     fits.open('data/PSFSTD_WFC3IR_F140W.fits')[0].data[1,:,:],
                     fits.open('data/PSFSTD_WFC3IR_F160W.fits')[0].data[1,:,:]))


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
            if(F_filter(x[i])>=0.5e-3):
                lambdas.append(x[i])

    return lambdas


def main(sim1,sim2):

    global input_data, lamb_filter, flux_noise_psf_std_sum, flux_psf_std_sum, N_sim_1, N_sim_2
    
    N_sim_1, N_sim_2 = sim1, sim2

    read_simulation_data()

    init_lum_tables()
    init_noise()
    init_PSF()

    filter_index           = 0
    flux_psf_std_sum       = np.zeros((nbins_min,nbins_min,len(z)))
    flux_noise_psf_std_sum = np.zeros((nbins_min,nbins_min,len(z)))

    for filter_name in ['125','140','160']:

        for i in range(0,len(z)):

            input_data  = np.load(lums[filter_index,i])
            lamb_filter = filter_init(filter_name,z[i])
            filter_flux(i,filter_index,filter_name)

            print(np.shape(input_data))
            print(z[i],nbins[i])

        plt.figure(2+3*filter_index)
        plt.savefig('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/figures/fig_f'+filter_name+'w_1.pdf', format='pdf')
        plt.figure(3+3*filter_index)
        plt.savefig('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/figures/fig_f'+filter_name+'w_2.pdf', format='pdf')
        plt.figure(4+3*filter_index)
        plt.savefig('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/figures/fig_f'+filter_name+'w_3.pdf', format='pdf')

        filter_index += 1

    for i in range(0,len(z)):

        if i<10:
            np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/processed_output/data_total_0_0' + str(i) +'.dat', flux_psf_std_sum[:,:,i],fmt='%1.5e')
            np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/processed_output/data_total_1_0' + str(i) +'.dat', flux_noise_psf_std_sum[:,:,i],fmt='%1.5e')
        else:
            np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/processed_output/data_total_0_' + str(i) +'.dat', flux_psf_std_sum[:,:,i],fmt='%1.5e')
            np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/processed_output/data_total_1_' + str(i) +'.dat', flux_noise_psf_std_sum[:,:,i],fmt='%1.5e')

        plt.figure(0)
        plt.subplot(5,5,i+1)
        plt.imshow(flux_psf_std_sum[:,:,i], interpolation='nearest',vmin=0, vmax=5)
        plt.yticks([])
        plt.xticks([])
        plt.title( str(round(z[i],2)))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.contour(flux_psf_std_sum[:,:,i], levels=np.array([2,5]), lw=0.4,colors='k')

        plt.figure(1)
        plt.subplot(5,5,i+1)
        plt.imshow(flux_noise_psf_std_sum[:,:,i], interpolation='nearest',vmin=0, vmax=5)
        plt.yticks([])
        plt.xticks([])
        plt.title( str(round(z[i],2)))
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.contour(flux_noise_psf_std_sum[:,:,i], levels=np.array([2,5]), lw=0.4,colors='k')

    plt.figure(0)
    plt.savefig('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/figures/fig_total_2.pdf', format='pdf')
    plt.figure(1)
    plt.savefig('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/figures/fig_total_3.pdf', format='pdf')

def filter_flux(i,l,name):

    nu = speed_of_light/(lamb_filter/1e8)
    total_flux = np.zeros_like(input_data[:,:,0])
    lum_dist = D_A(z[i]) * (1 + z[i]) * (1 + z[i])

    a = int((nbins[i] - nbins_min)/2)
 
    for j in range(0,len(total_flux[:,0])):
        for k in range(0,len(total_flux[:,0])):
            total_flux[j,k] = 1e23 * 1e9 * integrate.trapz( input_data[j,k,::-1] * F_filter(lamb_filter[::-1]*(1+z[i])), nu[::-1] ) / \
                             integrate.trapz( F_filter(lamb_filter[::-1]*(1+z[i])), nu[::-1] ) * \
                             (1+z[i]) / (4 * np.pi * np.power(lum_dist*cm_in_pc*1e6,2))

    flux_min_bin       = total_flux[a:a+nbins_min,a:a+nbins_min]
    flux_noise         = flux_min_bin + noise[:,:,l]
    flux_noise_psf     = signal.fftconvolve(flux_noise, PSF[:,:,l], mode='same')
    flux_psf_std       = signal.fftconvolve(flux_min_bin,PSF[:,:,l], mode='same')/noise_std[l]
    flux_noise_psf_std = flux_noise_psf/noise_std[l]

    plt.figure(2+3*l)
    plt.subplot(5,5,i+1)
    plt.imshow(flux_noise_psf, interpolation='nearest',cmap=plt.cm.gray)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.title( str(round(z[i],2)))

    plt.figure(3+3*l)
    plt.subplot(5,5,i+1)
    plt.imshow(flux_psf_std, interpolation='nearest',vmin=0, vmax=5)
    plt.yticks([])
    plt.xticks([])
    plt.title( str(round(z[i],2)))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.contour(flux_psf_std, levels=np.array([2,5]),lw=0.4, colors='k')

    plt.figure(4+3*l)
    plt.subplot(5,5,i+1)
    plt.imshow(flux_noise_psf_std, interpolation='nearest',vmin=0, vmax=5)
    plt.yticks([])
    plt.xticks([])
    plt.title( str(round(z[i],2)))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.contour(flux_noise_psf_std, levels=np.array([2,5]),lw=0.4, colors='k')

    flux_psf_std_sum[:,:,i]       += flux_psf_std
    flux_noise_psf_std_sum[:,:,i] += flux_noise_psf_std

    if(i<10):
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/processed_output/data_f' + name + 'w_1_0'+ str(i) +'.dat', flux_noise_psf,fmt='%1.5e')
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/processed_output/data_f' + name + 'w_2_0'+ str(i) +'.dat', flux_psf_std,fmt='%1.5e')
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/processed_output/data_f' + name + 'w_3_0'+ str(i) +'.dat', flux_noise_psf_std,fmt='%1.5e')
    else:
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/processed_output/data_f' + name + 'w_1_'+ str(i) +'.dat', flux_noise_psf,fmt='%1.5e')
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/processed_output/data_f' + name + 'w_2_'+ str(i) +'.dat', flux_psf_std,fmt='%1.5e')
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/processed_output/data_f' + name + 'w_2_'+ str(i) +'.dat', flux_noise_psf_std,fmt='%1.5e')

main(*external_param)
