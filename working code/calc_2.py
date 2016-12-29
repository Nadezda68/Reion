import glob
import numpy              as np
import matplotlib.pyplot  as plt
import argparse

from scipy.interpolate    import interp1d
from scipy                import integrate
from scipy                import signal
from scipy                import stats
from astropy.io           import fits

parser = argparse.ArgumentParser(description='Process floats.')
parser.add_argument('floats', metavar='N', type=int, nargs='+', help='a float for the accumulator')
args = parser.parse_args()
external_param = np.array(args.floats)

# detectors' pixel size
HST_WFC3CAM_pixel_size = 0.13   # arcsec per pixel
JWST_NIRCAM_pixel_size = 0.032  # arcsec per pixel

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


def init_lum_tables():

    '''
    Luminosity tables as a function of metallicity Z and stars' birth time t
    '''

    global lam_list, lookup, Z, logt

    muf_list = sorted(glob.glob("data/drt/muv.bin*"))
    lam_list = np.zeros(len(muf_list))
    lookup = np.zeros([len(muf_list), 188, 22])

    for i in range(len(muf_list)):
        f = open(muf_list[i])
        header = f.readline()
        lam_list[i] = float(header.split()[2])

        f.close()

        data = np.genfromtxt(muf_list[i], skip_header=1)
        lookup[i, :, :] = data[1:,1:]

    Z = data[0, 1:]  # metallicity [Sun_Z]
    logt = data[1:, 0]  # log10(t) [yr]


def filter_bandwidth(a,b,x):

    position_in_lam_array = []

    for i in range(0,len(x)):
        if (a<=x[i] and x[i]<=b):
            if(F_filter(x[i])>=0.5e-3):
                position_in_lam_array.append(i)

    return position_in_lam_array


def HST_filter_init(z, filter_name):

    '''
    Hubble Space Telescope filter initialization function
    '''

    global F_filter

    filter_b = np.loadtxt('data/filter_' + filter_name + '.dat')
    F_filter = interp1d(filter_b[:,0], filter_b[:,1],fill_value=0.0,bounds_error=False)
    a,b = np.min(filter_b[:,0]),np.max(filter_b[:,0])
    lamb_positions = filter_bandwidth(a,b,lam_list*(1+z))

    return lamb_positions


def JWST_filter_init(z, filter_name):

    '''
    James Webb Space Telescope filter initialization function
    '''

    global F_filter

    wavelengths,transmission = np.loadtxt('data/filter_' + filter_name + '.dat',skiprows=1).T
    wavelengths *= 1e4
    F_filter = interp1d(wavelengths, transmission,fill_value=0.0,bounds_error=False)
    a,b = np.min(wavelengths),np.max(wavelengths)
    lamb_positions = filter_bandwidth(a,b,lam_list*(1+z))

    return lamb_positions


def read_simulation_data(filter_name='f140w', telescope='HST'):

    global info, lums, z, nbins, nbins_min, z, indices, theta_min

    info = sorted(glob.glob('./output/info_rei000' + str(N_sim_1) + str(N_sim_2) + '_*'))
    lums = sorted(glob.glob('./output/lum_' + filter_name + '_rei000' + str(N_sim_1) + str(N_sim_2) + '_*'))

    z = []
    nbins = []
    angles = []

    for i in range(0,len(info)):
        z_input, DA_input, theta_input, c_x, c_y, c_z, r = np.loadtxt(info[i], skiprows=1)
        z.append(z_input)
        if telescope == 'HST':
            nbins.append(int(theta_input / HST_WFC3CAM_pixel_size))
        elif telescope == 'JWST':
            nbins.append(int(theta_input / JWST_NIRCAM_pixel_size))
        angles.append(theta_input)

    z = np.array(z)
    indices = np.argsort(z)[::-1]
    z = z[indices]
    nbins = np.array(nbins)
    nbins = nbins[indices]
    nbins_min = np.min(nbins)
    angles = np.array(angles)
    theta_min = np.min(angles)

    print(nbins_min)


def JWST_F150W_noise_spread(x):

    '''
    x - exposure time in seconds
    '''

    a,b,c,d,e = -0.35867483, 9.1144785, -75.60172247, 256.44825309, -275.1335303
    X = np.log10(x)
    return a + b/X + c/X/X + d/X/X/X + e/X/X/X/X


def init_noise(filter_name='f140w', telescope='HST',exp_time=31536000):

    '''
    Procedure to create noise for specific filter and exposure time (only for JWST)
    '''

    global noise, noise_std

    if telescope == 'HST':

        zero_point = np.array([26.23,26.45,25.94])
        coeff = 10 ** (0.4 * (zero_point + 48.6))

        coeff_125 = 1e23 * 1e9 / coeff[0]
        coeff_140 = 1e23 * 1e9 / coeff[1]
        coeff_160 = 1e23 * 1e9 / coeff[2]

        noise = np.vstack((stats.norm.rvs(-0.00015609*coeff_125,0.00275845*coeff_125,nbins_min*nbins_min),
                           stats.norm.rvs(-3.24841830e-05*coeff_140,3.26572605e-03*coeff_140,nbins_min*nbins_min),
                           stats.norm.rvs(-0.00020178*coeff_160,0.00239519*coeff_160,nbins_min*nbins_min)))

        if filter_name == 'f125w':
            noise_std = np.std(noise[0,:])
            noise = np.reshape(noise[0,:],(nbins_min,nbins_min))
        elif filter_name == 'f140w':
            noise_std = np.std(noise[1,:])
            noise = np.reshape(noise[1,:],(nbins_min,nbins_min))
        elif filter_name == 'f160w':
            noise_std = np.std(noise[2,:])
            noise = np.reshape(noise[2,:],(nbins_min,nbins_min))

    elif telescope == 'JWST':

        if filter_name == 'F150W':

            noise_spread = JWST_F150W_noise_spread(exp_time)
            #noise_loc = 4.3934
            noise_loc = 0.0
            noise = stats.norm.rvs(noise_loc, noise_spread, nbins_min*nbins_min)
            noise_std = np.std(noise)
            noise = np.reshape(noise,(nbins_min,nbins_min))


def init_PSF(filter_name='f140w', telescope='HST'):

    '''
    Point Spread function initialization function
    '''

    global PSF

    if telescope == 'HST':

        a = fits.open('data/psf_test_f125w00_psf.fits')[0].data[1:,1:]
        b = fits.open('data/psf_test_f140w00_psf.fits')[0].data[1:,1:]
        c = fits.open('data/psf_test_f160w00_psf.fits')[0].data[1:,1:]

        coords_a = np.linspace(-1.5 + 1.5/np.shape(a)[0],1.5 - 1.5/np.shape(a)[0],np.shape(a)[0])
        coords_b = np.linspace(-1.5 + 1.5/np.shape(b)[0],1.5 - 1.5/np.shape(b)[0],np.shape(b)[0])
        coords_c = np.linspace(-1.5 + 1.5/np.shape(c)[0],1.5 - 1.5/np.shape(c)[0],np.shape(c)[0])

        NNbins = int(3/HST_WFC3CAM_pixel_size)
        pix_edges = np.linspace(-1.5,1.5,NNbins+1)

        x, y = np.meshgrid(coords_a, coords_a)
        psf_f125w,X,Y = np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = a.flatten())
        x, y = np.meshgrid(coords_b, coords_b)
        psf_f140w,X,Y = np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = b.flatten())
        x, y = np.meshgrid(coords_c, coords_c)
        psf_f160w,X,Y = np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = c.flatten())

        if filter_name == 'f125w':
            PSF = psf_f125w
        elif filter_name == 'f140w':
            PSF = psf_f140w
        elif filter_name == 'f160w':
            PSF = psf_f160w

    elif telescope == 'JWST':

        A = fits.open('data/PSF_NIRCam_F150W.fits')[0].data
        B = fits.open('data/PSF_NIRCam_F200W.fits')[0].data

        coords_A = np.linspace(-1.5 + 1.5/np.shape(A)[0],1.5 - 1.5/np.shape(A)[0],np.shape(A)[0])
        coords_B = np.linspace(-1.5 + 1.5/np.shape(B)[0],1.5 - 1.5/np.shape(B)[0],np.shape(B)[0])

        NNbins = int(3/JWST_NIRCAM_pixel_size)
        pix_edges  = np.linspace(-1.5,1.5,NNbins+1)

        x, y = np.meshgrid(coords_A, coords_A)
        psf_F150W,X,Y = np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = A.flatten())
        x, y = np.meshgrid(coords_B, coords_B)
        psf_F200W,X,Y = np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = B.flatten())

        if filter_name == 'F150W':
            PSF = psf_F150W
        elif filter_name == 'F200W':
            PSF = psf_F200W


def plot(obj, index_1, index_2, contours=0):

    '''
    plotting function (was built for convenience)
    '''

    plt.figure(index_1)
    plt.subplot(5,6,index_2+1)
    if(contours):
         plt.imshow(obj, interpolation='nearest',vmin=0, vmax=5)
         plt.contour(obj, levels=np.array([2,5]),lw=0.4, colors='k')
    else:
        plt.imshow(obj, interpolation='nearest',cmap=plt.cm.gray)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.title(str(round(z[index_2],2)))


def main(sim1,sim2,prj_start,prj_stop):

    telescope = 'JWST'
    filter_name = 'F150W'

    global input_data, lamb_filter, N_sim_1, N_sim_2, prj, path
    
    N_sim_1, N_sim_2 = sim1, sim2

    path = 'processed/sim' + str(N_sim_1) + str(N_sim_2) + '/'
    prj = np.array(['_x','_y','_z'])

    # initializing data
    read_simulation_data(telescope=telescope,filter_name=filter_name)
    init_lum_tables()
    init_noise(telescope=telescope,filter_name=filter_name)
    init_PSF(telescope=telescope,filter_name=filter_name)

    for i in range(0,len(z)):

        for projection in range(prj_start,prj_stop):

            input_data = np.load(lums[i])[:,:,:,projection]
            if telescope == 'HST':
                lamb_filter = lam_list[HST_filter_init(z[i],filter_name=filter_name)]
            elif telescope == 'JWST':
                lamb_filter = lam_list[JWST_filter_init(z[i],filter_name=filter_name)]

            filter_flux(i,filter_name,projection)

        print(np.shape(input_data))
        print(z[i],nbins[i])

    # saving images
    for p in range(prj_start,prj_stop):
        plt.figure(0+p*3)
        plt.savefig(path + 'sim_' + str(N_sim_1) + str(N_sim_2) + '_' + filter_name + prj[p] + '1.pdf', format='pdf')
        plt.figure(1+p*3)
        plt.savefig(path + 'sim_' + str(N_sim_1) + str(N_sim_2) + '_' + filter_name + prj[p] + '2.pdf', format='pdf')
        plt.figure(2+p*3)
        plt.savefig(path + 'sim_' + str(N_sim_1) + str(N_sim_2) + '_' + filter_name + prj[p] + '3.pdf', format='pdf')

def filter_flux(i,filter_name, p):

    '''
    procedure to compute flux through filter bandwidth given luminosity and redshift
    '''

    nu = c/(lamb_filter/1e8)
    total_flux = np.zeros_like(input_data[:,:,0])
    lum_dist = D_A(z[i]) * (1 + z[i]) * (1 + z[i])

    a = int((nbins[i] - nbins_min)/2)

    # computing flux
    for j in range(0,len(total_flux[:,0])):
        for k in range(0,len(total_flux[:,0])):
            total_flux[j,k] = 1e23 * 1e9 * integrate.trapz( input_data[j, k, ::-1] * F_filter(lamb_filter[::-1]*(1+z[i]))/nu[::-1], nu[::-1] ) / \
                             integrate.trapz( F_filter(lamb_filter[::-1]*(1+z[i]))/nu[::-1], nu[::-1] ) * \
                             (1+z[i]) / (4 * np.pi * np.power(lum_dist*cm_in_pc*1e6,2))

    flux_min_bin = total_flux[a:a+nbins_min,a:a+nbins_min]
    # adding noise
    flux_noise = flux_min_bin + noise
    # executing convolution
    flux_noise_psf     = signal.fftconvolve(flux_noise, PSF, mode='same')
    # executing convolution for image w/o noise
    flux_psf_std       = signal.fftconvolve(flux_min_bin, PSF, mode='same')/noise_std
    # converting in noise std units
    flux_noise_psf_std = flux_noise_psf/noise_std

    # drawing images
    plot(obj=flux_noise_psf          ,index_1=(0+p*3),index_2=i, contours=0)
    plot(obj=flux_psf_std            ,index_1=(1+p*3),index_2=i, contours=1)
    plot(obj=flux_noise_psf_std      ,index_1=(2+p*3),index_2=i, contours=1)

    # saving data
    if(i<10):
        np.savetxt(path + 'data/sim_' + str(N_sim_1) + str(N_sim_2) + '_' + filter_name + prj[p] + '1_0' + str(i) +'.dat', flux_noise_psf,fmt='%1.5e')
        np.savetxt(path + 'data/sim_' + str(N_sim_1) + str(N_sim_2) + '_' + filter_name + prj[p] + '2_0' + str(i) +'.dat', flux_psf_std,fmt='%1.5e')
        np.savetxt(path + 'data/sim_' + str(N_sim_1) + str(N_sim_2) + '_' + filter_name + prj[p] + '3_0' + str(i) +'.dat', flux_noise_psf_std,fmt='%1.5e')
    else:
        np.savetxt(path + 'data/sim_' + str(N_sim_1) + str(N_sim_2) + '_' + filter_name + prj[p] + '1_' + str(i) +'.dat', flux_noise_psf,fmt='%1.5e')
        np.savetxt(path + 'data/sim_' + str(N_sim_1) + str(N_sim_2) + '_' + filter_name + prj[p] + '2_' + str(i) +'.dat', flux_psf_std,fmt='%1.5e')
        np.savetxt(path + 'data/sim_' + str(N_sim_1) + str(N_sim_2) + '_' + filter_name + prj[p] + '3_' + str(i) +'.dat', flux_noise_psf_std,fmt='%1.5e')

main(*external_param)
