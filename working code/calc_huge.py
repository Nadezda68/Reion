import yt
import glob
import argparse
import numpy              as np

from scipy.interpolate    import interp1d
from scipy.interpolate    import interp2d
from scipy                import integrate
from scipy                import signal
from scipy                import stats
from astropy.io           import fits

parser = argparse.ArgumentParser(description='Process floats.')
parser.add_argument('floats', metavar='N', type=float, nargs='+', help='a float for the accumulator')
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


def init_input_data(sim, sim2,  sim_start, sim_stop, telescope_input='HST', filter_input='f140w', projection_input='all'):

    '''
    sim, sim2 - two digits of a number, that determines which ART simulation to load (e.g. for
        №10 simulation you input sim = 1 and sim2 = 0

    sim_start, sim_stop - two numbers, which determine the left and the right boundary of snapshots to process
        (e.g. if we have 20 snapshots for a specific ART sim and we want to process snapshots only from 3 to 10,
        then we input sim_start = 3, sim_stop = 11)

    telescope_input - either 'JWST' or 'HST'

    filter_input - 'F150W' and 'F200W' for JWST and 'f125w', 'f140w' and 'f160w' for HST
    '''

    global input_start, input_stop, N_sim, N_sim_2, telescope, filter_name, projection

    N_sim, N_sim_2, input_start, input_stop = int(sim), int(sim2), int(sim_start), int(sim_stop)
    filter_name = filter_input
    telescope = telescope_input
    projection = projection_input
    print('Simulation number = ', N_sim, N_sim_2)

def init_transmission_function():

    '''
    Interstellar medium transmission function
    '''

    global F_ISM

    table = np.loadtxt('data/table_transmition_ISM.dat')
    lam_rest   = table[1:,0]
    z          = table[0,1:]
    trans_coef = table[1:,1:]
    F_ISM = interp2d(z, lam_rest, trans_coef)


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


def filter_bandwidth(a, b, x):

    position_in_lam_array = []

    for i in range(0,len(x)):
        if a <=x[i] and x[i] <= b:
            if F_filter(x[i])>=0.5e-3:
                position_in_lam_array.append(i)

    return position_in_lam_array


def HST_filter_init(z):

    '''
    Hubble Space Telescope filter initialization function
    '''

    global F_filter

    filter_b = np.loadtxt('data/filter_' + filter_name + '.dat')
    F_filter = interp1d(filter_b[:,0], filter_b[:,1],fill_value=0.0,bounds_error=False)
    a,b = np.min(filter_b[:,0]),np.max(filter_b[:,0])
    lamb_positions = filter_bandwidth(a,b,lam_list*(1+z))

    return lamb_positions


def JWST_filter_init(z):

    '''
    James Webb Space Telescope filter initialization function
    '''

    global F_filter

    wavelengths, transmission = np.loadtxt('data/filter_' + filter_name + '.dat',skiprows=1).T
    wavelengths *= 1e4
    F_filter = interp1d(wavelengths, transmission,fill_value=0.0,bounds_error=False)
    a,b = np.min(wavelengths),np.max(wavelengths)
    lamb_positions = filter_bandwidth(a,b,lam_list*(1+z))

    return lamb_positions


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
        coeff_140 = 1e23 * 1e9 / coeff[1]

        noise = stats.norm.rvs(0.0, 3.26572605e-03*coeff_140,nbins*nbins)

        if filter_name == 'f140w':
            noise_std = np.std(noise)
            noise = np.reshape(noise,(nbins,nbins))

    elif telescope == 'JWST':

        if filter_name == 'F150W':

            noise_spread = JWST_F150W_noise_spread(exp_time)
            #noise_loc = 4.3934
            noise_loc = 0.0
            noise = stats.norm.rvs(noise_loc, noise_spread, nbins*nbins)
            noise_std = np.std(noise)
            noise = np.reshape(noise,(nbins,nbins))


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


def main():

    global nbins

    init_input_data(*external_param)
    init_lum_tables()
    init_transmission_function()

    files = sorted(glob.glob('/home/kaurov/scratch/rei/random10/OUT_000' + str(N_sim) + str(N_sim_2) + '_010_010_random/rei000'
                             + str(N_sim) + str(N_sim_2) + '_010_010_random_a0*/rei000' + str(N_sim) + str(N_sim_2) + '_010_010_random_a0.*.art'))

    for simulation in range(input_start,input_stop):

        print('Loading data %i out of [%i...%i] (%i in total)' % (simulation, input_start, input_stop, len(files)))

        pf = yt.load(files[simulation])
        data = pf.all_data()
   
        x = np.array(data[('STAR', 'POSITION_X')])
        y = np.array(data[('STAR', 'POSITION_Y')])
        z = np.array(data[('STAR', 'POSITION_Z')])
        m = data[('STAR', 'MASS')].in_units('msun')
        met = data[('STAR', 'METALLICITY_SNIa')].in_units('Zsun') + data[('STAR', 'METALLICITY_SNII')].in_units('Zsun')
        t = (data[('STAR', 'age')].in_units('yr'))

        erase = np.where(t<=0)[0]
        
        x = np.delete(x,erase)
        y = np.delete(y,erase)
        z = np.delete(z,erase)

        met = np.delete(met,erase)
        t = np.log10(np.delete(t,erase))
        m = np.delete(m,erase)

        print('number of objects', len(t))

        redshift = pf.current_redshift
        simulation_name = 'z' + str(np.round(redshift,2))
        theta_arcsec = (2 * 10 * 1e6) / (D_A(redshift) * 1e6) * arcsec_in_rad

        if telescope == 'HST':
            nbins = int(theta_arcsec / HST_WFC3CAM_pixel_size)
        elif telescope == 'JWST':
            nbins = int(theta_arcsec / JWST_NIRCAM_pixel_size)

        print('Simulation name = ',	simulation_name)
        print('z = %1.3e, D_A = %1.3e [Mpc], Nbins = %i' % (redshift, D_A(redshift), nbins))
        print('filter name:' + filter_name)

        if telescope == 'HST':
            lamb_positions = HST_filter_init(redshift)
        elif telescope == 'JWST':
            lamb_positions = JWST_filter_init(redshift)

        lookup_averaged = np.zeros((len(Z),len(logt)))
        lamb_filter = lam_list[lamb_positions]
        nu = c/(lamb_filter/1e8)
        lum_dist = D_A(redshift) * (1 + redshift) * (1 + redshift)
        init_noise(telescope=telescope,filter_name=filter_name)
        init_PSF(telescope=telescope,filter_name=filter_name)
        path = 'processed/sim' + str(N_sim) + str(N_sim_2) + '_random/'

        for ii in range(len(Z)):
            for jj in range(len(logt)):
                lookup_averaged[ii,jj] = integrate.trapz( lookup[ii, jj, lamb_positions][::-1] * F_ISM(redshift,lamb_filter)[::-1,0] * \
                                                          F_filter(lamb_filter[::-1]*(1+redshift))/nu[::-1], nu[::-1] ) * 1e23 * 1e9 * \
                                                          (1+redshift) / (4 * np.pi * np.power(lum_dist*cm_in_pc*1e6,2)) * sun_luminosity / \
                                         integrate.trapz( F_filter(lamb_filter[::-1]*(1+redshift))/nu[::-1], nu[::-1] )
        print('CHECKPOINT')

        xedges = np.linspace(0, 64, nbins+1)
        yedges = np.linspace(0, 64, nbins+1)

        interp = interp2d(Z, logt, lookup_averaged)
        print('INTERPOLATION IS COMPLETED')

        Flux = np.zeros_like(m)
        for j in range(0,len(m)):
            Flux[j] = interp(met[j], t[j])[0] * m[j]

        if projection == 'all':

            H, X1, X2 = np.histogram2d(y, z, bins=(xedges, yedges), weights = Flux)

            flux_noise = np.rot90(H) + noise
            # executing convolution
            flux_noise_psf     = signal.fftconvolve(flux_noise, PSF, mode='same')
            # executing convolution for image w/o noise
            flux_psf_std       = signal.fftconvolve(np.rot90(H), PSF, mode='same')/noise_std
            # converting in noise std units
            flux_noise_psf_std = flux_noise_psf/noise_std

            np.savetxt(path + 'data/sim_' + str(N_sim) + str(N_sim_2) + '_' + filter_name + '_' + simulation_name + '_x1' + '.dat', flux_noise_psf,fmt='%1.5e')
            np.savetxt(path + 'data/sim_' + str(N_sim) + str(N_sim_2) + '_' + filter_name + '_' + simulation_name + '_x2' + '.dat', flux_psf_std,fmt='%1.5e')
            np.savetxt(path + 'data/sim_' + str(N_sim) + str(N_sim_2) + '_' + filter_name + '_' + simulation_name + '_x3' + '.dat', flux_noise_psf_std,fmt='%1.5e')

            H, X1, X2 = np.histogram2d(z, x, bins=(xedges, yedges), weights = Flux)

            flux_noise = np.rot90(H) + noise
            # executing convolution
            flux_noise_psf     = signal.fftconvolve(flux_noise, PSF, mode='same')
            # executing convolution for image w/o noise
            flux_psf_std       = signal.fftconvolve(np.rot90(H), PSF, mode='same')/noise_std
            # converting in noise std units
            flux_noise_psf_std = flux_noise_psf/noise_std
            
            np.savetxt(path + 'data/sim_' + str(N_sim) + str(N_sim_2) + '_' + filter_name + '_' + simulation_name + '_y1' + '.dat', flux_noise_psf,fmt='%1.5e')
            np.savetxt(path + 'data/sim_' + str(N_sim) + str(N_sim_2) + '_' + filter_name + '_' + simulation_name + '_y2' + '.dat', flux_psf_std,fmt='%1.5e')
            np.savetxt(path + 'data/sim_' + str(N_sim) + str(N_sim_2) + '_' + filter_name + '_' + simulation_name + '_y3' + '.dat', flux_noise_psf_std,fmt='%1.5e')

            H, X1, X2 = np.histogram2d(x, y, bins=(xedges, yedges), weights = Flux)

            flux_noise = np.rot90(H) + noise
            # executing convolution
            flux_noise_psf     = signal.fftconvolve(flux_noise, PSF, mode='same')
            # executing convolution for image w/o noise
            flux_psf_std       = signal.fftconvolve(np.rot90(H), PSF, mode='same')/noise_std
            # converting in noise std units
            flux_noise_psf_std = flux_noise_psf/noise_std
            
            np.savetxt(path + 'data/sim_' + str(N_sim) + str(N_sim_2) + '_' + filter_name + '_' + simulation_name + '_z1' + '.dat', flux_noise_psf,fmt='%1.5e')
            np.savetxt(path + 'data/sim_' + str(N_sim) + str(N_sim_2) + '_' + filter_name + '_' + simulation_name + '_z2' + '.dat', flux_psf_std,fmt='%1.5e')
            np.savetxt(path + 'data/sim_' + str(N_sim) + str(N_sim_2) + '_' + filter_name + '_' + simulation_name + '_z3' + '.dat', flux_noise_psf_std,fmt='%1.5e')

        np.savetxt('output/info_' + simulation_name + '.dat', np.array([redshift,D_A(redshift),theta_arcsec]), header='Redshift, Angular distance [Mpc], theta [arc-sec]')

main()