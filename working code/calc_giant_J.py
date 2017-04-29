import yt
import glob
import argparse
import numpy as np

from scipy.interpolate    import interp1d
from scipy.interpolate    import interp2d
from scipy                import integrate
from scipy                import signal
from scipy                import stats
from astropy.io           import fits

from photutils import detect_sources
from photutils import CircularAperture
from photutils import aperture_photometry

parser = argparse.ArgumentParser(description='Process floats.')
parser.add_argument('floats', metavar='N', type=float, nargs='+', help='a float for the accumulator')
args = parser.parse_args()
external_param = np.array(args.floats)

# detectors' pixel size
JWST_NIRCAM_pixel_size = 0.032  # arcsec per pixel

# Constants 0
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


def init_input_data(nsim, n_cyl1, n_cyl2, orient=0):

    global N_sim_2, telescope, filter_name, prj, N1, N2

    N1, N2 = int(n_cyl1), int(n_cyl2)
    N_sim_2 = int(nsim)

    if orient==0:
        prj='x'
    elif orient==1:
        prj='y'
    elif orient==2:
        prj='z'


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


def JWST_filter_init(z, filter_name):

    '''
    James Webb Space Telescope filter initialization function
    '''

    global F_filter

    wavelengths, transmission = np.loadtxt('data/filter_' + filter_name + '2.dat',skiprows=1).T
    wavelengths *= 1e4
    F_filter = interp1d(wavelengths, transmission,fill_value=0.0,bounds_error=False)
    a,b = np.min(wavelengths),np.max(wavelengths)
    lamb_positions = filter_bandwidth(a,b,lam_list*(1+z))

    return lamb_positions

# TO CREATE MORE UNIFORM NOISE
def noise_adv_HST():

    i = 0
    while i < 1000:

        init_noise()
        init_PSF()
        f115 = signal.fftconvolve(noise115, PSF115, mode='same')/noise_std115
        f150 = signal.fftconvolve(noise150, PSF150, mode='same')/noise_std150
        f200 = signal.fftconvolve(noise200, PSF200, mode='same')/noise_std200
        fsum = f115 + f150 + f200
        inf = detect_sources(fsum, 2.3, 3)

        print(nbins, inf.nlabels)

        if inf.nlabels == 0:
            print('CONDITION IS SATISFIED')
            break

        i += 1


def JWST_noise_spread(x=(30*60*60), filt='none'):

    '''
    x - exposure time in seconds
    '''
    
    if filt == '150':
        a, b, c, d, e = -0.35867483, 9.1144785, -75.60172247, 256.44825309, -275.1335303
        print('150')
    elif filt == '115':
        a, b, c, d, e = -0.38425561, 9.91023494, -83.24295973, 285.06457776, -306.59387461
        print('115')
    elif filt == '200':
        a, b, c, d, e = -0.36536499, 9.14144209, -74.73426839, 250.6354125, -267.83637018
        print('200')
        
    X = np.log10(x)
    return a + b/X + c/X/X + d/X/X/X + e/X/X/X/X


def init_noise(exp_time=(30*60*60)):

    '''
    Procedure to create noise for specific filter and exposure time (only for JWST)
    '''

    global noise115, noise150, noise200, noise_std115, noise_std150, noise_std200

    noise_loc = 0.0
    
    noise_spread = JWST_noise_spread(exp_time,filt='115')
    noise = stats.norm.rvs(noise_loc, noise_spread, nbins*nbins)
    noise_std115 = np.std(noise)
    noise115 = np.reshape(noise,(nbins,nbins))
    
    noise_spread = JWST_noise_spread(exp_time,filt='150')
    noise = stats.norm.rvs(noise_loc, noise_spread, nbins*nbins)
    noise_std150 = np.std(noise)
    noise150 = np.reshape(noise,(nbins,nbins))
    
    noise_spread = JWST_noise_spread(exp_time,filt='200')
    noise = stats.norm.rvs(noise_loc, noise_spread, nbins*nbins)
    noise_std200 = np.std(noise)
    noise200 = np.reshape(noise,(nbins,nbins))


def init_PSF():

    '''
    Point Spread function initialization function
    '''

    global PSF115, PSF150, PSF200

    A = fits.open('data/PSF_NIRCam_F115W.fits')[0].data
    B = fits.open('data/PSF_NIRCam_F150W.fits')[0].data
    C = fits.open('data/PSF_NIRCam_F200W.fits')[0].data

    coords_A = np.linspace(-1.5 + 1.5/np.shape(A)[0],1.5 - 1.5/np.shape(A)[0],np.shape(A)[0])
    coords_B = np.linspace(-1.5 + 1.5/np.shape(B)[0],1.5 - 1.5/np.shape(B)[0],np.shape(B)[0])
    coords_C = np.linspace(-1.5 + 1.5/np.shape(C)[0],1.5 - 1.5/np.shape(C)[0],np.shape(C)[0])

    JWST_NIRCAM_pixel_size = 0.032  # arcsec per pixel
    NNbins = int(3/JWST_NIRCAM_pixel_size)
    pix_edges  = np.linspace(-1.5,1.5,NNbins+1)

    x, y = np.meshgrid(coords_A, coords_A)
    psf_F115W,X,Y =np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = A.flatten())
    x, y = np.meshgrid(coords_B, coords_B)
    psf_F150W,X,Y =np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = B.flatten())
    x, y = np.meshgrid(coords_C, coords_C)
    psf_F200W,X,Y =np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = C.flatten())

    PSF115 = psf_F115W
    PSF150 = psf_F150W
    PSF200 = psf_F200W


def fluxes(temp, data_sources, N, X, Y, group=0, dist_max=3.0):

    if N > 1:

        apertures_values = np.zeros(N)
        params = np.zeros(N)
        x_coord = np.zeros(N)
        y_coord = np.zeros(N)

        # fluxes
        for j in range(1, N+1):  # coords of all the sources

            A = np.argwhere(data_sources==j)
            x_coord[j-1] = np.mean(X[A[:, 0], A[:, 1]])
            y_coord[j-1] = np.mean(Y[A[:, 0], A[:, 1]])

        for i in range(0, N-1):  # for every sources except the last in matrix NxN

            A = np.sqrt(np.power((x_coord[i+1:]-x_coord[i]), 2) + np.power((y_coord[i+1:]-y_coord[i]), 2))  # find distances to the sources to the right
            B = np.where(A <= dist_max)  # choose those with dist <= dist_max

            for ii in range(len(B[0])):  # counting
                params[i] = 1  # 1st paired object (main)
                params[i+1+B[0][ii]] = 1  # 2nd paired object

        # apertures
        for j in range(1, N+1):
            A = np.argwhere(data_sources == j)
            if np.mean(A[:,0]) <= 1 or np.mean(A[:,1]) <= 1:
                apertures_values[j-1] = temp[A[0,0], A[0,1]]
            else:
                aperture = CircularAperture([np.mean(A[:,1]), np.mean(A[:, 0])], r=np.sqrt(len(A)))
                flux = aperture_photometry(temp, aperture)
                apertures_values[j-1] = flux['aperture_sum']

        if group:
            return apertures_values[np.where(params > 0)[0]]
        else:
            return apertures_values[np.where(params == 0)[0]]

    elif N == 1:

        A = np.argwhere(data_sources == 1)
        aperture = CircularAperture([np.mean(A[:,1]),np.mean(A[:,0])], r=np.sqrt(len(A)))
        flux = aperture_photometry(temp, aperture)
        apertures_value = flux['aperture_sum']

        if group:
            return []
        else:
            return apertures_value

    else:
        return []


def main():

    global nbins

    # INITIALIZING DATA
    init_input_data(*external_param)
    init_lum_tables()
    init_transmission_function()

    # LOADING DATA
    files = sorted(glob.glob('/scratch/kaurov/40/rei40_a0.*/rei40_a0.*.art'))
    path = 'processed_giant/JWST/' + prj + '/'
    telescope = 'JWST'

    print('path', path)
    print(len(files))

    # BOX SIZES
    x_length = 8  # code length
    y_length = 8  # code length
    z_length = 8  # code length

    # LOADING DIFFERENT SNAPSHOTS OF ART SIMULATION (at different redshifts)
    for simulation in range(N_sim_2, N_sim_2+1):

        # ART SIM LOAD
        pf = yt.load(files[simulation])

        # SIMULATION INFO
        print('Loading data %i out of (%i in total)' % (simulation, len(files)))
        print(files[simulation])
        print('box size in each dimension', pf.domain_right_edge[0]/32)
        print('box size in each dimension', pf.domain_right_edge.in_units('kpc')[0]/32)

        # EXTRACTING REDSHIFT AND COMPUTING ANGULAR SIZE OF THE BOX
        redshift = pf.current_redshift
        simulation_name = str(simulation)
        theta_arcsec = (1e3 * pf.domain_right_edge.in_units('kpc')[0]/32) / (D_A(redshift) * 1e6) * arcsec_in_rad

        # CREATION NOISE AND PSF FOR A SPECIFIC CAMERA (TELESCOPE)
        nbins = int(theta_arcsec / JWST_NIRCAM_pixel_size)
        noise_adv_HST()

        # CREATING COORDS MESH
        ang = theta_arcsec / 2
        pixels_ang_coords = (np.linspace(-ang, ang, nbins + 1)[1:] + np.linspace(-ang, ang, nbins + 1)[:-1]) / 2
        X, Y = np.meshgrid(pixels_ang_coords, pixels_ang_coords)

        # THRESHOLD FOR DETECTION
        npixels = 3

        # WRITING INFO INTO A FILE
        info = open(path + 'info_' + simulation_name + '.dat', 'a')
        info.write('--------------------\n')
        info.write('Telescope: JWST\n')
        info.write('Simulation name: %d \n' % simulation)
        info.write('Redshift: %.6f \n' % redshift)
        info.write('N of boxes to process: %d, %d\n' % (N1, N2))
        info.write('Theta [arcsec]: %.6f \n' % theta_arcsec)
        info.write('Flux threshold: %.3f, %.3f, %.3f, %.3f, %.3f \n' % (2.5, 2.75, 3.0, 3.5, 4.0))
        info.write('Npix: %.3f \n' % npixels)
        info.close()

        # PRINTING OUT INFO
        print('Simulation name = ', simulation_name)
        print('z = %1.3e, D_A = %1.3e [Mpc], Nbins = %i' % (redshift, D_A(redshift), nbins))
        print('nbins = ', nbins)

        # LUMINOSITY DISTANCE = ANGULAR DISTANCE (1+z)**2
        lum_dist = D_A(redshift) * (1 + redshift) * (1 + redshift)

        # INTERPOLATION
        lookup_averaged = np.zeros((len(logt), len(Z), 3))

        for filter_name, filter_idx in zip(['F115W', 'F150W', 'F200W'], [0, 1, 2]):

            lamb_positions = JWST_filter_init(filter_name=filter_name, z=redshift)
            lamb_filter = lam_list[lamb_positions]
            nu = c / (lamb_filter / 1e8)

            for ii in range(len(Z)):
                for jj in range(len(logt)):
                    lookup_averaged[jj, ii, filter_idx] = integrate.trapz(
                        lookup[lamb_positions, jj, ii][::-1] * F_ISM(redshift, lamb_filter)[::-1, 0] * \
                        F_filter(lamb_filter[::-1] * (1 + redshift)) / nu[::-1], nu[::-1]) * 1e23 * 1e9 * \
                                                          (1 + redshift) / (
                                                          4 * np.pi * np.power(lum_dist * cm_in_pc * 1e6,
                                                                               2)) * sun_luminosity / \
                                                          integrate.trapz(
                                                              F_filter(lamb_filter[::-1] * (1 + redshift)) / nu[::-1],
                                                              nu[::-1])
            print(filter_name, filter_idx)

        interp115 = interp2d(Z, logt, lookup_averaged[:, :, 0])
        interp150 = interp2d(Z, logt, lookup_averaged[:, :, 1])
        interp200 = interp2d(Z, logt, lookup_averaged[:, :, 2])

        print('INTERPOLATION IS COMPLETED')

        # COMPUTATION STARTS... (4*4*4 BOXES [L=16] FULL BOX SIZE: [0:64,0:64,0:64])

        for r_x in partition[N1:N2]:
            for r_y in partition:
                for r_z in partition:

                    data = pf.box([r_x, r_y, r_z], [r_x + x_length, r_y + y_length, r_z + z_length])

                    x = np.array(data[('STAR', 'POSITION_X')] - data.center[0])
                    y = np.array(data[('STAR', 'POSITION_Y')] - data.center[1])
                    z = np.array(data[('STAR', 'POSITION_Z')] - data.center[2])

                    print('left edge', data.left_edge)
                    print('center', data.center)
                    print('right edge', data.right_edge)

                    m = data[('STAR', 'MASS')].in_units('msun')
                    met = data[('STAR', 'METALLICITY_SNIa')].in_units('Zsun') + data[('STAR', 'METALLICITY_SNII')].in_units('Zsun')
                    t = (data[('STAR', 'age')].in_units('yr'))

                    erase = np.where(t <= 0)[0]

                    x = np.delete(x, erase)
                    y = np.delete(y, erase)
                    z = np.delete(z, erase)

                    met = np.delete(met, erase)
                    t = np.log10(np.delete(t, erase))
                    m = np.delete(m, erase)

                    print('number of objects', len(t))

                    xedges = np.linspace(-x_length / 2, x_length / 2, nbins + 1)
                    yedges = np.linspace(-y_length / 2, y_length / 2, nbins + 1)

                    Flux = np.zeros((len(m), 3))

                    for j in range(0, len(m)):
                        Flux[j, 0] = interp115(met[j], t[j])[0] * m[j]
                        Flux[j, 1] = interp150(met[j], t[j])[0] * m[j]
                        Flux[j, 2] = interp200(met[j], t[j])[0] * m[j]

                    if prj == 'x':
                        J115, X1, X2 = np.histogram2d(y, z, bins=(xedges, yedges), weights=Flux[:, 0])
                        J150, X1, X2 = np.histogram2d(y, z, bins=(xedges, yedges), weights=Flux[:, 1])
                        J200, X1, X2 = np.histogram2d(y, z, bins=(xedges, yedges), weights=Flux[:, 2])
                    elif prj == 'y':
                        J115, X1, X2 = np.histogram2d(z, x, bins=(xedges, yedges), weights=Flux[:, 0])
                        J150, X1, X2 = np.histogram2d(z, x, bins=(xedges, yedges), weights=Flux[:, 1])
                        J200, X1, X2 = np.histogram2d(z, x, bins=(xedges, yedges), weights=Flux[:, 2])
                    elif prj == 'z':
                        J115, X1, X2 = np.histogram2d(x, y, bins=(xedges, yedges), weights=Flux[:, 0])
                        J150, X1, X2 = np.histogram2d(x, y, bins=(xedges, yedges), weights=Flux[:, 1])
                        J200, X1, X2 = np.histogram2d(x, y, bins=(xedges, yedges), weights=Flux[:, 2])

                    flux_noise115 = np.rot90(J115) + noise115
                    flux_noise_PSF115 = signal.fftconvolve(flux_noise115, PSF115, mode='same')
                    flux_noise_psf_std115 = flux_noise_PSF115 / noise_std115

                    flux_noise150 = np.rot90(J150) + noise150
                    flux_noise_PSF150 = signal.fftconvolve(flux_noise150, PSF150, mode='same')
                    flux_noise_psf_std150 = flux_noise_PSF150 / noise_std150

                    flux_noise200 = np.rot90(J200) + noise200
                    flux_noise_PSF200 = signal.fftconvolve(flux_noise200, PSF200, mode='same')
                    flux_noise_psf_std200 = flux_noise_PSF200 / noise_std200

                    flux_noise_psf_std = flux_noise_psf_std150 + flux_noise_psf_std200 + flux_noise_psf_std115

                    # SOURCES DETECTION

                    for threshold, thr_number in zip([2.5, 2.75, 3, 3.5, 4], ['0', '1', '2', '3', '4']):

                        print(threshold, thr_number)
                        inf = detect_sources(flux_noise_psf_std, threshold, npixels)

                        for iso_gr, gr_dist, gr_name in zip([0, 1, 1], [3.0, 3.0, 1.0], ['_iso_', '_gr_3_', '_gr_1_']):
                            print('iso_or_gr:', iso_gr)
                            print('gr_dist:', gr_dist)
                            print('file_name:', gr_name)

                            obj_115_data = fluxes(flux_noise_PSF115, np.array(inf), inf.nlabels, X, Y, group=iso_gr, dist_max=gr_dist)
                            obj_150_data = fluxes(flux_noise_PSF150, np.array(inf), inf.nlabels, X, Y, group=iso_gr, dist_max=gr_dist)
                            obj_200_data = fluxes(flux_noise_PSF200, np.array(inf), inf.nlabels, X, Y, group=iso_gr, dist_max=gr_dist)

                            obj_115_filename = open(path + 'objects' + gr_name + simulation_name + '_115_' + telescope + '_' + thr_number + '.dat', 'ab')
                            obj_150_filename = open(path + 'objects' + gr_name + simulation_name + '_150_' + telescope + '_' + thr_number + '.dat', 'ab')
                            obj_200_filename = open(path + 'objects' + gr_name + simulation_name + '_200_' + telescope + '_' + thr_number + '.dat', 'ab')

                            np.savetxt(obj_115_filename, np.array(obj_115_data).T, fmt='%1.5e')
                            np.savetxt(obj_150_filename, np.array(obj_150_data).T, fmt='%1.5e')
                            np.savetxt(obj_200_filename, np.array(obj_200_data).T, fmt='%1.5e')

                            obj_115_filename.close()
                            obj_150_filename.close()
                            obj_200_filename.close()

main()

