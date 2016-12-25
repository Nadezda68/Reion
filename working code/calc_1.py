
import yt
import glob
import argparse
import numpy              as np

from scipy.interpolate    import interp1d
from scipy.interpolate    import interp2d
from scipy                import integrate

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


def init_input_data(sim, sim2,  sim_start, sim_stop, center_x, center_y, center_z,
                    telescope_input='HST', filter_input='f140w', sim_radius=30, projection_input='all'):

    '''
    sim, sim2 - two digits of a number, that determines which ART simulation to load (e.g. for
        №10 simulation you input sim = 1 and sim2 = 0

    sim_start, sim_stop - two numbers, which determine the left and the right boundary of snapshots to process
        (e.g. if we have 20 snapshots for a specific ART sim and we want to process snapshots only from 3 to 10,
        then we input sim_start = 3, sim_stop = 11)

    center_x, center_y, center_z - coordinates of a center of a sphere

    sim_radius - radius of a sphere

    telescope_input - either 'JWST' or 'HST'

    filter_input - 'F150W' and 'F200W' for JWST and 'f125w', 'f140w' and 'f160w' for HST
    '''

    global input_start, input_stop, input_radius, input_coord, N_sim, N_sim_2, telescope, cam_filter, projection

    N_sim, N_sim_2, input_start, input_stop, input_radius = int(sim), int(sim2), int(sim_start), int(sim_stop), sim_radius
    input_coord = np.array([center_x, center_y, center_z])
    cam_filter = filter_input
    telescope = telescope_input
    projection = projection_input

    print('Simulation number = ', N_sim, N_sim_2)
    print('Radius of a sphere = ', input_radius)
    print('Coordinates of a center of a sphere = ', input_coord)

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


def filter_bandwidth(a,b,x):

    position_in_lam_array = []

    for i in range(0,len(x)):
        if (a<=x[i] and x[i]<=b):
            if(F_filter(x[i])>=0.5e-3):
                position_in_lam_array.append(i)

    return position_in_lam_array


def HST_filter_init(z):

    '''
    Hubble Space Telescope filter initialization function
    '''

    global F_filter

    filter_b = np.loadtxt('data/filter_' + cam_filter + '.dat')
    F_filter = interp1d(filter_b[:,0], filter_b[:,1],fill_value=0.0,bounds_error=False)
    a,b = np.min(filter_b[:,0]),np.max(filter_b[:,0])
    lamb_positions = filter_bandwidth(a,b,lam_list*(1+z))

    return lamb_positions


def JWST_filter_init(z):

    '''
    James Webb Space Telescope filter initialization function
    '''

    global F_filter

    wavelengths,transmission = np.loadtxt('data/filter_' + cam_filter + '.dat',skiprows=1).T
    wavelengths *= 1e4
    F_filter = interp1d(wavelengths, transmission,fill_value=0.0,bounds_error=False)
    a,b = np.min(wavelengths),np.max(wavelengths)
    lamb_positions = filter_bandwidth(a,b,lam_list*(1+z))

    return lamb_positions


def main():

    init_input_data(*external_param)
    init_lum_tables()
    init_transmission_function()

    files = sorted(glob.glob('/home/kaurov/scratch/rei/code05/OUT_000' + str(N_sim) + str(N_sim_2) + '_010_080B/rei000'
                             +  str(N_sim) + str(N_sim_2) + '_010_080B_a0*/rei000' + str(N_sim) + str(N_sim_2) + '_010_080B_a0.*.art'))

    for simulation in range(input_start,input_stop):

        print('Loading data %i out of [%i...%i] (%i in total)' % (simulation,input_start,input_stop,len(files)))

        pf = yt.load(files[simulation])
        simulation_name = files[simulation][-29:-4]
        data = pf.sphere(input_coord, (input_radius, "kpc"))
   
        x = np.array(data[('STAR', 'POSITION_X')] - data.center[0])
        y = np.array(data[('STAR', 'POSITION_Y')] - data.center[1])
        z = np.array(data[('STAR', 'POSITION_Z')] - data.center[2])
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
        theta_arcsec = (2 * input_radius * 1e3) / (D_A(redshift) * 1e6) * arcsec_in_rad

        if telescope == 'HST':
            nbins = int(theta_arcsec / HST_WFC3CAM_pixel_size)
        elif telescope == 'JWST':
            nbins = int(theta_arcsec / JWST_NIRCAM_pixel_size)

        print('Simulation name = ',	simulation_name)
        print('z = %1.3e, D_A = %1.3e [Mpc], Nbins = %i' % (redshift, D_A(redshift), nbins))

        print('filter name:' + cam_filter)

        if telescope == 'HST':
            lamb_positions = HST_filter_init(redshift)
        elif telescope == 'JWST':
            lamb_positions = JWST_filter_init(redshift)

        if projection == 'all':
            image = np.zeros([nbins, nbins, len(lamb_positions),3])
        else:
            image = np.zeros([nbins, nbins, len(lamb_positions)])

        index = 0
        xedges = np.linspace(-data.radius, data.radius, nbins+1)
        yedges = np.linspace(-data.radius, data.radius, nbins+1)

        for i in lamb_positions:

            print('Computing luminosity...  step %i out of %i, lambda = %1.3e' % (index+1,len(lamb_positions),lam_list[i]))
            interp = interp2d(Z, logt, lookup[i, :, :])
            L = np.zeros_like(m)

            for j in range(0,len(m)):
                L[j] = F_ISM(redshift,lam_list[i])[0] * interp(met[j], t[j])[0] * m[j] * sun_luminosity

            if projection == 'z':
                H, X1, X2 = np.histogram2d(x, y, bins=(xedges, yedges), weights = L)
                image[:, :, index] = np.rot90(H)
            elif projection == 'y':
                H, X1, X2 = np.histogram2d(z, x, bins=(xedges, yedges), weights = L)
                image[:, :, index] = np.rot90(H)
            elif projection == 'x':
                H, X1, X2 = np.histogram2d(y, z, bins=(xedges, yedges), weights = L)
                image[:, :, index] = np.rot90(H)
            elif projection == 'all':

                H, X1, X2 = np.histogram2d(y, z, bins=(xedges, yedges), weights = L)
                image[:, :, index, 0] = np.rot90(H)
                H, X1, X2 = np.histogram2d(z, x, bins=(xedges, yedges), weights = L)
                image[:, :, index, 1] = np.rot90(H)
                H, X1, X2 = np.histogram2d(x, y, bins=(xedges, yedges), weights = L)
                image[:, :, index, 2] = np.rot90(H)

            index += 1

        np.save('output/lum_' + cam_filter + '_' + simulation_name + '.npy', image)
        np.savetxt('output/info_'  + simulation_name + '.dat',np.array([nbins,redshift,D_A(redshift),theta_arcsec,
                                                                         input_coord[0], input_coord[1], input_coord[2],
                                                                        input_radius]),
                   header='Nbins, Redshift, Angular distance [Mpc], theta [arc-sec], x_center, y_center, z_center, radius [kpc]')

main()
