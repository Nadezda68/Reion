import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from photutils import detect_sources
from photutils import CircularAperture
from photutils import aperture_photometry
from scipy import stats

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



def JWST_F150W_noise_spread(x):

    '''
    x - exposure time in seconds
    '''

    a,b,c,d,e = -0.35867483, 9.1144785, -75.60172247, 256.44825309, -275.1335303
    X = np.log10(x)
    return a + b/X + c/X/X + d/X/X/X + e/X/X/X/X


def init_noise_JWST(exp_time=31536000):

    nbins_min=1000
    noise_spread = JWST_F150W_noise_spread(exp_time)
    #noise_loc = 4.3934
    noise_loc = 0.0
    noise = stats.norm.rvs(noise_loc, noise_spread, nbins_min*nbins_min)
    noise_std = np.std(noise)
    print(noise_std)
    return noise_std


def init_noise():

    nbins_min=1000

    zero_point = np.array([26.23,26.45,25.94])
    coeff = 10 ** (0.4 * (zero_point + 48.6))

    coeff_125 = 1e23 * 1e9 / coeff[0]
    coeff_140 = 1e23 * 1e9 / coeff[1]
    coeff_160 = 1e23 * 1e9 / coeff[2]

    noise = np.vstack((stats.norm.rvs(-0.00015609*coeff_125,0.00275845*coeff_125,nbins_min*nbins_min),
                       stats.norm.rvs(-3.24841830e-05*coeff_140,3.26572605e-03*coeff_140,nbins_min*nbins_min),
                       stats.norm.rvs(-0.00020178*coeff_160,0.00239519*coeff_160,nbins_min*nbins_min)))

    return np.std(noise[1,:])


def plot_style(xticks=5,yticks=5):

    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['figure.figsize'] = 8, 7.5

    fig,ax = plt.subplots()
    x_minor_locator = AutoMinorLocator(xticks)
    y_minor_locator = AutoMinorLocator(yticks)
    plt.tick_params(which='both', width=1.7)
    plt.tick_params(which='major', length=9)
    plt.tick_params(which='minor', length=5)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)


def sources_HST():

    plot_style()

    HST_WFC3CAM_pixel_size = 0.13
    theta = 39.476407
    Nbins_HST = int(theta/HST_WFC3CAM_pixel_size)
    ang = theta/2
    pixels_ang_coords_HST = (np.linspace(-ang, ang, Nbins_HST + 1) + np.linspace(-ang, ang, Nbins_HST + 1))/2
    X,Y = np.meshgrid(pixels_ang_coords_HST,pixels_ang_coords_HST)

    image = np.loadtxt('17z/total.dat')
    flux160 = np.loadtxt('17z/160.dat')

    data = detect_sources(image, 3, 3)
    N = data.nlabels

    extent = np.array([-ang,ang,-ang,ang])
    sigma = init_noise()

    plt.imshow(image,interpolation='nearest',cmap=plt.cm.gray,vmax=3,extent=extent)

    for iso_gr, gr_dist, gr_name in zip([0, 1, 1], [3.0, 3.0, 1.0], ['_iso_', '_gr_3_', '_gr_1_']):
        flux = fluxes(flux160, np.array(data), N, X, Y, group=iso_gr, dist_max=gr_dist)
        print(gr_name)
        print(flux)

    plt.xlim(-ang, ang)
    plt.ylim(-ang, ang)

    plt.figure(2)
    plt.imshow(data, interpolation='nearest',cmap=plt.cm.gray)

    for j in range(1, N+1):
        A = np.argwhere(np.array(data) == j)
        apertures = CircularAperture([np.mean(A[:,1]), np.mean(A[:, 0])], r=np.sqrt(len(A)))
        apertures.plot(color='blue', lw=2.5, alpha=0.5)

    plt.grid()
    plt.show()

sources_HST()
