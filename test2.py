import yt
import glob
import numpy              as np
import healpy             as hp
import cosmolopy          as cp
import matplotlib.pyplot  as plt
import matplotlib.cm      as cm
import scipy.ndimage      as ndimage

from scipy                import signal
from scipy.interpolate    import interp1d
from scipy.interpolate    import interp2d
from scipy.interpolate    import RegularGridInterpolator
from scipy                import integrate
from astropy.io           import fits
from astropy              import wcs
from mpl_toolkits.mplot3d import Axes3D




def pixel_integration(pixel,lam):

    '''

     To integrate each pixel over all the lambdas with f160w filter transmition function (F_b)

     pixel: 3d array, (pixel SED(lambdas) 2d map, lambdas) [x direction, y direction, lambdas],
     units [erg/s/Hz/Solar_luminosity]

     lam: 1d array, containing lambdas to integrate over, redshifted,
     units [angstrom]

     mtr_int: matrix, total flux from the object at redshift z in each pixel,
     units [erg/s/cm^2]

    '''

    sun_lumi = 3.828e33
    c = 2.99792458e10
    nu = c/(lam/1e8)

    mtr_int = np.zeros_like(pixel[:,:,0])

    for i in range(0,len(pixel[:,0,0])):
        for j in range(0,len(pixel[0,:,0])):

            mtr_int[i,j] = integrate.trapz( pixel[i,j,::-1] * sun_lumi * F_b(lam[::-1]), nu[::-1]) / (4 * np.pi * np.power(lum_dist*3.0857e18*1e6,2))

    np.savetxt('f160w_filter_matrix.dat',mtr_int,fmt='%1.5e')

    return mtr_int

def transmition_ISM():

    '''

    Transmition function throughout the ISM, which depends on redshift z and radiation wavelength

    '''

    global transmition_function,table

    table = np.loadtxt('table_transmition_ISM.dat')
    lam_rest   = table[1:,0]
    z          = table[0,1:]
    trans_coef = table[1:,1:]

    transmition_function = interp2d(z, lam_rest, trans_coef)

    plt.figure(3)
    for i in range(1,11):
        plt.plot(lam_rest*(1+i),transmition_function(i,lam_rest))

def noise_PSF():

    '''

    Point spread function (from WFC3 PSFs)
    (check out http://www.stsci.edu/hst/wfc3/analysis/PSF for more information)

    Noise (from HLS GOODS-S region)
    (check out https://archive.stsci.edu/prepds/hlf/ for more information)

    It is also important to convect our simulation data and noise data into the same units (by using AB magnitudes)

    data: (energy flux density for each pixel without noise and PSF)
    units [erg/s/cm^2]

    '''

    data = np.loadtxt('f160w_filter_matrix.dat')

    plt.figure(4)
    plt.imshow(np.log10(data), interpolation='nearest')

    pixels_with_noise = fits.open('hlsp_hlf_hst_wfc3-60mas_goodss_f160w_v1.0_sci.fits')[0].data[15000:15000+nbins,10000:10000+nbins]
    zero_point = 25.94
    coeff = 10 ** (0.4 * (zero_point + 48.6))

    data *= coeff
    data += pixels_with_noise

    PSF = fits.open('psf_wfc3ir_f160w.fits')[0].data
    plt.figure(5)
    plt.imshow(PSF, interpolation='nearest',cmap=cm.Greys)
    blurred = signal.fftconvolve(data, PSF, mode='same')

    np.savetxt('f160w_filter.dat',blurred,fmt='%1.5e')

    plt.figure(6)
    plt.imshow(blurred, interpolation='nearest',cmap=plt.cm.gray,vmin=np.min(blurred), vmax=np.max(blurred))
    plt.show()

def filter_range(a,b,x): # x - array, a,b - number (boundaries)

    '''

    Initially we have SED tables for vast range of wavelengths and this function picks out those wavelengths, which
    are in filter range (for our current purposes it is f160w).

    '''

    position_in_lam_array = []
    lambdas               = []

    for i in range(0,len(x)):

        if (a<=x[i] and x[i]<=b):

            if(F_b(x[i]) >= 1e-3):

                position_in_lam_array.append(i)
                lambdas.append(x[i])

    lambdas = np.array(lambdas)
    position_in_lam_array = np.array(position_in_lam_array)

    indices = np.argsort(lambdas)

    plt.figure(2)
    plt.plot(filter_b[:,0],F_b(filter_b[:,0]),'b-')
    plt.plot(lambdas,F_b(lambdas),'r^')
    plt.show()

    return position_in_lam_array[indices]

def gal_data():

    '''

    To load and thereafter calculate all the necessary data for simulation

    '''

    global x, y, z, muf_list, lookup, dx, dy, redshift, secinrad, lam_list, m, t, met, data

    sun_lumi = 3.828e33 # [erg/s]
    sun_mass = 1.989e33 # [g]
    secinrad = 206265.0

    muf_list = glob.glob("./drt/muv.bin*")             # Solar lumi/ Hz / Solar mass (log10 time in yr, metallicity, lambda) tables
    files = glob.glob("./rei05B_a0*/rei05B_a*.art")    # 3D HST Data
    pf = yt.load(files[0])

    lam_list = np.zeros(len(muf_list))
    lookup = np.zeros([len(muf_list), 188, 22])

    for i in range(len(muf_list)):

        f = open(muf_list[i])
        header = f.readline()
        f.close()
        d1 = header.split()[0]
        d2 = header.split()[1]
        lam_list[i] = float(header.split()[2])

        data = np.genfromtxt(muf_list[i], skip_header=1)
        lookup[i, :, :] = data[1:,1:]

    dx = data[0, 1:] # metallicity in Solar metal. units
    dy = data[1:, 0] # log10 time in yr

    # to specify a region of interest in the sky
    data = pf.sphere([1.52643699e+2/7.20807692,  1.08564619e+2/7.20807692,  9.16425327e+1/7.20807692], (25.0, "kpc")) # " 1 ---> 7.20807692e+22 "

    x = np.array(data[('STAR', 'POSITION_X')]) - 1.52643699e+2/7.20807692
    y = np.array(data[('STAR', 'POSITION_Y')]) - 1.08564619e+2/7.20807692
    z = np.array(data[('STAR', 'POSITION_Z')]) - 9.16425327e+1/7.20807692

    m = data[('STAR', 'MASS')].in_units('g')/sun_mass
    met = data[('STAR', 'METALLICITY_SNIa')] + data[('STAR', 'METALLICITY_SNII')]
    t = np.log10(data[('STAR', 'age')].in_units('yr'))
    redshift = pf.current_redshift

    plt.figure(0)
    plt.hist2d(x,y,100)
    plt.show()

def gal():

    '''

    The main procedure to create a fake HST image

    nbins: number of pixels in our image (both for x axix and y axix [z axis])

    ang_dist: angular distance
    units [Mpc]

    lum_dist: luminosity distance
    units [Mpc]

    filter_lam: lambdas, which are in the filter range
    units [Angstom]

    theta_milliarcsec: selected region as it is seen by HST
    units [milli-arcseconds]

    We define the following resolution for the fake image: 60 milli-arcsecond/pixel

    '''

    global nbins, lam_list, luminosity_distance, lum_dist

    lam_list *= (1+redshift) # to get redshifted lam
    ang_dist = cp.distance.angular_diameter_distance(redshift, **cp.fidcosmo)
    lum_dist = ang_dist * (1+redshift) * (1+redshift)

    theta_milliarcsec = ( 2 * 25.0 * 1e3 ) / ( ang_dist * 1e6  ) * ( secinrad * 1e3 )
    nbins =  int(theta_milliarcsec * 1/60)

    print('number of pixels', nbins)

    filter_lam = filter_range(left,right,lam_list)

    image = np.zeros([nbins, nbins, 3, len(filter_lam)])
    image_int = np.zeros([nbins, nbins, 3, 0])

    index = 0

    for i in filter_lam:

        print(i,lam_list[i],lam_list[i]/(1+redshift),transmition_function(redshift,lam_list[i]/(1+redshift))[0])

        interp = interp2d(dx, dy, lookup[i, :, :])
        temp = m.copy()

        for j in range(len(m)):
            temp[j] *= transmition_function(redshift,lam_list[i]/(1+redshift))[0]*interp(met[j], t[j])[0]  # [erg/s/Hz] in Solar luminosity

        xedges = np.linspace(-data.radius, data.radius, nbins+1)
        yedges = np.linspace(-data.radius, data.radius, nbins+1)

        H, X, Y = np.histogram2d(x, y, bins=(xedges, yedges), weights = temp)
        image[:, :, 0, index] = H

        index += 1

    image_int = pixel_integration(image[:,:,0,:],lam_list[filter_lam])

    plt.figure(1)
    plt.imshow(np.log10(image_int), interpolation='nearest')
    plt.show()

def filt():

    '''

    To get the filter of interest (f160w), its transmition function, the lowest and the highest wavelength

    '''

    global filter_b, F_b, left, right

    filters_info = []
    for line in open('data/FILTER.RES.latest.info', 'r'):
        filters_info.append(np.array(line.split()))
    filters_info = np.array(filters_info)

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

    F_b = interp1d(filter_b[:,0], filter_b[:,1],fill_value=0.0)
    left,right = np.min(filter_b[:,0]),np.max(filter_b[:,0])

def sphere():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]

    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)

    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    NSIDE = 2
    a = hp.nside2npix(NSIDE)
    theta,phi = hp.pix2ang(NSIDE,np.arange(a))
    pix_vec   = hp.pix2vec(NSIDE,np.arange(a))

    points = np.zeros((len(theta),3))
    points[:,0] = np.sin(theta)*np.cos(phi)
    points[:,1] = np.sin(theta)*np.sin(phi)
    points[:,2] = np.cos(theta)


    #ax.plot(points[:,0], points[:,1], points[:,2],c='b')
    ax.scatter(points[:,0], points[:,1], points[:,2],c='k')
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()




sphere()

'''
transmition_ISM()
filt()
gal_data()
gal()
noise_PSF()
'''
