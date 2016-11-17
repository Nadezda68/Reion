import yt
import glob
import numpy              as np
import healpy             as hp
import cosmolopy          as cp
import matplotlib.pyplot  as plt

from scipy                import signal
from scipy.interpolate    import interp1d
from scipy.interpolate    import interp2d
from scipy                import integrate
from astropy.io           import fits


# [parameters for arg pars]

filter_name = 'hst/wfc3/IR/f160w.dat'
start,stop = 1,2                        # [number of 3d boxes with different data]
radius = 25                             # [radius of a sphere in kpc]
mars2pix = 60                           # [milli-arcsec in pixel]
zero_point = 25.94                      # [noise zero point]

def transmition_ISM():

    global transmition_function,table

    table = np.loadtxt('table_transmition_ISM.dat')
    lam_rest   = table[1:,0]
    z          = table[0,1:]
    trans_coef = table[1:,1:]

    transmition_function = interp2d(z, lam_rest, trans_coef)

def filt():

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
        if filters_names[ifilt][1] == filter_name:
            filter_b = np.array([filters[ifilt][:,1],filters[ifilt][:,2]])
            filter_b = np.transpose(filter_b)

    F_b = interp1d(filter_b[:,0], filter_b[:,1],fill_value=0.0)
    left,right = np.min(filter_b[:,0]),np.max(filter_b[:,0])

def pixel_integration(pixel,lam):

    nu = 2.99792458e10/(lam/1e8)

    mtr_int = np.zeros_like(pixel[:,:,0])

    for i in range(0,len(pixel[:,0,0])):
        for j in range(0,len(pixel[0,:,0])):

            mtr_int[i,j] = integrate.trapz( pixel[i,j,::-1] * 3.828e33 * F_b(lam[::-1]), nu[::-1]) \
                           / (4 * np.pi * np.power(lum_dist*3.0857e18*1e6,2))

    return mtr_int

def noise_PSF(theta, phi, angle_idx, angles, sim_idx):

    global image_int

    pixels_with_noise = fits.open('hlsp_hlf_hst_wfc3-60mas_goodss_f160w_v1.0_sci.fits')[0].data[15000:15000+nbins,10000:10000+nbins]
    coeff = 10 ** (0.4 * (zero_point + 48.6))

    plt.figure(2*sim_idx+1)
    ax1 = plt.subplot(int(angles/6),6,angle_idx+1)
    plt.imshow(np.log10(image_int), interpolation='nearest')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_title(str(round(theta,1)) + ' ' + str(round(phi,1)))

    image_int *= coeff
    image_int += pixels_with_noise

    PSF = fits.open('psf_wfc3ir_f160w.fits')[0].data
    blurred = signal.fftconvolve(image_int, PSF, mode='same')

    #np.savetxt('f160w_filter.dat',blurred,fmt='%1.5e')

    plt.figure(2*sim_idx+2)
    ax2 = plt.subplot(int(angles/6),6,angle_idx+1)
    plt.imshow(blurred, interpolation='nearest',cmap=plt.cm.gray,vmin=np.min(blurred), vmax=np.max(blurred))
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title(str(round(theta,1)) + ' ' + str(round(phi,1)))

def filter_range(a,b,x):

    position_in_lam_array = []
    lambdas               = []

    for i in range(0,len(x)):

        if (a<=x[i] and x[i]<=b):

            position_in_lam_array.append(i)
            lambdas.append(x[i])

    lambdas = np.array(lambdas)
    position_in_lam_array = np.array(position_in_lam_array)

    indices = np.argsort(lambdas)

    return position_in_lam_array[indices]

def gal(theta,phi):

    global nbins, lam_list, luminosity_distance, lum_dist, image_int

    cartesian_vec_coord = np.vstack([x,y,z])
    rotated_vec_coord = np.array(lab2rot(cartesian_vec_coord,theta=theta,phi=phi))

    coord_prj_e_r     = rotated_vec_coord[0,:]
    coord_prj_e_theta = rotated_vec_coord[1,:]
    coord_prj_e_phi   = rotated_vec_coord[2,:]

    ang_dist = cp.distance.angular_diameter_distance(redshift, **cp.fidcosmo)
    lum_dist = ang_dist * (1+redshift) * (1+redshift)

    theta_milliarcsec = ( 2 * radius * 1e3 ) / ( ang_dist * 1e6  ) * ( 206265.0 * 1e3 )
    nbins =  int(theta_milliarcsec * 1/mars2pix)

    filter_lam = filter_range(left,right,lam_list)

    image = np.zeros([nbins, nbins, 3, len(filter_lam)])
    image_int = np.zeros([nbins, nbins, 3, 0])

    index = 0

    for i in filter_lam:

        interp = interp2d(dx, dy, lookup[i, :, :])
        temp = m.copy()

        for j in range(len(m)):
            temp[j] *= transmition_function(redshift,lam_list[i]/(1+redshift))[0]*interp(met[j], t[j])[0]

        e_theta_edges = np.linspace(-data.radius, data.radius, nbins+1)
        e_phi_edges   = np.linspace(-data.radius, data.radius, nbins+1)

        H, X, Y = np.histogram2d(coord_prj_e_theta, coord_prj_e_phi, bins=(e_theta_edges, e_phi_edges), weights = temp)
        image[:, :, 0, index] = H

        index += 1

    image_int = pixel_integration(image[:,:,0,:],lam_list[filter_lam])


def rot2lab(V,theta,phi):

    mtr = np.matrix([[ np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi),-np.sin(phi)],
                     [ np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)],
                     [ np.cos(theta)            ,-np.sin(theta)            , 0          ]])


    return np.squeeze(np.array(np.dot(mtr,V)))

def lab2rot(V,theta,phi):

    mtr = np.matrix([[ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)],
                     [ np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi),-np.sin(theta)],
                     [-np.sin(phi)              , np.cos(phi)              , 0            ]])

    return np.squeeze(np.array(np.dot(mtr,V)))


NSIDE = 1
a = hp.nside2npix(NSIDE)
theta,phi = hp.pix2ang(NSIDE,np.arange(a))
pix_vec   = np.array(hp.pix2vec(NSIDE,np.arange(a)))

transmition_ISM()
filt()

for i in range(start,stop):

    muf_list = glob.glob("./drt/muv.bin*")
    files = glob.glob("./rei05B_a0*/rei05B_a*.art")
    sorted(files)
    pf = yt.load(files[i])

    lam_list = np.zeros(len(muf_list))
    lookup = np.zeros([len(muf_list), 188, 22])

    for l in range(len(muf_list)):

        f = open(muf_list[l])
        header = f.readline()
        f.close()
        d1 = header.split()[0]
        d2 = header.split()[1]
        lam_list[l] = float(header.split()[2])

        data = np.genfromtxt(muf_list[l], skip_header=1)
        lookup[l, :, :] = data[1:,1:]

    dx = data[0, 1:]
    dy = data[1:, 0]

    data = pf.sphere("c", (radius, "kpc"))

    x = np.array(data[('STAR', 'POSITION_X')] - data.center[0])
    y = np.array(data[('STAR', 'POSITION_Y')] - data.center[1])
    z = np.array(data[('STAR', 'POSITION_Z')] - data.center[2])

    m = data[('STAR', 'MASS')].in_units('g')/1.989e33
    met = data[('STAR', 'METALLICITY_SNIa')] + data[('STAR', 'METALLICITY_SNII')]
    t = np.log10(data[('STAR', 'age')].in_units('yr'))
    redshift = pf.current_redshift

    lam_list *= (1+redshift)

    print('Data № ', i+1)

    for j in range(0,len(theta)):

        print('Projection [№%i out of %i] (phi, theta) = (%4.2f, %4.2f) deg' %  (j+1, len(theta),
              phi[j]*360/2/np.pi, theta[j]*360/2/np.pi))

        gal(theta=theta[j],phi=phi[j])
        noise_PSF(theta=theta[j]*360/2/np.pi,phi=phi[j]*360/2/np.pi,angle_idx=j,angles=len(theta),sim_idx=i)

plt.show()















