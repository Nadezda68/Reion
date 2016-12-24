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
name_save = '_010_080B/processed_output/data_'

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

    global info, lums, z, nbins, nbins_min, z, indices, theta_min

    info = sorted(glob.glob("./output/info_rei000" + str(N_sim_1) + str(N_sim_2) + "_*"))
    lums = np.vstack((sorted(glob.glob("./output/lum_125_rei000" + str(N_sim_1) + str(N_sim_2) + "_*")),
                      sorted(glob.glob("./output/lum_140_rei000" + str(N_sim_1) + str(N_sim_2) + "_*")),
                      sorted(glob.glob("./output/lum_160_rei000" + str(N_sim_1) + str(N_sim_2) + "_*"))))

    z      = []
    nbins  = []
    angles = []

    for i in range(0,len(info)):
        N, redshift, D_A, theta = np.loadtxt(info[i], skiprows=1)
        z.append(redshift)
        nbins.append(int(N))
        angles.append(theta)

    z       = np.array(z)
    indices = np.argsort(z)[::-1]
    z       = z[indices]

    nbins     = np.array(nbins)
    nbins     = nbins[indices]
    nbins_min = np.min(nbins)

    angles = np.array(angles)
    theta_min = np.min(angles)

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

    global PSF, from_arcsec_to_pix_num,  from_pix_num_to_arcsec

    a = fits.open('./tinytim/test/psf_test_f125w00_psf.fits')[0].data
    b = fits.open('./tinytim/test/psf_test_f140w00_psf.fits')[0].data
    c = fits.open('./tinytim/test/psf_test_f160w00_psf.fits')[0].data

    coords_a = np.linspace(-1.5,1.5,np.shape(a)[0]) # from -10 to 10 arcsec
    coords_b = np.linspace(-1.5,1.5,np.shape(b)[0])
    coords_c = np.linspace(-1.5,1.5,np.shape(c)[0])

    NNbins = int(3/0.13)
    pix_edges  = np.linspace(-1.5,1.5,NNbins+1)
    pix_coords = (pix_edges[1:] + pix_edges[:-1])/2
    pix_num    = np.arange(NNbins)

    from_arcsec_to_pix_num = interp1d(pix_coords,pix_num)
    from_pix_num_to_arcsec = interp1d(pix_num,pix_coords)

    x, y = np.meshgrid(coords_a, coords_a)
    psf_f125w,X,Y =np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = a.flatten())
    x, y = np.meshgrid(coords_b, coords_b)
    psf_f140w,X,Y =np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = b.flatten())
    x, y = np.meshgrid(coords_c, coords_c)
    psf_f160w,X,Y =np.histogram2d(x.flatten(), y.flatten(), bins=(pix_edges, pix_edges), weights = c.flatten())

    PSF = np.dstack((psf_f125w,psf_f140w,psf_f160w))

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


def plot(obj, index_1, index_2, title=' ', contours=0):

    plt.figure(index_1)
    plt.subplot(5,5,index_2+1)
    if(contours):
         plt.imshow(obj, interpolation='nearest',vmin=0, vmax=5)
         plt.contour(obj, levels=np.array([2,5]),lw=0.4, colors='k')
    else:
        plt.imshow(obj, interpolation='nearest',cmap=plt.cm.gray)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.title( str(round(z[index_2],2)))


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
        plt.figure(100+filter_index)
        plt.savefig('./rei000' + str(N_sim_1) + str(N_sim_2) + '_010_080B/figures/fig_f'+filter_name+'w_4.pdf', format='pdf')

        filter_index += 1

    for i in range(0,len(z)):

        if i<10:
            np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + name_save + 'total_0_0' + str(i) +'.dat', flux_psf_std_sum[:,:,i],fmt='%1.5e')
            np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + name_save + 'total_1_0' + str(i) +'.dat', flux_noise_psf_std_sum[:,:,i],fmt='%1.5e')
        else:
            np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + name_save + 'total_0_' + str(i) +'.dat', flux_psf_std_sum[:,:,i],fmt='%1.5e')
            np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + name_save + 'total_1_' + str(i) +'.dat', flux_noise_psf_std_sum[:,:,i],fmt='%1.5e')

        plot(obj=flux_psf_std_sum[:,:,i]      ,index_1=0,index_2=i,title=str(round(z[i],2)),contours=1)
        plot(obj=flux_noise_psf_std_sum[:,:,i],index_1=1,index_2=i,title=str(round(z[i],2)),contours=1)

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
            total_flux[j,k] = 1e23 * 1e9 * integrate.trapz( input_data[j,k,::-1] * F_filter(lamb_filter[::-1]*(1+z[i]))/nu[::-1], nu[::-1] ) / \
                             integrate.trapz( F_filter(lamb_filter[::-1]*(1+z[i]))/nu[::-1], nu[::-1] ) * \
                             (1+z[i]) / (4 * np.pi * np.power(lum_dist*cm_in_pc*1e6,2))
    
    flux_min_bin       = total_flux[a:a+nbins_min,a:a+nbins_min]
    flux_noise         = flux_min_bin + noise[:,:,l]

    if nbins_min>=np.shape(PSF[:,:,l])[0]:
        Kernel = PSF[:,:,l]
        print('works condition 1')
    else:
        left  = int(from_arcsec_to_pix_num(-theta_min/2))
        right = int(from_arcsec_to_pix_num(theta_min/2))
        if (right+1-left)%2==0:
            right += 1

        centr = int(from_arcsec_to_pix_num(0))
        print('info:',left,right,centr-left,right-centr,right+1-left)
        Kernel = PSF[left:(right+1),left:(right+1),l]

    flux_noise_psf     = signal.fftconvolve(flux_noise, Kernel, mode='same')
    flux_psf_std       = signal.fftconvolve(flux_min_bin, Kernel, mode='same')/noise_std[l]
    flux_noise_psf_std = flux_noise_psf/noise_std[l]

    plot(obj=flux_noise_psf          ,index_1=(2+3*l),index_2=i,title=str(round(z[i],2)),contours=0)
    plot(obj=flux_psf_std            ,index_1=(3+3*l),index_2=i,title=str(round(z[i],2)),contours=1)
    plot(obj=flux_noise_psf_std      ,index_1=(4+3*l),index_2=i,title=str(round(z[i],2)),contours=1)
    plot(obj=np.log10(flux_noise_psf),index_1=(100+l),index_2=i,title=str(round(z[i],2)),contours=0)

    flux_psf_std_sum[:,:,i]       += flux_psf_std
    flux_noise_psf_std_sum[:,:,i] += flux_noise_psf_std

    if(i<10):
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + name_save + 'f' + name + 'w_1_0'+ str(i) +'.dat', flux_noise_psf,fmt='%1.5e')
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + name_save + 'f' + name + 'w_2_0'+ str(i) +'.dat', flux_psf_std,fmt='%1.5e')
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + name_save + 'f' + name + 'w_3_0'+ str(i) +'.dat', flux_noise_psf_std,fmt='%1.5e')
    else:
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + name_save + 'f' + name + 'w_1_'+ str(i) +'.dat', flux_noise_psf,fmt='%1.5e')
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + name_save + 'f' + name + 'w_2_'+ str(i) +'.dat', flux_psf_std,fmt='%1.5e')
        np.savetxt('./rei000' + str(N_sim_1) + str(N_sim_2) + name_save + 'f' + name + 'w_2_'+ str(i) +'.dat', flux_noise_psf_std,fmt='%1.5e')

main(*external_param)
