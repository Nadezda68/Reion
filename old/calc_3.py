import glob
import numpy as np
from photutils import detect_sources
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from photutils import CircularAperture
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cosmolopy import distance as d
from scipy import integrate
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

cosmo = {'omega_M_0' : 0.2726, 'omega_lambda_0' : 0.7274, 'h' : 0.704}
cosmo = d.set_omega_k_0(cosmo)

dashes = np.array([[12,4],[22,4],[7,4], [10,10], [12,4],[5,12], [2,2]])
line_thickness = np.array([4.5, 4.1, 3.3, 1.7, 2.9, 2.5, 2.1, 3.7, 2])
colors = np.array(['r', 'purple', 'b', 'm', 'c', 'chocolate', 'darkgreen', 'orange', 'k'])

def fixlogax(ax, a='x'):
    if a == 'x':
        labels = [item.get_text() for item in ax.get_xticklabels()]
        positions = ax.get_xticks()
        # print positions
        # print labels
        for i in range(len(positions)):
            labels[i] = '$10^{'+str(int(np.log10(positions[i])))+'}$'
        if np.size(np.where(positions == 1)) > 0:
            labels[np.where(positions == 1)[0][0]] = '$1$'
        if np.size(np.where(positions == 10)) > 0:
            labels[np.where(positions == 10)[0][0]] = '$10$'
        if np.size(np.where(positions == 0.1)) > 0:
            labels[np.where(positions == 0.1)[0][0]] = '$0.1$'
        # print positions
        # print labels
        ax.set_xticklabels(labels)
    if a == 'y':
        labels = [item.get_text() for item in ax.get_yticklabels()]
        positions = ax.get_yticks()
        # print positions
        # print labels
        for i in range(len(positions)):
            labels[i] = '$10^{'+str(int(np.log10(positions[i])))+'}$'
        if np.size(np.where(positions == 1)) > 0:
            labels[np.where(positions == 1)[0][0]] = '$1$'
        if np.size(np.where(positions == 10)) > 0:
            labels[np.where(positions == 10)[0][0]] = '$10$'
        if np.size(np.where(positions == 0.1)) > 0:
            labels[np.where(positions == 0.1)[0][0]] = '$0.1$'
        # print positions
        # print labels
        ax.set_yticklabels(labels)

def plot_style(xticks=5,yticks=5):

    global ax

    plt.rcParams.update({'figure.autolayout': True})
    #plt.tight_layout()
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['figure.figsize'] = 8, 7.5

    fig,ax = plt.subplots()
    #x_minor_locator = AutoMinorLocator(xticks)
    #y_minor_locator = AutoMinorLocator(yticks)
    #plt.tick_params(which='both', width=1.7)
    #plt.tick_params(which='major', length=9)
    #plt.tick_params(which='minor', length=5)
    #ax.xaxis.set_minor_locator(x_minor_locator)
    #ax.yaxis.set_minor_locator(y_minor_locator)


from scipy import stats


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

def sources_JWST():

    plot_style()

    JWST_data1 = sorted(glob.glob('processed/sim06_100/data/sim_06_F150W_z1_*'))

    image1 = np.loadtxt(JWST_data1[17])
    sim06_info = sorted(glob.glob('processed/sim06_100/info/info_rei00006_*'))

    sim_06_redshifts = []
    for i in range(0,len(JWST_data1)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim06_info[i], skiprows=1)
        sim_06_redshifts.append(redshift)
        if(i==17):
            D_Angular = D_A
        sim_06_ang_size = Angular_size

    JWST_NIRCAM_pixel_size = 0.032  # arcsec per pixel
    Nbins_JWST = int(sim_06_ang_size/JWST_NIRCAM_pixel_size)
    ang = sim_06_ang_size/2
    pixels_ang_coords_JWST = (np.linspace(-ang, ang, Nbins_JWST + 1) + np.linspace(-ang, ang, Nbins_JWST + 1))/2
    X,Y = np.meshgrid(pixels_ang_coords_JWST,pixels_ang_coords_JWST)
    theta_arcsec = (100 * 1e3) / (D_Angular * 1e6) * 206265/2

    image = np.loadtxt('totalj.dat')
    data = detect_sources(image, 3, 3)
    N = data.nlabels
    if(N>1):
        x_coord = np.zeros(N)
        y_coord = np.zeros(N)
        for j in range(1,N+1):
            A = np.argwhere(np.array(data)==j)
            x_coord[j-1] = np.mean(X[A[:,0],A[:,1]]) -3.1
            y_coord[j-1] = np.mean(Y[A[:,0],A[:,1]]) -3.1

    extent = np.array([-ang,ang,-ang,ang])
    sigma = init_noise_JWST()

    plt.imshow(image1, interpolation='nearest',cmap=plt.cm.gray,vmax=2*sigma,extent=extent)
    plt.xlim(-theta_arcsec,theta_arcsec)
    plt.ylim(-theta_arcsec,theta_arcsec)

    positions = (x_coord, -y_coord)
    apertures = CircularAperture(positions, r=0.4)
    apertures.plot(color='red', lw=2.2, alpha=0.6)
    plt.yticks([])
    plt.xticks([])

    axins = zoomed_inset_axes(ax, 2.5, loc=1)
    axins.imshow(image1, interpolation='nearest',cmap=plt.cm.gray,vmax=2*sigma,extent=extent)
    apertures = CircularAperture(positions, r=0.2)
    apertures.plot(color='red', lw=2.5, alpha=0.6)

    # sub region of the original image
    x1, x2, y1, y2 =  -3.4, -0.4,-3.9, -0.9
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.yticks([])
    plt.xticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.8")
    plt.draw()

    plt.savefig('JWST_sources.pdf',format='pdf')
    plt.show()

sources_JWST()

def sources_HST():

    plot_style()

    HST_data1 = sorted(glob.glob('processed/sim06_100/data/sim_06_f140w_z1_*'))
    image1 = np.loadtxt(HST_data1[17])
    sim06_info = sorted(glob.glob('processed/sim06_100/info/info_rei00006_*'))

    sim_06_redshifts = []
    for i in range(0,len(HST_data1)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim06_info[i], skiprows=1)
        sim_06_redshifts.append(redshift)
        if(i==17):
            D_Angular = D_A
        sim_06_ang_size = Angular_size

    HST_WFC3CAM_pixel_size = 0.13# arcsec per pixel
    Nbins_HST = int(sim_06_ang_size/HST_WFC3CAM_pixel_size)
    ang = sim_06_ang_size/2
    pixels_ang_coords_HST = (np.linspace(-ang, ang, Nbins_HST + 1) + np.linspace(-ang, ang, Nbins_HST + 1))/2
    X,Y = np.meshgrid(pixels_ang_coords_HST,pixels_ang_coords_HST)
    theta_arcsec = (100 * 1e3) / (D_Angular * 1e6) * 206265/2


    image = np.loadtxt('total.dat')
    data = detect_sources(image, 3, 3)
    N = data.nlabels
    if(N>1):
        x_coord = np.zeros(N)
        y_coord = np.zeros(N)
        for j in range(1,N+1):
            A = np.argwhere(np.array(data)==j)
            x_coord[j-1] = np.mean(X[A[:,0],A[:,1]]) -3.1
            y_coord[j-1] = np.mean(Y[A[:,0],A[:,1]]) -3.1

    extent = np.array([-ang,ang,-ang,ang])
    sigma = init_noise()
    print(np.shape(image1))
    plt.imshow(image1,interpolation='nearest',cmap=plt.cm.gray,vmax=2*sigma,extent=extent)
    plt.xlim(-theta_arcsec,theta_arcsec)
    plt.ylim(-theta_arcsec,theta_arcsec)

    positions = (x_coord,-y_coord)
    apertures = CircularAperture(positions, r=0.7)
    apertures.plot(color='blue', lw=2.5, alpha=0.6)

    plt.yticks([])
    plt.xticks([])
    plt.savefig('HST_sources.pdf',format='pdf')
    plt.show()

sources_HST()


def four_figure_plot():

    fig = plt.figure(figsize=(15,14))
    plt.rcParams['figure.figsize'] = 10,20

    plt.subplot(2,2,1)

    JWST_stars_data = sorted(glob.glob('processed/sim06_100/data/sim_06_F150W_z2_*'))
    HST_stars_data = sorted(glob.glob('processed/sim06_100/data/sim_06_f140w_z2_*'))
    sim06_info = sorted(glob.glob('processed/sim06_100/info/info_rei00006_*'))

    sim_06_redshifts = []
    for i in range(0,len(JWST_stars_data)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim06_info[i], skiprows=1)
        sim_06_redshifts.append(redshift)
        if(i==17):
            D_Angular = D_A

    HST_WFC3CAM_pixel_size = 0.13   # arcsec per pixel
    theta_arcsec = (100 * 1e3) / (D_Angular * 1e6) *  206265
    ang = theta_arcsec/2
    nbins_needed = int(theta_arcsec/  HST_WFC3CAM_pixel_size)
    extent=np.array([-ang,ang,-ang,ang])
    image = np.loadtxt(HST_stars_data[17])
    a = np.shape(image)[0]
    dif = int((a-nbins_needed)/2)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['figure.figsize'] = 8, 7.5
    x_minor_locator = AutoMinorLocator(3)
    y_minor_locator = AutoMinorLocator(3)
    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=10)
    plt.tick_params(which='minor', length=5)
    #plt.xlabel('arcsec',fontsize=27)
    plt.xticks([-9,-6,-3,0,3,6,9],fontsize=36)
    plt.yticks([-9,-6,-3,0,3,6,9],fontsize=36)
    plt.ylabel('arcsec',fontsize=36)

    ax = plt.gca()
    im = ax.imshow(image[dif:-dif,dif:-dif],interpolation='nearest',vmax=3,cmap='plasma',extent=extent)
    divider = make_axes_locatable(ax)
    ax.set_xticklabels([])

    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    #ax.set_yticklabels([])
    cb = plt.colorbar(im,cax=cax, ticks=[0,1,2,3])
    cb.ax.tick_params(labelsize=32, width=2, length=9)

    plt.subplot(2,2,2)

    JWST_stars_data = sorted(glob.glob('processed/sim06_100/data/sim_06_F150W_z2_*'))
    HST_stars_data = sorted(glob.glob('processed/sim06_100/data/sim_06_f140w_z2_*'))

    sim06_info = sorted(glob.glob('processed/sim06_100/info/info_rei00006_*'))
    sim_06_redshifts = []
    for i in range(0,len(JWST_stars_data)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim06_info[i], skiprows=1)
        sim_06_redshifts.append(redshift)
        if(i==17):
            D_Angular = D_A

    JWST_NIRCAM_pixel_size = 0.032  # arcsec per pixel
    theta_arcsec = (100 * 1e3) / (D_Angular * 1e6) *  206265
    ang = theta_arcsec/2
    nbins_needed = int(theta_arcsec/JWST_NIRCAM_pixel_size )
    extent=np.array([-ang,ang,-ang,ang])
    image = np.loadtxt(JWST_stars_data[17])
    a = np.shape(image)[0]
    dif = int((a-nbins_needed)/2)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['figure.figsize'] = 8, 7.5
    x_minor_locator = AutoMinorLocator(3)
    y_minor_locator = AutoMinorLocator(3)
    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=10)
    plt.tick_params(which='minor', length=5)
    #plt.xlabel('arcsec',fontsize=27)
    plt.xticks([-9,-6,-3,0,3,6,9],fontsize=36)
    plt.yticks([-9,-6,-3,0,3,6,9],fontsize=36)
    #plt.ylabel('arcsec',fontsize=36)

    ax = plt.gca()
    im = ax.imshow(image[dif:-dif,dif:-dif],interpolation='nearest',vmax=3,cmap='plasma',extent=extent)
    divider = make_axes_locatable(ax)
    ax.set_xticklabels([])

    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    ax.set_yticklabels([])


    cb = plt.colorbar(im,cax=cax, ticks=[0,1,2,3])
    cb.ax.tick_params(labelsize=32, width=2, length=9)

    plt.subplot(2,2,3)
    JWST_stars_data = sorted(glob.glob('processed/sim06_100/data/sim_06_F150W_z1_*'))
    HST_stars_data = sorted(glob.glob('processed/sim06_100/data/sim_06_f140w_z1_*'))

    sim06_info = sorted(glob.glob('processed/sim06_100/info/info_rei00006_*'))
    sim_06_redshifts = []
    for i in range(0,len(JWST_stars_data)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim06_info[i], skiprows=1)
        sim_06_redshifts.append(redshift)
        if(i==17):
            D_Angular = D_A

    HST_WFC3CAM_pixel_size = 0.13   # arcsec per pixel
    theta_arcsec = (100 * 1e3) / (D_Angular * 1e6) *  206265
    ang = theta_arcsec/2
    nbins_needed = int(theta_arcsec/HST_WFC3CAM_pixel_size )
    extent=np.array([-ang,ang,-ang,ang])
    image = np.loadtxt(HST_stars_data[17])
    a = np.shape(image)[0]
    dif = int((a-nbins_needed)/2)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['figure.figsize'] = 8, 7.5
    x_minor_locator = AutoMinorLocator(3)
    y_minor_locator = AutoMinorLocator(3)
    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=10)
    plt.tick_params(which='minor', length=5)
    plt.xlabel('arcsec',fontsize=36)
    plt.xticks([-9,-6,-3,0,3,6,9],fontsize=36)
    plt.yticks([-9,-6,-3,0,3,6,9],fontsize=36)
    plt.ylabel('arcsec',fontsize=36)

    ax = plt.gca()
    sigma = init_noise()
    im = ax.imshow(image[dif:-dif,dif:-dif]/sigma,interpolation='nearest',cmap=plt.cm.gray,vmax=3,extent=extent)
    divider = make_axes_locatable(ax)
    #ax.set_xticklabels([])

    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    #ax.set_yticklabels([])

    cb = plt.colorbar(im,cax=cax, ticks=[-2,-1,0,1,2,3])
    cb.ax.tick_params(labelsize=32, width=2, length=9)

    plt.subplot(2,2,4)

    sim06_info = sorted(glob.glob('processed/sim06_100/info/info_rei00006_*'))
    sim_06_redshifts = []
    for i in range(0,len(JWST_stars_data)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim06_info[i], skiprows=1)
        sim_06_redshifts.append(redshift)
        if(i==17):
            D_Angular = D_A

    JWST_NIRCAM_pixel_size = 0.032  # arcsec per pixel
    theta_arcsec = (100 * 1e3) / (D_Angular * 1e6) *  206265
    ang = theta_arcsec/2
    nbins_needed = int(theta_arcsec/JWST_NIRCAM_pixel_size)
    extent=np.array([-ang,ang,-ang,ang])
    image = np.loadtxt(JWST_stars_data[17])
    a = np.shape(image)[0]
    dif = int((a-nbins_needed)/2)
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['figure.figsize'] = 8, 7.5
    x_minor_locator = AutoMinorLocator(3)
    y_minor_locator = AutoMinorLocator(3)
    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=10)
    plt.tick_params(which='minor', length=5)
    plt.xlabel('arcsec',fontsize=36)
    plt.xticks([-9,-6,-3,0,3,6,9],fontsize=36)
    plt.yticks([-9,-6,-3,0,3,6,9],fontsize=36)
    #plt.ylabel('arcsec',fontsize=36)

    ax = plt.gca()
    sigma = init_noise_JWST()
    im = ax.imshow(image[dif:-dif,dif:-dif]/sigma,interpolation='nearest',cmap=plt.cm.gray,vmax=3,extent=extent)
    divider = make_axes_locatable(ax)
    #ax.set_xticklabels([])

    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    ax.set_yticklabels([])

    cb = plt.colorbar(im,cax=cax, ticks=[-2,-1,0,1,2,3])
    cb.ax.tick_params(labelsize=32, width=2, length=9)

    fig.tight_layout()
    plt.savefig('fig2panel3.pdf',format='pdf',bbox_inches='tight')
    plt.show()


