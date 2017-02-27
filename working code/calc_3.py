import glob
import numpy as np
from photutils import detect_sources
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from photutils import CircularAperture
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cosmolopy import distance as d
from scipy import integrate

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

    plt.rcParams.update({'figure.autolayout': True})
    #plt.tight_layout()
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


def step_1():

    sim01_info = sorted(glob.glob('processed/sim01_100/info/info_rei00001_*'))
    sim02_info = sorted(glob.glob('processed/sim02_100/info/info_rei00002_*'))
    sim03_info = sorted(glob.glob('processed/sim03_100/info/info_rei00003_*'))
    sim04_info = sorted(glob.glob('processed/sim04_100/info/info_rei00004_*'))
    sim06_info = sorted(glob.glob('processed/sim06_100/info/info_rei00006_*'))
    sim09_info = sorted(glob.glob('processed/sim09_100/info/info_rei00009_*'))

    sim_01_HST_data_x = sorted(glob.glob('processed/sim01_100/data/sim_01_f140w_x3_*'))
    sim_01_HST_data_y = sorted(glob.glob('processed/sim01_100/data/sim_01_f140w_y3_*'))
    sim_01_HST_data_z = sorted(glob.glob('processed/sim01_100/data/sim_01_f140w_z3_*'))
    sim_01_JWST_data_x = sorted(glob.glob('processed/sim01_100/data/sim_01_F150W_x3_*'))
    sim_01_JWST_data_y = sorted(glob.glob('processed/sim01_100/data/sim_01_F150W_y3_*'))
    sim_01_JWST_data_z = sorted(glob.glob('processed/sim01_100/data/sim_01_F150W_z3_*'))

    sim_02_HST_data_x = sorted(glob.glob('processed/sim02_100/data/sim_02_f140w_x3_*'))
    sim_02_HST_data_y = sorted(glob.glob('processed/sim02_100/data/sim_02_f140w_y3_*'))
    sim_02_HST_data_z = sorted(glob.glob('processed/sim02_100/data/sim_02_f140w_z3_*'))
    sim_02_JWST_data_x = sorted(glob.glob('processed/sim02_100/data/sim_02_F150W_x3_*'))
    sim_02_JWST_data_y = sorted(glob.glob('processed/sim02_100/data/sim_02_F150W_y3_*'))
    sim_02_JWST_data_z = sorted(glob.glob('processed/sim02_100/data/sim_02_F150W_z3_*'))

    sim_03_HST_data_x = sorted(glob.glob('processed/sim03_100/data/sim_03_f140w_x3_*'))
    sim_03_HST_data_y = sorted(glob.glob('processed/sim03_100/data/sim_03_f140w_y3_*'))
    sim_03_HST_data_z = sorted(glob.glob('processed/sim03_100/data/sim_03_f140w_z3_*'))
    sim_03_JWST_data_x = sorted(glob.glob('processed/sim03_100/data/sim_03_F150W_x3_*'))
    sim_03_JWST_data_y = sorted(glob.glob('processed/sim03_100/data/sim_03_F150W_y3_*'))
    sim_03_JWST_data_z = sorted(glob.glob('processed/sim03_100/data/sim_03_F150W_z3_*'))

    sim_04_HST_data_x = sorted(glob.glob('processed/sim04_100/data/sim_04_f140w_x3_*'))
    sim_04_HST_data_y = sorted(glob.glob('processed/sim04_100/data/sim_04_f140w_y3_*'))
    sim_04_HST_data_z = sorted(glob.glob('processed/sim04_100/data/sim_04_f140w_z3_*'))
    sim_04_JWST_data_x = sorted(glob.glob('processed/sim04_100/data/sim_04_F150W_x3_*'))
    sim_04_JWST_data_y = sorted(glob.glob('processed/sim04_100/data/sim_04_F150W_y3_*'))
    sim_04_JWST_data_z = sorted(glob.glob('processed/sim04_100/data/sim_04_F150W_z3_*'))

    sim_06_HST_data_x = sorted(glob.glob('processed/sim06_100/data/sim_06_f140w_x3_*'))
    sim_06_HST_data_y = sorted(glob.glob('processed/sim06_100/data/sim_06_f140w_y3_*'))
    sim_06_HST_data_z = sorted(glob.glob('processed/sim06_100/data/sim_06_f140w_z3_*'))
    sim_06_JWST_data_x = sorted(glob.glob('processed/sim06_100/data/sim_06_F150W_x3_*'))
    sim_06_JWST_data_y = sorted(glob.glob('processed/sim06_100/data/sim_06_F150W_y3_*'))
    sim_06_JWST_data_z = sorted(glob.glob('processed/sim06_100/data/sim_06_F150W_z3_*'))

    sim_09_HST_data_x = sorted(glob.glob('processed/sim09_100/data/sim_09_f140w_x3_*'))
    sim_09_HST_data_y = sorted(glob.glob('processed/sim09_100/data/sim_09_f140w_y3_*'))
    sim_09_HST_data_z = sorted(glob.glob('processed/sim09_100/data/sim_09_f140w_z3_*'))
    sim_09_JWST_data_x = sorted(glob.glob('processed/sim09_100/data/sim_09_F150W_x3_*'))
    sim_09_JWST_data_y = sorted(glob.glob('processed/sim09_100/data/sim_09_F150W_y3_*'))
    sim_09_JWST_data_z = sorted(glob.glob('processed/sim09_100/data/sim_09_F150W_z3_*'))

    sim_01_number_of_sources = np.zeros((len(sim_01_HST_data_x), 2, 3))
    sim_02_number_of_sources = np.zeros((len(sim_02_HST_data_x), 2, 3))
    sim_03_number_of_sources = np.zeros((len(sim_03_HST_data_x), 2, 3))
    sim_04_number_of_sources = np.zeros((len(sim_04_HST_data_x), 2, 3))
    sim_06_number_of_sources = np.zeros((len(sim_06_HST_data_x), 2, 3))
    sim_09_number_of_sources = np.zeros((len(sim_09_HST_data_x), 2, 3))

    sim_01_redshifts = []
    for i in range(0,len(sim_01_HST_data_x)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim01_info[i], skiprows=1)
        sim_01_redshifts.append(redshift)
    sim_01_ang_size = Angular_size

    sim_02_redshifts = []
    for i in range(0,len(sim_02_HST_data_x)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim02_info[i], skiprows=1)
        sim_02_redshifts.append(redshift)
    sim_02_ang_size = Angular_size

    sim_03_redshifts = []
    for i in range(0,len(sim_03_HST_data_x)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim03_info[i], skiprows=1)
        sim_03_redshifts.append(redshift)
    sim_03_ang_size = Angular_size

    sim_04_redshifts = []
    for i in range(0,len(sim_04_HST_data_x)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim04_info[i], skiprows=1)
        sim_04_redshifts.append(redshift)
    sim_04_ang_size = Angular_size

    sim_06_redshifts = []
    for i in range(0,len(sim_06_HST_data_x)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim06_info[i], skiprows=1)
        sim_06_redshifts.append(redshift)
    sim_06_ang_size = Angular_size

    sim_09_redshifts = []
    for i in range(0,len(sim_09_HST_data_x)):
        redshift, D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim09_info[i], skiprows=1)
        sim_09_redshifts.append(redshift)
    sim_09_ang_size = Angular_size

    print(1)

    threshold = 2
    npixels = 3

    for i in range(0,len(sim_01_HST_data_x)):

        temp = np.loadtxt(sim_01_HST_data_x[i])
        sim_01_number_of_sources[i, 0, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_01_HST_data_y[i])
        sim_01_number_of_sources[i, 0, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_01_HST_data_z[i])
        sim_01_number_of_sources[i, 0, 2] = detect_sources(temp, threshold, npixels).nlabels

        temp = np.loadtxt(sim_01_JWST_data_x[i])
        sim_01_number_of_sources[i, 1, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_01_JWST_data_y[i])
        sim_01_number_of_sources[i, 1, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_01_JWST_data_z[i])
        sim_01_number_of_sources[i, 1, 2] = detect_sources(temp, threshold, npixels).nlabels

    print(2)

    for i in range(0,len(sim_02_HST_data_x)):

        temp = np.loadtxt(sim_02_HST_data_x[i])
        sim_02_number_of_sources[i, 0, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_02_HST_data_y[i])
        sim_02_number_of_sources[i, 0, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_02_HST_data_z[i])
        sim_02_number_of_sources[i, 0, 2] = detect_sources(temp, threshold, npixels).nlabels

        temp = np.loadtxt(sim_02_JWST_data_x[i])
        sim_02_number_of_sources[i, 1, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_02_JWST_data_y[i])
        sim_02_number_of_sources[i, 1, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_02_JWST_data_z[i])
        sim_02_number_of_sources[i, 1, 2] = detect_sources(temp, threshold, npixels).nlabels

    print(3)

    for i in range(0,len(sim_03_HST_data_x)):

        temp = np.loadtxt(sim_03_HST_data_x[i])
        sim_03_number_of_sources[i, 0, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_03_HST_data_y[i])
        sim_03_number_of_sources[i, 0, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_03_HST_data_z[i])
        sim_03_number_of_sources[i, 0, 2] = detect_sources(temp, threshold, npixels).nlabels

        temp = np.loadtxt(sim_03_JWST_data_x[i])
        sim_03_number_of_sources[i, 1, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_03_JWST_data_y[i])
        sim_03_number_of_sources[i, 1, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_03_JWST_data_z[i])
        sim_03_number_of_sources[i, 1, 2] = detect_sources(temp, threshold, npixels).nlabels

    print(4)

    for i in range(0,len(sim_04_HST_data_x)):

        temp = np.loadtxt(sim_04_HST_data_x[i])
        sim_04_number_of_sources[i, 0, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_04_HST_data_y[i])
        sim_04_number_of_sources[i, 0, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_04_HST_data_z[i])
        sim_04_number_of_sources[i, 0, 2] = detect_sources(temp, threshold, npixels).nlabels

        temp = np.loadtxt(sim_04_JWST_data_x[i])
        sim_04_number_of_sources[i, 1, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_04_JWST_data_y[i])
        sim_04_number_of_sources[i, 1, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_04_JWST_data_z[i])
        sim_04_number_of_sources[i, 1, 2] = detect_sources(temp, threshold, npixels).nlabels

    print(5)

    for i in range(0,len(sim_06_HST_data_x)):

        temp = np.loadtxt(sim_06_HST_data_x[i])
        sim_06_number_of_sources[i, 0, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_06_HST_data_y[i])
        sim_06_number_of_sources[i, 0, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_06_HST_data_z[i])
        sim_06_number_of_sources[i, 0, 2] = detect_sources(temp, threshold, npixels).nlabels

        temp = np.loadtxt(sim_06_JWST_data_x[i])
        sim_06_number_of_sources[i, 1, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_06_JWST_data_y[i])
        sim_06_number_of_sources[i, 1, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_06_JWST_data_z[i])
        sim_06_number_of_sources[i, 1, 2] = detect_sources(temp, threshold, npixels).nlabels

    print(6)

    for i in range(0,len(sim_09_HST_data_x)):

        temp = np.loadtxt(sim_09_HST_data_x[i])
        sim_09_number_of_sources[i, 0, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_09_HST_data_y[i])
        sim_09_number_of_sources[i, 0, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_09_HST_data_z[i])
        sim_09_number_of_sources[i, 0, 2] = detect_sources(temp, threshold, npixels).nlabels

        temp = np.loadtxt(sim_09_JWST_data_x[i])
        sim_09_number_of_sources[i, 1, 0] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_09_JWST_data_y[i])
        sim_09_number_of_sources[i, 1, 1] = detect_sources(temp, threshold, npixels).nlabels
        temp = np.loadtxt(sim_09_JWST_data_z[i])
        sim_09_number_of_sources[i, 1, 2] = detect_sources(temp, threshold, npixels).nlabels

    print(7)

    HST_WFC3CAM_pixel_size = 0.13   # arcsec per pixel
    JWST_NIRCAM_pixel_size = 0.032  # arcsec per pixel

    def pairs(data_sources,N,X,Y):

        if(N>1):
            x_coord = np.zeros(N)
            y_coord = np.zeros(N)

            sources_distances = np.zeros(N)
            pairs_distances = []
            pairs = 0

            for j in range(1,N+1):
                A = np.argwhere(data_sources==j)
                x_coord[j-1] = np.mean(X[A[:,0],A[:,1]])
                y_coord[j-1] = np.mean(Y[A[:,0],A[:,1]])

            for i in range(0,N-1):
                A = np.sqrt(np.power((x_coord[i+1:]-x_coord[i]),2) + np.power((y_coord[i+1:]-y_coord[i]),2))
                B = np.where(A<=3)
                pairs += len(B[0])
                for ii in range(len(B[0])):
                    sources_distances[i] += 1
                    sources_distances[i+1+B[0][ii]] += 1

                pairs_distances = np.concatenate((pairs_distances, A[B[0]]))

            if(len(pairs_distances)>0):
                return pairs, np.mean(pairs_distances), len(np.where(sources_distances>0)[0])
            else:
                return pairs, 0, 0
        else:
            return 0,0,0

    # ------------------------------------------------------------------------------------------------

    print(8)

    Nbins_HST = int(sim_01_ang_size/HST_WFC3CAM_pixel_size)
    ang = sim_01_ang_size/2

    pixels_ang_coords_HST = (np.linspace(-ang, ang, Nbins_HST + 1) + np.linspace(-ang, ang, Nbins_HST + 1))/2
    X_HST,Y_HST = np.meshgrid(pixels_ang_coords_HST,pixels_ang_coords_HST)
    sources_sim01_HST = np.zeros((len(sim_01_HST_data_x),3,3))

    for i in range(0,len(sim_01_HST_data_x)):

        temp = np.loadtxt(sim_01_HST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim01_HST[i,0,0], sources_sim01_HST[i,0,1], sources_sim01_HST[i,0,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_01_HST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim01_HST[i,1,0], sources_sim01_HST[i,1,1], sources_sim01_HST[i,1,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_01_HST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim01_HST[i,2,0], sources_sim01_HST[i,2,1], sources_sim01_HST[i,2,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

    Nbins_JWST = int(sim_01_ang_size/JWST_NIRCAM_pixel_size)
    ang = sim_01_ang_size/2

    pixels_ang_coords_JWST = (np.linspace(-ang, ang, Nbins_JWST + 1) + np.linspace(-ang, ang, Nbins_JWST + 1))/2
    X_JWST,Y_JWST = np.meshgrid(pixels_ang_coords_JWST,pixels_ang_coords_JWST)
    sources_sim01_JWST = np.zeros((len(sim_01_HST_data_x),3,3))

    for i in range(0,len(sim_01_HST_data_x)):

        temp = np.loadtxt(sim_01_JWST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim01_JWST[i,0,0], sources_sim01_JWST[i,0,1], sources_sim01_JWST[i,0,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_01_JWST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim01_JWST[i,1,0], sources_sim01_JWST[i,1,1], sources_sim01_JWST[i,1,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_01_JWST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim01_JWST[i,2,0], sources_sim01_JWST[i,2,1], sources_sim01_JWST[i,2,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

    # -------------------------------------------------------------------------------------------

    print(9)

    Nbins_HST = int(sim_02_ang_size/HST_WFC3CAM_pixel_size)
    ang = sim_02_ang_size/2

    pixels_ang_coords_HST = (np.linspace(-ang, ang, Nbins_HST + 1) + np.linspace(-ang, ang, Nbins_HST + 1))/2
    X_HST,Y_HST = np.meshgrid(pixels_ang_coords_HST,pixels_ang_coords_HST)
    sources_sim02_HST = np.zeros((len(sim_02_HST_data_x),3,3))

    for i in range(0,len(sim_02_HST_data_x)):

        temp = np.loadtxt(sim_02_HST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim02_HST[i,0,0], sources_sim02_HST[i,0,1], sources_sim02_HST[i,0,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_02_HST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim02_HST[i,1,0], sources_sim02_HST[i,1,1], sources_sim02_HST[i,1,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_02_HST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim02_HST[i,2,0], sources_sim02_HST[i,2,1], sources_sim02_HST[i,2,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

    Nbins_JWST = int(sim_02_ang_size/JWST_NIRCAM_pixel_size)
    ang = sim_02_ang_size/2

    pixels_ang_coords_JWST = (np.linspace(-ang, ang, Nbins_JWST + 1) + np.linspace(-ang, ang, Nbins_JWST + 1))/2
    X_JWST,Y_JWST = np.meshgrid(pixels_ang_coords_JWST,pixels_ang_coords_JWST)
    sources_sim02_JWST = np.zeros((len(sim_02_HST_data_x),3,3))

    for i in range(0,len(sim_02_HST_data_x)):

        temp = np.loadtxt(sim_02_JWST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim02_JWST[i,0,0], sources_sim02_JWST[i,0,1], sources_sim02_JWST[i,0,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_02_JWST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim02_JWST[i,1,0], sources_sim02_JWST[i,1,1], sources_sim02_JWST[i,1,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_02_JWST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim02_JWST[i,2,0], sources_sim02_JWST[i,2,1], sources_sim02_JWST[i,2,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

    # ----------------------------------------------------------------------------------------------------

    print(10)

    Nbins_HST = int(sim_03_ang_size/HST_WFC3CAM_pixel_size)
    ang = sim_03_ang_size/2

    pixels_ang_coords_HST = (np.linspace(-ang, ang, Nbins_HST + 1) + np.linspace(-ang, ang, Nbins_HST + 1))/2
    X_HST,Y_HST = np.meshgrid(pixels_ang_coords_HST,pixels_ang_coords_HST)
    sources_sim03_HST = np.zeros((len(sim_03_HST_data_x),3,3))

    for i in range(0,len(sim_03_HST_data_x)):

        temp = np.loadtxt(sim_03_HST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim03_HST[i,0,0], sources_sim03_HST[i,0,1], sources_sim03_HST[i,0,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_03_HST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim03_HST[i,1,0], sources_sim03_HST[i,1,1], sources_sim03_HST[i,1,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_03_HST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim03_HST[i,2,0], sources_sim03_HST[i,2,1], sources_sim03_HST[i,2,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

    Nbins_JWST = int(sim_03_ang_size/JWST_NIRCAM_pixel_size)
    ang = sim_03_ang_size/2
    pixels_ang_coords_JWST = (np.linspace(-ang, ang, Nbins_JWST + 1) + np.linspace(-ang, ang, Nbins_JWST + 1))/2
    X_JWST,Y_JWST = np.meshgrid(pixels_ang_coords_JWST,pixels_ang_coords_JWST)
    sources_sim03_JWST = np.zeros((len(sim_03_HST_data_x),3,3))

    for i in range(0,len(sim_03_HST_data_x)):

        temp = np.loadtxt(sim_03_JWST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim03_JWST[i,0,0], sources_sim03_JWST[i,0,1], sources_sim03_JWST[i,0,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_03_JWST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim03_JWST[i,1,0], sources_sim03_JWST[i,1,1], sources_sim03_JWST[i,1,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_03_JWST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim03_JWST[i,2,0], sources_sim03_JWST[i,2,1], sources_sim03_JWST[i,2,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

    # -----------------------------------------------------------------------------------------------------------------------------

    print(11)

    Nbins_HST = int(sim_04_ang_size/HST_WFC3CAM_pixel_size)
    ang = sim_04_ang_size/2

    pixels_ang_coords_HST = (np.linspace(-ang, ang, Nbins_HST + 1) + np.linspace(-ang, ang, Nbins_HST + 1))/2
    X_HST,Y_HST = np.meshgrid(pixels_ang_coords_HST,pixels_ang_coords_HST)
    sources_sim04_HST = np.zeros((len(sim_04_HST_data_x),3,3))

    for i in range(0,len(sim_04_HST_data_x)):

        temp = np.loadtxt(sim_04_HST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim04_HST[i,0,0], sources_sim04_HST[i,0,1], sources_sim04_HST[i,0,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_04_HST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim04_HST[i,1,0], sources_sim04_HST[i,1,1], sources_sim04_HST[i,1,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_04_HST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim04_HST[i,2,0], sources_sim04_HST[i,2,1], sources_sim04_HST[i,2,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

    Nbins_JWST = int(sim_04_ang_size/JWST_NIRCAM_pixel_size)
    ang = sim_04_ang_size/2
    pixels_ang_coords_JWST = (np.linspace(-ang, ang, Nbins_JWST + 1) + np.linspace(-ang, ang, Nbins_JWST + 1))/2
    X_JWST,Y_JWST = np.meshgrid(pixels_ang_coords_JWST,pixels_ang_coords_JWST)
    sources_sim04_JWST = np.zeros((len(sim_04_HST_data_x),3,3))

    for i in range(0,len(sim_04_HST_data_x)):

        temp = np.loadtxt(sim_04_JWST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim04_JWST[i,0,0], sources_sim04_JWST[i,0,1], sources_sim04_JWST[i,0,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_04_JWST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim04_JWST[i,1,0], sources_sim04_JWST[i,1,1], sources_sim04_JWST[i,1,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_04_JWST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim04_JWST[i,2,0], sources_sim04_JWST[i,2,1], sources_sim04_JWST[i,2,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

    # ------------------------------------------------------------------------------------------------------------------------

    print(12)

    Nbins_HST = int(sim_06_ang_size/HST_WFC3CAM_pixel_size)
    ang = sim_06_ang_size/2

    pixels_ang_coords_HST = (np.linspace(-ang, ang, Nbins_HST + 1) + np.linspace(-ang, ang, Nbins_HST + 1))/2
    X_HST,Y_HST = np.meshgrid(pixels_ang_coords_HST,pixels_ang_coords_HST)
    sources_sim06_HST = np.zeros((len(sim_06_HST_data_x),3,3))

    for i in range(0,len(sim_06_HST_data_x)):

        temp = np.loadtxt(sim_06_HST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim06_HST[i,0,0], sources_sim06_HST[i,0,1], sources_sim06_HST[i,0,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_06_HST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim06_HST[i,1,0], sources_sim06_HST[i,1,1], sources_sim06_HST[i,1,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_06_HST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim06_HST[i,2,0], sources_sim06_HST[i,2,1], sources_sim06_HST[i,2,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)


    Nbins_JWST = int(sim_06_ang_size/JWST_NIRCAM_pixel_size)
    ang = sim_06_ang_size/2
    pixels_ang_coords_JWST = (np.linspace(-ang, ang, Nbins_JWST + 1) + np.linspace(-ang, ang, Nbins_JWST + 1))/2
    X_JWST,Y_JWST = np.meshgrid(pixels_ang_coords_JWST,pixels_ang_coords_JWST)
    sources_sim06_JWST = np.zeros((len(sim_06_HST_data_x),3,3))

    for i in range(0,len(sim_06_HST_data_x)):

        temp = np.loadtxt(sim_06_JWST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim06_JWST[i,0,0], sources_sim06_JWST[i,0,1], sources_sim06_JWST[i,0,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_06_JWST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim06_JWST[i,1,0], sources_sim06_JWST[i,1,1], sources_sim06_JWST[i,1,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_06_JWST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim06_JWST[i,2,0], sources_sim06_JWST[i,2,1], sources_sim06_JWST[i,2,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

    # -----------------------------------------------------------------------------------------------------------------

    print(13)

    Nbins_HST = int(sim_09_ang_size/HST_WFC3CAM_pixel_size)
    ang = sim_09_ang_size/2

    pixels_ang_coords_HST = (np.linspace(-ang, ang, Nbins_HST + 1) + np.linspace(-ang, ang, Nbins_HST + 1))/2
    X_HST,Y_HST = np.meshgrid(pixels_ang_coords_HST,pixels_ang_coords_HST)
    sources_sim09_HST = np.zeros((len(sim_09_HST_data_x),3,3))

    for i in range(0,len(sim_09_HST_data_x)):

        temp = np.loadtxt(sim_09_HST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim09_HST[i,0,0], sources_sim09_HST[i,0,1], sources_sim09_HST[i,0,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_09_HST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim09_HST[i,1,0], sources_sim09_HST[i,1,1], sources_sim09_HST[i,1,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

        temp = np.loadtxt(sim_09_HST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim09_HST[i,2,0], sources_sim09_HST[i,2,1], sources_sim09_HST[i,2,2] = pairs(np.array(data),data.nlabels,X_HST,Y_HST)

    Nbins_JWST = int(sim_09_ang_size/JWST_NIRCAM_pixel_size)
    ang = sim_09_ang_size/2
    pixels_ang_coords_JWST = (np.linspace(-ang, ang, Nbins_JWST + 1) + np.linspace(-ang, ang, Nbins_JWST + 1))/2
    X_JWST,Y_JWST = np.meshgrid(pixels_ang_coords_JWST,pixels_ang_coords_JWST)
    sources_sim09_JWST = np.zeros((len(sim_09_HST_data_x),3,3))

    for i in range(0,len(sim_09_HST_data_x)):

        temp = np.loadtxt(sim_09_JWST_data_x[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim09_JWST[i,0,0], sources_sim09_JWST[i,0,1], sources_sim09_JWST[i,0,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_09_JWST_data_y[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim09_JWST[i,1,0], sources_sim09_JWST[i,1,1], sources_sim09_JWST[i,1,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

        temp = np.loadtxt(sim_09_JWST_data_z[i])
        data = detect_sources(temp, threshold, npixels)
        sources_sim09_JWST[i,2,0], sources_sim09_JWST[i,2,1], sources_sim09_JWST[i,2,2] = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)

    #  --------------------------------------------------------------------------------------------------------------
    #  --------------------------------------------------------------------------------------------------------------

    print(14)

    sources_sim_3_JWST_x = np.hstack([sim_03_number_of_sources[0,1,0], 0, sim_03_number_of_sources[1:-1,1,0], 0, sim_03_number_of_sources[-1,1,0], 0, 0])
    sources_sim_3_JWST_y = np.hstack([sim_03_number_of_sources[0,1,1], 0, sim_03_number_of_sources[1:-1,1,1], 0, sim_03_number_of_sources[-1,1,1], 0, 0])
    sources_sim_3_JWST_z = np.hstack([sim_03_number_of_sources[0,1,2], 0, sim_03_number_of_sources[1:-1,1,2], 0, sim_03_number_of_sources[-1,1,2], 0, 0])
    sources_sim_3_HST_x = np.hstack([sim_03_number_of_sources[0,0,0],  0, sim_03_number_of_sources[1:-1,0,0], 0, sim_03_number_of_sources[-1,0,0], 0, 0])
    sources_sim_3_HST_y = np.hstack([sim_03_number_of_sources[0,0,1],  0, sim_03_number_of_sources[1:-1,0,1], 0, sim_03_number_of_sources[-1,0,1], 0, 0])
    sources_sim_3_HST_z = np.hstack([sim_03_number_of_sources[0,0,2],  0, sim_03_number_of_sources[1:-1,0,2], 0, sim_03_number_of_sources[-1,0,2], 0, 0])

    sources_sim_6_JWST_x = np.hstack([sim_06_number_of_sources[:,1,0],0])
    sources_sim_6_JWST_y = np.hstack([sim_06_number_of_sources[:,1,1],0])
    sources_sim_6_JWST_z = np.hstack([sim_06_number_of_sources[:,1,2],0])
    sources_sim_6_HST_x = np.hstack([sim_06_number_of_sources[:,0,0], 0])
    sources_sim_6_HST_y = np.hstack([sim_06_number_of_sources[:,0,1], 0])
    sources_sim_6_HST_z = np.hstack([sim_06_number_of_sources[:,0,2], 0])

    groups_sim_3_JWST_x = np.hstack([sources_sim03_JWST[0,0,2], 0, sources_sim03_JWST[1:-1,0,2], 0, sources_sim03_JWST[-1,0,2], 0, 0])
    groups_sim_3_JWST_y = np.hstack([sources_sim03_JWST[0,1,2], 0, sources_sim03_JWST[1:-1,1,2], 0, sources_sim03_JWST[-1,1,2], 0, 0])
    groups_sim_3_JWST_z = np.hstack([sources_sim03_JWST[0,2,2], 0, sources_sim03_JWST[1:-1,2,2], 0, sources_sim03_JWST[-1,2,2], 0, 0])
    groups_sim_3_HST_x = np.hstack([sources_sim03_HST[0,0,2],   0, sources_sim03_HST[1:-1,0,2],  0, sources_sim03_HST[-1,0,2],  0, 0])
    groups_sim_3_HST_y = np.hstack([sources_sim03_HST[0,1,2],   0, sources_sim03_HST[1:-1,1,2],  0, sources_sim03_HST[-1,1,2],  0, 0])
    groups_sim_3_HST_z = np.hstack([sources_sim03_HST[0,2,2],   0, sources_sim03_HST[1:-1,2,2],  0, sources_sim03_HST[-1,2,2],  0, 0])

    groups_sim_6_JWST_x = np.hstack([sources_sim06_JWST[:,0,2],0])
    groups_sim_6_JWST_y = np.hstack([sources_sim06_JWST[:,1,2],0])
    groups_sim_6_JWST_z = np.hstack([sources_sim06_JWST[:,2,2],0])
    groups_sim_6_HST_x = np.hstack([sources_sim06_HST[:,0,2], 0])
    groups_sim_6_HST_y = np.hstack([sources_sim06_HST[:,1,2], 0])
    groups_sim_6_HST_z = np.hstack([sources_sim06_HST[:,2,2], 0])

    HST_03_temp_redshifts = np.hstack([sim_03_redshifts[0],0,sim_03_redshifts[1:-1],0,sim_03_redshifts[-1],0,0])
    HST_06_temp_redshifts = np.hstack([sim_06_redshifts,0])

    redshifts_save = np.vstack((np.array(sim_01_redshifts),
                                np.array(sim_02_redshifts),
                                HST_03_temp_redshifts,
                                np.array(sim_04_redshifts),
                                HST_06_temp_redshifts,
                                np.array(sim_09_redshifts)))

    JWST_groups_save = np.vstack((sources_sim01_JWST[:,0,2],
                                  sources_sim01_JWST[:,1,2],
                                  sources_sim01_JWST[:,2,2],
                                  sources_sim02_JWST[:,0,2],
                                  sources_sim02_JWST[:,1,2],
                                  sources_sim02_JWST[:,2,2],
                                  groups_sim_3_JWST_x,
                                  groups_sim_3_JWST_y,
                                  groups_sim_3_JWST_z,
                                  sources_sim04_JWST[:,0,2],
                                  sources_sim04_JWST[:,1,2],
                                  sources_sim04_JWST[:,2,2],
                                  groups_sim_6_JWST_x,
                                  groups_sim_6_JWST_y,
                                  groups_sim_6_JWST_z,
                                  sources_sim09_JWST[:,0,2],
                                  sources_sim09_JWST[:,1,2],
                                  sources_sim09_JWST[:,2,2]))

    HST_groups_save = np.vstack((sources_sim01_HST[:,0,2],
                                  sources_sim01_HST[:,1,2],
                                  sources_sim01_HST[:,2,2],
                                  sources_sim02_HST[:,0,2],
                                  sources_sim02_HST[:,1,2],
                                  sources_sim02_HST[:,2,2],
                                  groups_sim_3_HST_x,
                                  groups_sim_3_HST_y,
                                  groups_sim_3_HST_z,
                                  sources_sim04_HST[:,0,2],
                                  sources_sim04_HST[:,1,2],
                                  sources_sim04_HST[:,2,2],
                                  groups_sim_6_HST_x,
                                  groups_sim_6_HST_y,
                                  groups_sim_6_HST_z,
                                  sources_sim09_HST[:,0,2],
                                  sources_sim09_HST[:,1,2],
                                  sources_sim09_HST[:,2,2]))

    JWST_sources_save = np.vstack((sim_01_number_of_sources[:,1,0],
                                   sim_01_number_of_sources[:,1,1],
                                   sim_01_number_of_sources[:,1,2],
                                   sim_02_number_of_sources[:,1,0],
                                   sim_02_number_of_sources[:,1,1],
                                   sim_02_number_of_sources[:,1,2],
                                   sources_sim_3_JWST_x,
                                   sources_sim_3_JWST_y,
                                   sources_sim_3_JWST_z,
                                   sim_04_number_of_sources[:,1,0],
                                   sim_04_number_of_sources[:,1,1],
                                   sim_04_number_of_sources[:,1,2],
                                   sources_sim_6_JWST_x,
                                   sources_sim_6_JWST_y,
                                   sources_sim_6_JWST_z,
                                   sim_09_number_of_sources[:,1,0],
                                   sim_09_number_of_sources[:,1,1],
                                   sim_09_number_of_sources[:,1,2]))

    HST_sources_save = np.vstack((sim_01_number_of_sources[:,0,0],
                                  sim_01_number_of_sources[:,0,1],
                                  sim_01_number_of_sources[:,0,2],
                                  sim_02_number_of_sources[:,0,0],
                                  sim_02_number_of_sources[:,0,1],
                                  sim_02_number_of_sources[:,0,2],
                                  sources_sim_3_HST_x,
                                  sources_sim_3_HST_y,
                                  sources_sim_3_HST_z,
                                  sim_04_number_of_sources[:,0,0],
                                  sim_04_number_of_sources[:,0,1],
                                  sim_04_number_of_sources[:,0,2],
                                  sources_sim_6_HST_x,
                                  sources_sim_6_HST_y,
                                  sources_sim_6_HST_z,
                                  sim_09_number_of_sources[:,0,0],
                                  sim_09_number_of_sources[:,0,1],
                                  sim_09_number_of_sources[:,0,2]))

    np.savetxt('processed_data_groups_HST3.dat', HST_groups_save, fmt='%1.3e')
    np.savetxt('processed_data_groups_JWST3.dat', JWST_groups_save, fmt='%1.3e')
    np.savetxt('processed_data_sources_HST3.dat', HST_sources_save, fmt='%1.3e')
    np.savetxt('processed_data_sources_JWST3.dat', JWST_sources_save, fmt='%1.3e')
    np.savetxt('processed_data_redshifts.dat', redshifts_save, fmt='%1.3e')

def trial_function(x,a,b,d,e,c,f):
    return a + (b+e*x*c*x*x*f*x*x*x)*np.exp(-d*x)

from scipy import  stats


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
    return  noise_std


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

    JWST_data = sorted(glob.glob('processed/sim06_100/data/sim_06_F150W_z3_*'))
    JWST_data1 = sorted(glob.glob('processed/sim06_100/data/sim_06_F150W_z1_*'))

    image1 = np.loadtxt(JWST_data1[17])

    sim06_info = sorted(glob.glob('processed/sim06_100/info/info_rei00006_*'))

    sim_06_redshifts = []
    for i in range(0,len(JWST_data)):
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

    image = np.loadtxt(JWST_data[17])
    data = detect_sources(image, 2.5,3)
    N = data.nlabels
    if(N>1):
        x_coord = np.zeros(N)
        y_coord = np.zeros(N)
        for j in range(1,N+1):
            A = np.argwhere(np.array(data)==j)
            x_coord[j-1] = np.mean(X[A[:,0],A[:,1]])
            y_coord[j-1] = np.mean(Y[A[:,0],A[:,1]])

    extent = np.array([-ang,ang,-ang,ang])
    sigma = init_noise_JWST()

    plt.imshow(image1,interpolation='nearest',cmap=plt.cm.gray,vmax=2*sigma,extent=extent)
    plt.xlim(-theta_arcsec,theta_arcsec)
    plt.ylim(-theta_arcsec,theta_arcsec)

    positions = (x_coord, -y_coord)
    apertures = CircularAperture(positions, r=0.5)
    apertures.plot(color='red', lw=2, alpha=0.5)

    plt.yticks([])
    plt.xticks([])
    plt.savefig('JWST_sources.pdf',format='pdf')
    plt.show()

def sources_HST():

    plot_style()

    HST_data = sorted(glob.glob('processed/sim06_100/data/sim_06_f140w_z3_*'))
    HST_data1 = sorted(glob.glob('processed/sim06_100/data/sim_06_f140w_z1_*'))

    image1 = np.loadtxt(HST_data1[17])

    sim06_info = sorted(glob.glob('processed/sim06_100/info/info_rei00006_*'))

    sim_06_redshifts = []
    for i in range(0,len(HST_data)):
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

    image = np.loadtxt(HST_data[17])
    data = detect_sources(image, 2.0, 3)
    N = data.nlabels
    if(N>1):
        x_coord = np.zeros(N)
        y_coord = np.zeros(N)
        for j in range(1,N+1):
            A = np.argwhere(np.array(data)==j)
            x_coord[j-1] = np.mean(X[A[:,0],A[:,1]])
            y_coord[j-1] = -np.mean(Y[A[:,0],A[:,1]])

    extent = np.array([-ang,ang,-ang,ang])
    sigma = init_noise()

    plt.imshow(image1,interpolation='nearest',cmap=plt.cm.gray,vmax=2*sigma,extent=extent)
    plt.xlim(-theta_arcsec,theta_arcsec)
    plt.ylim(-theta_arcsec,theta_arcsec)

    positions = (x_coord,y_coord)
    apertures = CircularAperture(positions, r=0.7)
    apertures.plot(color='blue', lw=2.5, alpha=0.5)

    plt.yticks([])
    plt.xticks([])
    plt.savefig('HST_sources.pdf',format='pdf')
    plt.show()

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

def sources_groups_ratios(): #[N Groups]

    redshifts = np.loadtxt('processed_data_redshifts4.dat')
    redshift_sample = np.mean(redshifts, axis=0)

    sources_HST = np.loadtxt('processed_data_sources_HST4.dat')
    sources_JWST = np.loadtxt('processed_data_sources_JWST4.dat')
    groups_HST = np.loadtxt('processed_data_groups_HST4.dat')
    groups_JWST = np.loadtxt('processed_data_groups_JWST4.dat')

    sources_HST_mean  = np.zeros_like(sources_HST[:3,:])
    sources_JWST_mean = np.zeros_like(sources_HST[:3,:])

    HST_ratios = (groups_HST + 1e-26) / (sources_HST + 1e-13)
    JWST_ratios = (groups_JWST + 1e-26) / (sources_JWST + 1e-13)

    mean = np.mean(HST_ratios, axis=0)
    sigma = np.std(HST_ratios, axis=0)
    sources_HST_mean[1,:] = mean
    sources_HST_mean[0,:] = mean - sigma
    sources_HST_mean[2,:] = mean + sigma

    mean = np.mean(JWST_ratios , axis=0)
    sigma = np.std(JWST_ratios , axis=0)
    sources_JWST_mean[1,:] = mean
    sources_JWST_mean[0,:] = mean - sigma
    sources_JWST_mean[2,:] = mean + sigma

    plot_style()
    xx = np.array([6,7,8,9,10,11])
    yy = np.array([0,0.2,0.4,0.6,0.8,1.0])
    plt.yticks(yy, fontsize=24)
    plt.xticks(xx, fontsize=24)
    plt.xlim(6,11)
    plt.ylim(0,1)

    plt.fill_between(redshifts[-1,:],sources_JWST_mean[0,:], sources_JWST_mean[2,:], facecolor='red', interpolate=True, alpha=0.5)
    plt.fill_between(redshifts[-1,:],sources_HST_mean[0,:], sources_HST_mean[2,:], facecolor='blue', interpolate=True, alpha=0.5)
    plt.plot(redshift_sample,sources_JWST_mean[1,:], color='red',lw=3,label='JWST')
    plt.plot(redshift_sample,sources_HST_mean[1,:], color='blue',lw=4,label='HST',ls='--')

    plt.ylabel('Fraction of sources in groups',fontsize=22)
    plt.xlabel('$z$',fontsize=24)

    plt.legend(loc='upper right',fontsize=22)
    plt.savefig('groups.pdf',format='pdf')
    plt.show()

def sources_groups_ratios_culumative(): # [groups culumative]

    redshifts = np.loadtxt('processed_data_redshifts4.dat')
    redshift_sample = np.mean(redshifts, axis=0)

    sources_HST = np.loadtxt('processed_data_sources_HST4.dat')
    sources_JWST = np.loadtxt('processed_data_sources_JWST4.dat')
    groups_HST = np.loadtxt('processed_data_groups_HST4.dat')
    groups_JWST = np.loadtxt('processed_data_groups_JWST4.dat')

    HST_culumative = np.zeros_like(sources_HST)
    JWST_culumative = np.zeros_like(sources_HST)

    sources_HST_mean  = np.zeros_like(sources_HST[:3,:])
    sources_JWST_mean = np.zeros_like(sources_HST[:3,:])

    for i in range(np.shape(sources_HST)[0]):
        for j in range(np.shape(sources_HST)[1]):
            HST_culumative[i,j] = (integrate.trapz(y=groups_HST[i,:j][::-1],x=redshifts[i,:j][::-1]) + 1e-26) / \
                                  (integrate.trapz(y=sources_HST[i,:j][::-1],x=redshifts[i,:j][::-1]) + 1e-13)
            JWST_culumative[i,j] = (integrate.trapz(y=groups_JWST[i,:j][::-1],x=redshifts[i,:j][::-1]) + 1e-26) / \
                                   (integrate.trapz(y=sources_JWST[i,:j][::-1],x=redshifts[i,:j][::-1]) + 1e-13)

    mean = np.mean(HST_culumative, axis=0)
    sigma = np.std(HST_culumative, axis=0)
    sources_HST_mean[1,:] = mean
    sources_HST_mean[0,:] = mean - sigma
    sources_HST_mean[2,:] = mean + sigma

    mean = np.mean(JWST_culumative, axis=0)
    sigma = np.std(JWST_culumative, axis=0)
    sources_JWST_mean[1,:] = mean
    sources_JWST_mean[0,:] = mean - sigma
    sources_JWST_mean[2,:] = mean + sigma

    plot_style(yticks=5)
    xx = np.array([6,7,8,9,10,11])
    yy = np.array([0,0.2,0.4,0.6,0.8,1.0])
    plt.yticks(yy, fontsize=24)
    plt.xticks(xx, fontsize=24)
    plt.xlim(6,11)
    plt.ylim(0,1)

    plt.fill_between(redshifts[-1,:],sources_JWST_mean[0,:], sources_JWST_mean[2,:], facecolor='red', interpolate=True, alpha=0.5)
    plt.fill_between(redshifts[-1,:],sources_HST_mean[0,:], sources_HST_mean[2,:], facecolor='blue', interpolate=True, alpha=0.5)
    plt.plot(redshift_sample,sources_JWST_mean[1,:], color='red',lw=3,label='JWST')
    plt.plot(redshift_sample,sources_HST_mean[1,:], color='blue',lw=4,label='HST',ls='--')

    plt.plot(7,7/34,'ko',markersize=18)
    plt.plot(7,11/34,'ys',markersize=18)
    plt.ylabel('$ f^{m}_{>z} $',fontsize=26)
    plt.xlabel('$z$',fontsize=24)
    plt.legend(loc='upper right',fontsize=22)
    plt.savefig('groups_culumative.pdf',format='pdf')
    plt.show()

def sources_population():

    S_ang = np.loadtxt('S_ang.dat')
    dz_0 = np.loadtxt('dz.dat')
    redshifts = np.loadtxt('processed_data_redshifts4.dat')
    redshift_sample = np.mean(redshifts, axis=0)

    sources_HST = np.loadtxt('processed_data_sources_HST4.dat') / S_ang / dz_0
    sources_JWST = np.loadtxt('processed_data_sources_JWST4.dat') / S_ang / dz_0

    sources_HST_mean  = np.zeros_like(sources_HST[:3,:])
    sources_JWST_mean = np.zeros_like(sources_HST[:3,:])

    mean = np.mean(sources_HST, axis=0)
    sigma = np.std(sources_HST, axis=0)
    sources_HST_mean[1,:] = mean
    sources_HST_mean[0,:] = mean - sigma
    sources_HST_mean[2,:] = mean + sigma

    mean = np.mean(sources_JWST, axis=0)
    sigma = np.std(sources_JWST, axis=0)
    sources_JWST_mean[1,:] = mean
    sources_JWST_mean[0,:] = mean - sigma
    sources_JWST_mean[2,:] = mean + sigma

    plot_style()
    xx = np.array([6,7,8,9,10,11])
    yy = np.array([0,1,2,3,4,5])
    plt.yticks(yy, fontsize=24)
    plt.xticks(xx, fontsize=24)
    plt.xlim(6,11)
    plt.ylim(0,5)

    plt.fill_between(redshift_sample,sources_JWST_mean[0,:], sources_JWST_mean[2,:], facecolor='red', interpolate=True, alpha=0.5)
    plt.fill_between(redshift_sample,sources_HST_mean[0,:], sources_HST_mean[2,:], facecolor='blue', interpolate=True, alpha=0.5)
    plt.plot(redshift_sample,sources_JWST_mean[1,:], color='red',lw=3,label='JWST')
    plt.plot(redshift_sample,sources_HST_mean[1,:], color='blue',lw=4,label='HST',ls='--')

    plt.ylabel('$N / arcsec^{2} /dz$',fontsize=24)
    plt.xlabel('$z$',fontsize=24)

    plt.legend(loc='upper right',fontsize=22)
    plt.savefig('sources.pdf',format='pdf')
    plt.show()


def sources_population_culumative():

    dz_0 = np.loadtxt('dz.dat')
    S_ang = np.loadtxt('S_ang.dat')

    redshifts = np.loadtxt('processed_data_redshifts4.dat')
    redshift_sample = np.mean(redshifts, axis=0)

    sources_HST = np.loadtxt('processed_data_sources_HST4.dat') / S_ang / dz_0
    sources_JWST = np.loadtxt('processed_data_sources_JWST4.dat') / S_ang / dz_0

    sources_HST_culumative = np.zeros_like(sources_HST)
    sources_JWST_culumative = np.zeros_like(sources_HST)

    sources_HST_mean  = np.zeros_like(sources_HST[:3,:])
    sources_JWST_mean = np.zeros_like(sources_HST[:3,:])

    for i in range(np.shape(sources_HST)[0]):
        for j in range(np.shape(sources_HST)[1]):
            sources_HST_culumative[i,j] = integrate.trapz(y=sources_HST[i,:j][::-1],x=redshifts[i,:j][::-1])
            sources_JWST_culumative[i,j] = integrate.trapz(y=sources_JWST[i,:j][::-1],x=redshifts[i,:j][::-1])

    mean = np.mean(sources_HST_culumative, axis=0)
    sigma = np.std(sources_HST_culumative, axis=0)
    sources_HST_mean[1,:] = mean
    sources_HST_mean[0,:] = mean - sigma
    sources_HST_mean[2,:] = mean + sigma

    mean = np.mean(sources_JWST_culumative, axis=0)
    sigma = np.std(sources_JWST_culumative, axis=0)
    sources_JWST_mean[1,:] = mean
    sources_JWST_mean[0,:] = mean - sigma
    sources_JWST_mean[2,:] = mean + sigma

    xdf_data = np.loadtxt('XDF_data.dat')
    xdf_data = np.reshape(xdf_data,np.shape(xdf_data)[0]*np.shape(xdf_data)[1])

    plot_style()

    xx = np.array([6,7,8,9,10,11])
    yy = np.array([0.1,1,10,100,1000])
    plt.xticks(xx, fontsize=24)
    plt.xlim(6,11)
    A = np.max(sources_HST_mean[1,:])/155

    plt.hist(xdf_data, bins=np.linspace(5.9,11,1000), histtype='step', cumulative=-1,color='k',lw=3)
    plt.plot(redshift_sample,sources_JWST_mean[1,:]/A, color='red',lw=3,label='JWST')
    plt.plot(redshift_sample,sources_HST_mean[1,:]/A, color='blue',lw=4,label='HST',ls='--')
    plt.plot([1e4,1e5],[1e2,1e4],color='k',lw=3,label='XDF data')

    plt.fill_between(redshift_sample,sources_JWST_mean[0,:]/A, sources_JWST_mean[2,:]/A, facecolor='red', interpolate=True,alpha=0.5)
    plt.fill_between(redshift_sample,sources_HST_mean[0,:]/A, sources_HST_mean[2,:]/A, facecolor='blue', interpolate=True,alpha=0.5)

    plt.xlabel('$z$',fontsize=24)
    plt.ylabel('$ N_{>z} $',fontsize=26)
    plt.legend(loc='upper right',fontsize=22)
    plt.yscale('log')
    plt.ylim(0.1, 1000)
    plt.yticks(yy, fontsize=24)

    fixlogax(plt.gca(), a='y')

    plt.savefig('sources_culumative.pdf',format='pdf')
    plt.show()

def files_correction():

    file = np.loadtxt('dz.dat')
    #file = np.loadtxt('processed_data_redshifts3.dat')

    for j in range(np.shape(file)[1]):
        if(j==np.shape(file)[1]-1):
            numbers = np.array([6, 7, 8, 12, 13, 14])
            mean = np.mean(np.delete(file[:,j],numbers))
            file[:,j][numbers] = mean
        elif(j==np.shape(file)[1]-2):
            numbers = np.array([6, 7, 8])
            mean = np.mean(np.delete(file[:,j],numbers))
            file[:,j][numbers] = mean
        elif(j==np.shape(file)[1]-4):
            numbers = np.array([6, 7, 8])
            mean = np.mean(np.delete(file[:,j],numbers))
            file[:,j][numbers] = mean
        elif(j==1):
            numbers = np.array([6, 7, 8])
            mean = np.mean(np.delete(file[:,j],numbers))
            file[:,j][numbers] = mean

    #np.savetxt('processed_data_groups_HST4.dat', file, fmt='%1.3e')
    np.savetxt('dz.dat', file, fmt='%1.3e')


sources_groups_ratios()
sources_groups_ratios_culumative()

sources_population()
sources_population_culumative()


