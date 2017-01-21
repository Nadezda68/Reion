import glob
import numpy as np
from photutils import detect_sources
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
import matplotlib.cm as cm
from photutils.utils import random_cmap
from photutils import CircularAperture
from photutils import DAOStarFinder
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.scalebar import IMPERIAL_LENGTH
from scipy.interpolate import interp1d
import cosmolopy
from cosmolopy import distance as d
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

    print(sim_01_ang_size)
    print(sim_02_ang_size)
    print(sim_03_ang_size)
    print(sim_04_ang_size)
    print(sim_06_ang_size)
    print(sim_09_ang_size)

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

    HST_01 = (sim_01_number_of_sources[:, 0, 0] + sim_01_number_of_sources[:, 0, 1] + sim_01_number_of_sources[:, 0, 2])/3
    JWST_01 = (sim_01_number_of_sources[:, 1, 0] + sim_01_number_of_sources[:, 1, 1] + sim_01_number_of_sources[:, 1, 2])/3

    HST_02 = (sim_02_number_of_sources[:, 0, 0] + sim_02_number_of_sources[:, 0, 1] + sim_02_number_of_sources[:, 0, 2])/3
    JWST_02 = (sim_02_number_of_sources[:, 1, 0] + sim_02_number_of_sources[:, 1, 1] + sim_02_number_of_sources[:, 1, 2])/3

    HST_03 = (sim_03_number_of_sources[:, 0, 0] + sim_03_number_of_sources[:, 0, 1] + sim_03_number_of_sources[:, 0, 2])/3
    JWST_03 = (sim_03_number_of_sources[:, 1, 0] + sim_03_number_of_sources[:, 1, 1] + sim_03_number_of_sources[:, 1, 2])/3

    HST_04 = (sim_04_number_of_sources[:, 0, 0] + sim_04_number_of_sources[:, 0, 1] + sim_04_number_of_sources[:, 0, 2])/3
    JWST_04 = (sim_04_number_of_sources[:, 1, 0] + sim_04_number_of_sources[:, 1, 1] + sim_04_number_of_sources[:, 1, 2])/3

    HST_06 = (sim_06_number_of_sources[:, 0, 0] + sim_06_number_of_sources[:, 0, 1] + sim_06_number_of_sources[:, 0, 2])/3
    JWST_06 = (sim_06_number_of_sources[:, 1, 0] + sim_06_number_of_sources[:, 1, 1] + sim_06_number_of_sources[:, 1, 2])/3

    HST_09 = (sim_09_number_of_sources[:, 0, 0] + sim_09_number_of_sources[:, 0, 1] + sim_09_number_of_sources[:, 0, 2])/3
    JWST_09 = (sim_09_number_of_sources[:, 1, 0] + sim_09_number_of_sources[:, 1, 1] + sim_09_number_of_sources[:, 1, 2])/3

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

    redshift_mean = np.array(sim_01_redshifts) + \
                    np.array(sim_02_redshifts) + \
                    np.array(sim_04_redshifts) + \
                    np.array(sim_09_redshifts) +  \
                    HST_03_temp_redshifts + \
                    HST_06_temp_redshifts

    divider = np.hstack([6, 5, np.ones(len(HST_03[1:-1]))*6, 5, 6, 5, 4])
    redshift_mean /= divider

    print('Average Redshift:')
    print(np.round(redshift_mean,2))
    print(divider)

    redshifts_save = np.vstack((np.array(sim_01_redshifts),
                                np.array(sim_02_redshifts),
                                HST_03_temp_redshifts,
                                np.array(sim_04_redshifts),
                                HST_06_temp_redshifts,
                                np.array(sim_09_redshifts),
                                redshift_mean))

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

    print(15)

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

    extent=np.array([-ang,ang,-ang,ang])
    sigma = init_noise_JWST()
    plt.imshow(image1,interpolation='nearest',cmap=plt.cm.gray,vmax=2*sigma,extent=extent)
    plt.xlim(-theta_arcsec,theta_arcsec )
    plt.ylim(-theta_arcsec,theta_arcsec )
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

    extent=np.array([-ang,ang,-ang,ang])
    sigma = init_noise()
    plt.imshow(image1,interpolation='nearest',cmap=plt.cm.gray,vmax=2*sigma,extent=extent)
    plt.xlim(-theta_arcsec,theta_arcsec )
    plt.ylim(-theta_arcsec,theta_arcsec )
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

    redshifts = np.loadtxt('processed_data_redshifts.dat')

    redshifts_0 = np.vstack((redshifts[0,:],
                             redshifts[0,:],
                             redshifts[0,:],
                             redshifts[1,:],
                             redshifts[1,:],
                             redshifts[1,:],
                             redshifts[2,:],
                             redshifts[2,:],
                             redshifts[2,:],
                             redshifts[3,:],
                             redshifts[3,:],
                             redshifts[3,:],
                             redshifts[4,:],
                             redshifts[4,:],
                             redshifts[4,:],
                             redshifts[5,:],
                             redshifts[5,:],
                             redshifts[5,:]))


    sources_HST = np.loadtxt('processed_data_sources_HST3.dat')
    sources_JWST = np.loadtxt('processed_data_sources_JWST3.dat')
    groups_HST = np.loadtxt('processed_data_groups_HST3.dat')
    groups_JWST = np.loadtxt('processed_data_groups_JWST3.dat')

    sources_HST += 1e-15
    sources_JWST += 1e-15
    groups_HST += 1e-25
    groups_JWST += 1e-25

    print(np.max(groups_HST))
    gs_JWST = groups_JWST/sources_JWST
    gs_HST = groups_HST/sources_HST

    std_JWST = np.zeros(np.shape(redshifts)[1])
    std_HST = np.zeros(np.shape(redshifts)[1])
    mean_JWST = np.zeros(np.shape(redshifts)[1])
    mean_HST = np.zeros(np.shape(redshifts)[1])

    for j in range(np.shape(redshifts)[1]):
        if(j==np.shape(redshifts)[1]-1):
            std_JWST[j] = np.std(np.delete(gs_JWST[:,j],[6, 7, 8, 12, 13, 14]))
            std_HST[j] = np.std(np.delete(gs_HST[:,j],[6, 7, 8, 12, 13, 14]))
            mean_JWST[j] = np.mean(np.delete(gs_JWST[:,j],[6, 7, 8, 12, 13, 14]))
            mean_HST[j] = np.mean(np.delete(gs_HST[:,j],[6, 7, 8, 12, 13, 14]))
        elif(j==np.shape(redshifts)[1]-2):
            std_JWST[j] = np.std(np.delete(gs_JWST[:,j],[6, 7, 8]))
            std_HST[j] = np.std(np.delete(gs_HST[:,j],[6, 7, 8]))
            mean_JWST[j] = np.mean(np.delete(gs_JWST[:,j],[6, 7, 8]))
            mean_HST[j] = np.mean(np.delete(gs_HST[:,j],[6, 7, 8]))
        elif(j==np.shape(redshifts)[1]-4):
            std_JWST[j] = np.std(np.delete(gs_JWST[:,j],[6, 7, 8]))
            std_HST[j] = np.std(np.delete(gs_HST[:,j],[6, 7, 8]))
            mean_JWST[j] = np.mean(np.delete(gs_JWST[:,j],[6, 7, 8]))
            mean_HST[j] = np.mean(np.delete(gs_HST[:,j],[6, 7, 8]))
        else:
            std_JWST[j] = np.std(gs_JWST[:,j])
            std_HST[j] = np.std(gs_HST[:,j])
            mean_JWST[j] = np.mean(gs_JWST[:,j])
            mean_HST[j] = np.mean(gs_HST[:,j])

    plot_style()
    xx = np.array([6,7,8,9,10,11])
    yy = np.array([0,0.2,0.4,0.6,0.8,1.0])
    plt.yticks(yy, fontsize=24)
    plt.xticks(xx, fontsize=24)
    plt.xlim(6,11)
    plt.ylim(0,1)


    mean_JWST_sigma_1 = mean_JWST + std_JWST
    mean_JWST_sigma_2 = mean_JWST - std_JWST

    mean_HST_sigma_1 = mean_HST + std_HST
    mean_HST_sigma_2 = mean_HST - std_HST

    plt.fill_between(redshifts[-1,:],mean_JWST_sigma_2, mean_JWST_sigma_1, facecolor='red', interpolate=True,alpha=0.5)
    plt.fill_between(redshifts[-1,:],mean_HST_sigma_2, mean_HST_sigma_1, facecolor='blue', interpolate=True,alpha=0.5)
    plt.plot(redshifts[-1,:],mean_JWST, color='red',lw=3,label='JWST')
    plt.plot(redshifts[-1,:],mean_HST, color='blue',lw=4,label='HST',ls='--')

    plt.ylabel('Fraction of sources in groups',fontsize=22)
    plt.xlabel('z',fontsize=24)

    plt.legend(loc='upper right',fontsize=22)
    plt.savefig('groups.pdf',format='pdf')
    plt.show()

def grous_culumative(): # [groups culumative]

    redshifts = np.loadtxt('processed_data_redshifts.dat')
    redshift_sample = np.loadtxt('processed_data_redshifts.dat')[-1,:]

    redshifts_0 = np.vstack((redshifts[0,:],
                             redshifts[0,:],
                             redshifts[0,:],
                             redshifts[1,:],
                             redshifts[1,:],
                             redshifts[1,:],
                             redshifts[2,:],
                             redshifts[2,:],
                             redshifts[2,:],
                             redshifts[3,:],
                             redshifts[3,:],
                             redshifts[3,:],
                             redshifts[4,:],
                             redshifts[4,:],
                             redshifts[4,:],
                             redshifts[5,:],
                             redshifts[5,:],
                             redshifts[5,:]))


    s_HST = np.loadtxt('processed_data_sources_HST3.dat')
    s_JWST = np.loadtxt('processed_data_sources_JWST3.dat')
    g_HST = np.loadtxt('processed_data_groups_HST3.dat')
    g_JWST = np.loadtxt('processed_data_groups_JWST3.dat')

    std_JWST = np.zeros(np.shape(redshifts)[1])
    std_HST = np.zeros(np.shape(redshifts)[1])
    mean_JWST = np.zeros(np.shape(redshifts)[1])
    mean_HST = np.zeros(np.shape(redshifts)[1])
    stds_JWST = np.zeros(np.shape(redshifts)[1])
    stds_HST = np.zeros(np.shape(redshifts)[1])
    means_JWST = np.zeros(np.shape(redshifts)[1])
    means_HST = np.zeros(np.shape(redshifts)[1])

    for j in range(np.shape(redshifts)[1]):
        if(j==np.shape(redshifts)[1]-1):
            std_JWST[j] = np.std(np.delete(g_JWST[:,j],[6, 7, 8, 12, 13, 14]))
            std_HST[j] = np.std(np.delete(g_HST[:,j],[6, 7, 8, 12, 13, 14]))
            mean_JWST[j] = np.mean(np.delete(g_JWST[:,j],[6, 7, 8, 12, 13, 14]))
            mean_HST[j] = np.mean(np.delete(g_HST[:,j],[6, 7, 8, 12, 13, 14]))
        elif(j==np.shape(redshifts)[1]-2):
            std_JWST[j] = np.std(np.delete(g_JWST[:,j],[6, 7, 8]))
            std_HST[j] = np.std(np.delete(g_HST[:,j],[6, 7, 8]))
            mean_JWST[j] = np.mean(np.delete(g_JWST[:,j],[6, 7, 8]))
            mean_HST[j] = np.mean(np.delete(g_HST[:,j],[6, 7, 8]))
        elif(j==np.shape(redshifts)[1]-4):
            std_JWST[j] = np.std(np.delete(g_JWST[:,j],[6, 7, 8]))
            std_HST[j] = np.std(np.delete(g_HST[:,j],[6, 7, 8]))
            mean_JWST[j] = np.mean(np.delete(g_JWST[:,j],[6, 7, 8]))
            mean_HST[j] = np.mean(np.delete(g_HST[:,j],[6, 7, 8]))
        else:
            std_JWST[j] = np.std(g_JWST[:,j])
            std_HST[j] = np.std(g_HST[:,j])
            mean_JWST[j] = np.mean(g_JWST[:,j])
            mean_HST[j] = np.mean(g_HST[:,j])

    for j in range(np.shape(redshifts)[1]):
        if(j==np.shape(redshifts)[1]-1):
            stds_JWST[j] = np.std(np.delete(s_JWST[:,j],[6, 7, 8, 12, 13, 14]))
            stds_HST[j] = np.std(np.delete(s_HST[:,j],[6, 7, 8, 12, 13, 14]))
            means_JWST[j] = np.mean(np.delete(s_JWST[:,j],[6, 7, 8, 12, 13, 14]))
            means_HST[j] = np.mean(np.delete(s_HST[:,j],[6, 7, 8, 12, 13, 14]))
        elif(j==np.shape(redshifts)[1]-2):
            stds_JWST[j] = np.std(np.delete(s_JWST[:,j],[6, 7, 8]))
            stds_HST[j] = np.std(np.delete(s_HST[:,j],[6, 7, 8]))
            means_JWST[j] = np.mean(np.delete(s_JWST[:,j],[6, 7, 8]))
            means_HST[j] = np.mean(np.delete(s_HST[:,j],[6, 7, 8]))
        elif(j==np.shape(redshifts)[1]-4):
            stds_JWST[j] = np.std(np.delete(s_JWST[:,j],[6, 7, 8]))
            stds_HST[j] = np.std(np.delete(s_HST[:,j],[6, 7, 8]))
            means_JWST[j] = np.mean(np.delete(s_JWST[:,j],[6, 7, 8]))
            means_HST[j] = np.mean(np.delete(s_HST[:,j],[6, 7, 8]))
        else:
            stds_JWST[j] = np.std(s_JWST[:,j])
            stds_HST[j] = np.std(s_HST[:,j])
            means_JWST[j] = np.mean(s_JWST[:,j])
            means_HST[j] = np.mean(s_HST[:,j])

    plot_style(yticks=5)
    xx = np.array([6,7,8,9,10,11])
    yy = np.array([0,0.2,0.4,0.6,0.8,1.0])
    plt.yticks(yy, fontsize=24)
    plt.xticks(xx, fontsize=24)
    plt.xlim(6,11)
    plt.ylim(0,1)

    print(std_JWST)
    print(stds_JWST)
    mean_JWST_sigma_1 = mean_JWST + std_JWST
    mean_JWST_sigma_2 = mean_JWST - std_JWST

    mean_HST_sigma_1 = mean_HST + std_HST
    mean_HST_sigma_2 = mean_HST - std_HST

    f_JWST_value = np.zeros(len(redshift_sample)-1)
    f_HST_value = np.zeros(len(redshift_sample)-1)

    f_JWST_valuep = np.zeros(len(redshift_sample)-1)
    f_HST_valuep = np.zeros(len(redshift_sample)-1)

    f_JWST_valuem = np.zeros(len(redshift_sample)-1)
    f_HST_valuem = np.zeros(len(redshift_sample)-1)


    means_JWST_sigma_1 = means_JWST + stds_JWST
    means_JWST_sigma_2 = means_JWST - stds_JWST

    means_HST_sigma_1 = means_HST + stds_HST
    means_HST_sigma_2 = means_HST - stds_HST

    fs_JWST_value = np.zeros(len(redshift_sample)-1)
    fs_HST_value = np.zeros(len(redshift_sample)-1)

    fs_JWST_valuep = np.zeros(len(redshift_sample)-1)
    fs_HST_valuep = np.zeros(len(redshift_sample)-1)

    fs_JWST_valuem = np.zeros(len(redshift_sample)-1)
    fs_HST_valuem = np.zeros(len(redshift_sample)-1)


    dz = np.abs((redshift_sample[1:][::-1] - redshift_sample[:-1][::-1]))
    z_mean = (redshift_sample[1:] + redshift_sample[:-1])/2


    mean_JWST_int = (mean_JWST[1:] + mean_JWST[:-1])/2
    mean_HST_int = (mean_HST[1:] + mean_HST[:-1])/2

    plus_JWST_int = (mean_JWST_sigma_1[1:] + mean_JWST_sigma_1[:-1])/2
    plus_HST_int = (mean_HST_sigma_1[1:] + mean_HST_sigma_1[:-1])/2

    minus_JWST_int = (mean_JWST_sigma_2[1:] + mean_JWST_sigma_2[:-1])/2
    minus_HST_int = (mean_HST_sigma_2[1:] + mean_HST_sigma_2[:-1])/2


    means_JWST_int = (means_JWST[1:] + means_JWST[:-1])/2
    means_HST_int = (means_HST[1:] + means_HST[:-1])/2

    pluss_JWST_int = (means_JWST_sigma_1[1:] + means_JWST_sigma_1[:-1])/2
    pluss_HST_int = (means_HST_sigma_1[1:] + means_HST_sigma_1[:-1])/2

    minuss_JWST_int = (means_JWST_sigma_2[1:] + means_JWST_sigma_2[:-1])/2
    minuss_HST_int = (means_HST_sigma_2[1:] + means_HST_sigma_2[:-1])/2

    for i in range(len(redshift_sample)-1):

        f_JWST_value[i] = np.sum(mean_JWST_int[:i][::-1]*dz[:i]) + 1e-26
        f_HST_value[i] = np.sum(mean_HST_int[:i][::-1]*dz[:i]) + 1e-26

        f_JWST_valuep[i] = np.sum(plus_JWST_int[:i][::-1]*dz[:i]) + 1e-26
        f_HST_valuep[i] = np.sum(plus_HST_int[:i][::-1]*dz[:i]) + 1e-26

        f_JWST_valuem[i] = np.sum(minus_JWST_int[:i][::-1]*dz[:i]) + 1e-26
        f_HST_valuem[i] = np.sum(minus_HST_int[:i][::-1]*dz[:i]) + 1e-26

        fs_JWST_value[i] = np.sum(means_JWST_int[:i][::-1]*dz[:i]) + 1e-13
        fs_HST_value[i] = np.sum(means_HST_int[:i][::-1]*dz[:i]) + 1e-13

        fs_JWST_valuep[i] = np.sum(pluss_JWST_int[:i][::-1]*dz[:i]) + 1e-13
        fs_HST_valuep[i] = np.sum(pluss_HST_int[:i][::-1]*dz[:i]) + 1e-13

        fs_JWST_valuem[i] = np.sum(minuss_JWST_int[:i][::-1]*dz[:i]) + 1e-13
        fs_HST_valuem[i] = np.sum(minuss_HST_int[:i][::-1]*dz[:i]) + 1e-13

    for i in range(len(f_JWST_valuem)):
        if(f_JWST_valuem[i]<0):
            f_JWST_valuem[i] = 0

    for i in range(len(f_HST_valuem)):
        if(f_HST_valuem[i]<0):
            f_HST_valuem[i] = 0

    for i in range(len(f_JWST_valuem)):
        if(f_JWST_valuem[i]<0):
            f_JWST_valuem[i] = 0
    '''
    for i in range(len(f_HST_valuem)):
        if(0>=fs_HST_valuem[i]):
            fs_HST_valuem[i] = 1e7
        if(0>=fs_HST_valuep[i]):
            fs_HST_valuep[i] = 1e7
        if(0>=fs_HST_value[i]):
            fs_HST_value[i] = 1e7
        if(0>=fs_JWST_valuem[i]):
            fs_JWST_valuem[i] = 1e7
        if(0>=fs_JWST_valuep[i]):
            fs_JWST_valuep[i] = 1e7
        if(0>=fs_JWST_value[i]):
            fs_JWST_value[i] = 1e7
    '''



    plt.plot(z_mean,f_JWST_value/fs_JWST_value, color='red',lw=3,label='JWST')
    plt.plot(z_mean,f_HST_value/fs_HST_value, color='blue',lw=4,label='HST',ls='--')
    plt.fill_between(z_mean,f_JWST_valuem/fs_JWST_valuem, f_JWST_valuep/fs_JWST_valuep, facecolor='red', interpolate=True,alpha=0.5)
    plt.fill_between(z_mean,f_HST_valuem/fs_HST_valuem, f_HST_valuep/fs_HST_valuep, facecolor='blue', interpolate=True,alpha=0.5)
    plt.plot(7,7/34,'ko',markersize=18)
    plt.plot(7,11/34,'ys',markersize=18)
    plt.ylabel('$ \\rm f^{m}_{>z} $',fontsize=26)
    plt.xlabel('z',fontsize=24)
    plt.legend(loc='upper right',fontsize=22)
    plt.savefig('groups_culumative.pdf',format='pdf')
    plt.show()

def sources_population():

    redshifts = np.loadtxt('processed_data_redshifts.dat')

    redshifts_0 = np.vstack((redshifts[0,:],
                             redshifts[0,:],
                             redshifts[0,:],
                             redshifts[1,:],
                             redshifts[1,:],
                             redshifts[1,:],
                             redshifts[2,:],
                             redshifts[2,:],
                             redshifts[2,:],
                             redshifts[3,:],
                             redshifts[3,:],
                             redshifts[3,:],
                             redshifts[4,:],
                             redshifts[4,:],
                             redshifts[4,:],
                             redshifts[5,:],
                             redshifts[5,:],
                             redshifts[5,:]))
    '''
    dz = np.zeros_like(redshifts)
    D_phys = 0.2  # [Mpc]
    z_sample = np.linspace(0,0.02,5000)
    for i in range(np.shape(redshifts)[0]):
        print(i)
        for j in range(np.shape(redshifts)[1]):
            for k in range(len(z_sample)):
                D_comoving = D_phys*(1+redshifts[i,j])
                dist = d.comoving_distance(redshifts[i,j]+z_sample[k],redshifts[i,j], **cosmo)
                if dist>=D_comoving:
                    dz[i,j] = z_sample[k]
                    break

    dz_0 = np.vstack((dz[0,:],
                     dz[0,:],
                     dz[0,:],
                     dz[1,:],
                     dz[1,:],
                     dz[1,:],
                     dz[2,:],
                     dz[2,:],
                     dz[2,:],
                     dz[3,:],
                     dz[3,:],
                     dz[3,:],
                     dz[4,:],
                     dz[4,:],
                     dz[4,:],
                     dz[5,:],
                     dz[5,:],
                     dz[5,:]))

    np.savetxt('dz.dat', dz_0, fmt='%1.3e')
    '''
    dz_0 = np.loadtxt('dz.dat')

    '''
    S_ang_s1 = np.pi*np.power(32.3735689973/2,2) * np.ones(np.shape(redshifts)[1])
    S_ang_s2 = np.pi*np.power(32.4246750341/2,2) * np.ones(np.shape(redshifts)[1])
    S_ang_s3 = np.pi*np.power(33.9930433782/2,2) * np.ones(np.shape(redshifts)[1])
    S_ang_s4 = np.pi*np.power(32.4306058448/2,2) * np.ones(np.shape(redshifts)[1])
    S_ang_s6 = np.pi*np.power(33.213491504/2,2)  * np.ones(np.shape(redshifts)[1])
    S_ang_s9 = np.pi*np.power(32.4911839894/2,2) * np.ones(np.shape(redshifts)[1])

    S_ang = np.vstack((S_ang_s1,
                       S_ang_s1,
                       S_ang_s1,
                       S_ang_s2,
                       S_ang_s2,
                       S_ang_s2,
                       S_ang_s3,
                       S_ang_s3,
                       S_ang_s3,
                       S_ang_s4,
                       S_ang_s4,
                       S_ang_s4,
                       S_ang_s6,
                       S_ang_s6,
                       S_ang_s6,
                       S_ang_s9,
                       S_ang_s9,
                       S_ang_s9))

    np.savetxt('S_ang.dat', S_ang, fmt='%1.3e')
    '''
    S_ang = np.loadtxt('S_ang.dat')

    sources_HST = np.loadtxt('processed_data_sources_HST3.dat') / S_ang / dz_0
    sources_JWST = np.loadtxt('processed_data_sources_JWST3.dat') /S_ang / dz_0

    std_JWST = np.zeros(np.shape(redshifts)[1])
    std_HST = np.zeros(np.shape(redshifts)[1])
    mean_JWST = np.zeros(np.shape(redshifts)[1])
    mean_HST = np.zeros(np.shape(redshifts)[1])

    for j in range(np.shape(redshifts)[1]):
        if(j==np.shape(redshifts)[1]-1):
            std_JWST[j] = np.std(np.delete(sources_JWST[:,j],[6, 7, 8, 12, 13, 14]))
            std_HST[j] = np.std(np.delete(sources_HST[:,j],[6, 7, 8, 12, 13, 14]))
            mean_JWST[j] = np.mean(np.delete(sources_JWST[:,j],[6, 7, 8, 12, 13, 14]))
            mean_HST[j] = np.mean(np.delete(sources_HST[:,j],[6, 7, 8, 12, 13, 14]))
        elif(j==np.shape(redshifts)[1]-2):
            std_JWST[j] = np.std(np.delete(sources_JWST[:,j],[6, 7, 8]))
            std_HST[j] = np.std(np.delete(sources_HST[:,j],[6, 7, 8]))
            mean_JWST[j] = np.mean(np.delete(sources_JWST[:,j],[6, 7, 8]))
            mean_HST[j] = np.mean(np.delete(sources_HST[:,j],[6, 7, 8]))
        elif(j==np.shape(redshifts)[1]-4):
            std_JWST[j] = np.std(np.delete(sources_JWST[:,j],[6, 7, 8]))
            std_HST[j] = np.std(np.delete(sources_HST[:,j],[6, 7, 8]))
            mean_JWST[j] = np.mean(np.delete(sources_JWST[:,j],[6, 7, 8]))
            mean_HST[j] = np.mean(np.delete(sources_HST[:,j],[6, 7, 8]))
        else:
            std_JWST[j] = np.std(sources_JWST[:,j])
            std_HST[j] = np.std(sources_HST[:,j])
            mean_JWST[j] = np.mean(sources_JWST[:,j])
            mean_HST[j] = np.mean(sources_HST[:,j])

    plot_style()
    xx = np.array([6,7,8,9,10,11])
    yy = np.array([0,1,2,3,4,5])
    plt.yticks(yy, fontsize=24)
    plt.xticks(xx, fontsize=24)
    plt.xlim(6,11)
    plt.ylim(0,5)

    mean_JWST_sigma_1 = mean_JWST + std_JWST
    mean_JWST_sigma_2 = mean_JWST - std_JWST

    mean_HST_sigma_1 = mean_HST + std_HST
    mean_HST_sigma_2 = mean_HST - std_HST

    plt.fill_between(redshifts[-1,:],mean_JWST_sigma_2, mean_JWST_sigma_1, facecolor='red', interpolate=True,alpha=0.5)
    plt.fill_between(redshifts[-1,:],mean_HST_sigma_2, mean_HST_sigma_1, facecolor='blue', interpolate=True,alpha=0.5)
    plt.plot(redshifts[-1,:],mean_JWST, color='red',lw=3,label='JWST')
    plt.plot(redshifts[-1,:],mean_HST, color='blue',lw=4,label='HST',ls='--')


    plt.ylabel('$\\rm \\frac{Number \\thinspace  \\thinspace of \\thinspace   \\thinspace sources}{arcsec^{2} \\thinspace dz}$',fontsize=32)
    plt.xlabel('z',fontsize=22)

    plt.legend(loc='upper right',fontsize=22)
    plt.savefig('sources.pdf',format='pdf')
    plt.show()

def culumative_sources():

    dz_0 = np.loadtxt('dz.dat')
    S_ang = np.loadtxt('S_ang.dat')
    redshift_sample = np.loadtxt('processed_data_redshifts.dat')[-1,:]
    redshifts = np.loadtxt('processed_data_redshifts.dat')

    sources_HST = np.loadtxt('processed_data_sources_HST3.dat') / S_ang / dz_0
    sources_JWST = np.loadtxt('processed_data_sources_JWST3.dat') /S_ang / dz_0

    std_JWST = np.zeros(np.shape(redshifts)[1])
    std_HST = np.zeros(np.shape(redshifts)[1])
    mean_JWST = np.zeros(np.shape(redshifts)[1])
    mean_HST = np.zeros(np.shape(redshifts)[1])

    for j in range(np.shape(redshifts)[1]):
        if(j==np.shape(redshifts)[1]-1):
            std_JWST[j] = np.std(np.delete(sources_JWST[:,j],[6, 7, 8, 12, 13, 14]))
            std_HST[j] = np.std(np.delete(sources_HST[:,j],[6, 7, 8, 12, 13, 14]))
            mean_JWST[j] = np.mean(np.delete(sources_JWST[:,j],[6, 7, 8, 12, 13, 14]))
            mean_HST[j] = np.mean(np.delete(sources_HST[:,j],[6, 7, 8, 12, 13, 14]))
        elif(j==np.shape(redshifts)[1]-2):
            std_JWST[j] = np.std(np.delete(sources_JWST[:,j],[6, 7, 8]))
            std_HST[j] = np.std(np.delete(sources_HST[:,j],[6, 7, 8]))
            mean_JWST[j] = np.mean(np.delete(sources_JWST[:,j],[6, 7, 8]))
            mean_HST[j] = np.mean(np.delete(sources_HST[:,j],[6, 7, 8]))
        elif(j==np.shape(redshifts)[1]-4):
            std_JWST[j] = np.std(np.delete(sources_JWST[:,j],[6, 7, 8]))
            std_HST[j] = np.std(np.delete(sources_HST[:,j],[6, 7, 8]))
            mean_JWST[j] = np.mean(np.delete(sources_JWST[:,j],[6, 7, 8]))
            mean_HST[j] = np.mean(np.delete(sources_HST[:,j],[6, 7, 8]))
        else:
            std_JWST[j] = np.std(sources_JWST[:,j])
            std_HST[j] = np.std(sources_HST[:,j])
            mean_JWST[j] = np.mean(sources_JWST[:,j])
            mean_HST[j] = np.mean(sources_HST[:,j])

    mean_JWST_sigma_1 = mean_JWST + std_JWST
    mean_JWST_sigma_2 = mean_JWST - std_JWST

    mean_HST_sigma_1 = mean_HST + std_HST
    mean_HST_sigma_2 = mean_HST - std_HST

    print(redshift_sample)
    print(mean_HST_sigma_1)
    print(mean_HST)
    print(mean_HST_sigma_2)

    f_JWST_value = np.zeros(len(redshift_sample)-1)
    f_HST_value = np.zeros(len(redshift_sample)-1)

    f_JWST_valuep = np.zeros(len(redshift_sample)-1)
    f_HST_valuep = np.zeros(len(redshift_sample)-1)

    f_JWST_valuem = np.zeros(len(redshift_sample)-1)
    f_HST_valuem = np.zeros(len(redshift_sample)-1)

    dz = np.abs((redshift_sample[1:][::-1] - redshift_sample[:-1][::-1]))
    z_mean = (redshift_sample[1:] + redshift_sample[:-1])/2

    mean_JWST_int = (mean_JWST[1:] + mean_JWST[:-1])/2
    mean_HST_int = (mean_HST[1:] + mean_HST[:-1])/2

    plus_JWST_int = (mean_JWST_sigma_1[1:] + mean_JWST_sigma_1[:-1])/2
    plus_HST_int = (mean_HST_sigma_1[1:] + mean_HST_sigma_1[:-1])/2

    minus_JWST_int = (mean_JWST_sigma_2[1:] + mean_JWST_sigma_2[:-1])/2
    minus_HST_int = (mean_HST_sigma_2[1:] + mean_HST_sigma_2[:-1])/2

    for i in range(len(redshift_sample)-1):

        f_JWST_value[i] = np.sum(mean_JWST_int[:i][::-1]*dz[:i])
        f_HST_value[i] = np.sum(mean_HST_int[:i][::-1]*dz[:i])

        f_JWST_valuep[i] = np.sum(plus_JWST_int[:i][::-1]*dz[:i])
        f_HST_valuep[i] = np.sum(plus_HST_int[:i][::-1]*dz[:i])

        f_JWST_valuem[i] = np.sum(minus_JWST_int[:i][::-1]*dz[:i])
        f_HST_valuem[i] = np.sum(minus_HST_int[:i][::-1]*dz[:i])

    xdf_data = np.array([ 3.68,  3.91,  4.37,  4.43,  3.63,  3.72,  3.91,  4.27,  3.58,
        3.68,  3.4 ,  4.43,  3.23,  3.49,  3.63,  3.63,  3.77,  3.77,
        4.01,  3.68,  3.87,  3.15,  4.48,  3.32,  3.77,  4.48,  4.27,
        3.36,  2.76,  3.58,  3.68,  3.32,  3.15,  3.82,  3.77,  3.23,
        4.01,  3.77,  3.72,  3.58,  3.54,  3.49,  3.63,  4.27,  4.48,
        3.87,  2.95,  4.22,  4.22,  4.22,  3.68,  4.37,  4.16,  2.79,
        3.63,  4.27,  2.91,  3.63,  3.91,  3.58,  3.36,  3.63,  4.43,
        3.19,  3.77,  3.72,  4.27,  3.72,  3.54,  4.27,  3.58,  3.58,
        3.36,  3.68,  3.82,  3.87,  3.54,  4.27,  3.49,  4.22,  3.36,
        4.22,  4.11,  3.63,  3.45,  3.58,  4.37,  3.96,  4.16,  4.01,
        4.48,  4.22,  3.68,  3.07,  3.68,  3.63,  3.72,  4.32,  3.87,
        4.11,  3.87,  3.91,  3.77,  4.37,  3.87,  4.01,  3.77,  4.54,
        3.23,  4.43,  3.72,  3.32,  3.96,  3.49,  4.22,  3.82,  4.22,
        4.54,  3.03,  3.87,  3.87,  3.49,  3.36,  3.77,  3.23,  3.96,
        3.4 ,  3.82,  3.63,  4.54,  3.68,  3.77,  4.16,  3.72,  3.54,
        3.82,  4.43,  3.54,  3.32,  4.48,  4.01,  3.77,  3.58,  3.77,
        3.32,  3.45,  3.63,  3.68,  2.83,  4.11,  4.01,  4.32,  4.48,
        3.49,  3.58,  3.77,  3.87,  3.45,  3.32,  3.68,  3.91,  3.54,
        3.36,  3.72,  3.36,  3.58,  4.48,  3.07,  3.36,  3.77,  3.58,
        3.45,  3.32,  3.87,  3.49,  3.87,  4.11,  4.27,  3.68,  2.95,
        3.32,  4.11,  3.91,  3.36,  3.68,  3.58,  3.32,  3.91,  3.91,
        3.58,  3.23,  3.63,  3.4 ,  3.72,  3.58,  3.96,  3.77,  4.06,
        3.58,  4.06,  3.87,  3.68,  3.63,  3.82,  4.32,  3.72,  3.87,
        4.37,  3.4 ,  4.06,  4.01,  3.91,  3.58,  3.68,  4.11,  4.27,
        3.72,  3.4 ,  4.16,  3.68,  3.4 ,  4.43,  3.49,  4.11,  3.87,
        2.95,  3.96,  3.49,  4.22,  3.58,  4.37,  3.45,  3.11,  3.96,
        3.49,  4.22,  3.87,  3.82,  3.77,  4.43,  3.45,  3.72,  3.58,
        4.16,  3.82,  4.22,  3.77,  4.37,  4.11,  3.72,  4.43,  3.87,
        3.77,  3.4 ,  3.82,  4.01,  3.45,  4.27,  2.83,  3.91,  4.37,
        3.91,  4.37,  4.59,  4.01,  4.06,  4.01,  3.72,  3.77,  3.54,
        3.82,  3.27,  3.77,  3.72,  3.03,  3.4 ,  3.49,  3.03,  3.36,
        2.95,  3.54,  3.82,  4.16,  3.49,  3.45,  3.58,  4.27,  3.36,
        3.72,  4.11,  3.49,  4.22,  3.27,  4.22,  4.06,  4.06,  4.37,
        4.22,  3.45,  3.91,  4.01,  3.68,  2.91,  3.49,  3.63,  3.49,
        4.06,  3.4 ,  3.96,  3.36,  3.54,  3.27,  3.77,  3.49,  4.22,
        4.27,  3.82,  3.36,  4.06,  2.79,  3.72,  4.11,  3.77,  4.01,
        3.54,  3.4 ,  4.37,  4.01,  3.96,  4.32,  3.19,  4.16,  2.91,
        4.11,  3.11,  4.43,  3.23,  4.43,  3.72,  3.32,  4.32,  3.72,
        3.4 ,  4.11,  3.45,  2.76,  3.72,  3.63,  4.27,  3.36,  3.49,
        3.68,  3.03,  3.11,  3.96,  3.27,  3.63,  3.32,  3.58,  3.58,
        3.91,  3.58,  4.01,  4.06,  3.49,  4.32,  3.45,  3.4 ,  4.76,
        4.32,  4.76,  4.65,  4.54,  5.06,  4.65,  4.7 ,  4.59,  4.94,
        5.24,  5.18,  4.65,  5.49,  4.94,  5.36,  5.43,  5.36,  4.94,
        4.76,  4.88,  4.43,  4.94,  4.59,  5.06,  5.36,  4.76,  5.  ,
        4.76,  5.06,  4.54,  5.  ,  4.7 ,  4.54,  4.76,  4.82,  4.88,
        4.82,  4.65,  5.3 ,  5.06,  5.43,  4.76,  4.65,  4.54,  4.76,
        4.48,  4.94,  4.82,  5.36,  4.06,  5.43,  5.24,  5.12,  5.24,
        4.65,  5.43,  4.65,  4.65,  4.76,  4.54,  4.76,  4.94,  5.3 ,
        4.94,  5.43,  4.65,  5.06,  5.36,  4.59,  5.56,  5.49,  5.36,
        4.88,  4.54,  4.65,  5.36,  4.7 ,  4.48,  4.43,  4.94,  4.48,
        5.06,  4.7 ,  4.82,  5.49,  4.54,  5.3 ,  5.06,  5.3 ,  5.12,
        4.43,  5.18,  5.06,  4.65,  4.65,  4.94,  4.7 ,  4.94,  5.06,
        5.3 ,  5.69,  5.24,  4.22,  4.48,  4.65,  5.18,  4.82,  4.76,
        4.59,  5.  ,  5.36,  5.49,  5.12,  5.18,  5.18,  4.54,  4.27,
        4.65,  5.  ,  5.49,  4.59,  4.54,  5.24,  4.65,  5.18,  4.82,
        4.65,  5.49,  4.59,  5.24,  5.18,  5.  ,  5.18,  5.49,  5.3 ,
        4.37,  5.82,  6.1 ,  6.1 ,  5.62,  5.62,  6.1 ,  5.49,  5.49,
        5.96,  5.82,  5.62,  6.32,  5.89,  6.1 ,  5.89,  5.76,  5.56,
        5.82,  5.49,  5.76,  5.62,  6.39,  5.96,  5.76,  5.82,  5.69,
        6.17,  6.24,  5.62,  6.1 ,  6.1 ,  5.96,  6.39,  6.1 ,  5.89,
        5.82,  5.82,  5.82,  6.1 ,  5.62,  6.03,  5.56,  6.17,  6.17,
        6.1 ,  6.03,  6.24,  5.76,  6.17,  6.24,  6.17,  6.17,  5.89,
        6.03,  6.32,  5.62,  5.69,  6.03,  6.03,  6.17,  6.17,  6.1 ,
        6.1 ,  5.96,  6.24,  6.32,  5.82,  6.1 ,  6.46,  5.76,  5.96,
        6.24,  6.03,  6.54,  5.69,  5.69,  6.03,  6.03,  6.32,  5.62,
        5.76,  6.03,  6.1 ,  6.1 ,  6.1 ,  5.24,  6.46,  6.46,  6.32,
        6.77,  6.32,  6.69,  6.39,  6.69,  7.33,  6.32,  7.08,  6.77,
        6.84,  7.  ,  6.24,  6.24,  6.61,  6.69,  7.  ,  6.39,  6.46,
        6.46,  6.61,  6.39,  6.69,  6.1 ,  6.32,  6.54,  6.77,  7.49,
        6.84,  6.92,  6.46,  7.75,  6.46,  6.54,  6.24,  7.08,  6.84,
        6.1 ,  6.54,  7.  ,  6.24,  6.92,  7.  ,  6.46,  6.84,  6.39,
        6.32,  6.84,  7.16,  6.17,  7.49,  6.54,  6.39,  7.75,  7.93,
        7.08,  7.49,  7.75,  7.93,  7.49,  7.33,  7.49,  7.16,  7.49,
        7.49,  7.58,  7.84,  7.49,  8.29,  8.11,  8.02,  8.2 ,  8.29,
        7.66,  7.75,  7.33,  7.58,  7.93,  9.78,  9.68])

    plot_style()
    xx = np.array([6,7,8,9,10,11])
    yy = np.array([0.1,1,10,100,1000])

    plt.xticks(xx, fontsize=24)
    plt.xlim(6,11)
    A = np.max(f_HST_value)/125
    plt.hist(xdf_data, bins=np.linspace(5.9,11,1000), histtype='step', cumulative=-1,color='k',lw=3)
    plt.plot(z_mean,f_JWST_value/A, color='red',lw=3,label='JWST')
    plt.plot(z_mean,f_HST_value/A, color='blue',lw=4,label='HST',ls='--')
    plt.plot([1e4,1e5],[1e2,1e4],color='k',lw=3,label='XDF data')
    plt.fill_between(z_mean,f_JWST_valuem/A, f_JWST_valuep/A, facecolor='red', interpolate=True,alpha=0.5)
    plt.fill_between(z_mean,f_HST_valuem/A, f_HST_valuep/A, facecolor='blue', interpolate=True,alpha=0.5)
    plt.xlabel('z',fontsize=22)
    plt.ylabel('$\\rm N_{>z} $',fontsize=26)
    plt.legend(loc='upper right',fontsize=22)
    plt.yscale('log')
    plt.ylim(0.1,1000)
    plt.yticks(yy, fontsize=24)
    fixlogax(plt.gca(), a='y')
    plt.savefig('sources_culumative.pdf',format='pdf')
    plt.show()

