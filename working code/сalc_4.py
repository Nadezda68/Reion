import numpy as np
import matplotlib.pylab as plt
import glob

from photutils import detect_sources
from photutils import CircularAperture
from photutils import aperture_photometry
from matplotlib.ticker import AutoMinorLocator
from astropy.io           import fits

HST_WFC3CAM_pixel_size = 0.13   # arcsec per pixel
JWST_NIRCAM_pixel_size = 0.032  # arcsec per pixel

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

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

def prob_int():
    xdf_data = np.loadtxt('XDF_data.dat')
    xdf_data = np.reshape(xdf_data,np.shape(xdf_data)[0]*np.shape(xdf_data)[1])
    n, bins = np.histogram(xdf_data, bins=np.linspace(5., 11, 100), normed=False)
    N = np.sum(n)
    n, bins = np.histogram(xdf_data, bins=np.linspace(5., 11, 100), normed=True)
    print(N)


    bin_centres = (bins[1:] + bins[:-1])/2
    S_field = 4 * 60 * 60  # arcsec

    def Gal_distribution(z1, z2):
        idx_1 = find_nearest(bin_centres,z1)
        idx_2 = find_nearest(bin_centres,z2)
        if idx_2 <= idx_1:
            return 0.0
        return 3 * 3 / 4 * np.pi / S_field * np.trapz(y = n[idx_1:idx_2+1], x = bin_centres[idx_1:idx_2+1])

    z_1 = np.array([5.8])
    z_2 = np.linspace(z_1[0], 11, 60)
    W = np.zeros((len(z_1), len(z_2))) # prob

    for k in range(len(z_1)):
        for i in range(len(z_2)):
            W[k,i] = np.power(Gal_distribution(z_1[k], z_2[i]), 2) * (N-1) * (N-2) / 2

    plot_style()
    plt.yscale('log')

    thickness = np.array([1.5, 3, 4.5])
    color = np.array(['darkred', 'sandybrown', 'blue'])
    labels = np.array(['$z_{0} \\thinspace = \\thinspace 5$',
                       '$z_{0} \\thinspace = \\thinspace 6$',
                       '$z_{0} \\thinspace = \\thinspace 7$'])

    for k in range(len(z_1)):
        plt.plot(z_2, W[k,:],color=color[k],lw=thickness[k],label=labels[k])

    xx = np.array([5,6,7,8,9,10,11])
    yy = np.array([0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000])/1e7
    plt.xticks(xx, fontsize=24)
    plt.yticks(yy, fontsize=24)
    plt.xlabel('$z$',fontsize=22)
    plt.ylabel('$W_{\\rm{gr}}$',fontsize=22)
    plt.legend(loc='lower right', fontsize=21)
    plt.ylim(1e-6,1e-2)
    plt.savefig('W.pdf',format='pdf')

    print(np.power(Gal_distribution(5.82, 7.49), 2) * (N-1) * (N-2) / 2)
    print(np.power(Gal_distribution(5.82, 7.49), 1) * (N-1))
    print(np.power(Gal_distribution(6, 11), 1) * (N-1))
    plt.show()

def pairs(data_sources,N,X,Y):

    if N>1:
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
                sources_distances[i] += 1 # 1st paired object (main)
                sources_distances[i+1+B[0][ii]] += 1 # 2nd paired object

            pairs_distances = np.concatenate((pairs_distances, A[B[0]]))

        if(len(pairs_distances)>0):
            return len(np.where(sources_distances>0)[0])
        else:
            return 0
    else:
        return 0

def fluxes(temp, data_sources, N, X, Y):

    if N>1:

        apertures_values = np.zeros(N)
        params = np.zeros(N)
        x_coord = np.zeros(N)
        y_coord = np.zeros(N)
        for j in range(1,N+1):
            A = np.argwhere(data_sources==j)
            x_coord[j-1] = np.mean(X[A[:,0],A[:,1]])
            y_coord[j-1] = np.mean(Y[A[:,0],A[:,1]])

        for i in range(0,N-1):
            A = np.sqrt(np.power((x_coord[i+1:]-x_coord[i]),2) + np.power((y_coord[i+1:]-y_coord[i]),2))
            B = np.where(A<=3)
            for ii in range(len(B[0])):
                params[i] = 1 # 1st paired object (main)
                params[i+1+B[0][ii]] = 1 # 2nd paired object

        for j in range(1,N+1):
            A = np.argwhere(data_sources==j)
            if np.mean(A[:,0])<=1 or np.mean(A[:,1]) <=1:
                aperture = CircularAperture([np.mean(A[:,1]),np.mean(A[:,0])], r = 1)
                apertures_values[j-1] = temp[A[0,0],A[0,1]]
                print(apertures_values[j-1])
            else:
                aperture = CircularAperture([np.mean(A[:,1]),np.mean(A[:,0])], r = (np.sqrt(len(A))+2))
                flux = aperture_photometry(temp, aperture)
                apertures_values[j-1] = flux['aperture_sum']

        return apertures_values[np.where(params==0)[0]]


    elif N==1:
        A = np.argwhere(data_sources==1)
        aperture = CircularAperture([np.mean(A[:,1]),np.mean(A[:,0])], r = (np.sqrt(len(A))+2))
        flux = aperture_photometry(temp, aperture)
        apertures_value = flux['aperture_sum']

        return apertures_value

    else:
        return []

def fluxes_group(temp, data_sources, N, X, Y):

    if N>1:
        apertures_values = np.zeros(N)
        params = np.zeros(N)
        x_coord = np.zeros(N)
        y_coord = np.zeros(N)

        for j in range(1,N+1):
            A = np.argwhere(data_sources==j)
            x_coord[j-1] = np.mean(X[A[:,0],A[:,1]])
            y_coord[j-1] = np.mean(Y[A[:,0],A[:,1]])

        for i in range(0,N-1):
            A = np.sqrt(np.power((x_coord[i+1:]-x_coord[i]),2) + np.power((y_coord[i+1:]-y_coord[i]),2))
            B = np.where(A<=3)
            for ii in range(len(B[0])):
                params[i] = 1 # 1st paired object (main)
                params[i+1+B[0][ii]] = 1 # 2nd paired object

        for j in range(1,N+1):
            A = np.argwhere(data_sources==j)
            if np.mean(A[:,0])<=1 or np.mean(A[:,1]) <=1:
                aperture = CircularAperture([np.mean(A[:,1]),np.mean(A[:,0])], r = 1)
                apertures_values[j-1] = temp[A[0,0],A[0,1]]
                print(apertures_values[j-1])
            else:
                aperture = CircularAperture([np.mean(A[:,1]),np.mean(A[:,0])], r = (np.sqrt(len(A))+2))
                flux = aperture_photometry(temp, aperture)
                apertures_values[j-1] = flux['aperture_sum']

        return apertures_values[np.where(params>0)[0]]
    else:
        return []

def lum(num=1, threshold = 2, npixels = 3):

    sim_info = sorted(glob.glob('processed/sim0' + str(num) + '_100/info/info_rei0000' + str(1) + '_*'))
    sources_HST = np.zeros((len(sim_info),18))
    sources_JWST = np.zeros((len(sim_info),18))
    groups_HST = np.zeros((len(sim_info),18))
    groups_JWST = np.zeros((len(sim_info),18))
    redshifts = np.zeros((len(sim_info),18))

    counter = 0

    ojb_gr_HST = open('objects_group_HST.dat','ab')
    ojb_gr_JWST = open('objects_group_JWST.dat','ab')
    ojb_is_HST = open('objects_alone_HST.dat','ab')
    ojb_is_JWST = open('objects_alone_JWST.dat','ab')

    for num in [1,2,3,4,6,9]:  # simulations

        print(num)

        angles = np.array([0, 32.3735689973, 32.4246750341, 33.9930433782, 32.4306058448, 0, 33.213491504, 0, 0, 32.4911839894])
        ang = angles[num]/2

        HST_WFC3CAM_pixel_size = 0.13   # arcsec per pixel
        JWST_NIRCAM_pixel_size = 0.032  # arcsec per pixel
        Nbins_HST = int(angles[num]/HST_WFC3CAM_pixel_size)
        Nbins_JWST = int(angles[num]/JWST_NIRCAM_pixel_size)

        pixels_ang_coords_HST = (np.linspace(-ang, ang, Nbins_HST + 1)[1:] + np.linspace(-ang, ang, Nbins_HST + 1)[:-1])/2
        pixels_ang_coords_JWST = (np.linspace(-ang, ang, Nbins_JWST + 1)[1:] + np.linspace(-ang, ang, Nbins_JWST + 1)[:-1])/2

        sim_info = sorted(glob.glob('processed/sim0' + str(num) + '_100/info/info_rei0000' + str(num) + '_*'))

        HST_data_x = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_f140w_x3_*'))
        HST_data_y = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_f140w_y3_*'))
        HST_data_z = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_f140w_z3_*'))

        JWST_data_x = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_F150W_x3_*'))
        JWST_data_y = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_F150W_y3_*'))
        JWST_data_z = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_F150W_z3_*'))

        HST_data_xv = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_f140w_x1_*'))
        HST_data_yv = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_f140w_y1_*'))
        HST_data_zv = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_f140w_z1_*'))

        JWST_data_xv = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_F150W_x1_*'))
        JWST_data_yv = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_F150W_y1_*'))
        JWST_data_zv = sorted(glob.glob('processed/sim0'+ str(num) +'_100/data/sim_0'+ str(num) +'_F150W_z1_*'))

        redshifts = np.zeros(len(sim_info))
        for i in range(len(HST_data_x)):
            redshifts[i], D_A, Angular_size, c_x, c_y, c_z, r = np.loadtxt(sim_info[i], skiprows=1)

        X_HST,Y_HST = np.meshgrid(pixels_ang_coords_HST,pixels_ang_coords_HST) # coord mesh
        X_JWST,Y_JWST = np.meshgrid(pixels_ang_coords_JWST,pixels_ang_coords_JWST)

        number_of_sources = np.zeros((len(HST_data_x), 2, 3))
        number_of_groups = np.zeros((len(HST_data_x), 2, 3))
        object_alone_fluxes_HST = []
        object_group_fluxes_HST = []
        object_alone_fluxes_JWST = []
        object_group_fluxes_JWST = []

        for i in range(len(sim_info)):


            # HST ------------------------------------------------
            temp = np.loadtxt(HST_data_x[i])
            temp2 = np.loadtxt(HST_data_xv[i])
            data = detect_sources(temp, threshold, npixels)
            number_of_sources[i, 0, 0] = data.nlabels
            number_of_groups[i, 0, 0]  = pairs(np.array(data),data.nlabels,X_HST,Y_HST)
            object_alone_fluxes_HST = np.concatenate((object_alone_fluxes_HST,fluxes(temp2,np.array(data),data.nlabels,X_HST,Y_HST)))
            object_group_fluxes_HST = np.concatenate((object_group_fluxes_HST,fluxes_group(temp2,np.array(data),data.nlabels,X_HST,Y_HST)))

            temp = np.loadtxt(HST_data_y[i])
            temp2 = np.loadtxt(HST_data_yv[i])
            data = detect_sources(temp, threshold, npixels)
            number_of_sources[i, 0, 1] = data.nlabels
            number_of_groups[i, 0, 1]  = pairs(np.array(data),data.nlabels,X_HST,Y_HST)
            object_alone_fluxes_HST = np.concatenate((object_alone_fluxes_HST,fluxes(temp2,np.array(data),data.nlabels,X_HST,Y_HST)))
            object_group_fluxes_HST = np.concatenate((object_group_fluxes_HST,fluxes_group(temp2,np.array(data),data.nlabels,X_HST,Y_HST)))

            temp = np.loadtxt(HST_data_z[i])
            temp2 = np.loadtxt(HST_data_zv[i])
            data = detect_sources(temp, threshold, npixels)
            number_of_sources[i, 0, 2] = data.nlabels
            number_of_groups[i, 0, 2]  = pairs(np.array(data),data.nlabels,X_HST,Y_HST)
            object_alone_fluxes_HST = np.concatenate((object_alone_fluxes_HST,fluxes(temp2,np.array(data),data.nlabels,X_HST,Y_HST)))
            object_group_fluxes_HST = np.concatenate((object_group_fluxes_HST,fluxes_group(temp2,np.array(data),data.nlabels,X_HST,Y_HST)))

            # JWST -----------------------------------------------
            temp = np.loadtxt(JWST_data_x[i])
            temp2 = np.loadtxt(JWST_data_xv[i])
            data = detect_sources(temp, threshold, npixels)
            number_of_sources[i, 1, 0] = data.nlabels
            number_of_groups[i, 1, 0]  = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)
            object_alone_fluxes_JWST = np.concatenate((object_alone_fluxes_JWST,fluxes(temp2,np.array(data),data.nlabels,X_JWST,Y_JWST)))
            object_group_fluxes_JWST = np.concatenate((object_group_fluxes_JWST,fluxes_group(temp2,np.array(data),data.nlabels,X_JWST,Y_JWST)))

            temp = np.loadtxt(JWST_data_y[i])
            temp2 = np.loadtxt(JWST_data_yv[i])
            data = detect_sources(temp, threshold, npixels)
            number_of_sources[i, 1, 1] = data.nlabels
            number_of_groups[i, 1, 1]  = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)
            object_alone_fluxes_JWST = np.concatenate((object_alone_fluxes_JWST,fluxes(temp2,np.array(data),data.nlabels,X_JWST,Y_JWST)))
            object_group_fluxes_JWST = np.concatenate((object_group_fluxes_JWST,fluxes_group(temp2,np.array(data),data.nlabels,X_JWST,Y_JWST)))

            temp = np.loadtxt(JWST_data_z[i])
            temp2 = np.loadtxt(JWST_data_zv[i])
            data = detect_sources(temp, threshold, npixels)
            number_of_sources[i, 1, 2] = data.nlabels
            number_of_groups[i, 1, 2]  = pairs(np.array(data),data.nlabels,X_JWST,Y_JWST)
            object_alone_fluxes_JWST = np.concatenate((object_alone_fluxes_JWST,fluxes(temp2,np.array(data),data.nlabels,X_JWST,Y_JWST)))
            object_group_fluxes_JWST = np.concatenate((object_group_fluxes_JWST,fluxes_group(temp2,np.array(data),data.nlabels,X_JWST,Y_JWST)))


        if num==3:

            sources_HST[:,3*counter] = np.hstack([number_of_sources[0, 0, 0], 0, number_of_sources[1:-1, 0, 0], 0, number_of_sources[-1, 0, 0], 0, 0])
            sources_HST[:,3*counter+1] = np.hstack([number_of_sources[0, 0, 1], 0, number_of_sources[1:-1, 0, 1], 0, number_of_sources[-1, 0, 1], 0, 0])
            sources_HST[:,3*counter+2] = np.hstack([number_of_sources[0, 0, 2], 0, number_of_sources[1:-1, 0, 2], 0, number_of_sources[-1, 0, 2], 0, 0])

            sources_JWST[:,3*counter] = np.hstack([number_of_sources[0, 1, 0], 0, number_of_sources[1:-1, 1, 0], 0, number_of_sources[-1, 1, 0], 0, 0])
            sources_JWST[:,3*counter+1] = np.hstack([number_of_sources[0, 1, 1], 0, number_of_sources[1:-1, 1, 1], 0, number_of_sources[-1, 1, 1], 0, 0])
            sources_JWST[:,3*counter+2] = np.hstack([number_of_sources[0, 1, 2], 0, number_of_sources[1:-1, 1, 2], 0, number_of_sources[-1, 1, 2], 0, 0])

            groups_HST[:,3*counter] = np.hstack([number_of_groups[0, 0, 0], 0, number_of_groups[1:-1, 0, 0], 0, number_of_groups[-1, 0, 0], 0, 0])
            groups_HST[:,3*counter+1] = np.hstack([number_of_groups[0, 0, 1], 0, number_of_groups[1:-1, 0, 1], 0, number_of_groups[-1, 0, 1], 0, 0])
            groups_HST[:,3*counter+2] = np.hstack([number_of_groups[0, 0, 2], 0, number_of_groups[1:-1, 0, 2], 0, number_of_groups[-1, 0, 2], 0, 0])

            groups_JWST[:,3*counter] = np.hstack([number_of_groups[0, 1, 0], 0, number_of_groups[1:-1, 1, 0], 0, number_of_groups[-1, 1, 0], 0, 0])
            groups_JWST[:,3*counter+1] = np.hstack([number_of_groups[0, 1, 1], 0, number_of_groups[1:-1, 1, 1], 0, number_of_groups[-1, 1, 1], 0, 0])
            groups_JWST[:,3*counter+2] = np.hstack([number_of_groups[0, 1, 2], 0, number_of_groups[1:-1, 1, 2], 0, number_of_groups[-1, 1, 2], 0, 0])

            counter += 1

        elif num==6:
            sources_HST[:,3*counter] = np.hstack([number_of_sources[:, 0, 0], 0])
            sources_HST[:,3*counter+1] = np.hstack([number_of_sources[:, 0, 1], 0])
            sources_HST[:,3*counter+2] = np.hstack([number_of_sources[:, 0, 2], 0])

            sources_JWST[:,3*counter] = np.hstack([number_of_sources[:, 1, 0], 0])
            sources_JWST[:,3*counter+1] = np.hstack([number_of_sources[:, 1, 1], 0])
            sources_JWST[:,3*counter+2] = np.hstack([number_of_sources[:, 1, 2], 0])

            groups_HST[:,3*counter] = np.hstack([number_of_groups[:, 0, 0], 0])
            groups_HST[:,3*counter+1] = np.hstack([number_of_groups[:, 0, 1], 0])
            groups_HST[:,3*counter+2] = np.hstack([number_of_groups[:, 0, 2], 0])

            groups_JWST[:,3*counter] = np.hstack([number_of_groups[:, 1, 0], 0])
            groups_JWST[:,3*counter+1] = np.hstack([number_of_groups[:, 1, 1], 0])
            groups_JWST[:,3*counter+2] = np.hstack([number_of_groups[:, 1, 2], 0])

            counter += 1

        else:
            sources_HST[:,3*counter] = number_of_sources[:, 0, 0]
            sources_HST[:,3*counter+1] = number_of_sources[:, 0, 1]
            sources_HST[:,3*counter+2] = number_of_sources[:, 0, 2]

            sources_JWST[:,3*counter] = number_of_sources[:, 1, 0]
            sources_JWST[:,3*counter+1] = number_of_sources[:, 1, 1]
            sources_JWST[:,3*counter+2] = number_of_sources[:, 1, 2]

            groups_HST[:,3*counter] = number_of_groups[:, 0, 0]
            groups_HST[:,3*counter+1] = number_of_groups[:, 0, 1]
            groups_HST[:,3*counter+2] = number_of_groups[:, 0, 2]

            groups_JWST[:,3*counter] = number_of_groups[:, 1, 0]
            groups_JWST[:,3*counter+1] = number_of_groups[:, 1, 1]
            groups_JWST[:,3*counter+2] = number_of_groups[:, 1, 2]

            counter += 1

        print(object_alone_fluxes_HST)
        print(object_group_fluxes_HST)
        print(object_alone_fluxes_JWST)
        print(object_group_fluxes_JWST)
        print('-------------------------------')
        np.savetxt(ojb_is_JWST, object_alone_fluxes_JWST.T,fmt='%1.5e')
        np.savetxt(ojb_is_HST, object_alone_fluxes_HST.T,fmt='%1.5e')
        np.savetxt(ojb_gr_JWST, object_group_fluxes_JWST.T,fmt='%1.5e')
        np.savetxt(ojb_gr_HST, object_group_fluxes_HST.T,fmt='%1.5e')

    #np.savetxt('processed_data_sources_HST5.dat', sources_HST, fmt='%1.3e')
    #np.savetxt('processed_data_sources_JWST5.dat', sources_JWST, fmt='%1.3e')
    #np.savetxt('processed_data_groups_HST5.dat', groups_HST, fmt='%1.3e')
    #np.savetxt('processed_data_groups_JWST5.dat', groups_JWST, fmt='%1.3e')


#lum()


def func(x,a,b,c):


    return (a + b*x*x*x)*np.exp(-c*x*x)

def lum_show():

    lum_HST_alone = np.loadtxt('data_processed_files/objects_alone_HST.dat')
    lum_HST_group = np.loadtxt('data_processed_files/objects_group_HST.dat')
    lum_JWST_alone = np.loadtxt('data_processed_files/objects_alone_JWST.dat')
    lum_JWST_group = np.loadtxt('data_processed_files/objects_group_JWST.dat')

    plot_style()

    Hiso, bins = np.histogram(lum_HST_alone,bins=np.logspace(-1,3,70))
    Hgru, bins = np.histogram(lum_HST_group,bins=np.logspace(-1,3,70))

    Jiso, bins = np.histogram(lum_JWST_alone,bins=np.logspace(-1,3,70))
    Jgru, bins = np.histogram(lum_JWST_group,bins=np.logspace(-1,3,70))

    bins_c = (bins[1:] + bins[:-1])/2

    plt.xscale('log')
    xx = np.array([1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5])

    yy = np.array([0,10,20,30,40,50,60])
    yy = np.array([0,40,80,120,160,200])
    yy = np.array([0,50,100,150,200,250,300])

    plt.xticks(xx, fontsize=24)
    plt.yticks(yy, fontsize=24)

    plt.xlim(1e-1,1e3)
    plt.ylim(0,200)

    #plt.step(bins_c, Jiso,c='orange',label='JWST iso',lw=2.5)
    #plt.step(bins_c, Jgru,c='blue',label='JWST gr',lw=2.5)
    #plt.step(bins_c, Hiso,c='red',label='HST iso',lw=3.5)
    #plt.step(bins_c, Hgru,c='green',label='HST gr',lw=3.5)
    plt.step(bins_c, Hiso+Hgru,c='blue',label='HST total',lw=3.5)
    plt.step(bins_c, Jiso+Jgru,c='red',label='JWST total',lw=3.5)

    plt.ylabel('$\\rm Number \\thinspace of \\thinspace objects$',fontsize=23)
    plt.xlabel('$\\rm Flux \\thinspace nJy$',fontsize=23)
    plt.legend(loc='upper right',fontsize=22)
    #plt.savefig('JWST_hist.pdf',format='pdf')
    #plt.savefig('HST_hist.pdf',format='pdf')
    plt.savefig('lum_total.pdf',format='pdf')




    #plt.figure(2)
    #plt.hist(lum_HST_group,bins=np.linspace(0,1000,200))
    #plt.hist(lum_HST_alone,bins=np.linspace(0,1000,200))
    plt.show()

lum_show()
#prob_int()


