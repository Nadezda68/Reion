import numpy as np
import matplotlib.pylab as plt

from matplotlib.ticker import AutoMinorLocator
from scipy import integrate
from scipy import interpolate
from cosmolopy import distance as d


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


HST_WFC3CAM_pixel_size = 0.13   # arcsec per pixel
JWST_NIRCAM_pixel_size = 0.032  # arcsec per pixel

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def plot_style(xticks=5,yticks=5):

    global ax

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


def redshift():
    cosmo = {'omega_M_0' : 0.2726, 'omega_lambda_0' :0.7274, 'h' : 0.704}
    cosmo = d.set_omega_k_0(cosmo)

    z = np.loadtxt('processed_random/Redshifts.dat')
    D_phys = np.loadtxt('processed_random/D_phys.dat')
    dz = np.zeros_like(D_phys)
    for ii in range(6):
        D_comove = D_phys[:,ii] * (1 + z[:,ii]) /1e6 # Mpc
        print(D_comove)
        z_sample = np.linspace(0.01,0.09,5000)
        for i in range(len(D_phys[:,ii])):
            print(i)
            for k in range(len(z_sample)):
                dist = d.comoving_distance(z[i,ii]+z_sample[k],z[i,ii], **cosmo)
                if dist >= D_comove[i]:
                    dz[i,ii] = z_sample[k]
                    print(k)
                    print(z_sample[k])
                    break
    np.savetxt('processed_random/dz.dat',dz,fmt='%1.5e')

path = 'processed_random/'
S_ang = np.loadtxt(path + 'S_angle.dat')
d_z = np.loadtxt(path + 'dz.dat')
redshift = np.loadtxt(path + 'Redshifts.dat')

def sources_population_cumulative():

    plot_style()

    xx = np.array([6,7,8,9,10])
    yy = np.array([0.1,1,10,100,1000])
    plt.xticks(xx, fontsize=24)
    ax.tick_params(axis='x', which='major', pad=7)
    ax.tick_params(axis='y', which='major', pad=7)
    plt.xlim(6,10)

    sources_HST_cumul_full = np.zeros((28, 6))
    sources_JWST_cumul_full = np.zeros((28, 6))
    sources_HST_diff_full = np.zeros((28, 6))
    sources_JWST_diff_full = np.zeros((28, 6))

    for sim, sim_num in zip(['0','1','2','4','5','6'], [0,1,2,3,4,5]):

        # RANDOM
        sources_HST = np.zeros((28, 3))
        sources_JWST = np.zeros((28, 3))

        for i in range(28):
            for k, proj in zip([0,1,2], ['x','y','z']):
                A_H = np.loadtxt(path + 'HST/' + sim + '/' + proj + '/objects_iso_' + str(i) + '_' + filter + '_HST_' + sigma + '.dat')
                B_H = np.loadtxt(path + 'HST/' + sim + '/' + proj + '/objects_gr_3_' + str(i) + '_' + filter + '_HST_' + sigma + '.dat')
                sources_HST[i, k] = (A_H.size + B_H.size) / S_ang[i, sim_num] / d_z[i, sim_num] * 3600 * 4

                A_J = np.loadtxt(path + 'JWST/' + sim + '/' + proj + '/objects_iso_' + str(i) + '_' + '150' + '_JWST_' + '2' + '.dat')
                B_J = np.loadtxt(path + 'JWST/' + sim + '/' + proj + '/objects_gr_3_' + str(i) + '_' + '150' + '_JWST_' + '2' + '.dat')
                sources_JWST[i, k] += (A_J.size + B_J.size) / S_ang[i, sim_num] / d_z[i, sim_num] * 3600 * 4

        sources_HST = np.mean(sources_HST, axis=1)
        sources_JWST = np.mean(sources_JWST, axis=1)

        for j in range(len(sources_HST)):  # redshifts
            sources_HST_cumul_full[j, sim_num] = np.abs(integrate.trapz(y=sources_HST[:j+1], x=redshift[:j+1, sim_num]))
            sources_JWST_cumul_full[j, sim_num] = np.abs(integrate.trapz(y=sources_JWST[:j+1], x=redshift[:j+1, sim_num]))
            sources_HST_diff_full[j, sim_num] = sources_HST[j]
            sources_JWST_diff_full[j, sim_num] = sources_JWST[j]

    data = np.vstack([np.mean(redshift[:, 1:], axis=1),
                      np.mean(sources_HST_diff_full[:, 1:]*S_ang[:,1:]*d_z[:,1:], axis=1)*64/3600/4,
                      np.mean(sources_JWST_diff_full[:, 1:]*S_ang[:,1:]*d_z[:,1:], axis=1)*64/3600/4,
                      np.mean(sources_HST_diff_full[:, 1:], axis=1), np.mean(sources_JWST_diff_full[:, 1:], axis=1),
                      np.mean(sources_HST_cumul_full[:, 1:], axis=1), np.mean(sources_JWST_cumul_full[:, 1:], axis=1)])

    np.savetxt('data3.dat',data.T,fmt='%1.5e')
    plt.plot(np.mean(redshift[:, 1:], axis=1), np.mean(sources_HST_cumul_full[:, 1:], axis=1), '-', color='blue', lw=3,label= 'HST',zorder=3)
    plt.plot(np.mean(redshift[:, 1:], axis=1), np.mean(sources_JWST_cumul_full[:, 1:], axis=1), '--', color='red', lw=4, label='JWST', zorder=3)

    xdf_data = np.loadtxt('real/XDF_data.dat')
    xdf_data = xdf_data.flatten()
    xdf_data, bins = np.histogram(xdf_data, bins=np.linspace(5.25, 11, 1000))

    bins_c = (bins[:-1] + bins[1:])/2
    xdf_data_cum = np.cumsum(xdf_data[::-1])[::-1]

    # COSMIC VAR FROM UNI COL
    N_var = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    N_error = [0, 2, 2, 3, 5, 8, 11, 14, 17, 19 ,21, 24, 26, 29, 33, 37]
    f_var = interpolate.interp1d(N_var, N_error, kind='cubic')

    plt.step(bins_c, xdf_data_cum,'k',lw='3',zorder=2)
    plt.fill_between(bins_c, xdf_data_cum - f_var(xdf_data_cum), xdf_data_cum + f_var(xdf_data_cum), facecolor='black', interpolate=True, alpha=0.4, zorder=1)
    plt.plot([1e4, 1e5], [1e2, 1e4], color='k', lw=3, label='XDF data')

    output = np.vstack([np.mean(redshift[:, 1:], axis=1),np.mean(sources_HST_cumul_full[:, 1:], axis=1),np.mean(sources_JWST_cumul_full[:, 1:], axis=1)])
    np.savetxt('HST_JWST_data.dat',output.T,fmt='%.4e')

    plt.xlabel('$z$',fontsize=26)
    plt.ylabel('$ N_{>z} $',fontsize=26)
    plt.yscale('log')
    plt.yticks(yy, fontsize=24)
    plt.legend(loc='upper right', fontsize=20)
    fixlogax(plt.gca(), a='y')
    plt.ylim(1, 1000)
    #plt.savefig('N_sources_cum.pdf', fmt='pdf')

    plt.show()


def sources_groups_cumulative(gr_crit=3):

    plot_style()

    xx = np.array([6,7,8,9,10,11])
    yy = np.array([0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0])
    plt.xticks(xx, fontsize=24)
    plt.xlim(6, 10)

    sources_HST_cumulative_full_3 = np.zeros((28, 6))
    sources_JWST_cumulative_full_3 = np.zeros((28, 6))

    for sim, sim_num in zip(['0', '1', '2', '4', '5', '6'], [0, 1, 2, 3, 4, 5]):

        # RANDOM
        sources_HST = np.zeros((28, 3))
        groups_HST_3 = np.zeros((28, 3))
        sources_JWST = np.zeros((28,3))
        groups_JWST_3 = np.zeros((28,3))

        for i in range(28):
            for k, proj in zip([0,1,2],['x','y','z']):

                A_H = np.loadtxt(path + 'HST/' + sim + '/' + proj + '/objects_iso_' + str(i) + '_' + filter + '_HST_' + sigma + '.dat')
                B_H = np.loadtxt(path + 'HST/' + sim + '/' + proj + '/objects_gr_3_' + str(i) + '_' + filter + '_HST_' + sigma + '.dat')
                sources_HST[i, k] += (A_H.size + B_H.size)
                groups_HST_3[i, k] += (B_H.size)
                A_J = np.loadtxt(path + 'JWST/'+ sim +'/' + proj + '/objects_iso_' + str(i) + '_' + '150' + '_JWST_' + '2' + '.dat')
                B_J = np.loadtxt(path + 'JWST/'+ sim +'/' + proj + '/objects_gr_3_' + str(i) + '_' + '150' + '_JWST_' + '2' + '.dat')
                sources_JWST[i] += (A_J.size + B_J.size)
                groups_JWST_3[i] += B_J.size

        sources_HST = np.mean(sources_HST, axis=1)
        groups_3_HST = np.mean(groups_HST_3, axis=1)
        sources_JWST = np.mean(sources_JWST, axis=1)
        groups_3_JWST = np.mean(groups_JWST_3, axis=1)

        sources_HST_cumulative = np.zeros_like(sources_HST)
        groups_HST_3_cumulative = np.zeros_like(groups_3_HST)
        sources_JWST_cumulative = np.zeros_like(sources_HST)
        groups_JWST_3_cumulative = np.zeros_like(groups_3_HST)

        for j in range(len(sources_HST)):  # redshifts
            sources_HST_cumulative[j] = integrate.trapz(y=sources_HST[:j+1][::-1], x=redshift[:j+1, sim_num][::-1])
            groups_HST_3_cumulative[j] = integrate.trapz(y=groups_3_HST[:j+1][::-1], x=redshift[:j+1, sim_num][::-1])
            sources_JWST_cumulative[j] = integrate.trapz(y=sources_JWST[:j + 1][::-1], x=redshift[:j + 1, sim_num][::-1])
            groups_JWST_3_cumulative[j] = integrate.trapz(y=groups_3_JWST[:j + 1][::-1], x=redshift[:j + 1, sim_num][::-1])

        sources_HST_cumulative_full_3[:,sim_num] = (groups_HST_3_cumulative+1e-26)/(sources_HST_cumulative+1e-13)
        sources_JWST_cumulative_full_3[:, sim_num] = (groups_JWST_3_cumulative + 1e-26) / (sources_JWST_cumulative + 1e-13)

    plt.plot(np.mean(redshift[:,1:],axis=1), np.mean(sources_HST_cumulative_full_3[:,1:], axis=1), lw=3, color='blue', label='HST')
    plt.plot(np.mean(redshift[:, 1:], axis=1), np.mean(sources_JWST_cumulative_full_3[:, 1:], axis=1), '--',lw=4, color='red', label='JWST')
    plt.plot(7, 6/27, 'ks', markersize=18, label='XDF data')
    plt.errorbar(7, 6/27, 0.08, lw=3.3, color='black')

    plt.xlabel('$z$', fontsize=26)
    plt.ylabel('$ f^{m}_{>z} $', fontsize=26)
    plt.yticks(yy, fontsize=24)
    plt.ylim(0, 0.4)
    plt.legend(loc='upper right', fontsize=20, numpoints=1)
    plt.savefig('N_groups_cum.pdf', fmt='pdf')

    plt.show()


def luminosity(z_min=6):

    A = np.array([(6.1, 28.38), (6.1, 29.37), (6.1, 29.24), (6.32, 26.77),
                  (6.1, 28.8), (6.39, 30.02), (6.17, 29.51), (6.24, 29.08),
                  (6.1, 28.93), (6.1, 29.58), (6.39, 29.47), (6.1, 29.81),
                  (6.1, 28.61), (6.03, 29.17), (6.17, 29.45), (6.17, 29.03),
                  (6.1, 29.17), (6.03, 26.16), (6.24, 28.49), (6.17, 27.63),
                  (6.24, 29.63), (6.17, 29.49), (6.17, 29.74), (6.03, 28.16),
                  (6.32, 29.32), (6.03, 28.34), (6.03, 28.84), (6.17, 29.11),
                  (6.17, 29.25), (6.1, 27.19), (6.1, 28.24), (6.24, 29.45),
                  (6.32, 28.52), (6.1, 29.88), (6.46, 29.78), (6.24, 29.62),
                  (6.03, 25.51), (6.54, 26.34), (6.03, 27.26), (6.03, 29.29),
                  (6.32, 29.36), (6.46, 29.55), (6.46, 29.07), (6.32, 29.42),
                  (6.77, 28.35), (6.32, 29.91), (6.69, 29.77), (6.39, 29.62),
                  (6.69, 28.85), (7.33, 27.64), (6.32, 27.86), (7.08, 29.23),
                  (6.77, 30.12), (6.84, 25.95), (7.0, 29.53), (6.24, 30.35),
                  (6.24, 28.34), (6.61, 29.51), (6.69, 29.82), (7.0, 28.93),
                  (6.39, 29.98), (6.46, 28.2), (6.46, 27.74), (6.61, 30.89),
                  (6.39, 29.56), (6.69, 29.22), (6.1, 29.52), (6.32, 28.98),
                  (6.54, 29.03), (6.77, 28.08), (7.49, 28.74), (6.84, 28.96),
                  (6.92, 29.75), (6.46, 30.08), (7.75, 28.81), (6.46, 27.29),
                  (6.54, 28.12), (6.24, 30.55), (7.08, 28.53), (6.84, 29.55),
                  (6.1, 30.52), (6.54, 29.61), (7.0, 28.34), (6.24, 29.29),
                  (6.92, 29.25), (7.0, 29.35), (6.46, 29.49), (6.84, 30.11),
                  (6.39, 29.77), (6.32, 28.87), (6.84, 30.15), (7.75, 30.23),
                  (7.93, 29.69), (7.08, 28.64), (7.49, 28.33), (7.75, 29.67),
                  (7.93, 29.13), (7.49, 26.22), (7.33, 29.73), (7.49, 26.74),
                  (7.16, 28.98), (7.49, 27.11), (7.49, 27.74), (7.58, 27.51),
                  (7.84, 28.83), (7.49, 30.19), (8.29, 29.0), (8.11, 28.16),
                  (8.02, 28.21), (8.2, 29.87), (8.29, 27.67), (7.66, 30.31),
                  (7.75, 29.79)])

    fluxes = 1e9 * np.power(10, -2 / 5 * (A[:, 1] - 8.9))
    fluxes = fluxes[np.where(A[:,0]>z_min)]

    HSTreal160_1, bins = np.histogram(fluxes, bins=np.logspace(-1.8, 3, 30))
    HST_real160_int_1 = np.cumsum(HSTreal160_1[::-1])[::-1]

    plot_style()
    plt.xscale('log')

    plt.ylabel('$N_{>F_{\\nu}}$',fontsize=26)
    plt.xlabel('$F_{\\nu}$ $\\rm nJy$',fontsize=26)

    xx = np.array([1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5])
    yy = np.array([1e-1,1e0,1e1,1e2,1e3,1e4])

    plt.xticks(xx, fontsize=24)

    plt.xlim(1e-1,180)
    plt.yscale('log')
    plt.yticks(yy, fontsize=24)
    plt.ylim(1e-1, 300)

    ax.tick_params(axis='x', which='major', pad=8)
    ax.tick_params(axis='y', which='major', pad=7)

    if z_min == 6:
        N = 25
    elif z_min == 7:
        N = 21
    elif z_min == 8:
        N = 17

    N_obj_full = np.zeros((29, 6, N))
    N_obj_full_z_int = np.zeros((29, 6))
    N_obj_full_cumul = np.zeros((29, 6))

    N_obj_full_J = np.zeros((29, 6, N))
    N_obj_full_z_int_J = np.zeros((29, 6))
    N_obj_full_cumul_J = np.zeros((29, 6))

    for sim, sim_num in zip(['0', '1', '2', '4', '5', '6'], [0, 1, 2, 3, 4, 5]):

        for i in range(N):
            for k, proj in zip([0, 1, 2], ['x', 'y', 'z']):
                A = np.loadtxt(path + 'HST/' + sim + '/' + proj + '/objects_iso_' + str(i) + '_' + filter + '_HST_' + sigma + '.dat')
                B = np.loadtxt(path + 'HST/' + sim + '/' + proj + '/objects_gr_3_' + str(i) + '_' + filter + '_HST_' + sigma + '.dat')
                A_J = np.loadtxt(path + 'JWST/' + sim + '/' + proj + '/objects_iso_' + str(i) + '_' + '150' + '_JWST_' + '2' + '.dat')
                B_J = np.loadtxt(path + 'JWST/' + sim + '/' + proj + '/objects_gr_3_' + str(i) + '_' + '150' + '_JWST_' + '2' + '.dat')

                if proj == 'x':
                    obj_x = np.hstack([A, B])
                    obj_x_J = np.hstack([A_J, B_J])
                elif proj == 'y':
                    obj_y = np.hstack([A, B])
                    obj_y_J = np.hstack([A_J, B_J])
                else:
                    obj_z = np.hstack([A, B])
                    obj_z_J = np.hstack([A_J, B_J])

            N_obj_x, bins = np.histogram(obj_x, bins=np.logspace(-1.8, 3, 30))
            N_obj_y, bins = np.histogram(obj_y, bins=np.logspace(-1.8, 3, 30))
            N_obj_z, bins = np.histogram(obj_z, bins=np.logspace(-1.8, 3, 30))

            N_obj_x_J, bins = np.histogram(obj_x_J, bins=np.logspace(-1.8, 3, 30))
            N_obj_y_J, bins = np.histogram(obj_y_J, bins=np.logspace(-1.8, 3, 30))
            N_obj_z_J, bins = np.histogram(obj_z_J, bins=np.logspace(-1.8, 3, 30))

            N_obj = (N_obj_x + N_obj_y + N_obj_z) / 3 / S_ang[i, sim_num] / d_z[i, sim_num] * 3600 * 4
            N_obj_J = (N_obj_x_J + N_obj_y_J + N_obj_z_J) / 3 / S_ang[i, sim_num] / d_z[i, sim_num] * 3600 * 4
            N_obj_full[:, sim_num, i] = N_obj
            N_obj_full_J[:, sim_num, i] = N_obj_J

            if i == 0:
                bins_c = (bins[1:] + bins[:-1])/2

        for j in range(29):  # redshifts
            N_obj_full_z_int[j, sim_num] = np.abs(integrate.trapz(y=N_obj_full[j, sim_num, :], x=redshift[:N, sim_num]))
            N_obj_full_z_int_J[j, sim_num] = np.abs(integrate.trapz(y=N_obj_full_J[j, sim_num, :], x=redshift[:N, sim_num]))

        N_obj_full_cumul[:, sim_num] = np.cumsum(N_obj_full_z_int[::-1, sim_num])[::-1]
        N_obj_full_cumul_J[:, sim_num] = np.cumsum(N_obj_full_z_int_J[::-1, sim_num])[::-1]

    plt.plot(bins_c, np.mean(N_obj_full_cumul[:,1:], axis=1), lw=4, color='blue', label='HST')
    plt.plot(bins_c, np.mean(N_obj_full_cumul_J[:,1:], axis=1), lw=4, color='red', ls='--',label='JWST')

    # REAL DATA
    real_f160_comp = np.loadtxt('real/3f/flux_160_amp_gt_6.dat')[:,2]
    real_f160_comp_z = np.loadtxt('real/3f/flux_160_amp_gt_6.dat')[:, 3]
    real_f160_comp = real_f160_comp[np.where(real_f160_comp_z>z_min)]
    HSTreal160_2, bins = np.histogram(real_f160_comp, bins=np.logspace(-1.8, 3, 30))
    HST_real160_int_2 = np.cumsum(HSTreal160_2[::-1])[::-1]

    # PLOT REAL DATA
    # cosmic var
    N_var = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140]
    N_error = [0, 2, 2, 3, 5, 8, 11, 14, 17, 19 ,21, 24, 26, 29, 33, 37]
    f_var = interpolate.interp1d(N_var, N_error, kind='cubic')

    plt.plot(bins_c, HST_real160_int_2, c='black', lw=4, zorder=2,label='XDF data')
    # plt.plot(bins_c, HST_real160_int_1, c='orange', lw=4, zorder=2, label='XDF data')
    plt.fill_between(bins_c, HST_real160_int_2 - f_var(HST_real160_int_2), HST_real160_int_2 + f_var(HST_real160_int_2), color = 'black', alpha=0.3)

    plt.legend(loc='upper right',fontsize=20)
    fixlogax(plt.gca(), a='y')
    fixlogax(plt.gca(), a='x')
    plt.savefig('fluxes_z'+str(z_min)+'_v1.pdf',format='pdf')
    plt.show()

def sources_groups_cumulative_diff(gr_crit=3):

    plot_style()

    xx = np.array([6,7,8,9,10,11])
    yy = np.array([0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0])
    plt.xticks(xx, fontsize=24)
    plt.xlim(6, 10)

    sources_HST_full_3 = np.zeros((28, 6))
    sources_JWST_full_3 = np.zeros((28, 6))

    for sim, sim_num in zip(['0', '1', '2', '4', '5', '6'], [0, 1, 2, 3, 4, 5]):

        # RANDOM
        sources_HST = np.zeros((28, 3))
        groups_HST_3 = np.zeros((28, 3))
        sources_JWST = np.zeros((28, 3))
        groups_JWST_3 = np.zeros((28, 3))

        for i in range(28):
            for k, proj in zip([0,1,2],['x','y','z']):

                A_H = np.loadtxt(path + 'HST/' + sim + '/' + proj + '/objects_iso_' + str(i) + '_' + filter + '_HST_' + sigma + '.dat')
                B_H = np.loadtxt(path + 'HST/' + sim + '/' + proj + '/objects_gr_3_' + str(i) + '_' + filter + '_HST_' + sigma + '.dat')
                sources_HST[i, k] += (A_H.size + B_H.size)
                groups_HST_3[i, k] += (B_H.size)
                A_J = np.loadtxt(path + 'JWST/'+ sim +'/' + proj + '/objects_iso_' + str(i) + '_' + '150' + '_JWST_' + '2' + '.dat')
                B_J = np.loadtxt(path + 'JWST/'+ sim +'/' + proj + '/objects_gr_3_' + str(i) + '_' + '150' + '_JWST_' + '2' + '.dat')
                sources_JWST[i] += (A_J.size + B_J.size)
                groups_JWST_3[i] += B_J.size

        sources_HST = np.mean(sources_HST, axis=1)
        groups_3_HST = np.mean(groups_HST_3, axis=1)
        sources_JWST = np.mean(sources_JWST, axis=1)
        groups_3_JWST = np.mean(groups_JWST_3, axis=1)

        sources_HST_full_3[:,sim_num] = (groups_3_HST + 1e-26)/(sources_HST + 1e-13)
        sources_JWST_full_3[:, sim_num] = (groups_3_JWST + 1e-26) / (sources_JWST + 1e-13)

    plt.plot(np.mean(redshift[:, 1:],axis=1),  np.mean(sources_HST_full_3[:, 1:], axis=1), lw=3, color='blue', label='HST')
    plt.plot(np.mean(redshift[:, 1:], axis=1), np.mean(sources_JWST_full_3[:, 1:], axis=1), '--',lw=4, color='red', label='JWST')

    plt.xlabel('$z$', fontsize=26)
    plt.ylabel('$ f^{m}_{z} $', fontsize=26)
    plt.yticks(yy, fontsize=24)
    plt.ylim(0, 0.4)
    plt.legend(loc='upper right', fontsize=20, numpoints=1)
    plt.savefig('N_groups_diff.pdf', fmt='pdf')

    plt.show()


sigma = '2'  # threshold = [2.5, 2.75, 3.0, 3.5, 4.0]
filter = '160'

sources_population_cumulative()
#luminosity(z_min=7)
#sources_groups_cumulative(gr_crit=3)
#sources_groups_cumulative_diff(gr_crit=3)
