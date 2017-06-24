
# PATH TO THE DATA /home/kaurov/scratch/40/a=0.1282
import numpy as np
from scipy.interpolate    import interp1d
from scipy import integrate
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pylab as plt
import glob


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


def f1():

    initial_data = np.loadtxt('data.dat')
    N_HST = interp1d(initial_data[:, 0], initial_data[:, 1], fill_value='extrapolate')
    N_JWST = interp1d(initial_data[:, 0], initial_data[:, 2], fill_value='extrapolate')
    path = '/home/kaurov/scratch/40/'
    data = np.array(sorted(glob.glob(path + 'a=*')))
    data = np.delete(data,3)[:6]
    print(data)
    output_data = np.zeros((len(data), 9))


    for file_name,i in zip(data,range(len(data))):

        z = 1/float(file_name[-6:])-1
        print('redshift:',z)

        hlist = np.genfromtxt(path + 'a=0.' + file_name[-4:] + '/halo_catalog_a0.' + file_name[-4:] + '.dat')
        gallum = np.genfromtxt(path + 'a=0.' + file_name[-4:] + '/gallums.res')

        N_of_gal_HST = int(N_HST(z) + 1)
        N_of_gal_JWST = int(N_JWST(z) + 1)

        filt_HST = gallum[:,4] > np.sort(gallum[:,4])[-N_of_gal_HST]
        filt_JWST = gallum[:, 4] > np.sort(gallum[:, 4])[-N_of_gal_JWST]

        print('N Obj HST:', np.sum(filt_HST))
        print('N Obj JWST:', np.sum(filt_JWST))

        output_data[i, 0] = z
        output_data[i, 1] = np.sum(filt_HST)
        output_data[i, 5] = np.sum(filt_JWST)

        for crit, j in zip([0.02, 0.03, 0.1], [2, 3, 4]):

            if(np.sum(filt_HST)==0):
                output_data[i, j] = 0.0
            else:
                # Matrix of distances:
                dist = np.ones([np.sum(filt_HST), np.sum(filt_HST)])*100.

                # Filling up the matrix:
                for ii in range(np.sum(filt_HST)):
                    dist[ii,:] = np.sqrt((hlist[filt_HST,1][ii] - hlist[filt_HST,1][:])**2+(hlist[filt_HST,2][ii]-hlist[filt_HST,2][:])**2)
                    dist[ii,ii] = 100.

                filt_ready = dist.min(axis=1) <= crit
                print('N pair HST:', np.sum(filt_ready))
                output_data[i, j] = np.sum(filt_ready)

            #Matrix of distances:
            dist = np.ones([np.sum(filt_JWST), np.sum(filt_JWST)]) * 100.

            # Filling up the matrix:
            for ii in range(np.sum(filt_JWST)):
                dist[ii, :] = np.sqrt((hlist[filt_JWST, 1][ii] - hlist[filt_JWST, 1][:]) ** 2 + (hlist[filt_JWST, 2][ii] - hlist[filt_JWST, 2][:]) ** 2)
                dist[ii, ii] = 100.

            filt_ready = dist.min(axis=1) <= crit
            print('N pair JWST:', np.sum(filt_ready))
            output_data[i, j+4] = np.sum(filt_ready)

    np.savetxt('output_data.dat', output_data, fmt='%1.5e')

def f2():

    output = np.loadtxt('data2.dat')

    plot_style()

    xx = np.array([6, 7, 8, 9, 10, 11])
    yy = np.array([0.001,0.01,0.1,1.0])
    plt.xticks(xx, fontsize=24)
    plt.xlim(6, 10)

    for j, label, color in zip([4,3,2],['$\\rm 100 \\thinspace h^{-1}kpc$',
                                        '$\\rm 30 \\thinspace h^{-1}kpc$',
                                        '$\\rm 20 \\thinspace h^{-1}kpc$'],
                                                  ['dodgerblue', 'orangered','navy']):

        sources_HST = (output[:, j] + 1e-26) / (output[:, 1] + 1e-13)
        sources_JWST = (output[:, 4+j] + 1e-26) / (output[:, 5] + 1e-13)

        plt.plot(output[:, 0], sources_HST, lw=3, color=color, label=label)
        plt.plot(output[:, 0], sources_JWST, '--', lw=4, color=color)

    plt.plot([-30, -40], [-30, -40], '--', lw=3, color='black', label='JWST')
    plt.plot([-30,-40], [-30,-40], lw=3, color='black', label='HST')

    plt.xlabel('$z$', fontsize=26)
    plt.ylabel('$ f^{m}_{z} $', fontsize=26)
    plt.yticks(yy, fontsize=24)
    plt.yscale('log')
    plt.ylim(0.001, 1.0)
    plt.legend(loc='upper right', fontsize=20, numpoints=1,frameon=False)
    plt.savefig('pairs_diff_log.pdf', fmt='pdf')
    plt.show()

def f3():

    output = np.loadtxt('data2.dat')
    sources_HST_INT = np.zeros(len(output[:, 1]))
    sources_JWST_INT  = np.zeros(len(output[:, 1]))
    groups_HST_INT  = np.zeros(len(output[:, 1]))
    groups_JWST_INT  = np.zeros(len(output[:, 1]))

    plot_style()

    xx = np.array([6, 7, 8, 9, 10, 11])
    yy = np.array([0.001, 0.01, 0.1, 1.0])
    plt.xticks(xx, fontsize=24)
    plt.xlim(6, 10)

    for j, label, color in zip([4,3,2],['$\\rm 100 \\thinspace h^{-1}kpc$',
                                        '$\\rm 30 \\thinspace h^{-1}kpc$',
                                        '$\\rm 20 \\thinspace h^{-1}kpc$'],
                                                  ['dodgerblue', 'orangered','navy']):

        for jj in range(len(output[:, 1])):
            sources_HST_INT[jj] = integrate.trapz(y=output[:jj+1, 1][::-1], x=output[:jj+1, 0][::-1])
            sources_JWST_INT[jj] = integrate.trapz(y=output[:jj+1, 5][::-1], x=output[:jj+1, 0][::-1])
            groups_HST_INT[jj] = integrate.trapz(y=output[:jj+1, j][::-1], x=output[:jj+1, 0][::-1])
            groups_JWST_INT[jj] = integrate.trapz(y=output[:jj+1, j+4][::-1], x=output[:jj+1, 0][::-1])

        sources_HST = (groups_HST_INT + 1e-26) / (sources_HST_INT + 1e-13)
        sources_JWST = (groups_JWST_INT + 1e-26) / (sources_JWST_INT + 1e-13)

        plt.plot(output[:, 0], sources_HST, lw=3, color=color, label=label)
        plt.plot(output[:, 0], sources_JWST, '--', lw=4, color=color)

    plt.plot([-30, -40], [-30, -40], '--', lw=3, color='black', label='JWST')
    plt.plot([-30,-40], [-30,-40], lw=3, color='black', label='HST')

    plt.xlabel('$z$', fontsize=26)
    plt.ylabel('$ f^{m}_{>z} $', fontsize=26)
    plt.yticks(yy, fontsize=24)
    plt.yscale('log')
    plt.ylim(0.001, 1.0)
    plt.legend(loc='upper right', fontsize=20, numpoints=1,frameon=False)
    plt.savefig('pairs_cumul_log.pdf', fmt='pdf')
    plt.show()

f2()
f3()