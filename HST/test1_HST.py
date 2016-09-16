import yt
import glob
import argparse
import numpy              as np
from scipy.interpolate    import interp1d
from scipy.interpolate    import interp2d
from scipy                import integrate

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=float, nargs='+', help='an integer for the accumulator')
args = parser.parse_args()
external_param = np.array(args.integers)

input_start,input_stop,input_radius = int(external_param[0]),int(external_param[1]),float(150)
if(len(external_param)==5):
    input_coord = float(external_param[2]),float(external_param[3]),float(external_param[4])
else:
    input_coord = np.array([20.9130859375,14.9189453125,12.5673828125])

table = np.loadtxt('data/table_transmition_ISM.dat')
lam_rest   = table[1:,0]
z          = table[0,1:]
trans_coef = table[1:,1:]
F_IMS = interp2d(z, lam_rest, trans_coef)

E   = lambda x: 1/np.sqrt(Omega_M_0*np.power(1+x,3)+Omega_lam+Omega_k*np.power(1+x,2))
D_m = lambda x: D_c(x)
D_c = lambda x: (9.26e27/h)*integrate.quad(E, 0, x)[0]
D_A = lambda x: D_m(x)/(1+x)/3.0857e18/1e6

muf_list = sorted(glob.glob("/home/kaurov/kicp/code05A/drt/muv.bin*"))
files = sorted(glob.glob("/home/kaurov/kicp/code05A/OUT/rei05B_a0*/rei05B_a*.art"))

lam_list = np.zeros(len(muf_list)-1)
lookup = np.zeros([len(muf_list), 188, 22])

for i in range(len(muf_list)-1):

    f = open(muf_list[i])
    header = f.readline()
    f.close()
    d1 = header.split()[0]
    d2 = header.split()[1]
    lam_list[i] = float(header.split()[2])

    data = np.genfromtxt(muf_list[i], skip_header=1)
    lookup[i, :, :] = data[1:,1:]

dx = data[0, 1:]
dy = data[1:, 0]

def filter_bandwidth(a,b,x):

    position_in_lam_array = []
    lambdas               = []

    for i in range(0,len(x)):
        if (a<=x[i] and x[i]<=b):
            if(F_filter(x[i])>=1e-3):
                position_in_lam_array.append(i)
                lambdas.append(x[i])

    lambdas = np.array(lambdas)
    position_in_lam_array = np.array(position_in_lam_array)
    indices = np.argsort(lambdas)

    return position_in_lam_array[indices]

def filter_init(name,z):

    global F_filter

    filter_b = np.loadtxt('data/filter_f' + name + 'w.dat')
    F_filter = interp1d(filter_b[:,0], filter_b[:,1],fill_value=0.0,bounds_error=False)
    a,b = np.min(filter_b[:,0]),np.max(filter_b[:,0])
    lamb_positions = filter_bandwidth(a,b,lam_list*(1+z))

    return lamb_positions


for simulation in range(input_start,input_stop):

    print('Loading data %i out of [%i...%i] (%i in total)' % (simulation,input_start,input_stop,len(files)))

    pf = yt.load(files[simulation])
    simulation_name = files[simulation][-18:-4]
    data = pf.sphere(input_coord, (input_radius, "kpc"))

    print('Simulation name = ',	simulation_name)

    x = np.array(data[('STAR', 'POSITION_X')] - data.center[0])
    y = np.array(data[('STAR', 'POSITION_Y')] - data.center[1])
    z = np.array(data[('STAR', 'POSITION_Z')] - data.center[2])

    Omega_lam = 0.7274
    Omega_M_0 = 0.2726
    Omega_k   = 0.0
    h         = 0.704

    m = data[('STAR', 'MASS')].in_units('msun')
    met = data[('STAR', 'METALLICITY_SNIa')].in_units('Zsun') + data[('STAR', 'METALLICITY_SNII')].in_units('Zsun')
    t = np.log10(data[('STAR', 'age')].in_units('yr'))
    redshift = pf.current_redshift

    theta_arcsec = ( 2 * input_radius * 1e3 ) / ( D_A(redshift) * 1e6  ) * ( 206265.0 )
    nbins =  int(theta_arcsec * 1/0.13)

    print('z = %1.3e, D_A = %1.3e [Mpc], Nbins = %i' % (redshift,D_A(redshift), nbins))

    for filter_name in ['125','140','160']:

        print('filter name: f' + filter_name + 'w')
        lamb_positions = filter_init(filter_name,redshift)
        print('lam max, lam min, N of lams: ',np.min(lam_list[lamb_positions]),np.max(lam_list[lamb_positions]),len(lamb_positions))
        image = np.zeros([nbins, nbins, len(lamb_positions)])
        index = 0

        for i in lamb_positions:

            print('Computing luminosity...  step %i out of %i, lambda = %1.3e' % (index+1,len(lamb_positions),lam_list[i]))
            interp = interp2d(dx, dy, lookup[i, :, :])
            L = np.zeros_like(m)

            for j in range(0,len(m)):
                L[j] = F_IMS(redshift,lam_list[i])[0] * interp(met[j], t[j])[0] * m[j] * 3.828e33

            xedges = np.linspace(-data.radius, data.radius, nbins+1)
            yedges = np.linspace(-data.radius, data.radius, nbins+1)
            H,X,Y = np.histogram2d(x, y, bins=(xedges, yedges), weights = L)
            image[:, :, index] = np.rot90(H)

            index += 1

        np.save('output/lum_' + filter_name + '_' + simulation_name + '.npy', image)
        np.savetxt('output/info_'  + simulation_name + '.dat',np.array([nbins,redshift,D_A(redshift),theta_arcsec]),header='Nbins, Redshift, Angular distance [Mpc], theta [arc-sec]')




