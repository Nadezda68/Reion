import yt
import glob
import argparse
import numpy              as np
from scipy.interpolate    import interp2d
from scipy                import integrate

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=float, nargs='+', help='an integer for the accumulator')
args = parser.parse_args()
external_param = np.array(args.integers)

input_start,input_stop,input_radius = int(external_param[0]),int(external_param[1]),float(external_param[2])
if(len(external_param)==6):
    input_coord = float(external_param[3]),float(external_param[4]),float(external_param[5])
else:
    input_coord = np.array([20.9130859375,14.9189453125,12.5673828125])

table = np.loadtxt('table_transmition_ISM.dat')
lam_rest   = table[1:,0]
z          = table[0,1:]
trans_coef = table[1:,1:]
F_IMS = interp2d(z, lam_rest, trans_coef)

def D_m(z):
    if(Omega_k>0):
        return  (9.26e27/h)/np.sqrt(Omega_k)*np.sinh(np.sqrt(Omega_k)*D_c(z)/(9.26e27/h))
    else:
        return D_c(z)

E = lambda x: 1/np.sqrt(Omega_M_0*np.power(1+x,3)+Omega_lam+Omega_k*np.power(1+x,2))
D_c = lambda z: (9.26e27/h)*integrate.quad(E, 0, z)[0]
D_A = lambda z: D_m(z)/(1+z)/3.0857e18/1e6

muf_list = glob.glob("/home/kaurov/kicp/code05A/drt/muv.bin*")
files = glob.glob("/home/kaurov/kicp/code05A/OUT/rei05B_a0*/rei05B_a*.art")
sorted(files)

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

dx = data[0, 1:]
dy = data[1:, 0]


for simulation in range(input_start,input_stop):

    print('Loading data %i out of [%i...%i] (%i in total)' % (simulation,input_stop,input_start,len(files)))

    pf = yt.load(files[simulation])
    simulation_name = files[simulation][-18:-4]
    data = pf.sphere(input_coord, (input_radius, "kpc"))

    x = np.array(data[('STAR', 'POSITION_X')] - data.center[0])
    y = np.array(data[('STAR', 'POSITION_Y')] - data.center[1])
    z = np.array(data[('STAR', 'POSITION_Z')] - data.center[2])

    Omega_lam = pf.omega_lambda
    Omega_M_0 = pf.omega_matter
    Omega_k   = 1 - Omega_lam - Omega_M_0
    h         = pf.hubble_constant

    m = data[('STAR', 'MASS')].in_units('msun')
    met = data[('STAR', 'METALLICITY_SNIa')].in_units('Zsun') + data[('STAR', 'METALLICITY_SNII')].in_units('Zsun')
    t = np.log10(data[('STAR', 'age')].in_units('yr'))
    redshift = pf.current_redshift

    theta_arcsec = ( 2 * input_radius * 1e3 ) / ( D_A(redshift) * 1e6  ) * ( 206265.0 )
    nbins =  int(theta_arcsec * 1/0.13)

    print('z = %1.3e, D_A = %1.3e [Mpc], Nbins = %i' % (redshift,D_A(redshift), nbins))

    image = np.zeros([nbins, nbins, len(lam_list)])
    index = 0

    for i in np.argsort(lam_list):

        print('Computing luminosity...  step %i out of %i, lambda = %1.3e' % (index+1,len(lam_list),lam_list[i]))
        interp = interp2d(dx, dy, lookup[i, :, :])
        L = np.zeros_like(m)

        for j in range(0,len(m)):
            L[j] =  F_IMS(redshift,lam_list[i])[0] * interp(met[j], t[j])[0] * m[j] * 3.828e33

        xedges = np.linspace(-data.radius, data.radius, nbins+1)
        yedges = np.linspace(-data.radius, data.radius, nbins+1)
        H,X,Y = np.histogram2d(x, y, bins=(xedges, yedges), weights = L)
        image[:, :, index] = np.rot90(H)

        index += 1

    np.save('output/lum_' + simulation_name + '.npy',image)




