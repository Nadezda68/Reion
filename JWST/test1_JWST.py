import json
import numpy as np
import matplotlib
from scipy.interpolate    import interp1d
from scipy                import integrate
import glob
import scipy 
from matplotlib import style
style.use('ggplot')
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
import os
os.environ['pandeia_refdata'] = "/home/maryhallow/Desktop/python/Reionizatoin/JWSTUserTraining2016/pandeia_data"
os.environ['PYSYN_CDBS'] = "/home/maryhallow/Desktop/python/Reionizatoin/JWSTUserTraining2016/cdbs.23.1.rc3"

from pandeia.engine.perform_calculation import perform_calculation
from pandeia.engine.calc_utils import build_default_calc

cm_in_pc            = 3.0857e18
Omega_lam           = 0.7274
Omega_M_0           = 0.2726
Omega_k             = 0.0
h                   = 0.704

E   = lambda x: 1/np.sqrt(Omega_M_0*np.power(1+x,3)+Omega_lam+Omega_k*np.power(1+x,2))
D_m = lambda x: D_c(x)
D_c = lambda x: (9.26e27/h)*integrate.quad(E, 0, x)[0]
D_A = lambda x: D_m(x)/(1+x)/cm_in_pc/1e6  # Angular distance [Mpc]

N_sim_1 = 0
N_sim_2 = 1

def read_simulation_data():

    global info, lums, z, nbins, nbins_min, z, indices, angles, theta_min

    info = sorted(glob.glob("./output/info_rei000" + str(N_sim_1) + str(N_sim_2) + "_*"))
    lums = np.vstack((sorted(glob.glob("./output/lum_150_rei000" + str(N_sim_1) + str(N_sim_2) + "_*")),
                      sorted(glob.glob("./output/lum_200_rei000" + str(N_sim_1) + str(N_sim_2) + "_*"))))

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
    angles    = np.array(angles)
    nbins     = nbins[indices]
    angles    = angles[indices]
    nbins_min = np.min(nbins)
    theta_min = np.min(angles)

    print(nbins_min)
    print(theta_min)

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


def filter_bandwidth(a,b,x):

    global lambdas

    lambdas = []

    for i in range(0,len(x)):
        if (a<=x[i] and x[i]<=b):
            if(F_filter(x[i])>=0.5e-3):
                lambdas.append(x[i])

    return lambdas

def filter_init(name,z):

    global F_filter

    wavelengths,transmission = np.loadtxt('data/filter_F' + name + 'W.dat',skiprows=1).T  
    F_filter = interp1d(wavelengths*10000, transmission,fill_value=0.0,bounds_error=False)
    a,b = np.min(wavelengths*10000),np.max(wavelengths*10000)
    lamb_filter = filter_bandwidth(a,b,lamb*(1+z))

    return lamb_filter

def input_calc_init():
    
    global calc_input

    calc_input = build_default_calc('jwst', 'nircam', 'sw_imaging')

    #The 0th index is taken to be wavelength in units of 'mJy'.
    #The 1st index is taken to be the flux in units of 'microns'.

    calc_input['configuration']['max_scene_size'] = 30
    calc_input['configuration']['readmode'] = 'deep8'
    calc_input['background'] = 'low'

    calc_input['configuration']['instrument']['filter'] = 'f150w'
    calc_input['configuration']['detector']['nexp'] = 1

    calc_input['strategy']['aperture_size'] = 1.2
    calc_input['strategy']['sky_annulus'] = [1.22, 1.4]

    calc_input['scene'][0]['spectrum']['normalization']['norm_flux'] = 0e0  # mJy (flat spectrum)
    calc_input['scene'][0]['spectrum']['normalization']['norm_wave'] = 1.5   # microns
    
def filter_flux(i,l,name):
    
    global calc_input
    
    lum_dist = D_A(z[i]) * (1 + z[i]) * (1 + z[i])
    print(np.shape(input_data))

    spec = [np.array(lamb_filter[::3])/1e4,np.ones(len(lamb_filter[::3]))/1e25]

    source = {
               'id': 1,
               'target': True,
               'position': {
                         'ang_unit': 'arcsec',
                         'x_offset': pixels_angle_coord[0],
                         'y_offset': pixels_angle_coord[0],
                         },
               'shape': {
                         'pa': 0.0,
                         'major': 0.0,
                         'minor': 0.0
                         },
               'spectrum': {
                            'sed': {
                                     "sed_type": "input",
                                     "spectrum":  spec
                                   },
                            'normalization': {
                                               'type': 'none'
                                             },
                            'redshift': 0.0
                            }
              }

    calc_input['scene'].append(source)

    for j in range(len(coords_x)):

        flux = 1e26*(1 + z[i])*input_data[coords_y[j],coords_x[j],:]/ (4 * np.pi * np.power(lum_dist*cm_in_pc*1e6,2))
        spec = [np.array(lamb_filter[::3])/1e4,flux[::3]]

        source = {
               'id': j+2,
               'target': True,
               'position': {
                         'ang_unit': 'arcsec',
                         'x_offset': pixels_angle_coord[coords_x[j]],
                         'y_offset': pixels_angle_coord[coords_y[j]],
                         },
               'shape': {
                         'pa': 0.0,
                         'major': 0.0,
                         'minor': 0.0
                         },
               'spectrum': {
                            'sed': {
                                     "sed_type": "input",
                                     "spectrum":  spec
                                   },
                            'normalization': {
                                               'type': 'none'
                                             },
                            'redshift': 0.0
                            }
              }

        calc_input['scene'].append(source)


read_simulation_data()
wavelengths,transmission = np.loadtxt('data/filter_F150W.dat',skiprows=1).T 

print('Redshifts:')
print(np.round(z,2))
print()
print('Number of pixels:')
print(nbins)
print()
print('Angular sizes:')
print(np.round(angles,2))

filter_index = 0
filter_name  = '150'
init_lum_tables()

for i in range(len(z)-1,-1,-1):

    image_size = angles[i]/2
    pixels_angle_coord = np.linspace(-image_size+image_size/nbins[i], image_size-image_size/nbins[i], nbins[i])

    print('z',z[i])

    input_data  = np.load(lums[filter_index,i])
    input_data  = input_data[::-1,:,:]
    lamb_filter = filter_init(filter_name,z[i])

    spec_intensity_sum = np.sum(input_data,axis=2)
    maxx = np.max(spec_intensity_sum)/1e5

    plt.figure(1)
    plt.subplot(5,5,i+1)
    plt.imshow(spec_intensity_sum[::-1,:], interpolation='nearest',vmax=maxx)
    plt.yticks([])
    plt.xticks([])
    plt.title( str(round(z[i],2)))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    objects = 0

    coords_x = []
    coords_y = []

    for ii in range(nbins_min):
            for jj in range(nbins_min):
                if spec_intensity_sum[ii,jj]>=maxx:

                    objects += 1
                    coords_y.append(ii)
                    coords_x.append(jj)

    print('satisfying criteria ',objects)

    input_calc_init()
    filter_flux(i,filter_index,filter_name)
    
    print('calc works')
    report = perform_calculation(calc_input, dict_report=False, webapp=True)
    report_dict = report.as_dict()

    a = int((nbins[i] - nbins_min)/2)

    print(np.max(report_dict['2d']['detector']))
    print(np.max(report_dict['2d']['detector'][a:a+nbins_min,a:a+nbins_min]))
    print(np.max(report_dict['2d']['snr']))

    if i>9:
        np.savetxt('test_det_' + str(i) + '.dat', report_dict['2d']['detector'][a:a+nbins_min,a:a+nbins_min],fmt='%1.5e')
        np.savetxt('test_snr_' + str(i) + '.dat',report_dict['2d']['snr'][a:a+nbins_min,a:a+nbins_min],fmt='%1.5e')
    else:
        np.savetxt('test_det_0' + str(i) + '.dat', report_dict['2d']['detector'][a:a+nbins_min,a:a+nbins_min],fmt='%1.5e')
        np.savetxt('test_snr_0' + str(i) + '.dat',report_dict['2d']['snr'][a:a+nbins_min,a:a+nbins_min],fmt='%1.5e')

plt.figure(1)
plt.savefig('maryhallow.pdf', format = 'pdf')

