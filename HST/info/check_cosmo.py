__author__ = 'maryhallow'

import numpy as np
import cosmolopy as cp
from scipy import integrate
import matplotlib.pyplot as plt


# [parameters]
from scipy.interpolate    import interp2d

table = np.loadtxt('data/table_transmition_ISM.dat')
lam_rest   = table[1:,0]
z          = table[0,1:]
trans_coef = table[1:,1:]
F_IMS = interp2d(z, lam_rest, trans_coef)

h = 0.704
pc = 3.0856776e18      # [cm]
t_h = 3.09e17/h        # [sec]
D_h = 9.26e27/h        # [cm]
Omega_lam = 0.7274     # 0.7274
Omega_b_0 = 0.0456
Omega_M_0 = 0.2726     # 2726
Omega_k = 1-Omega_lam-Omega_M_0


E = lambda x: 1/np.sqrt(Omega_M_0*np.power(1+x,3)+Omega_lam+Omega_k*np.power(1+x,2))

def D_m(z):

    if(Omega_k>0):
        return  D_h/np.sqrt(Omega_k)*np.sinh(np.sqrt(Omega_k)*D_c(z)/D_h)
    else:
        return D_c(z)

D_c = lambda z: D_h*integrate.quad(E, 0, z)[0]

D_A = lambda z: D_m(z)/(1+z)

'''
test = np.linspace(0,50,500)
D_test = []
for i in range(0,len(test)):
    D_test = np.append(D_test,D_A(test[i])/pc/1e6)
plt.plot(test,D_test)
plt.show()

redshift = 8.0
print(cp.distance.angular_diameter_distance(redshift, **cp.fidcosmo))
print(D_A(redshift)/pc/1e6)
print(cp.fidcosmo)

'''

from scipy.interpolate    import interp1d
import glob

muf_list = glob.glob("./drt/muv.bin*")
lam_list = np.zeros(len(muf_list))

for i in range(len(muf_list)-1):
    f = open(muf_list[i])
    header = f.readline()
    f.close()
    lam_list[i] = float(header.split()[2])

lamb = lam_list[np.argsort(lam_list)]

def filter_bandwidth(a,b,x):

    global lambdas

    '''

    Initially we have SED tables for vast range of wavelengths and this function picks out those wavelengths, which
    are in filter bandwidth.

    '''

    lambdas               = []

    for i in range(0,len(x)):
        if (a<=x[i] and x[i]<=b):
            if(F_filter(x[i])>=1e-3):
                lambdas.append(x[i])

    print(len(lambdas))
    return lambdas


def filter_init(name):

    global F_filter
    '''
    temp_filters = open('data/FILTER.RES.latest', 'r')
    filters = []
    filters_names = []
    first = True

    for line in temp_filters:

        if line[0] == ' ':
            if not first:
                filters.append(np.array(temp))

            first = False
            filters_names.append(line.split())
            temp = []

        else:
            temp.append(np.array(line.split()).astype('float'))

    filters.append(np.array(temp))
    filters = np.array(filters)

    for ifilt in range(len(filters_names)):
        if filters_names[ifilt][1] == 'hst/wfc3/IR/f' + name + 'w.dat':
            filter_b = np.array([filters[ifilt][:,1],filters[ifilt][:,2]])
            filter_b = np.transpose(filter_b)
    '''
    filter_b = np.loadtxt('data/filter_f' + name + 'w.dat')
    F_filter = interp1d(filter_b[:,0], filter_b[:,1],fill_value=0.0,bounds_error=False)


    aaa = np.linspace(np.min(filter_b[:,0])/10,np.max(filter_b[:,0])*10,1000)
    plt.figure(2)
    plt.plot(filter_b[:,0],F_filter(filter_b[:,0]))
    plt.plot(lam_rest*(12.46),F_IMS(11.46,lam_rest))


    a,b = np.min(filter_b[:,0]),np.max(filter_b[:,0])

    aa = filter_bandwidth(a,b,lamb*(12.46))
    print(aa)

    plt.plot(aa,F_filter(aa),'k*')

    return a,b

for i in [125,140,160]:
    print(i)
    print(filter_init(str(i)))

plt.show()


'''
for i in range(0,19):
    a = np.loadtxt('./output2_processed/data_f125w_'+ str(i) +'.dat')
    b = np.loadtxt('./output2_processed/data_f140w_'+ str(i) +'.dat')
    c = np.loadtxt('./output2_processed/data_f160w_'+ str(i) +'.dat')
    d = np.loadtxt('./output2_processed/data_total_'+ str(i) +'.dat')
    print(np.max(a),np.max(b),np.max(c),np.max(d))

files = glob.glob("./drt/muv.bin*")
print(files)
files = sorted(files)
print(files)
'''

