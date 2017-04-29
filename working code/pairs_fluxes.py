import numpy as np
from photutils import CircularAperture
from photutils import aperture_photometry
import matplotlib.pylab as plt

# TO COMPUTE FLUX FOR ISOLATED OBJECTS
def fluxes(temp, data_sources, N, X, Y, group=0, dist_max=3.0):

    if N > 1:

        apertures_values = np.zeros(N)
        params = np.zeros(N)
        x_coord = np.zeros(N)
        y_coord = np.zeros(N)

        # fluxes
        for j in range(1, N+1):  # coords of all the sources

            A = np.argwhere(data_sources==j)
            x_coord[j-1] = np.mean(X[A[:, 0], A[:, 1]])
            y_coord[j-1] = np.mean(Y[A[:, 0], A[:, 1]])

        for i in range(0, N-1):  # for every sources except the last in matrix NxN

            A = np.sqrt(np.power((x_coord[i+1:]-x_coord[i]), 2) + np.power((y_coord[i+1:]-y_coord[i]), 2))  # find distances to the sources to the right
            B = np.where(A <= dist_max)  # choose those with dist <= dist_max

            for ii in range(len(B[0])):  # counting
                params[i] = 1  # 1st paired object (main)
                params[i+1+B[0][ii]] = 1  # 2nd paired object

        # apertures
        for j in range(1, N+1):
            A = np.argwhere(data_sources == j)
            if np.mean(A[:,0]) <= 1 or np.mean(A[:,1]) <= 1:
                apertures_values[j-1] = temp[A[0,0], A[0,1]]
            else:
                aperture = CircularAperture([np.mean(A[:,1]), np.mean(A[:, 0])], r=np.sqrt(len(A)))
                flux = aperture_photometry(temp, aperture)
                apertures_values[j-1] = flux['aperture_sum']

        if group:
            return apertures_values[np.where(params > 0)[0]]
        else:
            return apertures_values[np.where(params == 0)[0]]

    elif N == 1:

        A = np.argwhere(data_sources == 1)
        aperture = CircularAperture([np.mean(A[:,1]),np.mean(A[:,0])], r=np.sqrt(len(A)))
        flux = aperture_photometry(temp, aperture)
        apertures_value = flux['aperture_sum']

        if group:
            return []
        else:
            return apertures_value

    else:
        return []

def check():

    ang = 5
    nbins = int(2*5/0.13)
    pixels_ang_coords = (np.linspace(-ang, ang, nbins + 1)[1:] + np.linspace(-ang, ang, nbins + 1)[:-1])/2
    X, Y = np.meshgrid(pixels_ang_coords,pixels_ang_coords)
    print(np.round(X,2))
    print(np.round(Y,2))
    im = np.zeros((nbins,nbins))
    coords_x = [5,5,20,50]
    coords_y = [10,15,20,50]
    im[coords_x,coords_y] = [1,2,3,4]

    print(X[coords_x,coords_y])
    print(Y[coords_x,coords_y])
    print('GROUP')
    print(fluxes(im,im,len(coords_x),X,Y,group=1,dist_max=1))
    print('GROUP')
    print(fluxes(im,im,len(coords_x),X,Y,group=1))
    print('ISO')
    print(fluxes(im,im,len(coords_x),X,Y,group=0))

    for i in range(len(coords_x)):
        dist = np.sqrt((X.T - pixels_ang_coords[coords_x[i]])**2 + (Y.T - pixels_ang_coords[coords_y[i]])**2 )
        dots = np.where(dist<3)
        im[dots] += 0.5

    for i in range(len(coords_x)):
        dist = np.sqrt((X.T - pixels_ang_coords[coords_x[i]])**2 + (Y.T - pixels_ang_coords[coords_y[i]])**2 )
        dots = np.where(dist<1)
        im[dots] += 0.5

    plt.imshow(im,interpolation='nearest',extent=[-ang,ang,-ang,ang])
    plt.show()
