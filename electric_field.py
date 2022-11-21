"""
Reuben Allen
10/2/2021

This program will predict the electric field (direction not magnitude) surrounding
stationary charged particules using Coulombs's Law. The tracer particle
used to calculate the field was set to be positive.
"""

# import python libraries
import numpy as np
from numpy.core.function_base import linspace
from scipy import constants
from matplotlib import pyplot as plt

def check_str():
    """This function fetches user inputs and checks them before proceeding."""
    coord_arr = 0
    while True:
        print("Enter the number of charged particules you wish to simulate:")
        num_str = input()
        try:
            num_particles = int(num_str)
            if num_particles > 0:
                coord_arr = np.empty((num_particles,3))
                for i in range(num_particles):
                    while True:
                        print("Enter the coordinates and charge of particle {0} using the format 'x,y,q':".format(i+1))
                        coord_str = input().split(",")
                        try:
                            coord_i = [float(x) for x in coord_str]
                            if not coord_i[:-1] in coord_arr[:,:-1].tolist() or i == 0:
                                coord_arr[i,0:4] = coord_i
                            else:
                                print("Invalid point entered. Only unique coordinates are accepted.")
                                continue
                        except ValueError:
                            print("Invalid characters entered. Please try again!")
                            continue
                        break
            else:
                print("The number of particles must be greater than zero. Please try again!")
                continue
        except ValueError:
            print("Invalid characters entered. Please try again!")
            continue
        break
    return coord_arr

def coulomb(q1,r):
    """Coulombs law to find magnitude of electric field of a single particle (using test particle with charge +1, no constant)"""
    return q1 / r ** 2

def main(coord_arr):
    """Computes electric field using coulombs law """
    # determining scale of output vector field
    if coord_arr.shape[0] > 1:
        lims_multiplier = np.max(np.multiply(np.ptp(coord_arr[:,0:-1], axis = 0),2)) # the last number in the parenthases is the scale factor
        limits = np.array([np.add(np.max(coord_arr[:,0:-1],axis=0),lims_multiplier),np.subtract(np.min(coord_arr[:,0:-1],axis=0),lims_multiplier)])
    else:
        limits = np.array([np.add(coord_arr[:,:-1].flatten(),5),np.subtract(coord_arr[:,:-1].flatten(),5)])

    # generate vector origin grid
    X,Y = np.meshgrid(linspace(limits[1,0],limits[0,0],50),linspace(limits[1,1],limits[0,1],50))
    U = np.zeros(X.shape) # horizontal component array
    V = U.copy() # vertical component array

    for i in range(coord_arr.shape[0]):
        # find direction vectors
        X_diff = np.subtract(X,coord_arr[i,0])
        Y_diff = np.subtract(Y,coord_arr[i,1])
        magnitude = np.hypot(X_diff,Y_diff)
        X_unit = np.divide(X_diff,magnitude)
        Y_unit = np.divide(Y_diff,magnitude)
        F = np.zeros(X.shape) # electromagnetic force array
        for index in np.ndindex(X.shape):
            force = coulomb(coord_arr[i,2],magnitude[index])
            F[index] = force
        U = np.add(U,np.multiply(F,X_unit))
        V = np.add(V,np.multiply(F,Y_unit))
    
    # calculate unit vectors from field components
    # although not strictly necessary, this step improves performance in the immediate area surrounding a charge
    sum_magnitude = np.hypot(U,V)
    U_unit = np.divide(U,sum_magnitude)
    V_unit = np.divide(V,sum_magnitude)

    # plotting
    plt.streamplot(X,Y,U_unit,V_unit,density= 1.2)
    for i in range(coord_arr.shape[0]):
        if coord_arr[i,2] < 0:
            plt.plot(coord_arr[i,0],coord_arr[i,1],'-or')
        elif coord_arr[i,2] > 0:
            plt.plot(coord_arr[i,0],coord_arr[i,1],'-og')
        else:
            plt.plot(coord_arr[i,0],coord_arr[i,1],'-ok')
    plt.title("Electric Field")
    plt.show()
if __name__ == "__main__":
    a = check_str()
    main(a)