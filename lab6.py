# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:59:05 2026

@author: sorla
"""
import rinexReader as rr
import SatOrbits as so
import frame2kml as fk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

clight = 299792458 # m/s

def create_kernel(obs, satpos, x):
    """ 
    Create Kernel matrix, A, and
    Data vector, L,
    for NLLS positioning solution
    
    Parameters
    ----------
    obs : TYPE
        DESCRIPTION.
    satpos : Pandas dataframe
        Contains satellite positions of all satellites currently in view.
    x : numpy array, size [4,]
        Contains position and clock of current iteration of the NLLS solution.

    Returns
    -------
    L : TYPE
        Data vector.
    A : TYPE
        Kernel matrix.

    """
        
    # Determine ranges to all satellites

    dx = satpos.iloc[:, 0].to_numpy() - x[0]
    dy = satpos.iloc[:, 1].to_numpy() - x[1]
    dz = satpos.iloc[:, 2].to_numpy() - x[2]

    rng1 = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Determine unit vectors towards the satellites 
    # (should be a 2D array containing all the unit vectors in the kernel)
    a1 = np.column_stack((dx / rng1, dy / rng1, dz / rng1))
    
    # Create a clock error vector and create the full kernel, A
    clkErr = np.ones((len(a1), 1))
    A = np.hstack((-a1, clkErr))
    
    # Determine the data vector: Obs - ranges - current clock error
    L = obs.iloc[:, 0].to_numpy() - rng1 - x[3]
    return L, A

def spp(obs, satpos, x0):
    """
    

    Parameters
    ----------
    obs : TYPE
        DESCRIPTION.
    satpos : TYPE
        DESCRIPTION.
    x0 : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    """
    
    ##### Non-Linear Least Squares SPP #####
    tol = 0.001 # Solution must change less than "tol" in 3D to end the loop
    maxiter = 50 # Maximum number of iterations (10 should be enough)
    
    x = x0
    
    # Initialize the NLLS solution
    curiter = 0 # count number of iterations
    h = np.array([100, 100, 100]) # Select inital h far above tolerance

    while np.sum(np.abs(h)) > tol and (curiter < maxiter):

        # Setup the LS system
        L, A = create_kernel(obs, satpos, x)

        # Solve LS system
        dx = np.linalg.inv(A.T @ A) @ A.T @ L # Solve for dx using the normal equations
        
        # Update solution
        x = x+dx
        h = dx[:3]
        curiter += 1 # Keep track of number of iterations

    x = pd.Series(x, index=['X', 'Y', 'Z', 'cdt'], name='Solution') # Setup output
    
    return x

# Path to file (or file name if file is in the current working directory)
filepath = r"d:\M.Sc. Autonomous Systems - DTU\Spring Semester\30554 GNSS\Lab\lab6\SEPT0640.26O"

# Create rinex reader object
rinexFile = rr.rinexReader(filepath)
# Create a svpos object for getting satellite positions from sp3
svpos = so.sp3Orbits(r"d:\M.Sc. Autonomous Systems - DTU\Spring Semester\30554 GNSS\Lab\lab6\COD0OPSRAP_20260640000_01D_05M_ORB.SP3")

# Select constellations and observations to read
consts = ['G'] # Load GPS data
sigTypes = ["C1C"] # Load only code observations

# Now read the selected contents from the file
rinexFile.readFile(consts, sigTypes)

# Initial guess for position
x0 = [0, 0, 0, 0] # Guess of x, y, z, AND dt

# Initialize a dictionary for storing all solutions
sol = {}

startrun = time.time()

print("Computing solutions: ", end="")

for epoch in rinexFile.timelist:
    
    # print(f"Computing solution for epoch: {epoch}")
    print(".", end="")

    # Load data from current epoch
    obs = rinexFile.get_epoch_data(epoch, oTypes=sigTypes)
    obs.dropna() # Remove potential nans
    
    # Get signal travel time
    tau = obs.loc[:,'C1C'] / clight
    
    # Get satellite positions (only for satellites with observations)
    satpos = svpos.getSvPos(epoch, tau)
    
    # Split satellite positions and clock errors
    cdts = satpos.iloc[:, 3] * clight # Get satellite clock errors
    satpos = satpos.iloc[:, :3] # Satellite positions (no clocks)
    
    # Account for satellite clock errors
    obs = obs + cdts.values[:, None]
    
    # Get single point positioning
    x = spp(obs, satpos, x0=x0)
    
    # Store SPP in a dictionary
    sol[epoch] = x
    
endrun = time.time()
processingtime = round(endrun-startrun, 3)
print("")
print(f"Computed {len(rinexFile.timelist)} solutions in: {processingtime} seconds")

soldf = pd.DataFrame(sol).T

fk.dataframe_to_kml(
    soldf,
    filename="Session_1.kml",
    altitudeMode="clampToGround",
    color="green",
    placeorline=3
)

plt.figure(figsize=(10, 5))
plt.plot(soldf.index, soldf["X"])
plt.xlabel("Epoch")
plt.ylabel("X [m]")
plt.title("Receiver X Coordinate")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(soldf.index, soldf["Y"])
plt.xlabel("Epoch")
plt.ylabel("Y [m]")
plt.title("Receiver Y Coordinate")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(soldf.index, soldf["Z"])
plt.xlabel("Epoch")
plt.ylabel("Z [m]")
plt.title("Receiver Z Coordinate")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Convert to dataframe ###
# altitudeMode: 
#    clampToGround (Height set to 0)
#    absolute      (Height is plotted as is)
#    relativeToGround (Should account for terrain??)
# color: (pick one)
#   purple, green, yellow, red, blue, orange, teal, pink, white
# placeorline:
#   1: Plot positions as placemarks (dots/scatter plot)
#   2: Plot positions as a lineplot
#   3: Plot both placemarks and lines (dots with lines between them)
