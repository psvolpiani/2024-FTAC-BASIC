#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
Author: Pedro Stefanin Volpiani
mpiexec -n 2 python main.py
"""

# Library imports
import os
import numpy as np
import yaml
import math
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors
from mpi4py import MPI
from math import sqrt, sin, cos, acos, atan, tan, atan2, pi
from numpy import *

import control_param
from control_param import *
import control_mpi
from control_mpi import *
import control_mesh
from control_mesh import *
import control_scheme
from control_scheme import *
import control_init
from control_init import *
import control_bcs
from control_bcs import *
import control_fluxes_v
from control_fluxes_v import *
import control_fluxes_i
from control_fluxes_i import *


start = time.time()

# Initialize MPI
mpi = Mpi(param)
mpi.initialize_comm3d(param)
mpi.create_subarray_datatype(param)

# Build global and local meshes
mesh=Mesh(param)
mesh.compute_local_mesh(param, mpi)

if (param.les.sgs_model != "Implicit"): mesh.compute_deltac(param,mpi);

# Initialization of conservative variables
if param.time.new_simulation == True:
  # From scratch
  rho,rhou,rhov,rhow,rhoe = initialisation(param,mesh,mpi)
  rho0,rhou0,rhov0,rhow0,rhoe0 = initialisation0(param,mesh,mpi)
  t  = 0.; nn = 0; ni = 0
else:
  # Read initial solution
  rho,rhou,rhov,rhow,rhoe,nn,t = mpi.read_start_parallel(param,mesh)
  rho0,rhou0,rhov0,rhow0,rhoe0 = initialisation0(param,mesh,mpi)
  ni = nn
  
# For debug
#plot_image_per_proc(rho,rhou,rhov,rhow,rhoe,mesh)

Q = [rho, rhou, rhov, rhow, rhoe]
Q = update_boundary_condition(mpi,Q)

Q0 = [rho0,rhou0,rhov0,rhow0,rhoe0]
Q0 = update_boundary_condition(mpi,Q0)
  
# For debug
#plot_image_per_proc_test(rhou,mesh)

# Write solution at iteration 0
#mpi.write_start_parallel_2(param,mesh,t,nn,Q)

# Open temporal file
file = open("./temporal.txt","w")

# Temporal loop
dt = param.time.dt
niterations = param.time.niterations + 1
ndump = param.time.ndump

for nn in range(ni+1,niterations):

  Q = calc_rk(param,mesh,mpi,Q,Q0)
  t += dt
  
# For debug
#  rhou=Q[1]
#  if (nn%100 == 0): plot_image_per_proc_test(rhou,mesh)
  
  # Write files
  mpi.write_temporal(param,mesh,nn,t,Q,file)
  if (nn%ndump == 0):
    mpi.write_start_parallel_2(param,mesh,t,nn,Q)
    message = "Dumping instantaneous field at iteration "+str(nn)
    print_message(message,mpi.rank)
      
# Close temporal file
file.close()

# Free memory
mpi.finalize()

end = time.time()
print("Total time to run the code:")
print(end - start)
