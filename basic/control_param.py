#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import math
import yaml
from math import sqrt, sin, cos, acos, pi

class Param:
  """ CFD parameters """

  def __init__(self, data, config):
  
    self.casename = data[config]['casename']
    self.neq = data[config]['nequations']
    
    class Mpi:
      def __init__(self, data, config):
        self.ndomains = np.array(data[config]['mpi']['ndomains'])
    self.mpi = Mpi(data,config)
    
    class Mesh:
      def __init__(self, data, config):
        self.nxg = data[config]['mesh']['nx']
        self.nyg = data[config]['mesh']['ny']
        self.nzg = data[config]['mesh']['nz']
        self.xmin_xmax = np.array(data[config]['mesh']['xmin_xmax'])
        self.ymin_ymax = np.array(data[config]['mesh']['ymin_ymax'])
        self.zmin_zmax = np.array(data[config]['mesh']['zmin_zmax'])
        self.rx_ry_rz = np.array(data[config]['mesh']['rx_ry_rz'])
    self.mesh = Mesh(data,config)
      
    class Time:
      def __init__(self, data, config):
        self.niterations = data[config]['time']['niterations']
        self.dt = data[config]['time']['dt']
        self.ndump = data[config]['time']['ndump']
        self.new_simulation = data[config]['time']['new_simulation']
        self.initial_solution = data[config]['time']['initial_solution']
    self.time = Time(data,config)
    
    class Scheme:
      def __init__(self, data, config):
        self.nrk = data[config]['scheme']['nrk']
        self.split_form = data[config]['scheme']['split_form']
        self.nscheme = data[config]['scheme']['nscheme']
        self.nfilter = data[config]['scheme']['nfilter']
        self.relaxation_coeff = data[config]['scheme']['relaxation_coeff']
        self.stencil = 5 #max(self.nscheme, self.nfilter)
    self.scheme = Scheme(data,config)
    
    class Flow:
      def __init__(self, data, config):
        self.dimension = data[config]['flow']['dimension']
        self.Mach_number = data[config]['flow']['Mach_number']
        self.gamma = data[config]['flow']['gamma']
        self.R_inf = data[config]['flow']['R_inf']
        self.U_inf = data[config]['flow']['U_inf']
        self.T_inf = data[config]['flow']['T_inf']
        
        if self.dimension == 'adim':
          cteRg = 1./(self.gamma*self.Mach_number**2)
          Cp = cteRg*self.gamma/(self.gamma-1.)
          Cv = Cp/self.gamma
          P_inf = self.R_inf*cteRg*self.T_inf
          c_inf=sqrt(self.gamma*P_inf/self.R_inf)
        
        self.cteRg = cteRg
        self.Cp = Cp
        self.Cv = Cv
        self.P_inf = P_inf
        self.c_inf = c_inf
        
    self.flow = Flow(data,config)
    
    class Transport:
      def __init__(self, data, config):
        self.visclaw_type = data[config]['transport']['visclaw_type']
        self.Reynolds_number = data[config]['transport']['Reynolds_number']
        self.Prandtl_number = data[config]['transport']['Prandtl_number']
    self.transport = Transport(data,config)
            
    class Shock:
      def __init__(self, data, config):
        self.shock_capture = data[config]['shock']['shock_capture']
    self.shock = Shock(data,config)
    
    class Sponge:
      def __init__(self, data, config):
        self.compute_sponge = data[config]['sponge']['compute_sponge']
    self.sponge = Sponge(data,config)
    
    class Les:
      def __init__(self, data, config):
        self.sgs_model = data[config]['les']['sgs_model']
    self.les = Les(data,config)
  
  
# For MPI domains
PERIODIC = 1                            # BC periodic
WALL = 2                                # BC wall
INLET = 3                               # BC inflow
OUTLET = 4                              # BC outflow
XNEG = 0; XPOS = 1                      # X limits
YNEG = 2; YPOS = 3                      # Y limits
ZNEG = 4; ZPOS = 5                      # Z limits

# For numerical schemes
A = np.zeros((6,6))
D = np.zeros((6,6))

# Euler o1
rk1 = np.array([1.])

# Runge Kutta coeffs. o4
rk4 = np.array([1./4., 1./3., 1./2., 1.])

# RKo6s Runge Kutta coeffs. optimised 2004
g1 =   1.000000000000
g2 =   0.500000000000
g3 =   0.165919771368
g4 =   0.040919732041
g5 =   0.007555704391
g6 =   0.000891421261

# RKo6s Runge Kutta coeffs. optimised 2024
g1 =   1.000000000000
g2 =   0.500000000000
g3 =   0.16605113027324025
g4 =   0.04095008253773693
g5 =   0.007606263159697804
g6 =   0.0008461111690354384

rk6 = np.zeros(6)
rk6[5] = 1;
rk6[4] = g2;
rk6[3] = g3/rk6[4];
rk6[2] = g4/(rk6[3]*rk6[4]);
rk6[1] = g5/(rk6[3]*rk6[2]*rk6[4])
rk6[0] = g6/(rk6[3]*rk6[2]*rk6[4]*rk6[1])

# RKo5s Runge Kutta coeffs. optimised 2024
rk5 = np.zeros(5)
g1 =   1.000000000000
g2 =   0.500000000000
g3 =   0.1655100819071849
g4 =   0.03997993949554004
g5 =   0.007156751645644292

rk5[4] = 1;
rk5[3] = g2;
rk5[2] = g3/rk5[3];
rk5[1] = g4/(rk5[2]*rk5[3]);
rk5[0] = g5/(rk5[2]*rk5[1]*rk5[3])

# Derivative and filter coeffs. o2
a3 = np.array([0.,  1./2., 0., 0., 0., 0.])
d3 = np.array([0.5, -0.25, 0., 0., 0., 0.])
A[1,:] = a3 ; D[1,:] = d3 ;

# Derivative and filter coeffs. o4
a5 = np.array([0.,    2./3., -1./12.,  0., 0., 0. ])
d5 = np.array([0.375, -0.25,  0.0625,  0., 0., 0. ])
A[2,:] = a5 ; D[2,:] = d5 ;

# Derivative and filter coeffs. o6
a7 = np.array([0.,         3./4., -3./20.,    1./60., 0., 0.])
d7 = np.array([0.3125, -0.234375, 0.09375, -0.015625, 0., 0.])
A[3,:] = a7 ; D[3,:] = d7 ;

# Derivative and filter coeffs. o8
a9 = np.array([0.,        4./5. , -1./5., 4./105., -1./280., 0.])
d9 = np.array([35./128., -7./32., 7./64., -1./32.,  1./256., 0.])
A[4,:] = a9 ; D[4,:] = d9 ;

# Derivative and filter coeffs. o10 (Standard)
a11 = np.array([0., 5./6, -5./21, 5./84, -5./504, 1./1260])
d11 = np.array([63./256,-105./512, 15./128,-45./1024, 5./512,-1./1024])

# Optimised from Bogey et Bailly (2004)
a11_2004 = np.array([0., 0.872756993962, -0.286511173973, 0.090320001280, -0.020779405824, 0.002484594688])
d11_2004 = np.array([0.215044884112, -0.187772883589, 0.123755948787, -0.059227575576, 0.018721609157, -0.002999540835])

# Optimised from Bogey et Bailly (2009)
a11_2009 = a11_2004
d11_2009 = np.array([0.234810479761700, -0.199250131285813, 0.120198310245186, -0.049303775636020, 0.012396449873964, -0.001446093078167])

# New coefficients (2024)
a11_2024 = np.array([0., 0.8635414482176262, -0.274305332247656, 0.08147311338115067, -0.01708000680758487, 0.0017939806729146523])
d11_2024 = np.array([0.22107796514583924, -0.19150151148699646, 0.12311347506118175, -0.05624682921948344, 0.0163475525925704, -0.002251669520191884])
A[5,:] = a11 ; D[5,:] = d11 ;
#A[5,:] = a11_2004 ; D[5,:] = d11_2004 ;
#A[5,:] = a11_2009 ; D[5,:] = d11_2009 ;
A[5,:] = a11_2024 ; D[5,:] = d11_2024 ;

# Derivative and filter coeffs. o4 (not centered)
a13 = np.array([-1./4., -5./6., 3./2., -1./2., 1./12.]) ; a31 = -a13 ;
a04 = np.array([-25./12., 4., -3., 4./3., -1./4.]) ; a40 = -a04 ;
a06 = np.array([-2.192280339, 4.748611401, -5.108851915, 4.461567104, -2.833498741, 1.128328861, -0.203876371]) ;
a24 = np.array([0.049041958, -0.468840357, -0.474760914, 1.273274737, -0.518484526, 0.166138533, -0.026369431]) ;
a15 = np.array([-0.209337622, -1.084875676, 2.147776050, -1.388928322, 0.768949766, -0.281814650, 0.048230454]) ;

# ====================
# Choose configuration
# ====================
config='tgv'
#config = input('Insert configuration:')

# Read input file
with open('inputs.in') as f:
    
    data = yaml.load(f, Loader=yaml.FullLoader)
    #print(data)
        
# Initialize parameters
param = Param(data,config)

if   (param.scheme.nrk == 1): rk = rk1;
elif (param.scheme.nrk == 4): rk = rk4;
elif (param.scheme.nrk == 5): rk = rk5;
elif (param.scheme.nrk == 6): rk = rk6;
else : exit();
