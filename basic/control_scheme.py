#! /usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import time

import control_param
from control_param import *
import control_bcs
from control_bcs import *
import control_fluxes_v
from control_fluxes_v import *
import control_fluxes_i
from control_fluxes_i import *


def calc_rk(param,mesh,mpi,Q,Q0):

  deltat = param.time.dt
  neq = param.neq
  Q2 = Q.copy()
  Q3 = Q.copy()

  for irk in range(0, len(rk)):
    
    Fi = calc_inviscid_fluxes(param,mesh,Q2)
    Fv = calc_viscous_fluxes(param,mpi,mesh,Q2)
    #Fv = calc_viscous_fluxes_split(param,mpi,mesh,Q2)

    alpha = deltat*rk[irk]
    
    for i in range(0,neq):
      Q3[i] = Q[i] - alpha * ( Fi[i] +Fv[i] )
    
    Q3 = update_boundary_condition(mpi,Q3)
    
    Q2 = Q3.copy()
  
  if (param.scheme.nfilter>0): Q3 = calc_filter(param,mesh,Q3,Q0)

  return Q3
  

def calc_filter(param,mesh,Q,Q0):

  for i in range(len(Q)):
    Q[i] = explicit_filtering(param,mesh,Q[i],Q0[i])
    
  return Q
  

def explicit_filtering(param,mesh,U,U0):
    
  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  csi = param.scheme.relaxation_coeff
  nf = param.scheme.nfilter
  stencil = 5
  a = D[nf,:]
  
  # Separate ref field
  Uf = U - U0

  Dx = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  Dy = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  Dz = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  
  Dx[xi:xf, yi:yf, zi:zf] = (a[0] * (Uf[xi:xf, yi:yf, zi:zf]) + \
                             a[1] * (Uf[xi+1:xf+1, yi:yf, zi:zf] + Uf[xi-1:xf-1, yi:yf, zi:zf]) + \
                             a[2] * (Uf[xi+2:xf+2, yi:yf, zi:zf] + Uf[xi-2:xf-2, yi:yf, zi:zf]) + \
                             a[3] * (Uf[xi+3:xf+3, yi:yf, zi:zf] + Uf[xi-3:xf-3, yi:yf, zi:zf]) + \
                             a[4] * (Uf[xi+4:xf+4, yi:yf, zi:zf] + Uf[xi-4:xf-4, yi:yf, zi:zf]) + \
                             a[5] * (Uf[xi+5:xf+5, yi:yf, zi:zf] + Uf[xi-5:xf-5, yi:yf, zi:zf]) )

  Dy[xi:xf, yi:yf, zi:zf] = (a[0] * (Uf[xi:xf, yi:yf, zi:zf]) + \
                             a[1] * (Uf[xi:xf, yi+1:yf+1, zi:zf] + Uf[xi:xf, yi-1:yf-1, zi:zf]) + \
                             a[2] * (Uf[xi:xf, yi+2:yf+2, zi:zf] + Uf[xi:xf, yi-2:yf-2, zi:zf]) + \
                             a[3] * (Uf[xi:xf, yi+3:yf+3, zi:zf] + Uf[xi:xf, yi-3:yf-3, zi:zf]) + \
                             a[4] * (Uf[xi:xf, yi+4:yf+4, zi:zf] + Uf[xi:xf, yi-4:yf-4, zi:zf]) + \
                             a[5] * (Uf[xi:xf, yi+5:yf+5, zi:zf] + Uf[xi:xf, yi-5:yf-5, zi:zf]) )
                             
  Dz[xi:xf, yi:yf, zi:zf] = (a[0] * (Uf[xi:xf, yi:yf, zi:zf]) + \
                             a[1] * (Uf[xi:xf, yi:yf, zi+1:zf+1] + Uf[xi:xf, yi:yf, zi-1:zf-1]) + \
                             a[2] * (Uf[xi:xf, yi:yf, zi+2:zf+2] + Uf[xi:xf, yi:yf, zi-2:zf-2]) + \
                             a[3] * (Uf[xi:xf, yi:yf, zi+3:zf+3] + Uf[xi:xf, yi:yf, zi-3:zf-3]) + \
                             a[4] * (Uf[xi:xf, yi:yf, zi+4:zf+4] + Uf[xi:xf, yi:yf, zi-4:zf-4]) + \
                             a[5] * (Uf[xi:xf, yi:yf, zi+5:zf+5] + Uf[xi:xf, yi:yf, zi-5:zf-5]) )
                            
  Uf = U - csi*Dx - csi*Dy - csi*Dz

  return Uf
