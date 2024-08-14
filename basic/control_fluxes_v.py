#! /usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np

import control_scheme
from control_scheme import *
import control_bcs
from control_bcs import *

  

def calc_viscous_fluxes(param,mpi,mesh,Q):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz ; x = mesh.x ; y = mesh.y ; z = mesh.z
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  gamma = param.flow.gamma
  rg = param.flow.cteRg
  Chp = param.flow.Cp
  Re = param.transport.Reynolds_number
  Pr = param.transport.Prandtl_number
  
  rho  = Q[0]
  rhou = Q[1]
  rhov = Q[2]
  rhow = Q[3]
  rhoe = Q[4]
  
  u = rhou/rho
  v = rhov/rho
  w = rhow/rho
  p = (gamma-1.)*(rhoe-0.5*rho*(u**2+v**2+w**2))
  T = p/(rho*rg)
  
  if (param.les.sgs_model == "Implicit"):
    Frhou,Frhov,Frhow,Frhoe,Grhou,Grhov,Grhow,Grhoe,Hrhou,Hrhov,Hrhow,Hrhoe = no_model(param,mesh,Q)
    
  elif (param.les.sgs_model == "Smagorinsky"):
    Frhou,Frhov,Frhow,Frhoe,Grhou,Grhov,Grhow,Grhoe,Hrhou,Hrhov,Hrhow,Hrhoe = Smagorinsky_model(param,mesh,Q)
    
  elif (param.les.sgs_model == "MSSM-LS"):
    Frhou,Frhov,Frhow,Frhoe,Grhou,Grhov,Grhow,Grhoe,Hrhou,Hrhov,Hrhow,Hrhoe = multi_scale_Smagorinsky_model_LS(param,mpi,mesh,Q)
        
  elif (param.les.sgs_model == "MSSM-SS"):
    Frhou,Frhov,Frhow,Frhoe,Grhou,Grhov,Grhow,Grhoe,Hrhou,Hrhov,Hrhow,Hrhoe = multi_scale_Smagorinsky_model_SS(param,mpi,mesh,Q)
            
  elif (param.les.sgs_model == "DSM"):
    Frhou,Frhov,Frhow,Frhoe,Grhou,Grhov,Grhow,Grhoe,Hrhou,Hrhov,Hrhow,Hrhoe = dynamic_Smagorinsky_model(param,mpi,mesh,Q)
    
  else :
    print("ERROR in routine: control_scheme"); exit();
    
  Krho  = 0.*rho
  Krhou = calc_xderiv_o4(mesh,Frhou,0)+calc_yderiv_o4(mesh,Grhou,0)+calc_zderiv_o4(mesh,Hrhou,0)
  Krhov = calc_xderiv_o4(mesh,Frhov,0)+calc_yderiv_o4(mesh,Grhov,0)+calc_zderiv_o4(mesh,Hrhov,0)
  Krhow = calc_xderiv_o4(mesh,Frhow,0)+calc_yderiv_o4(mesh,Grhow,0)+calc_zderiv_o4(mesh,Hrhow,0)
  Krhoe = calc_xderiv_o4(mesh,Frhoe,0)+calc_yderiv_o4(mesh,Grhoe,0)+calc_zderiv_o4(mesh,Hrhoe,0)
  
  K = [Krho,Krhou,Krhov,Krhow,Krhoe]

  return K
  
  
  
def calc_viscous_fluxes_split(param,mpi,mesh,Q):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz ; x = mesh.x ; y = mesh.y ; z = mesh.z
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  gamma = param.flow.gamma
  rg = param.flow.cteRg
  Chp = param.flow.Cp
  Re = param.transport.Reynolds_number
  Pr = param.transport.Prandtl_number
  
  rho  = Q[0]
  rhou = Q[1]
  rhov = Q[2]
  rhow = Q[3]
  rhoe = Q[4]
  
  u = rhou/rho
  v = rhov/rho
  w = rhow/rho
  p = (gamma-1.)*(rhoe-0.5*rho*(u**2+v**2+w**2))
  T = p/(rho*rg)
  
  visco = 1./Re           # viscosity
  kappa = Chp*visco/Pr    # termal coefficient k=mu*Cp/Pr
  
  S11 = calc_xderiv_o4(mesh,u,2)
  S22 = calc_yderiv_o4(mesh,v,2)
  S33 = calc_zderiv_o4(mesh,w,2)
  S12 = 0.5 * ( calc_xderiv_o4(mesh,v,2) + calc_yderiv_o4(mesh,u,2) )
  S13 = 0.5 * ( calc_xderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,u,2) )
  S23 = 0.5 * ( calc_yderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,v,2) )
  dTdx= calc_xderiv_o4(mesh,T,2)
  dTdy= calc_yderiv_o4(mesh,T,2)
  dTdz= calc_zderiv_o4(mesh,T,2)
  
  Frhoe = - visco * 2.*( rhou*(S11 - (S11+S22+S33)/3.) + rhov*S12 + rhow*S13 ) /rho - kappa*dTdx
  Grhoe = - visco * 2.*( rhou*S12 + rhov*(S22 - (S11+S22+S33)/3.) + rhow*S23 ) /rho - kappa*dTdy
  Hrhoe = - visco * 2.*( rhou*S13 + rhov*S23 + rhow*(S33 - (S11+S22+S33)/3.) ) /rho - kappa*dTdz
    
  Krho  = 0.*rho
  Krhoe = calc_xderiv_o4(mesh,Frhoe,0)+calc_yderiv_o4(mesh,Grhoe,0)+calc_zderiv_o4(mesh,Hrhoe,0)

  Sii = S11+S22+S33
  Krhou = - visco/3. * calc_xderiv_o4(mesh,Sii,0)
  Krhov = - visco/3. * calc_yderiv_o4(mesh,Sii,0)
  Krhow = - visco/3. * calc_zderiv_o4(mesh,Sii,0)
  
  d2udx2 = calc_2deriv_o4(mesh,u,0,0)
  d2udy2 = calc_2deriv_o4(mesh,u,0,1)
  d2udz2 = calc_2deriv_o4(mesh,u,0,2)
  d2vdx2 = calc_2deriv_o4(mesh,v,0,0)
  d2vdy2 = calc_2deriv_o4(mesh,v,0,1)
  d2vdz2 = calc_2deriv_o4(mesh,v,0,2)
  d2wdx2 = calc_2deriv_o4(mesh,w,0,0)
  d2wdy2 = calc_2deriv_o4(mesh,w,0,1)
  d2wdz2 = calc_2deriv_o4(mesh,w,0,2)
  
  Krhou = Krhou - visco*(d2udx2+d2udy2+d2udz2)
  Krhov = Krhov - visco*(d2vdx2+d2vdy2+d2vdz2)
  Krhow = Krhow - visco*(d2wdx2+d2wdy2+d2wdz2)
  
#  dviscodx = calc_xderiv_o4(mesh,visco,0)
#  dviscody = calc_yderiv_o4(mesh,visco,0)
#  dviscodz = calc_zderiv_o4(mesh,visco,0)
#  
#  Krhou = Krhou - dviscodx*2*(S11) - dviscody*2*(S12) - dviscodz*2*(S13)
#  Krhov = Krhov - dviscodx*2*(S12) - dviscody*2*(S22) - dviscodz*2*(S23)
#  Krhow = Krhow - dviscodx*2*(S13) - dviscody*2*(S23) - dviscodz*2*(S33)
#
#  Krhou = Krhou + dviscodx*2*(Sii)/3
#  Krhov = Krhov + dviscody*2*(Sii)/3
#  Krhow = Krhow + dviscodz*2*(Sii)/3
  
  K = [Krho,Krhou,Krhov,Krhow,Krhoe]

  return K
  
  
def calc_xderiv_o4(mesh,U,stencil2):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  stencil = 5
  dx = mesh.dx4
  a = a5
  
  Ux = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  
  # I need to update these variables to compute the first derivatives in calc_viscous_fluxes
  xi = xi - stencil2
  xf = xf + stencil2
  yi = yi - stencil2
  yf = yf + stencil2
  zi = zi - stencil2
  zf = zf + stencil2

  Ux[xi:xf, yi:yf, zi:zf] = (a[1] * (U[xi+1:xf+1, yi:yf, zi:zf] - U[xi-1:xf-1, yi:yf, zi:zf]) + \
                             a[2] * (U[xi+2:xf+2, yi:yf, zi:zf] - U[xi-2:xf-2, yi:yf, zi:zf]) ) * dx[xi:xf, yi:yf, zi:zf]
  
  return Ux
  

def calc_yderiv_o4(mesh,U,stencil2):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  stencil = 5
  dy = mesh.dy4
  a = a5
  
  Uy = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  
  # I need to update these variables to compute the first derivatives in calc_viscous_fluxes
  xi = xi - stencil2
  xf = xf + stencil2
  yi = yi - stencil2
  yf = yf + stencil2
  zi = zi - stencil2
  zf = zf + stencil2

  Uy[xi:xf, yi:yf, zi:zf] = (a[1] * (U[xi:xf, yi+1:yf+1, zi:zf] - U[xi:xf, yi-1:yf-1, zi:zf]) + \
                             a[2] * (U[xi:xf, yi+2:yf+2, zi:zf] - U[xi:xf, yi-2:yf-2, zi:zf]) ) * dy[xi:xf, yi:yf, zi:zf]
  
  return Uy


def calc_zderiv_o4(mesh,U,stencil2):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  stencil = 5
  dz = mesh.dz4
  a = a5
  
  Uz = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  
  # I need to update these variables to compute the first derivatives in calc_viscous_fluxes
  xi = xi - stencil2
  xf = xf + stencil2
  yi = yi - stencil2
  yf = yf + stencil2
  zi = zi - stencil2
  zf = zf + stencil2

  Uz[xi:xf, yi:yf, zi:zf] = (a[1] * (U[xi:xf, yi:yf, zi+1:zf+1] - U[xi:xf, yi:yf, zi-1:zf-1]) + \
                             a[2] * (U[xi:xf, yi:yf, zi+2:zf+2] - U[xi:xf, yi:yf, zi-2:zf-2]) ) * dz[xi:xf, yi:yf, zi:zf]
  
  return Uz




def no_model(param,mesh,Q):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz ; x = mesh.x ; y = mesh.y ; z = mesh.z
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  gamma = param.flow.gamma
  rg = param.flow.cteRg
  Chp = param.flow.Cp
  Re = param.transport.Reynolds_number
  Pr = param.transport.Prandtl_number
  stencil = param.scheme.stencil
  
  rho  = Q[0]
  rhou = Q[1]
  rhov = Q[2]
  rhow = Q[3]
  rhoe = Q[4]
  
  u = rhou/rho
  v = rhov/rho
  w = rhow/rho
  p = (gamma-1.)*(rhoe-0.5*rho*(u**2+v**2+w**2))
  T = p/(rho*rg)
  
  visco = 1./Re           # viscosity
  kappa = Chp*visco/Pr    # termal coefficient k=mu*Cp/Pr
  
  S11 = calc_xderiv_o4(mesh,u,2)
  S22 = calc_yderiv_o4(mesh,v,2)
  S33 = calc_zderiv_o4(mesh,w,2)
  S12 = 0.5 * ( calc_xderiv_o4(mesh,v,2) + calc_yderiv_o4(mesh,u,2) )
  S13 = 0.5 * ( calc_xderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,u,2) )
  S23 = 0.5 * ( calc_yderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,v,2) )
  dTdx= calc_xderiv_o4(mesh,T,2)
  dTdy= calc_yderiv_o4(mesh,T,2)
  dTdz= calc_zderiv_o4(mesh,T,2)
  
  Frhou = - visco * 2.*(S11 - (S11+S22+S33)/3.)
  Frhov = - visco * 2.*S12
  Frhow = - visco * 2.*S13
  Frhoe = - visco * 2.*( rhou*(S11 - (S11+S22+S33)/3.) + rhov*S12 + rhow*S13 ) /rho - kappa*dTdx

  Grhou = - visco * 2.*S12
  Grhov = - visco * 2.*(S22 - (S11+S22+S33)/3.)
  Grhow = - visco * 2.*S23
  Grhoe = - visco * 2.*( rhou*S12 + rhov*(S22 - (S11+S22+S33)/3.) + rhow*S23 ) /rho - kappa*dTdy

  Hrhou = - visco * 2.*S13
  Hrhov = - visco * 2.*S23
  Hrhow = - visco * 2.*(S33 - (S11+S22+S33)/3.)
  Hrhoe = - visco * 2.*( rhou*S13 + rhov*S23 + rhow*(S33 - (S11+S22+S33)/3.) ) /rho - kappa*dTdz

  return Frhou,Frhov,Frhow,Frhoe,Grhou,Grhov,Grhow,Grhoe,Hrhou,Hrhov,Hrhow,Hrhoe




def Smagorinsky_model(param,mesh,Q):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz ; x = mesh.x ; y = mesh.y ; z = mesh.z
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  gamma = param.flow.gamma
  rg = param.flow.cteRg
  Chp = param.flow.Cp
  Re = param.transport.Reynolds_number
  Pr = param.transport.Prandtl_number
  stencil = param.scheme.stencil
  
  rho  = Q[0]
  rhou = Q[1]
  rhov = Q[2]
  rhow = Q[3]
  rhoe = Q[4]
  
  u = rhou/rho
  v = rhov/rho
  w = rhow/rho
  p = (gamma-1.)*(rhoe-0.5*rho*(u**2+v**2+w**2))
  T = p/(rho*rg)
  
  visco = 1./Re           # viscosity
  kappa = Chp*visco/Pr    # termal coefficient k=mu*Cp/Pr
  
  S11 = calc_xderiv_o4(mesh,u,2)
  S22 = calc_yderiv_o4(mesh,v,2)
  S33 = calc_zderiv_o4(mesh,w,2)
  S12 = 0.5 * ( calc_xderiv_o4(mesh,v,2) + calc_yderiv_o4(mesh,u,2) )
  S13 = 0.5 * ( calc_xderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,u,2) )
  S23 = 0.5 * ( calc_yderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,v,2) )
  dTdx= calc_xderiv_o4(mesh,T,2)
  dTdy= calc_yderiv_o4(mesh,T,2)
  dTdz= calc_zderiv_o4(mesh,T,2)

  deltac = mesh.deltac
  Cs = 0.18; Ci = 0.0066; Prt = 0.6; # Smagorinsky constant
  visco_sgs = rho *((Cs*deltac)**2)*(2.*(S11**2+S22**2+S33**2+2.*(S12**2+S13**2+S23**2)))**0.5
  k_sgs = Ci *rho *((   deltac)**2)*(2.*(S11**2+S22**2+S33**2+2.*(S12**2+S13**2+S23**2)))

  visco = visco + visco_sgs
  kappa = kappa + Chp*visco_sgs/Prt
#  visco = (visco + abs(visco)) / 2.
#  k_sgs = (k_sgs + abs(k_sgs)) / 2.
  
  Frhou = - visco * 2.*(S11 - (S11+S22+S33)/3.) + 2.*k_sgs/3.
  Frhov = - visco * 2.*S12
  Frhow = - visco * 2.*S13
  Frhoe = - visco * 2.*( rhou*(S11 - (S11+S22+S33)/3.) + rhov*S12 + rhow*S13 ) /rho - kappa*dTdx + 2.*k_sgs/3. * u

  Grhou = - visco * 2.*S12
  Grhov = - visco * 2.*(S22 - (S11+S22+S33)/3.) + 2.*k_sgs/3.
  Grhow = - visco * 2.*S23
  Grhoe = - visco * 2.*( rhou*S12 + rhov*(S22 - (S11+S22+S33)/3.) + rhow*S23 ) /rho - kappa*dTdy + 2.*k_sgs/3. * v

  Hrhou = - visco * 2.*S13
  Hrhov = - visco * 2.*S23
  Hrhow = - visco * 2.*(S33 - (S11+S22+S33)/3.) + 2.*k_sgs/3.
  Hrhoe = - visco * 2.*( rhou*S13 + rhov*S23 + rhow*(S33 - (S11+S22+S33)/3.) ) /rho - kappa*dTdz + 2.*k_sgs/3. * w

  return Frhou,Frhov,Frhow,Frhoe,Grhou,Grhov,Grhow,Grhoe,Hrhou,Hrhov,Hrhow,Hrhoe




def multi_scale_Smagorinsky_model_LS(param,mpi,mesh,Q):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz ; x = mesh.x ; y = mesh.y ; z = mesh.z
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  gamma = param.flow.gamma
  rg = param.flow.cteRg
  Chp = param.flow.Cp
  Re = param.transport.Reynolds_number
  Pr = param.transport.Prandtl_number
  stencil = param.scheme.stencil
  
  rho  = Q[0]
  rhou = Q[1]
  rhov = Q[2]
  rhow = Q[3]
  rhoe = Q[4]
  
  u = rhou/rho
  v = rhov/rho
  w = rhow/rho
  p = (gamma-1.)*(rhoe-0.5*rho*(u**2+v**2+w**2))
  T = p/(rho*rg)
  
  visco = 1./Re           # viscosity
  kappa = Chp*visco/Pr    # termal coefficient k=mu*Cp/Pr
  
  S11 = calc_xderiv_o4(mesh,u,2)
  S22 = calc_yderiv_o4(mesh,v,2)
  S33 = calc_zderiv_o4(mesh,w,2)
  S12 = 0.5 * ( calc_xderiv_o4(mesh,v,2) + calc_yderiv_o4(mesh,u,2) )
  S13 = 0.5 * ( calc_xderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,u,2) )
  S23 = 0.5 * ( calc_yderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,v,2) )
  
  #print(S11[:,10,10])
  S = [S11, S22, S33, S12, S13, S23]
  S = update_boundary_condition(mpi,S)
  #print(S11[:,10,10])
  Sf = S.copy()
  Sf = calc_filter_Sij(param,mesh,Sf)
  Sf = update_boundary_condition(mpi,Sf)
  [S11f, S22f, S33f, S12f, S13f, S23f] = Sf
  #print(S11f[:,10,10])
  
  dTdx= calc_xderiv_o4(mesh,T,2)
  dTdy= calc_yderiv_o4(mesh,T,2)
  dTdz= calc_zderiv_o4(mesh,T,2)

  # SGS quantities based on the large scales
  deltac = mesh.deltac
  Cs = 0.18; Ci = 0.0066; Prt = 0.6; # Smagorinsky constant
  visco_sgs = rho *((Cs*deltac)**2)*(2.*(S11**2+S22**2+S33**2+2.*(S12**2+S13**2+S23**2)))**0.5
  k_sgs = Ci *rho *((   deltac)**2)*(2.*(S11**2+S22**2+S33**2+2.*(S12**2+S13**2+S23**2)))
  kappa_sgs = Chp*visco_sgs/Prt

  # Viscous fluxes
  Frhou = - visco * 2.*(S11 - (S11+S22+S33)/3.)
  Frhov = - visco * 2.*S12
  Frhow = - visco * 2.*S13
  Frhoe = - visco * 2.*( rhou*(S11 - (S11+S22+S33)/3.) + rhov*S12 + rhow*S13 ) /rho - kappa*dTdx

  Grhou = - visco * 2.*S12
  Grhov = - visco * 2.*(S22 - (S11+S22+S33)/3.)
  Grhow = - visco * 2.*S23
  Grhoe = - visco * 2.*( rhou*S12 + rhov*(S22 - (S11+S22+S33)/3.) + rhow*S23 ) /rho - kappa*dTdy

  Hrhou = - visco * 2.*S13
  Hrhov = - visco * 2.*S23
  Hrhow = - visco * 2.*(S33 - (S11+S22+S33)/3.)
  Hrhoe = - visco * 2.*( rhou*S13 + rhov*S23 + rhow*(S33 - (S11+S22+S33)/3.) ) /rho - kappa*dTdz
  
  # Turbulent fluxes
  Frhou = Frhou - visco_sgs * 2.*(S11f - (S11f+S22f+S33f)/3.) + 2.*k_sgs/3.
  Frhov = Frhov - visco_sgs * 2.*S12f
  Frhow = Frhow - visco_sgs * 2.*S13f
  Frhoe = Frhoe - visco_sgs * 2.*( rhou*(S11f - (S11f+S22f+S33f)/3.) + rhov*S12f + rhow*S13f ) /rho - kappa_sgs*dTdx + 2.*k_sgs/3. * u

  Grhou = Grhou - visco_sgs * 2.*S12f
  Grhov = Grhov - visco_sgs * 2.*(S22f - (S11f+S22f+S33f)/3.) + 2.*k_sgs/3.
  Grhow = Grhow - visco_sgs * 2.*S23f
  Grhoe = Grhoe - visco_sgs * 2.*( rhou*S12f + rhov*(S22f - (S11f+S22f+S33f)/3.) + rhow*S23f ) /rho - kappa_sgs*dTdy + 2.*k_sgs/3. * v

  Hrhou = Hrhou - visco_sgs * 2.*S13f
  Hrhov = Hrhov - visco_sgs * 2.*S23f
  Hrhow = Hrhow - visco_sgs * 2.*(S33f - (S11f+S22f+S33f)/3.) + 2.*k_sgs/3.
  Hrhoe = Hrhoe - visco_sgs * 2.*( rhou*S13f + rhov*S23f + rhow*(S33f - (S11f+S22f+S33f)/3.) ) /rho - kappa_sgs*dTdz + 2.*k_sgs/3. * w

  return Frhou,Frhov,Frhow,Frhoe,Grhou,Grhov,Grhow,Grhoe,Hrhou,Hrhov,Hrhow,Hrhoe





def multi_scale_Smagorinsky_model_SS(param,mpi,mesh,Q):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz ; x = mesh.x ; y = mesh.y ; z = mesh.z
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  gamma = param.flow.gamma
  rg = param.flow.cteRg
  Chp = param.flow.Cp
  Re = param.transport.Reynolds_number
  Pr = param.transport.Prandtl_number
  stencil = param.scheme.stencil
  
  rho  = Q[0]
  rhou = Q[1]
  rhov = Q[2]
  rhow = Q[3]
  rhoe = Q[4]
  
  u = rhou/rho
  v = rhov/rho
  w = rhow/rho
  p = (gamma-1.)*(rhoe-0.5*rho*(u**2+v**2+w**2))
  T = p/(rho*rg)
  
  visco = 1./Re           # viscosity
  kappa = Chp*visco/Pr    # termal coefficient k=mu*Cp/Pr
  
  S11 = calc_xderiv_o4(mesh,u,2)
  S22 = calc_yderiv_o4(mesh,v,2)
  S33 = calc_zderiv_o4(mesh,w,2)
  S12 = 0.5 * ( calc_xderiv_o4(mesh,v,2) + calc_yderiv_o4(mesh,u,2) )
  S13 = 0.5 * ( calc_xderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,u,2) )
  S23 = 0.5 * ( calc_yderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,v,2) )
  
  #print(S11[:,10,10])
  S = [S11, S22, S33, S12, S13, S23]
  S = update_boundary_condition(mpi,S)
  #print(S11[:,10,10])
  Sf = S.copy()
  Sf = calc_filter_Sij(param,mesh,Sf)
  Sf = update_boundary_condition(mpi,Sf)
  [S11f, S22f, S33f, S12f, S13f, S23f] = Sf
  #print(S11f[:,10,10])
  
  dTdx= calc_xderiv_o4(mesh,T,2)
  dTdy= calc_yderiv_o4(mesh,T,2)
  dTdz= calc_zderiv_o4(mesh,T,2)

  # SGS quantities based on the small scales
  deltac = mesh.deltac
  Cs = 0.18; Ci = 0.0066; Prt = 0.6; # Smagorinsky constant
  visco_sgs = rho *((Cs*deltac)**2)*(2.*(S11f**2+S22f**2+S33f**2+2.*(S12f**2+S13f**2+S23f**2)))**0.5
  k_sgs = Ci *rho *((   deltac)**2)*(2.*(S11f**2+S22f**2+S33f**2+2.*(S12f**2+S13f**2+S23f**2)))
  kappa_sgs = Chp*visco_sgs/Prt

  # Viscous fluxes
  Frhou = - visco * 2.*(S11 - (S11+S22+S33)/3.)
  Frhov = - visco * 2.*S12
  Frhow = - visco * 2.*S13
  Frhoe = - visco * 2.*( rhou*(S11 - (S11+S22+S33)/3.) + rhov*S12 + rhow*S13 ) /rho - kappa*dTdx

  Grhou = - visco * 2.*S12
  Grhov = - visco * 2.*(S22 - (S11+S22+S33)/3.)
  Grhow = - visco * 2.*S23
  Grhoe = - visco * 2.*( rhou*S12 + rhov*(S22 - (S11+S22+S33)/3.) + rhow*S23 ) /rho - kappa*dTdy

  Hrhou = - visco * 2.*S13
  Hrhov = - visco * 2.*S23
  Hrhow = - visco * 2.*(S33 - (S11+S22+S33)/3.)
  Hrhoe = - visco * 2.*( rhou*S13 + rhov*S23 + rhow*(S33 - (S11+S22+S33)/3.) ) /rho - kappa*dTdz
  
  # Turbulent fluxes
  Frhou = Frhou - visco_sgs * 2.*(S11f - (S11f+S22f+S33f)/3.) + 2.*k_sgs/3.
  Frhov = Frhov - visco_sgs * 2.*S12f
  Frhow = Frhow - visco_sgs * 2.*S13f
  Frhoe = Frhoe - visco_sgs * 2.*( rhou*(S11f - (S11f+S22f+S33f)/3.) + rhov*S12f + rhow*S13f ) /rho - kappa_sgs*dTdx + 2.*k_sgs/3. * u

  Grhou = Grhou - visco_sgs * 2.*S12f
  Grhov = Grhov - visco_sgs * 2.*(S22f - (S11f+S22f+S33f)/3.) + 2.*k_sgs/3.
  Grhow = Grhow - visco_sgs * 2.*S23f
  Grhoe = Grhoe - visco_sgs * 2.*( rhou*S12f + rhov*(S22f - (S11f+S22f+S33f)/3.) + rhow*S23f ) /rho - kappa_sgs*dTdy + 2.*k_sgs/3. * v

  Hrhou = Hrhou - visco_sgs * 2.*S13f
  Hrhov = Hrhov - visco_sgs * 2.*S23f
  Hrhow = Hrhow - visco_sgs * 2.*(S33f - (S11f+S22f+S33f)/3.) + 2.*k_sgs/3.
  Hrhoe = Hrhoe - visco_sgs * 2.*( rhou*S13f + rhov*S23f + rhow*(S33f - (S11f+S22f+S33f)/3.) ) /rho - kappa_sgs*dTdz + 2.*k_sgs/3. * w

  return Frhou,Frhov,Frhow,Frhoe,Grhou,Grhov,Grhow,Grhoe,Hrhou,Hrhov,Hrhow,Hrhoe




def calc_filter_Sij(param,mesh,Q):

  for i in range(len(Q)):
    Q[i] = explicit_test_filtering_Sij(param,mesh,Q[i])
    
  return Q



def explicit_test_filtering_Sij(param,mesh,U):
    
  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  csi = param.scheme.relaxation_coeff
  nf = param.scheme.nfilter
  stencil = 5
  # Filter coeff 11 points kc dx=pi/3
  a = np.array([2./3., -0.26775782, -0.12016956, 0., 0.03683622, 0.01775782])
  
  # Separate ref field
  Uf = U.copy()

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
                            
  #Uf = U - csi*Dx - csi*Dy - csi*Dz
  # We keep the small scales
  Uf = Dx + Dy + Dz

  return Uf







def dynamic_Smagorinsky_model(param,mpi,mesh,Q):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz ; x = mesh.x ; y = mesh.y ; z = mesh.z
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  gamma = param.flow.gamma
  rg = param.flow.cteRg
  Chp = param.flow.Cp
  Re = param.transport.Reynolds_number
  Pr = param.transport.Prandtl_number
  stencil = param.scheme.stencil
  
  rho  = Q[0]
  rhou = Q[1]
  rhov = Q[2]
  rhow = Q[3]
  rhoe = Q[4]
  
  u = rhou/rho
  v = rhov/rho
  w = rhow/rho
  p = (gamma-1.)*(rhoe-0.5*rho*(u**2+v**2+w**2))
  T = p/(rho*rg)
  
  visco = 1./Re           # viscosity
  kappa = Chp*visco/Pr    # termal coefficient k=mu*Cp/Pr
  
  S11 = calc_xderiv_o4(mesh,u,2)
  S22 = calc_yderiv_o4(mesh,v,2)
  S33 = calc_zderiv_o4(mesh,w,2)
  S12 = 0.5 * ( calc_xderiv_o4(mesh,v,2) + calc_yderiv_o4(mesh,u,2) )
  S13 = 0.5 * ( calc_xderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,u,2) )
  S23 = 0.5 * ( calc_yderiv_o4(mesh,w,2) + calc_zderiv_o4(mesh,v,2) )
  
  S = [S11, S22, S33, S12, S13, S23]
  S = update_boundary_condition(mpi,S)
  
  dTdx= calc_xderiv_o4(mesh,T,2)
  dTdy= calc_yderiv_o4(mesh,T,2)
  dTdz= calc_zderiv_o4(mesh,T,2)
  
  deltac = mesh.deltac;
  Cs = 0.18; Ci = 0.0066; Prt = 0.6; # Smagorinsky constant
  
  Smag2 = 2*(S11**2+S22**2+S33**2+2*(S12**2+S13**2+S23**2))
  Smag  = np.sqrt(Smag2)
  Sii   = S11+S22+S33

  rhouu = rhou*rhou/rho
  rhovv = rhov*rhov/rho
  rhoww = rhow*rhow/rho
  rhouv = rhou*rhov/rho
  rhouw = rhou*rhow/rho
  rhovw = rhov*rhow/rho
  tau11d = 2*rho*(deltac**2)*Smag*(S11-Sii/3.)
  tau22d = 2*rho*(deltac**2)*Smag*(S22-Sii/3.)
  tau33d = 2*rho*(deltac**2)*Smag*(S33-Sii/3.)
  tau12d = 2*rho*(deltac**2)*Smag*S12
  tau13d = 2*rho*(deltac**2)*Smag*S13
  tau23d = 2*rho*(deltac**2)*Smag*S23
  Civar  = 2*rho*(deltac**2)*Smag2
  
  S11_hat  = explicit_test_filtering_DSM(param,mesh,S11)
  S22_hat  = explicit_test_filtering_DSM(param,mesh,S22)
  S33_hat  = explicit_test_filtering_DSM(param,mesh,S33)
  S12_hat  = explicit_test_filtering_DSM(param,mesh,S12)
  S13_hat  = explicit_test_filtering_DSM(param,mesh,S13)
  S23_hat  = explicit_test_filtering_DSM(param,mesh,S23)
  rho_hat  = explicit_test_filtering_DSM(param,mesh,rho)
  rhou_hat = explicit_test_filtering_DSM(param,mesh,rhou)
  rhov_hat = explicit_test_filtering_DSM(param,mesh,rhov)
  rhow_hat = explicit_test_filtering_DSM(param,mesh,rhow)
  Civar_hat  = explicit_test_filtering_DSM(param,mesh,Civar)
  
  L11 = explicit_test_filtering_DSM(param,mesh,rhouu) - rhou_hat*rhou_hat/rho_hat
  L22 = explicit_test_filtering_DSM(param,mesh,rhovv) - rhov_hat*rhov_hat/rho_hat
  L33 = explicit_test_filtering_DSM(param,mesh,rhoww) - rhow_hat*rhow_hat/rho_hat
  L12 = explicit_test_filtering_DSM(param,mesh,rhouv) - rhou_hat*rhov_hat/rho_hat
  L13 = explicit_test_filtering_DSM(param,mesh,rhouw) - rhou_hat*rhow_hat/rho_hat
  L23 = explicit_test_filtering_DSM(param,mesh,rhovw) - rhov_hat*rhow_hat/rho_hat
  
  Shat_mag2 = 2*(S11_hat**2+S22_hat**2+S33_hat**2+2*(S12_hat**2+S13_hat**2+S23_hat**2))
  Shat_mag  = np.sqrt(Shat_mag2)
  Sii_hat   = S11_hat+S22_hat+S33_hat
  
  M11 = explicit_test_filtering_DSM(param,mesh,tau11d) - 2*rho_hat*4*(deltac**2)*Shat_mag*(S11_hat-Sii_hat/3.)
  M22 = explicit_test_filtering_DSM(param,mesh,tau22d) - 2*rho_hat*4*(deltac**2)*Shat_mag*(S22_hat-Sii_hat/3.)
  M33 = explicit_test_filtering_DSM(param,mesh,tau33d) - 2*rho_hat*4*(deltac**2)*Shat_mag*(S33_hat-Sii_hat/3.)
  M12 = explicit_test_filtering_DSM(param,mesh,tau12d) - 2*rho_hat*4*(deltac**2)*Shat_mag*(S12_hat)
  M13 = explicit_test_filtering_DSM(param,mesh,tau13d) - 2*rho_hat*4*(deltac**2)*Shat_mag*(S13_hat)
  M23 = explicit_test_filtering_DSM(param,mesh,tau23d) - 2*rho_hat*4*(deltac**2)*Shat_mag*(S23_hat)
  
  Lii = L11 + L22 + L33
  Cdnom = (L11-Lii/3.)*M11+(L22-Lii/3.)*M22+(L33-Lii/3.)*M33+2*(L12*M12+L13*M13+L23*M23)
  Cdden = M11**2+M22**2+M33**2+2*(M12**2+M13**2+M23**2) + 1.e-06
  Ciden = 2*rho_hat*4*(deltac**2)*Shat_mag2 - Civar_hat + 1.e-06
  Cinom = Lii * Ciden
  Ciden = Ciden**2
  
  # Apply Gaussian filter to smooth the solution
  for l in range(2):
    dyn = [Cdnom, Cdden, Cinom, Ciden]; dyn = update_boundary_condition(mpi,dyn)
    Cdnom = gaussian_filter_DSM(param,mesh,Cdnom)
    Cdden = gaussian_filter_DSM(param,mesh,Cdden)
    Cinom = gaussian_filter_DSM(param,mesh,Cinom)
    Ciden = gaussian_filter_DSM(param,mesh,Ciden)
    
  Cd = Cdnom/Cdden
  Ci = Cinom/Ciden
  dyn = [Cd, Ci]; dyn = update_boundary_condition(mpi,dyn)
  #print(np.mean(Cd[xi:xf,yi:yf,zi:zf]),np.mean(Ci[xi:xf,yi:yf,zi:zf]))

  visco_sgs = Cd * rho *((deltac)**2)*(2.*(S11**2+S22**2+S33**2+2.*(S12**2+S13**2+S23**2)))**0.5
  k_sgs =     Ci * rho *((deltac)**2)*(2.*(S11**2+S22**2+S33**2+2.*(S12**2+S13**2+S23**2)))

  visco = visco + visco_sgs
  kappa = kappa + Chp*visco_sgs/Prt
  visco = (visco + abs(visco)) / 2.
  k_sgs = (k_sgs + abs(k_sgs)) / 2.
  
  Frhou = - visco * 2.*(S11 - (S11+S22+S33)/3.) + 2.*k_sgs/3.
  Frhov = - visco * 2.*S12
  Frhow = - visco * 2.*S13
  Frhoe = - visco * 2.*( rhou*(S11 - (S11+S22+S33)/3.) + rhov*S12 + rhow*S13 ) /rho - kappa*dTdx + 2.*k_sgs/3. * u

  Grhou = - visco * 2.*S12
  Grhov = - visco * 2.*(S22 - (S11+S22+S33)/3.) + 2.*k_sgs/3.
  Grhow = - visco * 2.*S23
  Grhoe = - visco * 2.*( rhou*S12 + rhov*(S22 - (S11+S22+S33)/3.) + rhow*S23 ) /rho - kappa*dTdy + 2.*k_sgs/3. * v

  Hrhou = - visco * 2.*S13
  Hrhov = - visco * 2.*S23
  Hrhow = - visco * 2.*(S33 - (S11+S22+S33)/3.) + 2.*k_sgs/3.
  Hrhoe = - visco * 2.*( rhou*S13 + rhov*S23 + rhow*(S33 - (S11+S22+S33)/3.) ) /rho - kappa*dTdz + 2.*k_sgs/3. * w

  return Frhou,Frhov,Frhow,Frhoe,Grhou,Grhov,Grhow,Grhoe,Hrhou,Hrhov,Hrhow,Hrhoe




def explicit_test_filtering_DSM(param,mesh,U):
    
  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  csi = 0.5 # to test !!!!
  nf = param.scheme.nfilter
  stencil = 5
  # Filter coeff 11 points cutoff kc dx=pi/3
  a = np.array([2./3., -0.26775782, -0.12016956, 0., 0.03683622, 0.01775782])
  #a = np.array([0.66875165147621, -0.25104249240477, -0.09700265367034, 0.00104249240477, 0.01262682793224, 0.])
  #a = np.array([0.5, -0.25, 0., 0., 0., 0.])

  
  # Separate ref field
  Uf = U.copy()

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
  # We keep the small scales
  #Uf = Dx + Dy + Dz

  return Uf



def gaussian_filter_DSM(param,mesh,U):
    
  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  csi = 0.5
  nf = param.scheme.nfilter
  stencil = 5
  # Gaussian filter
  a = np.array([0.5, -0.25, 0., 0., 0., 0.])
  
  # Separate ref field
  Uf = U.copy()

  Dx = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  Dy = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  Dz = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  
  Dx[xi:xf, yi:yf, zi:zf] = (a[0] * (Uf[xi:xf, yi:yf, zi:zf]) + \
                             a[1] * (Uf[xi+1:xf+1, yi:yf, zi:zf] + Uf[xi-1:xf-1, yi:yf, zi:zf]) )

  Dy[xi:xf, yi:yf, zi:zf] = (a[0] * (Uf[xi:xf, yi:yf, zi:zf]) + \
                             a[1] * (Uf[xi:xf, yi+1:yf+1, zi:zf] + Uf[xi:xf, yi-1:yf-1, zi:zf]) )
                             
  Dz[xi:xf, yi:yf, zi:zf] = (a[0] * (Uf[xi:xf, yi:yf, zi:zf]) + \
                             a[1] * (Uf[xi:xf, yi:yf, zi+1:zf+1] + Uf[xi:xf, yi:yf, zi-1:zf-1]) )
                            
  Uf = U - csi*Dx - csi*Dy - csi*Dz
  # We keep the small scales
  #Uf = Dx + Dy + Dz

  return Uf




def calc_2deriv_o4(mesh,U,stencil2,dir): # For regular mesh only !

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  stencil = 5
  dx = mesh.dx4**2#4
  dy = mesh.dy4**2#4
  dz = mesh.dz4**2#4
  a = [-30./12, 16./12, -1./12] # r = 2
#  a = [6., -4., 1.]             # r = 4
  
  Ux = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  
  # I need to update these variables to compute the first derivatives in calc_viscous_fluxes
  xi = xi - stencil2
  xf = xf + stencil2
  yi = yi - stencil2
  yf = yf + stencil2
  zi = zi - stencil2
  zf = zf + stencil2

  if dir == 0:
    Ux[xi:xf, yi:yf, zi:zf] = (a[0] * (U[xi:xf, yi:yf, zi:zf]) + \
                               a[1] * (U[xi+1:xf+1, yi:yf, zi:zf] + U[xi-1:xf-1, yi:yf, zi:zf]) + \
                               a[2] * (U[xi+2:xf+2, yi:yf, zi:zf] + U[xi-2:xf-2, yi:yf, zi:zf]) ) * dx[xi:xf, yi:yf, zi:zf]
  elif dir == 1:
    Ux[xi:xf, yi:yf, zi:zf] = (a[0] * (U[xi:xf, yi:yf, zi:zf]) + \
                               a[1] * (U[xi:xf, yi+1:yf+1, zi:zf] + U[xi:xf, yi-1:yf-1, zi:zf]) + \
                               a[2] * (U[xi:xf, yi+2:yf+2, zi:zf] + U[xi:xf, yi-2:yf-2, zi:zf]) ) * dy[xi:xf, yi:yf, zi:zf]
  elif dir == 2:
    Ux[xi:xf, yi:yf, zi:zf] = (a[0] * (U[xi:xf, yi:yf, zi:zf]) + \
                               a[1] * (U[xi:xf, yi:yf, zi+1:zf+1] + U[xi:xf, yi:yf, zi-1:zf-1]) + \
                               a[2] * (U[xi:xf, yi:yf, zi+2:zf+2] + U[xi:xf, yi:yf, zi-2:zf-2]) ) * dz[xi:xf, yi:yf, zi:zf]
  
  return Ux
