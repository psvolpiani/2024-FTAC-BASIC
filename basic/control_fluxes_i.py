#! /usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np

import control_scheme
from control_scheme import *


def calc_inviscid_fluxes(param,mesh,Q):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz ; x = mesh.x ; y = mesh.y ; z = mesh.z
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  stencil = 5 #param.scheme.stencil
  gamma = param.flow.gamma
  
  rho  = Q[0]
  rhou = Q[1]
  rhov = Q[2]
  rhow = Q[3]
  rhoe = Q[4]
  
  u = rhou/rho
  v = rhov/rho
  w = rhow/rho
  p = (gamma-1.)*(rhoe-0.5*rho*(u**2+v**2+w**2))
  
  Frho  = rhou
  Frhou = p + rhou*u
  Frhov = rhou*v
  Frhow = rhou*w
  Frhoe = (rhoe+p)*u

  Grho  = rhov
  Grhou = rhov*u
  Grhov = p + rhov*v
  Grhow = rhov*w
  Grhoe = (rhoe+p)*v

  Hrho  = rhow
  Hrhou = rhow*u
  Hrhov = rhow*v
  Hrhow = p + rhow*w
  Hrhoe = (rhoe+p)*w
  
  Krho  = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  Krhou = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  Krhov = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  Krhow = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  Krhoe = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  
        
  if (param.scheme.split_form == "Divergence"):

      # Non-split form : unstable => filtering procedure to stabilize the scheme
      Krho  = calc_xderiv(param,mesh,Frho )+calc_yderiv(param,mesh,Grho )+calc_zderiv(param,mesh,Hrho )
      Krhou = calc_xderiv(param,mesh,Frhou)+calc_yderiv(param,mesh,Grhou)+calc_zderiv(param,mesh,Hrhou)
      Krhov = calc_xderiv(param,mesh,Frhov)+calc_yderiv(param,mesh,Grhov)+calc_zderiv(param,mesh,Hrhov)
      Krhow = calc_xderiv(param,mesh,Frhow)+calc_yderiv(param,mesh,Grhow)+calc_zderiv(param,mesh,Hrhow)
      Krhoe = calc_xderiv(param,mesh,Frhoe)+calc_yderiv(param,mesh,Grhoe)+calc_zderiv(param,mesh,Hrhoe)

  
  elif (param.scheme.split_form == "Ducros"):

      # Split-form : stable !
      # -- x --
      Krho   = Krho  + 0.5 * ( u*calc_xderiv(param,mesh,rho ) + rho *calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frho ) )
      Krhou  = Krhou + 0.5 * ( u*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frhou) )
      Krhov  = Krhov + 0.5 * ( u*calc_xderiv(param,mesh,rhov) + rhov*calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frhov) )
      Krhow  = Krhow + 0.5 * ( u*calc_xderiv(param,mesh,rhow) + rhow*calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frhow) )
      Krhoe  = Krhoe + 0.5 * ( u*calc_xderiv(param,mesh,rhoe+p) + (rhoe+p)*calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frhoe) )

      # -- y --
      Krho   = Krho  + 0.5 * ( v*calc_yderiv(param,mesh,rho ) + rho *calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grho ) )
      Krhou  = Krhou + 0.5 * ( v*calc_yderiv(param,mesh,rhou) + rhou*calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grhou) )
      Krhov  = Krhov + 0.5 * ( v*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grhov) )
      Krhow  = Krhow + 0.5 * ( v*calc_yderiv(param,mesh,rhow) + rhow*calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grhow) )
      Krhoe  = Krhoe + 0.5 * ( v*calc_yderiv(param,mesh,rhoe+p) + (rhoe+p)*calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grhoe) )

      # -- z --
      Krho   = Krho  + 0.5 * ( w*calc_zderiv(param,mesh,rho ) + rho *calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrho ) )
      Krhou  = Krhou + 0.5 * ( w*calc_zderiv(param,mesh,rhou) + rhou*calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrhou) )
      Krhov  = Krhov + 0.5 * ( w*calc_zderiv(param,mesh,rhov) + rhov*calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrhov) )
      Krhow  = Krhow + 0.5 * ( w*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrhow) )
      Krhoe  = Krhoe + 0.5 * ( w*calc_zderiv(param,mesh,rhoe+p) + (rhoe+p)*calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrhoe) )

  
  elif (param.scheme.split_form == "Blaisdell"):

      # Split-form : unstable !
      rhoh = rhoe+p
      
      # -- x --
      Krho   = Krho  + calc_xderiv(param,mesh,Frho)
      Krhou  = Krhou + 0.5 * ( u*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frhou) )
      Krhov  = Krhov + 0.5 * ( u*calc_xderiv(param,mesh,rhov) + rhov*calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frhov) )
      Krhow  = Krhow + 0.5 * ( u*calc_xderiv(param,mesh,rhow) + rhow*calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frhow) )
      Krhoe  = Krhoe + 0.5 * ( u*calc_xderiv(param,mesh,rhoh) + rhoh*calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frhoe) )

      # -- y --
      Krho   = Krho  + calc_yderiv(param,mesh,Grho)
      Krhou  = Krhou + 0.5 * ( v*calc_yderiv(param,mesh,rhou) + rhou*calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grhou) )
      Krhov  = Krhov + 0.5 * ( v*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grhov) )
      Krhow  = Krhow + 0.5 * ( v*calc_yderiv(param,mesh,rhow) + rhow*calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grhow) )
      Krhoe  = Krhoe + 0.5 * ( v*calc_yderiv(param,mesh,rhoh) + rhoh*calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grhoe) )

      # -- z --
      Krho   = Krho  + calc_zderiv(param,mesh,Hrho)
      Krhou  = Krhou + 0.5 * ( w*calc_zderiv(param,mesh,rhou) + rhou*calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrhou) )
      Krhov  = Krhov + 0.5 * ( w*calc_zderiv(param,mesh,rhov) + rhov*calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrhov) )
      Krhow  = Krhow + 0.5 * ( w*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrhow) )
      Krhoe  = Krhoe + 0.5 * ( w*calc_zderiv(param,mesh,rhoh) + rhoh*calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrhoe) )
  
  
  elif (param.scheme.split_form == "Feiereisen"):

      # Split-form : stable !
      rhoh = rhoe+p
      h = rhoh/rho
      
      # -- x --
      Krho   = Krho  + calc_xderiv(param,mesh,Frho)
      Krhou  = Krhou + 0.5 * ( u*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frhou) )
      Krhov  = Krhov + 0.5 * ( v*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,v) + calc_xderiv(param,mesh,Frhov) )
      Krhow  = Krhow + 0.5 * ( w*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,w) + calc_xderiv(param,mesh,Frhow) )
      Krhoe  = Krhoe + 0.5 * ( h*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,h) + calc_xderiv(param,mesh,Frhoe) )

      # -- y --
      Krho   = Krho  + calc_yderiv(param,mesh,Grho)
      Krhou  = Krhou + 0.5 * ( u*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,u) + calc_yderiv(param,mesh,Grhou) )
      Krhov  = Krhov + 0.5 * ( v*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grhov) )
      Krhow  = Krhow + 0.5 * ( w*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,w) + calc_yderiv(param,mesh,Grhow) )
      Krhoe  = Krhoe + 0.5 * ( h*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,h) + calc_yderiv(param,mesh,Grhoe) )

      # -- z --
      Krho   = Krho  + calc_zderiv(param,mesh,Hrho)
      Krhou  = Krhou + 0.5 * ( u*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,u) + calc_zderiv(param,mesh,Hrhou) )
      Krhov  = Krhov + 0.5 * ( v*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,v) + calc_zderiv(param,mesh,Hrhov) )
      Krhow  = Krhow + 0.5 * ( w*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrhow) )
      Krhoe  = Krhoe + 0.5 * ( h*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,h) + calc_zderiv(param,mesh,Hrhoe) )
  
  

  elif (param.scheme.split_form == "KennedyGruber"):

      # Split-form : stable !
      rhoh = rhoe+p
      h = rhoh/rho
      
      # Mass equation
      Krho   = Krho  + 0.5 *  ( u*calc_xderiv(param,mesh,rho) + rho*calc_xderiv(param,mesh,u) + calc_xderiv(param,mesh,Frho ) )
      Krho   = Krho  + 0.5 *  ( v*calc_yderiv(param,mesh,rho) + rho*calc_yderiv(param,mesh,v) + calc_yderiv(param,mesh,Grho ) )
      Krho   = Krho  + 0.5 *  ( w*calc_zderiv(param,mesh,rho) + rho*calc_zderiv(param,mesh,w) + calc_zderiv(param,mesh,Hrho ) )
      
      # Divergence form
      # -- x --
      Krhou  = Krhou + 0.25 * calc_xderiv(param,mesh,Frhou)
      Krhov  = Krhov + 0.25 * calc_xderiv(param,mesh,Frhov)
      Krhow  = Krhow + 0.25 * calc_xderiv(param,mesh,Frhow)
      Krhoe  = Krhoe + 0.25 * calc_xderiv(param,mesh,Frhoe)

      # -- y --
      Krhou  = Krhou + 0.25 * calc_yderiv(param,mesh,Grhou)
      Krhov  = Krhov + 0.25 * calc_yderiv(param,mesh,Grhov)
      Krhow  = Krhow + 0.25 * calc_yderiv(param,mesh,Grhow)
      Krhoe  = Krhoe + 0.25 * calc_yderiv(param,mesh,Grhoe)

      # -- z --
      Krhou  = Krhou + 0.25 * calc_zderiv(param,mesh,Hrhou)
      Krhov  = Krhov + 0.25 * calc_zderiv(param,mesh,Hrhov)
      Krhow  = Krhow + 0.25 * calc_zderiv(param,mesh,Hrhow)
      Krhoe  = Krhoe + 0.25 * calc_zderiv(param,mesh,Hrhoe)
      
      # C^u form
      # -- x --
      Krhou  = Krhou + 0.25 * ( u*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,u) )
      Krhov  = Krhov + 0.25 * ( u*calc_xderiv(param,mesh,rhov) + rhov*calc_xderiv(param,mesh,u) )
      Krhow  = Krhow + 0.25 * ( u*calc_xderiv(param,mesh,rhow) + rhow*calc_xderiv(param,mesh,u) )
      Krhoe  = Krhoe + 0.25 * ( u*calc_xderiv(param,mesh,rhoh) + rhoh*calc_xderiv(param,mesh,u) )

      # -- y --
      Krhou  = Krhou + 0.25 * ( v*calc_yderiv(param,mesh,rhou) + rhou*calc_yderiv(param,mesh,v) )
      Krhov  = Krhov + 0.25 * ( v*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,v) )
      Krhow  = Krhow + 0.25 * ( v*calc_yderiv(param,mesh,rhow) + rhow*calc_yderiv(param,mesh,v) )
      Krhoe  = Krhoe + 0.25 * ( v*calc_yderiv(param,mesh,rhoh) + rhoh*calc_yderiv(param,mesh,v) )

      # -- z --
      Krhou  = Krhou + 0.25 * ( w*calc_zderiv(param,mesh,rhou) + rhou*calc_zderiv(param,mesh,w) )
      Krhov  = Krhov + 0.25 * ( w*calc_zderiv(param,mesh,rhov) + rhov*calc_zderiv(param,mesh,w) )
      Krhow  = Krhow + 0.25 * ( w*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,w) )
      Krhoe  = Krhoe + 0.25 * ( w*calc_zderiv(param,mesh,rhoh) + rhoh*calc_zderiv(param,mesh,w) )
      
      # C^phi form
      # -- x --
      Krhou  = Krhou + 0.25 * ( u*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,u) )
      Krhov  = Krhov + 0.25 * ( v*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,v) )
      Krhow  = Krhow + 0.25 * ( w*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,w) )
      Krhoe  = Krhoe + 0.25 * ( h*calc_xderiv(param,mesh,rhou) + rhou*calc_xderiv(param,mesh,h) )

      # -- y --
      Krhou  = Krhou + 0.25 * ( u*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,u) )
      Krhov  = Krhov + 0.25 * ( v*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,v) )
      Krhow  = Krhow + 0.25 * ( w*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,w) )
      Krhoe  = Krhoe + 0.25 * ( h*calc_yderiv(param,mesh,rhov) + rhov*calc_yderiv(param,mesh,h) )

      # -- z --
      Krhou  = Krhou + 0.25 * ( u*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,u) )
      Krhov  = Krhov + 0.25 * ( v*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,v) )
      Krhow  = Krhow + 0.25 * ( w*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,w) )
      Krhoe  = Krhoe + 0.25 * ( h*calc_zderiv(param,mesh,rhow) + rhow*calc_zderiv(param,mesh,h) )
      
      # C^rho form
      # -- x --
      Krhou  = Krhou + 0.25 * ( rho*calc_xderiv(param,mesh,u*u) + u*u*calc_xderiv(param,mesh,rho) )
      Krhov  = Krhov + 0.25 * ( rho*calc_xderiv(param,mesh,v*u) + v*u*calc_xderiv(param,mesh,rho) )
      Krhow  = Krhow + 0.25 * ( rho*calc_xderiv(param,mesh,w*u) + w*u*calc_xderiv(param,mesh,rho) )
      Krhoe  = Krhoe + 0.25 * ( rho*calc_xderiv(param,mesh,h*u) + h*u*calc_xderiv(param,mesh,rho) )

      # -- y --
      Krhou  = Krhou + 0.25 * ( rho*calc_yderiv(param,mesh,u*v) + u*v*calc_yderiv(param,mesh,rho) )
      Krhov  = Krhov + 0.25 * ( rho*calc_yderiv(param,mesh,v*v) + v*v*calc_yderiv(param,mesh,rho) )
      Krhow  = Krhow + 0.25 * ( rho*calc_yderiv(param,mesh,w*v) + w*v*calc_yderiv(param,mesh,rho) )
      Krhoe  = Krhoe + 0.25 * ( rho*calc_yderiv(param,mesh,h*v) + h*v*calc_yderiv(param,mesh,rho) )

      # -- z --
      Krhou  = Krhou + 0.25 * ( rho*calc_zderiv(param,mesh,u*w) + u*w*calc_zderiv(param,mesh,rho) )
      Krhov  = Krhov + 0.25 * ( rho*calc_zderiv(param,mesh,v*w) + v*w*calc_zderiv(param,mesh,rho) )
      Krhow  = Krhow + 0.25 * ( rho*calc_zderiv(param,mesh,w*w) + w*w*calc_zderiv(param,mesh,rho) )
      Krhoe  = Krhoe + 0.25 * ( rho*calc_zderiv(param,mesh,h*w) + h*w*calc_zderiv(param,mesh,rho) )
      
      
  else:
      print("ERROR in routine: control_scheme"); exit();
      
  K = [Krho,Krhou,Krhov,Krhow,Krhoe]

  return K
  
  
def calc_xderiv(param,mesh,U):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  ns = param.scheme.nscheme
  stencil = 5 #param.scheme.stencil
  dx = mesh.dx
  a = A[ns,:]
  
  Ux = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  
  Ux[xi:xf, yi:yf, zi:zf] = (a[1] * (U[xi+1:xf+1, yi:yf, zi:zf] - U[xi-1:xf-1, yi:yf, zi:zf]) + \
                             a[2] * (U[xi+2:xf+2, yi:yf, zi:zf] - U[xi-2:xf-2, yi:yf, zi:zf]) + \
                             a[3] * (U[xi+3:xf+3, yi:yf, zi:zf] - U[xi-3:xf-3, yi:yf, zi:zf]) + \
                             a[4] * (U[xi+4:xf+4, yi:yf, zi:zf] - U[xi-4:xf-4, yi:yf, zi:zf]) + \
                             a[5] * (U[xi+5:xf+5, yi:yf, zi:zf] - U[xi-5:xf-5, yi:yf, zi:zf]) ) * dx[xi:xf, yi:yf, zi:zf]

  return Ux

def calc_yderiv(param,mesh,U):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  ns = param.scheme.nscheme
  stencil = 5 #param.scheme.stencil
  dy = mesh.dy
  a = A[ns,:]
  
  Uy = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  
  Uy[xi:xf, yi:yf, zi:zf] = (a[1] * (U[xi:xf, yi+1:yf+1, zi:zf] - U[xi:xf, yi-1:yf-1, zi:zf]) + \
                             a[2] * (U[xi:xf, yi+2:yf+2, zi:zf] - U[xi:xf, yi-2:yf-2, zi:zf]) + \
                             a[3] * (U[xi:xf, yi+3:yf+3, zi:zf] - U[xi:xf, yi-3:yf-3, zi:zf]) + \
                             a[4] * (U[xi:xf, yi+4:yf+4, zi:zf] - U[xi:xf, yi-4:yf-4, zi:zf]) + \
                             a[5] * (U[xi:xf, yi+5:yf+5, zi:zf] - U[xi:xf, yi-5:yf-5, zi:zf]) ) * dy[xi:xf, yi:yf, zi:zf]
  
  return Uy

def calc_zderiv(param,mesh,U):

  nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz
  xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
  xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
  ns = param.scheme.nscheme
  stencil = 5 #param.scheme.stencil
  dz = mesh.dz
  a = A[ns,:]
  
  Uz = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
  
  Uz[xi:xf, yi:yf, zi:zf] = (a[1] * (U[xi:xf, yi:yf, zi+1:zf+1] - U[xi:xf, yi:yf, zi-1:zf-1]) + \
                             a[2] * (U[xi:xf, yi:yf, zi+2:zf+2] - U[xi:xf, yi:yf, zi-2:zf-2]) + \
                             a[3] * (U[xi:xf, yi:yf, zi+3:zf+3] - U[xi:xf, yi:yf, zi-3:zf-3]) + \
                             a[4] * (U[xi:xf, yi:yf, zi+4:zf+4] - U[xi:xf, yi:yf, zi-4:zf-4]) + \
                             a[5] * (U[xi:xf, yi:yf, zi+5:zf+5] - U[xi:xf, yi:yf, zi-5:zf-5]) ) * dz[xi:xf, yi:yf, zi:zf]
  
  return Uz


