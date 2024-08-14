#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import math
import struct
from math import sqrt, sin, cos, acos, pi


routine = "control_mesh"
  
class Mesh:
  """ Mesh routines """

  def __init__(self,param):
  
  
#  '''
#  Compute global mesh
#  '''
#  def compute_global_mesh(self,param):

    nxg=param.mesh.nxg; nyg=param.mesh.nyg; nzg=param.mesh.nzg
    xmin=param.mesh.xmin_xmax[0]; xmax=param.mesh.xmin_xmax[1]
    ymin=param.mesh.ymin_ymax[0]; ymax=param.mesh.ymin_ymax[1]
    zmin=param.mesh.zmin_zmax[0]; zmax=param.mesh.zmin_zmax[1]
    rx = param.mesh.rx_ry_rz[0]; ry = param.mesh.rx_ry_rz[1]; rz = param.mesh.rx_ry_rz[2];
    
    Lx = xmax-xmin
    Ly = ymax-ymin
    Lz = zmax-zmin
    
    deltax = Lx/nxg; deltay=Ly/nyg; deltaz=Lz/nzg;
    
    xg = np.arange(xmin, xmax, deltax)
    yg = np.arange(ymin, ymax, deltay)
    zg = np.arange(zmin, zmax, deltaz)

    if (ry > 1.):
      yg[0] = ymin
      for j in range(1, nyg):
        yg[j] = yg[j-1] + deltay
        deltay *= ry
      print(yg)
      
  #    if (self.sponge.compute_sponge == True):
  #      for i in range(nxg-29, nxg-19):
  #        xg[i] = xg[i-1] + deltax
  #        deltax *= 1.02
  #      for i in range(nxg-19, nxg):
  #        xg[i] = xg[i-1] + deltax
  #        deltax *= 1.06
  #    print(xg)

    # Compute dxg, dyg, dzg
    dxg=np.zeros(nxg); dyg=np.zeros(nyg); dzg=np.zeros(nzg);
    
    if( "bl" in param.casename ):
    
      nscheme = param.scheme.nscheme
      
      # Compute dxg
      i = 0
      for l in range(7): dxg[i] += a06[l] * xg[i+l]
      
      # i = 1, 2, 3, 4
      for i in range(1, nscheme):
        bpoints = i
        for l in range(1, bpoints + 1):
          dxg[i] += A[bpoints][l] * ( xg[i+l] - xg[i-l] )
      
      # i = nstencil ... nxg-nstencil-1 inclusive
      for i in range(nscheme, nxg-nscheme):
        for l in range(1, nscheme + 1):
          dxg[i] += A[nscheme][l] * ( xg[i+l] - xg[i-l] )
      
      # i = nxg-nscheme ... -2
      for i in range(nxg-nscheme, nxg-1):
        bpoints = nxg-1-i
        for l in range(1, bpoints + 1):
          dxg[i] += A[bpoints][l] * ( xg[i+l] - xg[i-l] )
          
      i = nxg-1
      for l in range(-6, 1):
        dxg[i] -= a06[-l] * xg[i+l]
        
      # Compute dyg
      i = 0
      for l in range(7):
        dyg[i] += a06[l] * yg[i+l]

      # i = 1, 2, 3, 4
      for i in range(1, nscheme):
        bpoints = i
        for l in range(1, bpoints + 1):
          dyg[i] += A[bpoints][l] * ( yg[i+l] - yg[i-l] )

      # i = nstencil ... nyg-nstencil-1 inclusive
      for i in range(nscheme, nyg-nscheme):
        for l in range(1, nscheme + 1):
          dyg[i] += A[nscheme][l] * ( yg[i+l] - yg[i-l] )
          
      # i = nyg-nscheme ... -2
      for i in range(nyg-nscheme, nyg-1):
        bpoints = nyg-1-i
        for l in range(1, bpoints + 1):
          dyg[i] += A[bpoints][l] * ( yg[i+l] - yg[i-l] )
          
      i = nyg-1
      for l in range(-6, 1):
        dyg[i] -= a06[-l] * yg[i+l]
        
      # Compute dzg
      dzg=np.ones(nzg)*Lz/nzg;
      
    else:
      dxg=np.ones(nxg)*Lx/nxg;
      dyg=np.ones(nyg)*Ly/nyg;
      dzg=np.ones(nzg)*Lz/nzg;
      
    dxg=1./dxg
    dyg=1./dyg
    dzg=1./dzg
    
    # Compute dxg4, dyg4, dz4 / FD 4 order
    dxg4=np.zeros(nxg); dyg4=np.zeros(nyg); dzg4=np.zeros(nzg);
    
    if( "bl" in param.casename ):
    
      nscheme = 2

      # Compute dxg4
      i = 0
      for l in range(5): dxg4[i] += a04[l] * xg[i+l]
      
      i = 1
      for l in range(-1,4): dxg4[i] += a13[l+1] * xg[i+l]
      
      # i = 2 ... nxg-3 inclusive
      for i in range(nscheme, nxg-nscheme):
        for l in range(1, nscheme + 1):
          dxg4[i] += A[nscheme][l] * ( xg[i+l] - xg[i-l] )
          
      i = nxg-nscheme
      for l in range(-3,2): dxg4[i] -= a13[-l+1] * xg[i+l]

      i = nxg-1
      for l in range(-4,1): dxg4[i] -= a04[-l] * xg[i+l]

      # Compute dyg4
      i = 0
      for l in range(5): dyg4[i] += a04[l] * yg[i+l]

      i = 1
      for l in range(-1,4): dyg4[i] += a13[l+1] * yg[i+l]
      
      # i = 2 ... nxg-3 inclusive
      for i in range(nscheme, nyg-nscheme):
        for l in range(1, nscheme + 1):
          dyg4[i] += A[nscheme][l] * ( yg[i+l] - yg[i-l] )
          
      i = nyg-nscheme
      for l in range(-3,2): dyg4[i] -= a13[-l+1] * yg[i+l]
      
      i = nyg-1
      for l in range(-4,1): dyg4[i] -= a04[-l] * yg[i+l]
              
      # Compute dzg
      dzg4=np.ones(nzg)*Lz/nzg;
      
    else:
      dxg4=np.ones(nxg)*Lx/nxg;
      dyg4=np.ones(nyg)*Ly/nyg;
      dzg4=np.ones(nzg)*Lz/nzg;
      
    dxg4=1./dxg4
    dyg4=1./dyg4
    dzg4=1./dzg4
    
    self.nxg=nxg; self.nyg=nyg; self.nzg=nzg;
    self.xg=xg; self.yg=yg; self.zg=zg;
    self.dxg=dxg; self.dyg=dyg; self.dzg=dzg;
    self.dxg4=dxg4; self.dyg4=dyg4; self.dzg4=dzg4;
    
    
  '''
  Compute local mesh
  '''
  def compute_local_mesh(self, param, mpi):
  
    nxg=param.mesh.nxg; nyg=param.mesh.nyg; nzg=param.mesh.nzg
    [ndomx, ndomy, ndomz] = param.mpi.ndomains
    coord3d = mpi.coord3d
    stencil = param.scheme.stencil
    xg=self.xg; yg=self.yg; zg=self.zg
    dxg=self.dxg; dyg=self.dyg; dzg=self.dzg
    dxg4=self.dxg4; dyg4=self.dyg4; dzg4=self.dzg4
    rank=mpi.rank
  
    nx = int(nxg/ndomx); ny = int(nyg/ndomy); nz = int(nzg/ndomz);

    # Compute mesh coordinates in each proc
    coordx = coord3d[0]*nx
    coordy = coord3d[1]*ny
    coordz = coord3d[2]*nz
    #print("Processor ",rank, "has coordx ", coordx)
    
    xi = stencil; xf = xi + nx;
    yi = stencil; yf = yi + ny;
    zi = stencil; zf = zi + nz;
    
    x = np.zeros(nx+2*stencil); y = np.zeros(ny+2*stencil); z = np.zeros(nz+2*stencil)
    dx = np.zeros(nx+2*stencil); dy = np.zeros(ny+2*stencil); dz = np.zeros(nz+2*stencil)
    dx4 = np.zeros(nx+2*stencil); dy4 = np.zeros(ny+2*stencil); dz4 = np.zeros(nz+2*stencil)
    
    x[xi:xf] = xg[coordx:coordx+nx]
    y[yi:yf] = yg[coordy:coordy+ny]
    z[zi:zf] = zg[coordz:coordz+nz]
    
    dx[xi:xf] = dxg[coordx:coordx+nx]
    dy[yi:yf] = dyg[coordy:coordy+ny]
    dz[zi:zf] = dzg[coordz:coordz+nz]
    
    # Special treatment for dx4, dy4, dz4
    dx4c = np.concatenate((dxg4[-3:-1], dxg4, dxg4[0:2]))
    dx4[xi-2:xf+2] = dx4c[coordx:coordx+nx+4]
    dy4c = np.concatenate((dyg4[-3:-1], dyg4, dyg4[0:2]))
    dy4[yi-2:yf+2] = dy4c[coordy:coordy+ny+4]
    dz4c = np.concatenate((dzg4[-3:-1], dzg4, dzg4[0:2]))
    dz4[zi-2:zf+2] = dz4c[coordz:coordz+nz+4]
    
    # I need dx, dy and dz in 3D to accelerate the computation
    dx3d = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    dy3d = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    dz3d = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    for i in range(xi, xf):
      for j in range(yi, yf):
          for k in range(zi, zf):
            dx3d[i,j,k] = dx[i]
            dy3d[i,j,k] = dy[j]
            dz3d[i,j,k] = dz[k]
            
    dx3d4 = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    dy3d4 = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    dz3d4 = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    for i in range(xi-2, xf+2):
      for j in range(yi-2, yf+2):
          for k in range(zi-2, zf+2):
            dx3d4[i,j,k] = dx4[i]
            dy3d4[i,j,k] = dy4[j]
            dz3d4[i,j,k] = dz4[k]
    
    self.nx=nx; self.ny=ny; self.nz=nz;
    self.x=x; self.y=y; self.z=z;
    self.dx=dx3d; self.dy=dy3d; self.dz=dz3d;
    self.dx4=dx3d4; self.dy4=dy3d4; self.dz4=dz3d4;
    self.xi=xi; self.yi=yi; self.zi=zi;
    self.xf=xf; self.yf=yf; self.zf=zf;
    
    if rank == 0:
      #print("Writing global mesh")
      
      # Write binary file
      with open('mesh.bin', 'wb') as file:
        file.write(struct.pack('i', nxg))
        file.write(struct.pack('i', nyg))
        file.write(struct.pack('i', nzg))
        file.write(struct.pack(f'{nxg}f', *xg))
        file.write(struct.pack(f'{nyg}f', *yg))
        file.write(struct.pack(f'{nzg}f', *zg))
        file.write(struct.pack(f'{nxg}f', *dxg))
        file.write(struct.pack(f'{nyg}f', *dyg))
        file.write(struct.pack(f'{nzg}f', *dzg))
        file.write(struct.pack(f'{nxg}f', *dxg4))
        file.write(struct.pack(f'{nyg}f', *dyg4))
        file.write(struct.pack(f'{nzg}f', *dzg4))
        
      print("Routine contol_mesh done")


  '''
  Compute deltac
  '''

  def compute_deltac(self,param,mpi):

    xg = self.xg ; yg = self.yg ; zg = self.zg
    nxg = self.nxg; nyg = self.nyg; nzg = self.nzg
    nx = self.nx; ny = self.ny; nz = self.nz
    xi = self.xi; yi = self.yi; zi = self.zi
    xf = self.xf; yf = self.yf; zf = self.zf
    stencil = param.scheme.stencil
    coord3d = mpi.coord3d
    
    # Compute global variables
    dx  = np.concatenate(([np.diff(xg)[0]],np.diff(xg)));           # np.diff returns array with size-1
    dxc = np.concatenate((dx[-(stencil+1):-1], dx, dx[0:stencil]))  # I guarantee periodicity
    dy  = np.concatenate(([np.diff(yg)[0]],np.diff(yg)));           # np.diff returns array with size-1
    dyc = np.concatenate((dy[-(stencil+1):-1], dy, dy[0:stencil]))  # I guarantee periodicity
    dz  = np.concatenate(([np.diff(zg)[0]],np.diff(zg)));           # np.diff returns array with size-1
    dzc = np.concatenate((dz[-(stencil+1):-1], dz, dz[0:stencil]))  # I guarantee periodicity
  #  print(np.diff(xg)[0])
  #  print(dxc.size)
    deltag = np.zeros((nxg+2*stencil,nyg+2*stencil,nzg+2*stencil))
    for i in range(0, nxg+2*stencil):
      for j in range(0, nyg+2*stencil):
        for k in range(0, nzg+2*stencil):
          deltag[i,j,k] = (dxc[i]*dyc[j]*dzc[k])**(1./3)
            
  #  print(deltag)

    # Compute mesh coordinates in each proc
    coordx = coord3d[0]*nx
    coordy = coord3d[1]*ny
    coordz = coord3d[2]*nz
      
    # Compute local variables
    deltac = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    deltac[xi-stencil:xf+stencil,yi-stencil:yf+stencil,zi-stencil:zf+stencil] = deltag[coordx:coordx+nx+2*stencil,coordy:coordy+ny+2*stencil,coordz:coordz+nz+2*stencil]
  #  for i in range(xi-stencil, xf+stencil):
  #    for j in range(yi-stencil, yf+stencil):
  #      for k in range(zi-stencil, zf+stencil):
  #        deltac[i,j,k] = deltag[coordx+i,coordy+j,coordz+k]
  #  print(deltac[:,0,0])

    self.deltac = deltac
    
    
