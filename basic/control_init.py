#! /usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, sin, cos, acos, pi

def initialisation(param,mesh,mpi):

    Rref=param.flow.R_inf ; Pref=param.flow.P_inf ; Uref=param.flow.U_inf ; gamma = param.flow.gamma
    nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz ; x = mesh.x ; y = mesh.y ; z = mesh.z
    xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
    xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
    stencil = param.scheme.stencil
    
    u = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    v = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    p = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    
    for i in range (xi,xf):
        for j in range (yi,yf):
            for k in range (zi,zf):
                u[i,j,k] =  Uref*sin(x[i])*cos(y[j])*cos(z[k])
                v[i,j,k] = -Uref*cos(x[i])*sin(y[j])*cos(z[k])
                p[i,j,k] =  Pref+Rref*Uref**2/16.*(cos(2*z[k])+2)*(cos(2*x[i])+cos(2*y[j]))

    rho  = np.full((nx+2*stencil,ny+2*stencil,nz+2*stencil),Rref)
    rhou = rho * u
    rhov = rho * v
    rhow = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    rhoe = p/(gamma-1.)+0.5*(rhou**2+rhov**2+rhow**2)/rho
    
    # For test
    #if mpi.rank == 0:
    #    rhou = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    #elif mpi.rank == 1:
    #    rhou = np.ones((nx+2*stencil,ny+2*stencil,nz+2*stencil))
   
    return rho,rhou,rhov,rhow,rhoe
    
def initialisation0(param,mesh,mpi):

    Rref=param.flow.R_inf ; Pref=param.flow.P_inf ; Uref=param.flow.U_inf ; gamma = param.flow.gamma
    nx = mesh.nx ; ny = mesh.ny ; nz = mesh.nz ; x = mesh.x ; y = mesh.y ; z = mesh.z
    xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
    xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf
    stencil = param.scheme.stencil

    rho0  = np.full((nx+2*stencil,ny+2*stencil,nz+2*stencil),Rref)
    rhou0 = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    rhov0 = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    rhow0 = np.zeros((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    rhoe0 = np.full((nx+2*stencil,ny+2*stencil,nz+2*stencil),Pref/(gamma-1.))
   
    return rho0,rhou0,rhov0,rhow0,rhoe0

def compute_cons2prim(rho,rhou,rhov,rhow,rhoe,param):

    gamma = param.flow.gamma
    u = rhou/rho
    v = rhov/rho
    w = rhow/rho
    p = (gamma-1.)*(rhoe-0.5*rho*(u**2+v**2+w**2))
    
    return u,v,w,p
    
def plot_image_per_proc(rho,rhou,rhov,rhow,rhoe,mesh):

    x = mesh.x ; y = mesh.y ; z = mesh.z
    xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
    xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf

    # Subplot
    colormap = 'jet' #'RdBu'
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True);
    axes[0][0].set_title(r"$u$")
    im=axes[0][0].pcolor(x[xi:xf],y[yi:yf],rhou[xi:xf,yi:yf,zi].transpose(), shading='auto', cmap=colormap)#, vmin=-1,vmax=1)
    #im=axes[0][0].pcolor(rhou[:,:,zi].transpose(), shading='auto', cmap=colormap, vmin=-1,vmax=1)
    axes[0][0].set_ylabel(r"$y$");
    fig.colorbar(im, ax=axes[0][0])

    axes[0][1].set_title(r"$v$")
    im=axes[0][1].pcolor(x[xi:xf],y[yi:yf],rhov[xi:xf,yi:yf,zi].transpose(), shading='auto', cmap=colormap)#, vmin=-1,vmax=1)
    #im=axes[0][1].pcolor(rhov[:,:,zi].transpose(), shading='auto', cmap=colormap, vmin=-1,vmax=1)
    fig.colorbar(im, ax=axes[0][1])

    axes[1][0].set_title(r"$rhoe$")
    im=axes[1][0].pcolor(x[xi:xf],y[yi:yf],rhoe[xi:xf,yi:yf,zi].transpose(), shading='auto', cmap=colormap)
    #im=axes[1][0].pcolor(rhoe[:,:,zi].transpose(), shading='auto', cmap=colormap)
    axes[1][0].set_xlabel(r"$x$");
    axes[1][0].set_ylabel(r"$y$");
    fig.colorbar(im, ax=axes[1][0])

    axes[1][1].set_title(r"$rho$")
    im=axes[1][1].pcolor(x[xi:xf],y[yi:yf],rho[xi:xf,yi:yf,zi].transpose(), shading='auto', cmap=colormap)
    #    im=axes[1][1].pcolor(rho[:,:,zi].transpose(), shading='auto', cmap=colormap)
    axes[1][1].set_xlabel(r"$x$");
    fig.colorbar(im, ax=axes[1][1])
    
    plt.show()


def plot_image_per_proc_test(rhou,mesh):

    x = mesh.x ; y = mesh.y ; z = mesh.z
    xi = mesh.xi ; yi = mesh.yi ; zi = mesh.zi
    xf = mesh.xf ; yf = mesh.yf ; zf = mesh.zf

    # Subplot
    colormap = 'jet' #'RdBu'
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False);
    axes[0][0].set_title(r"$rhou$")
    #im=axes[0][0].pcolor(x[xi:xf],y[yi:yf],rhou[xi:xf,yi:yf,zi].transpose(), shading='auto', cmap=colormap, vmin=-1,vmax=1)
    im=axes[0][0].pcolor(rhou[:,:,zi].transpose(), shading='auto', cmap=colormap)#, vmin=-1,vmax=1)
    axes[0][0].set_xlabel(r"$x$");
    axes[0][0].set_ylabel(r"$y$");
    fig.colorbar(im, ax=axes[0][0])

    axes[0][1].set_title(r"$rhou$")
    #im=axes[0][1].pcolor(x[xi:xf],y[yi:yf],rhov[xi:xf,yi:yf,zi].transpose(), shading='auto', cmap=colormap, vmin=-1,vmax=1)
    im=axes[0][1].pcolor(rhou[:,:,zi+5].transpose(), shading='auto', cmap=colormap)#, vmin=-1,vmax=1)
    axes[0][1].set_xlabel(r"$x$");
    axes[0][1].set_ylabel(r"$y$");
    fig.colorbar(im, ax=axes[0][1])

    axes[1][0].set_title(r"$rhou$")
    #im=axes[1][0].pcolor(x[xi:xf],y[yi:yf],rhoe[xi:xf,yi:yf,zi].transpose(), shading='auto', cmap=colormap)
    im=axes[1][0].pcolor(rhou[xi+2,:,:].transpose(), shading='auto', cmap=colormap)
    axes[1][0].set_xlabel(r"$y$");
    axes[1][0].set_ylabel(r"$z$");
    fig.colorbar(im, ax=axes[1][0])

    axes[1][1].set_title(r"$rhou$")
    #im=axes[1][1].pcolor(x[xi:xf],y[yi:yf],rho[xi:xf,yi:yf,zi].transpose(), shading='auto', cmap=colormap)
    im=axes[1][1].pcolor(rhou[:,yi+2,:].transpose(), shading='auto', cmap=colormap)
    axes[1][1].set_xlabel(r"$x$");
    axes[1][1].set_ylabel(r"$z$");
    fig.colorbar(im, ax=axes[1][1])
    
    plt.show()
