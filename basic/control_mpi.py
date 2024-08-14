#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
from mpi4py import MPI

import control_param
from control_param import *

import control_scheme
from control_scheme import *

routine = "control_mpi"
  
class Mpi:
  """ MPI routines """

  def __init__(self,param):
  
    # Create a global communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank();
    size = comm.Get_size();
    if rank == 0: print("COMM_WORLD has size = ",size)
    
    self.comm = comm
    self.rank = comm.Get_rank();
    self.size = comm.Get_size();

  def initialize_comm3d(self,param):
  
    casename = param.casename
    ndomains = param.mpi.ndomains
    comm = self.comm
    size = self.size
    rank = self.rank

    # Create a new communicator with cartesian topology: comm3d
    [ndomx, ndomy, ndomz] = ndomains
    dims = [ndomx,ndomy,ndomz]
    
    # Check if domain can be partitioned
    if ndomx * ndomy * ndomz != size:
      message = "Domain partition cannot be done because ndomx * ndomy * ndomz != size"
      print_error_and_quit(routine, message, rank)

    # Specify the periodicity
    if ( casename == 'tgv' ):
      periods = [True,True,True]
    elif ( casename == 'bl' or casename == 'sbli' ) :
      periods = [False,False,True]

    comm3d = comm.Create_cart(dims,periods,reorder=True)
    coord3d = comm3d.Get_coords(rank)
    print ("In 3D topology, Processor ",rank, " has coordinates ",coord3d)

    # Look for neighbour to the west and east/ north and south/ front and back
    W = -1; E = -1; S = -1; N = -1; B = -1; F = -1;
    W,E = comm3d.Shift(direction = 0,disp=1)
    S,N = comm3d.Shift(direction = 1,disp=1)
    B,F = comm3d.Shift(direction = 2,disp=1)
    print("Processor ",rank, "has his neighbour", W, " and ",E)

    # Look for neighbour NW, NE, SW, SE % constant z
    NE = -1; NW = -1; SE = -1; SW = -1;
    if (N>-1 and E>-1):
      coords = [coord3d[0]+1,coord3d[1]+1,coord3d[2]]
      NE = comm3d.Get_cart_rank(coords)
    if (N>-1 and W>-1):
      coords = [coord3d[0]-1,coord3d[1]+1,coord3d[2]]
      NW = comm3d.Get_cart_rank(coords)
    if (S>-1 and E>-1):
      coords = [coord3d[0]+1,coord3d[1]-1,coord3d[2]]
      SE = comm3d.Get_cart_rank(coords)
    if (S>-1 and W>-1):
      coords = [coord3d[0]-1,coord3d[1]-1,coord3d[2]]
      SW = comm3d.Get_cart_rank(coords)

    # Look for neighbour NF, NB, SF, SB % constant x
    NF = -1; NB = -1; SF = -1; SB = -1;
    if (N>-1 and B>-1):
      coords = [coord3d[0],coord3d[1]+1,coord3d[2]-1]
      NB = comm3d.Get_cart_rank(coords)
    if (N>-1 and F>-1):
      coords = [coord3d[0],coord3d[1]+1,coord3d[2]+1]
      NF = comm3d.Get_cart_rank(coords)
    if (S>-1 and B>-1):
      coords = [coord3d[0],coord3d[1]-1,coord3d[2]-1]
      SB = comm3d.Get_cart_rank(coords)
    if (S>-1 and F>-1):
      coords = [coord3d[0],coord3d[1]-1,coord3d[2]+1]
      SF = comm3d.Get_cart_rank(coords)

    # Look for neighbour BW, BE, FW, FE % constant y
    BE = -1; BW = -1; FE = -1; FW = -1;
    if (E>-1 and B>-1):
      coords = [coord3d[0]+1,coord3d[1],coord3d[2]-1]
      BE = comm3d.Get_cart_rank(coords)
    if (W>-1 and B>-1):
      coords = [coord3d[0]-1,coord3d[1],coord3d[2]-1]
      BW = comm3d.Get_cart_rank(coords)
    if (F>-1 and E>-1):
      coords = [coord3d[0]+1,coord3d[1],coord3d[2]+1]
      FE = comm3d.Get_cart_rank(coords)
    if (F>-1 and W>-1):
      coords = [coord3d[0]-1,coord3d[1],coord3d[2]+1]
      FW = comm3d.Get_cart_rank(coords)
    
    # Specify the boundary type for each processor
    bndType = np.ones(6) * PERIODIC

    if ( casename == 'tgv' ):
      bndType = np.ones(6) * PERIODIC
    elif ( casename == 'bl' or casename == 'sbli' ) :
      bndType = np.ones(6) * PERIODIC
      if ( coord3d[0] == 0 ): bndType[XNEG] = INLET
      if ( coord3d[0] == ndomx-1 ): bndType[XPOS] = OUTLET
      if ( coord3d[1] == 0 ): bndType[YNEG] = WALL
      if ( coord3d[1] == ndomy-1 ): bndType[YPOS] = OUTLET

    # Global variables
    self.comm3d=comm3d; self.coord3d=coord3d;
    self.W=W; self.E=E; self.S=S; self.N=N; self.B=B; self.F=F
    self.NE=NE; self.NW=NW; self.SE=SE; self.SW=SW;
    self.NF=NF; self.NB=NB; self.SF=SF; self.SB=SB;
    self.BE=BE; self.BW=BW; self.FE=FE; self.FW=FW;
    self.bndType = bndType

    message = "Routine "+routine+" done"
    print_message(message,rank)


  def create_subarray_datatype(self,param):

    nxg = param.mesh.nxg; nyg = param.mesh.nyg; nzg = param.mesh.nzg;
    stencil = param.scheme.stencil
    coord3d = self.coord3d
    [ndomx, ndomy, ndomz] = param.mpi.ndomains
    
    nx = int(nxg/ndomx);
    ny = int(nyg/ndomy);
    nz = int(nzg/ndomz);
    
    xi = stencil; xf = xi + nx;
    yi = stencil; yf = yi + ny;
    zi = stencil; zf = zi + nz;
    
    # Compute mesh coordinates in each proc
    coordx = coord3d[0]*nx
    coordy = coord3d[1]*ny
    coordz = coord3d[2]*nz

    # Slice in the x direction
    starts_list = [[xf-stencil, yi, zi], [0, yi, zi], [xi, yi, zi], [xf, yi, zi]]
    slicex = []
    for starts in starts_list:
      subarray_datatype = MPI.DOUBLE.Create_subarray(
      sizes=[nx+2*stencil, ny+2*stencil, nz+2*stencil],
      subsizes=[stencil, ny, nz],
      starts=starts,
      order=MPI.ORDER_C)
      subarray_datatype.Commit()
      slicex.append(subarray_datatype)
      
    # Slice in the y direction
    starts_list = [[xi, yf-stencil, zi], [xi, 0, zi], [xi, yi, zi], [xi, yf, zi]]
    slicey = []
    for starts in starts_list:
      subarray_datatype = MPI.DOUBLE.Create_subarray(
      sizes=[nx+2*stencil, ny+2*stencil, nz+2*stencil],
      subsizes=[nx, stencil, nz],
      starts=starts,
      order=MPI.ORDER_C)
      subarray_datatype.Commit()
      slicey.append(subarray_datatype)
      
    # Slice in the z direction
    starts_list = [[xi, yi, zf-stencil], [xi, yi, 0], [xi, yi, zi], [xi, yi, zf]]
    slicez = []
    for starts in starts_list:
      subarray_datatype = MPI.DOUBLE.Create_subarray(
      sizes=[nx+2*stencil, ny+2*stencil, nz+2*stencil],
      subsizes=[nx, ny, stencil],
      starts=starts,
      order=MPI.ORDER_C)
      subarray_datatype.Commit()
      slicez.append(subarray_datatype)
      
    # For the diffusive fluxes
    st2 = 2
    starts_list = [[xf-st2, yf-st2, 0], [xi-st2, yi-st2, 0], [xi, yi, 0], [xf, yf, 0]]
    linezNESW = []
    for starts in starts_list:
      subarray_datatype = MPI.DOUBLE.Create_subarray(
      sizes=[nx+2*stencil, ny+2*stencil, nz+2*stencil],
      subsizes=[st2, st2, nz+2*stencil],
      starts=starts,
      order=MPI.ORDER_C)
      subarray_datatype.Commit()
      linezNESW.append(subarray_datatype)
    
    starts_list = [[xi, yf-st2, 0], [xf, yi-st2, 0], [xf-st2, yi, 0], [xi-st2, yf, 0]]
    linezNWSE = []
    for starts in starts_list:
      subarray_datatype = MPI.DOUBLE.Create_subarray(
      sizes=[nx+2*stencil, ny+2*stencil, nz+2*stencil],
      subsizes=[st2, st2, nz+2*stencil],
      starts=starts,
      order=MPI.ORDER_C)
      subarray_datatype.Commit()
      linezNWSE.append(subarray_datatype)
      
    starts_list = [[0, yf-st2, zf-st2], [0, yi-st2, zi-st2], [0, yi, zi], [0, yf, zf]]
    linexNFSB = []
    for starts in starts_list:
      subarray_datatype = MPI.DOUBLE.Create_subarray(
      sizes=[nx+2*stencil, ny+2*stencil, nz+2*stencil],
      subsizes=[nx+2*stencil, st2, st2],
      starts=starts,
      order=MPI.ORDER_C)
      subarray_datatype.Commit()
      linexNFSB.append(subarray_datatype)
      
    starts_list = [[0, yf-st2, zi], [0, yi-st2, zf], [0, yi, zf-st2], [0, yf, zi-st2]]
    linexNBSF = []
    for starts in starts_list:
      subarray_datatype = MPI.DOUBLE.Create_subarray(
      sizes=[nx+2*stencil, ny+2*stencil, nz+2*stencil],
      subsizes=[nx+2*stencil, st2, st2],
      starts=starts,
      order=MPI.ORDER_C)
      subarray_datatype.Commit()
      linexNBSF.append(subarray_datatype)
      
    starts_list = [[xf-st2, 0, zf-st2], [xi-st2, 0, zi-st2], [xi, 0, zi], [xf, 0, zf]]
    lineyFEBW = []
    for starts in starts_list:
      subarray_datatype = MPI.DOUBLE.Create_subarray(
      sizes=[nx+2*stencil, ny+2*stencil, nz+2*stencil],
      subsizes=[st2, ny+2*stencil, st2],
      starts=starts,
      order=MPI.ORDER_C)
      subarray_datatype.Commit()
      lineyFEBW.append(subarray_datatype)
      
    starts_list = [[xi, 0, zf-st2], [xf, 0, zi-st2], [xf-st2, 0, zi], [xi-st2, 0, zf]]
    lineyFWBE = []
    for starts in starts_list:
      subarray_datatype = MPI.DOUBLE.Create_subarray(
      sizes=[nx+2*stencil, ny+2*stencil, nz+2*stencil],
      subsizes=[st2, ny+2*stencil, st2],
      starts=starts,
      order=MPI.ORDER_C)
      subarray_datatype.Commit()
      lineyFWBE.append(subarray_datatype)

    # Type to read/write in parallel
    filetype = MPI.DOUBLE.Create_subarray(
      sizes=[nxg, nyg, nzg],
      subsizes=[nx, ny, nz],
      starts=[coordx,coordy,coordz],
      order=MPI.ORDER_C)
    filetype.Commit()

    self.slicex = slicex
    self.slicey = slicey
    self.slicez = slicez
    self.linezNESW = linezNESW
    self.linezNWSE = linezNWSE
    self.linexNFSB = linexNFSB
    self.linexNBSF = linexNBSF
    self.lineyFEBW = lineyFEBW
    self.lineyFWBE = lineyFWBE
    self.filetype = filetype
    
  def comm_slices(self,v):
  
    comm3d = self.comm3d
    slicex = self.slicex
    slicey = self.slicey
    slicez = self.slicez
    W = self.W ; E = self.E ; S = self.S ; N = self.N ; B = self.B ; F = self.F ;
    NE = self.NE ; NW = self.NW ; SE = self.SE ; SW = self.SW ;
    NF = self.NF ; NB = self.NB ; SF = self.SF ; SB = self.SB ;
    BE = self.BE ; BW = self.BW ; FE = self.FE ; FW = self.FW ;
    
    comm3d.Sendrecv(sendbuf = [v, slicex[0]], dest = E, sendtag = 1,  recvbuf = [v, slicex[1]], source = W, recvtag = 1)
    comm3d.Sendrecv(sendbuf = [v, slicex[2]], dest = W, sendtag = 1,  recvbuf = [v, slicex[3]], source = E, recvtag = 1)

    comm3d.Sendrecv(sendbuf = [v, slicey[0]], dest = N, sendtag = 2,  recvbuf = [v, slicey[1]], source = S, recvtag = 2)
    comm3d.Sendrecv(sendbuf = [v, slicey[2]], dest = S, sendtag = 2,  recvbuf = [v, slicey[3]], source = N, recvtag = 2)

    comm3d.Sendrecv(sendbuf = [v, slicez[0]], dest = F, sendtag = 3,  recvbuf = [v, slicez[1]], source = B, recvtag = 3)
    comm3d.Sendrecv(sendbuf = [v, slicez[2]], dest = B, sendtag = 3,  recvbuf = [v, slicez[3]], source = F, recvtag = 3)
    
    comm3d.Barrier()
  
  def comm_lines(self,v):
  
    comm3d = self.comm3d
    slicex = self.slicex
    slicey = self.slicey
    slicez = self.slicez
    linezNESW = self.linezNESW
    linezNWSE = self.linezNWSE
    linexNFSB = self.linexNFSB
    linexNBSF = self.linexNBSF
    lineyFEBW = self.lineyFEBW
    lineyFWBE = self.lineyFWBE
    W = self.W ; E = self.E ; S = self.S ; N = self.N ; B = self.B ; F = self.F ;
    NE = self.NE ; NW = self.NW ; SE = self.SE ; SW = self.SW ;
    NF = self.NF ; NB = self.NB ; SF = self.SF ; SB = self.SB ;
    BE = self.BE ; BW = self.BW ; FE = self.FE ; FW = self.FW ;
    
    comm3d.Sendrecv(sendbuf = [v, linezNESW[0]], dest = NE, sendtag = 4,  recvbuf = [v, linezNESW[1]], source = SW, recvtag = 4)
    comm3d.Sendrecv(sendbuf = [v, linezNESW[2]], dest = SW, sendtag = 4,  recvbuf = [v, linezNESW[3]], source = NE, recvtag = 4)

    comm3d.Sendrecv(sendbuf = [v, linezNWSE[0]], dest = NW, sendtag = 5,  recvbuf = [v, linezNWSE[1]], source = SE, recvtag = 5)
    comm3d.Sendrecv(sendbuf = [v, linezNWSE[2]], dest = SE, sendtag = 5,  recvbuf = [v, linezNWSE[3]], source = NW, recvtag = 5)

    comm3d.Sendrecv(sendbuf = [v, linexNFSB[0]], dest = NF, sendtag = 6,  recvbuf = [v, linexNFSB[1]], source = SB, recvtag = 6)
    comm3d.Sendrecv(sendbuf = [v, linexNFSB[2]], dest = SB, sendtag = 6,  recvbuf = [v, linexNFSB[3]], source = NF, recvtag = 6)

    comm3d.Sendrecv(sendbuf = [v, linexNBSF[0]], dest = NB, sendtag = 7,  recvbuf = [v, linexNBSF[1]], source = SF, recvtag = 7)
    comm3d.Sendrecv(sendbuf = [v, linexNBSF[2]], dest = SF, sendtag = 7,  recvbuf = [v, linexNBSF[3]], source = NB, recvtag = 7)
    
    comm3d.Sendrecv(sendbuf = [v, lineyFEBW[0]], dest = FE, sendtag = 8,  recvbuf = [v, lineyFEBW[1]], source = BW, recvtag = 8)
    comm3d.Sendrecv(sendbuf = [v, lineyFEBW[2]], dest = BW, sendtag = 8,  recvbuf = [v, lineyFEBW[3]], source = FE, recvtag = 8)
    
    comm3d.Sendrecv(sendbuf = [v, lineyFWBE[0]], dest = FW, sendtag = 9,  recvbuf = [v, lineyFWBE[1]], source = BE, recvtag = 9)
    comm3d.Sendrecv(sendbuf = [v, lineyFWBE[2]], dest = BE, sendtag = 9,  recvbuf = [v, lineyFWBE[3]], source = FW, recvtag = 9)
    
    comm3d.Barrier()
  
  # Write start solution
  # In this routine I can pass a variable of type rho
  def write_start_parallel_1(self,mesh,v):
  
    comm3d = self.comm3d
    filetype = self.filetype
    rank = self.rank
    nxg = mesh.nxg; nyg = mesh.nyg; nzg = mesh.nzg;
    nx = mesh.nx; ny = mesh.ny; nz = mesh.nz;
    xi = mesh.xi; yi = mesh.yi; zi = mesh.zi;
    xf = mesh.xf; yf = mesh.yf; zf = mesh.zf;
    
    disp4 = MPI.INT.Get_size()
    disp8 = MPI.DOUBLE.Get_size()
    
    # Create the data array
    tab = np.empty((nx, ny, nz), dtype=np.double)
    
    iter = 1  # Iteration number for the output file name
    outputfile = "sol_inst_" + str(iter).zfill(6) + ".bin"
    fh = MPI.File.Open(comm3d, outputfile, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    disp = 0
    
    # Only rank 0 writes the header
    if rank == 0:
      nn = np.array([1], dtype=np.int32)  # Value of nn to write
      t = np.array([2.], dtype=np.float64)  # Value of t to write
      neqs = np.array([1], dtype=np.int32)  # Value of nspec to write

      fh.Write(nn)
      fh.Write(t)
      fh.Write(neqs)
  
    disp = disp4 + disp8 + disp4
    
    for i in range(1):
      tab[0:nx, 0:ny, 0:nz] = v[xi:xf,yi:yf,zi:zf]
      fh.Set_view(disp, etype=MPI.DOUBLE, filetype=filetype, info=MPI.INFO_NULL)
      fh.Write_all(tab, status=MPI.Status())
      disp=disp+nxg*nyg*nzg*disp8
    
    # All processors close the solution file
    fh.Close()
    
    if rank == 0:
      if os.path.exists('sol_inst_last.bin'):
          os.remove('sol_inst_last.bin')
      os.symlink(outputfile, 'sol_inst_last.bin')
  
  # Write start solution
  # In this routine I can pass a vector of type t = [rho,rhou,rhov,rhow,rhoe]
  def write_start_parallel_2(self,param,mesh,t,nn,v):
  
    neq = len(v) #param.neq
    comm3d = self.comm3d
    filetype = self.filetype
    rank = self.rank
    nxg = mesh.nxg; nyg = mesh.nyg; nzg = mesh.nzg;
    nx = mesh.nx; ny = mesh.ny; nz = mesh.nz;
    xi = mesh.xi; yi = mesh.yi; zi = mesh.zi;
    xf = mesh.xf; yf = mesh.yf; zf = mesh.zf;
    
    disp4 = MPI.INT.Get_size()
    disp8 = MPI.DOUBLE.Get_size()
    
    # Create the data array
    tab = np.empty((nx, ny, nz), dtype=np.double)
    
    iter = nn  # Iteration number for the output file name
    outputfile = "sol_inst_" + str(iter).zfill(6) + ".bin"
    fh = MPI.File.Open(comm3d, outputfile, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    disp = 0
    
    # Only rank 0 writes the header
    if rank == 0:
      iter = np.array([nn], dtype=np.int32)   # Value of nn to write
      time = np.array([t], dtype=np.float64)  # Value of t to write
      neqs = np.array([neq], dtype=np.int32)  # Value of nspec to write
      
      fh.Write(iter)
      fh.Write(time)
      fh.Write(neqs)
  
    disp = disp4 + disp8 + disp4
    
    for i in range(0,neq):
      tab[0:nx,0:ny,0:nz] = v[i][xi:xf,yi:yf,zi:zf]
      fh.Set_view(disp, etype=MPI.DOUBLE, filetype=filetype, info=MPI.INFO_NULL)
      fh.Write_all(tab, status=MPI.Status())
      disp=disp+nxg*nyg*nzg*disp8
  
    # All processors close the solution file
    fh.Close()
    
    if rank == 0:
      if os.path.exists('sol_inst_last.bin'):
          os.remove('sol_inst_last.bin')
      os.symlink(outputfile, 'sol_inst_last.bin')
    
  # Read start solution
  # In this routine I can pass a vector of type t = [rho,rhou,rhov,rhow,rhoe]
  def read_start_parallel(self,param,mesh):
  
    neq = param.neq
    initial_solution = param.time.initial_solution
    comm3d = self.comm3d
    filetype = self.filetype
    rank = self.rank
    nxg = mesh.nxg; nyg = mesh.nyg; nzg = mesh.nzg;
    nx = mesh.nx; ny = mesh.ny; nz = mesh.nz;
    xi = mesh.xi; yi = mesh.yi; zi = mesh.zi;
    xf = mesh.xf; yf = mesh.yf; zf = mesh.zf;
    stencil = param.scheme.stencil
    
    rho  = np.ones((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    rhou = np.ones((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    rhov = np.ones((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    rhow = np.ones((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    rhoe = np.ones((nx+2*stencil,ny+2*stencil,nz+2*stencil))
    
    disp4 = MPI.INT.Get_size()
    disp8 = MPI.DOUBLE.Get_size()
    
    # Create the data array
    tab = np.empty((nx, ny, nz), dtype=np.double)
    Q = []
    
    # Check if the file exists
    exist = os.path.isfile(initial_solution)

    if exist:
      # All processors open the solution file
      fh = MPI.File.Open(comm3d, initial_solution, MPI.MODE_RDONLY)

      # All processors read nn, t, and nspec to avoid broadcasting the data
      iter = np.zeros(1, dtype=np.int32)
      time = np.zeros(1, dtype=np.float64)
      neqs = np.zeros(1, dtype=np.int32)
      disp = 0
      fh.Read_at(disp, iter)
      disp += 4
      fh.Read_at(disp, time)
      disp += 8
      fh.Read_at(disp, neqs)
      disp += 4
      
      n = neqs.reshape(()); time = time.reshape(()); iter = iter.reshape(());
       
      for i in range(n):
        fh.Set_view(disp, MPI.DOUBLE, filetype=filetype, info=MPI.INFO_NULL)
        fh.Read_all(tab, status=MPI.Status())
        disp=disp+nxg*nyg*nzg*disp8
        Q.append(tab.copy())
        #print(Q)
        
      rho[xi:xf,yi:yf,zi:zf]  = Q[0][:,:,:].copy();
      rhou[xi:xf,yi:yf,zi:zf] = Q[1][:,:,:].copy();
      rhov[xi:xf,yi:yf,zi:zf] = Q[2][:,:,:].copy();
      rhow[xi:xf,yi:yf,zi:zf] = Q[3][:,:,:].copy();
      rhoe[xi:xf,yi:yf,zi:zf] = Q[4][:,:,:].copy();
      
      # Close the file
      fh.Close()
    else:
      message = "Initial file not encountered. STOP!"
      print_error_and_quit(routine, message, rank)
      
    return rho,rhou,rhov,rhow,rhoe,iter,time
    
    
  def write_temporal(self,param,mesh,n,t,Q,file):
  
    routine = "write_temporal"

    comm = self.comm
    rank = self.rank
    nxg = mesh.nxg; nyg = mesh.nyg; nzg = mesh.nzg;
    nx = mesh.nx; ny = mesh.ny; nz = mesh.nz;
    xi = mesh.xi; yi = mesh.yi; zi = mesh.zi;
    xf = mesh.xf; yf = mesh.yf; zf = mesh.zf;
    gamma = param.flow.gamma
    iRe = 1./param.transport.Reynolds_number

    r = Q[0]
    u = Q[1]/Q[0]
    v = Q[2]/Q[0]
    w = Q[3]/Q[0]
    rhoe = Q[4]
    p = (gamma-1.)*(rhoe-0.5*r*(u**2+v**2+w**2))
    
    rhouu = Q[1]**2/Q[0]
    rhovv = Q[2]**2/Q[0]
    rhoww = Q[3]**2/Q[0]
    
    # Compute resolved kinetic energy
    kc = 0.5*(rhouu[xi:xf,yi:yf,zi:zf] + rhovv[xi:xf,yi:yf,zi:zf] + rhoww[xi:xf,yi:yf,zi:zf])
    
    kc_local = np.sum(kc)
    kc_global = comm.allreduce(kc_local, op=MPI.SUM)
    kk = kc_global/(nxg * nyg * nzg)
    
    # Compute derivatives
    ux = calc_xderiv(param,mesh,u)
    uy = calc_yderiv(param,mesh,u)
    uz = calc_zderiv(param,mesh,u)
    vx = calc_xderiv(param,mesh,v)
    vy = calc_yderiv(param,mesh,v)
    vz = calc_zderiv(param,mesh,v)
    wx = calc_xderiv(param,mesh,w)
    wy = calc_yderiv(param,mesh,w)
    wz = calc_zderiv(param,mesh,w)

    # Compute vorticity
    vorx = wy - vz
    vory = uz - wx
    vorz = vx - uy
    vor2 = vorx**2 + vory**2 + vorz**2
    
    # Compute enstrophy
    en = 0.5 * r[xi:xf,yi:yf,zi:zf] * vor2[xi:xf,yi:yf,zi:zf]
    
    en_local = np.sum(en)
    en_global = comm.allreduce(en_local, op=MPI.SUM)
    en = en_global/(nxg * nyg * nzg)
    
    # Compute the ensemble enstrophy dissipation rate
    e0 = 2.*iRe*en
    
    S00 = 1./3.*(ux+vy+wz)
    S11 = ux - S00
    S22 = vy - S00
    S33 = wz - S00
    S12 = 0.5 * ( vx + uy )
    S13 = 0.5 * ( wx + uz )
    S23 = 0.5 * ( wy + vz )
    
    E  = S11**2+S22**2+S33**2 + 2.*(S12**2+S13**2+S23**2)
    E1 = 2.*iRe*E[xi:xf,yi:yf,zi:zf]
    e1_local = np.sum(E1)
    e1_global = comm.allreduce(e1_local, op=MPI.SUM)
    e1 = e1_global/(nxg * nyg * nzg)

    E  = (ux+vy+wz)**2
    E2 = 0*-2./3.*iRe*E[xi:xf,yi:yf,zi:zf]
    e2_local = np.sum(E2)
    e2_global = comm.allreduce(e2_local, op=MPI.SUM)
    e2 = e2_global/(nxg * nyg * nzg)

    E  = p * (ux+vy+wz)
    E3 = -E[xi:xf,yi:yf,zi:zf]
    e3_local = np.sum(E3)
    e3_global = comm.allreduce(e3_local, op=MPI.SUM)
    e3 = e3_global/(nxg * nyg * nzg)
    
    E  = r[xi:xf,yi:yf,zi:zf]
    e_local = np.sum(E)
    e_global = comm.allreduce(e_local, op=MPI.SUM)
    v1 = e_global/(nxg * nyg * nzg)
    
    E  = rhoe[xi:xf,yi:yf,zi:zf]
    e_local = np.sum(E)
    e_global = comm.allreduce(e_local, op=MPI.SUM)
    v2 = e_global/(nxg * nyg * nzg)
    
    if (param.les.sgs_model == "Smagorinsky"):
      Cs = 0.18
      Ci = 0.0066
      visco = iRe
      S11 = ux
      S22 = vy
      S33 = wz
      S12 = 0.5 * ( vx + uy )
      S13 = 0.5 * ( wx + uz )
      S23 = 0.5 * ( wy + vz )
      deltac = mesh.deltac
      k_sgs = Ci * r *(deltac**2)*(2.*(S11**2+S22**2+S33**2+2.*(S12**2+S13**2+S23**2)))
      E  = k_sgs[xi:xf,yi:yf,zi:zf]
      e_local = np.sum(E)
      e_global = comm.allreduce(e_local, op=MPI.SUM)
      kk_sgs = e_global/(nxg * nyg * nzg)
    
    if (n%50 == 0):
      message='n = '+str(n)+', t = '+str(t)+', kk = '+str(kk)+', en = '+str(en)+', e0 = '+str(e0)+', e1 = '+str(e1)+', e2 = '+str(e2)+', e3 = '+str(e3)+', v1 = '+str(v1)+', v2 = '+str(v2)
      print_message(message,rank)
    file.write('{:>25}'.format( n ) )
    file.write('{:>25}'.format( t ) )
    file.write('{:>25}'.format( kk ) )
    file.write('{:>25}'.format( en ) )
    file.write('{:>25}'.format( e0 ) )
    file.write('{:>25}'.format( e1 ) )
    file.write('{:>25}'.format( e2 ) )
    file.write('{:>25}'.format( e3 ) )
    file.write('{:>25}'.format( v1 ) )
    file.write('{:>25}'.format( v2 ) )
    if (param.les.sgs_model == "Smagorinsky"): file.write('{:>25}'.format( kk_sgs ) )
    file.write( '\n' )
    
    
    
  def finalize(self):
  
    for i in range(0,4):
      self.slicex[i].Free()
      self.slicey[i].Free()
      self.slicez[i].Free()
      self.linezNESW[i].Free()
      self.linezNWSE[i].Free()
      self.linexNFSB[i].Free()
      self.linexNBSF[i].Free()
      self.lineyFEBW[i].Free()
      self.lineyFWBE[i].Free()
    self.filetype.Free()
    self.comm3d.Free()
    MPI.Finalize()

'''
Print Error and quit
'''
def print_error_and_quit(routine, message, rank):
  if rank == 0: print("ERROR in routine: ", routine)
  if rank == 0: print(message)
  MPI.Finalize();
  exit();
  
'''
Print message
'''
def print_message(message, rank):
  if rank == 0: print(message)


