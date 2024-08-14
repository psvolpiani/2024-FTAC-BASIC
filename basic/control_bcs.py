#! /usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np

def update_boundary_condition(mpi,Q):

  for q in Q:
    mpi.comm_slices(q)
    mpi.comm_lines(q)
    
  return Q
