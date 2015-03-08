#
# File: geometry.py
# Author: Bo-Xiao Zheng <boxiao@princeton.edu>
#

import numpy as np
import numpy.linalg as la
import itertools as it
import re
import os

class FHam(object):
  def __init__(self):
    raise Exception("FHam::__init__ must be implemented in derived class")
  
  def get_h0(self):
    return self.H1e
  
  def get_Int2e(self):
    return self.Int2e

  def get_fock(self):
    raise Exception("FHam::get_fock must be implemented in derived class")

  def get_imp_corr(self):
    raise Exception("FHam::get_eff_imp must be implemented in derived class")

  def get_U(self):
    raise Exception("Not Hubbard model")

class FHamHubbard(FHam):
  def __init__(self, Params, inp_ctrl, lattice):
    
    # assign parameters
    self.t = Params["t"]
    self.t1 = Params["t1"]
    self.U = Params["U"]
    
    # build h0
    if lattice.bc in [-1, 0, 1]:
      nsc = lattice.nscells
      nscsites = lattice.supercell.nsites
      self.H1e = np.zeros((nsc, nscsites, nscsites))
      if abs(self.t) > 1e-7:
        pairs = lattice.get_NearNeighbor(sites = range(nscsites))
        for nn in pairs[0]:
          self.H1e[nn[1] / nscsites, nn[0], nn[1] % nscsites] = self.t
        for nn in pairs[1]:
          self.H1e[nn[1] / nscsites, nn[0], nn[1] % nscsites] = self.t * lattice.bc
      if abs(self.t1) > 1e-7:
        pairs = lattice.get_2ndNearNeighbor(sites = range(nscsites))
        for nn in pairs[0]:
          self.H1e[nn[1] / nscsites, nn[0], nn[1] % nscsites] = self.t1
        for nn in pairs[1]:
          self.H1e[nn[1] / nscsites, nn[0], nn[1] % nscsites] = self.t1 * lattice.bc
    else:
      raise Exception("Unsupported boundary condition")
    
    # build int2e matrix
    nsites = lattice.supercell.nsites
    self.Int2e = np.zeros((nsites, nsites, nsites, nsites))
    for i in range(nsites):
      self.Int2e[i,i,i,i] = self.U

  def get_U(self):
    return self.U

def Hamiltonian(inp_ham, inp_ctrl, lattice):
  if inp_ham.Type == "Hubbard":
    return FHamHubbard(inp_ham.Params, inp_ctrl, lattice)
  else:
    raise KeyError('HamiltonianType %s not exists' % inp_ham.Type)
