#
# File: geometry.py
# Author: Bo-Xiao Zheng <boxiao@princeton.edu>
#

import numpy as np
import numpy.linalg as la
import itertools as it
from cmath import sqrt

from utils import ToSpinOrb, ToSpatOrb

class FUnitCell(object):
  def __init__(self, size, sites): # sites is a list of tuples
    self.size = np.array(size) # unit cell shape
    assert(self.size.shape[0] == self.size.shape[1])
    self.dim = self.size.shape[0]
    self.sites = []
    self.names = []
    for site in sites:
      self.sites.append(np.array(site[0])) # coordination
      self.names.append(site[1])
    self.nsites = len(self.sites)
    self.site_dict = dict(zip([tuple(x) for x in self.sites], range(self.nsites)))

  def __str__(self):
    r = "UnitCell Shape\n%s\nSites:\n" % self.size
    for i in range(len(self.sites)):
      r += "%-10s%-10s\t" % (self.names[i], self.sites[i])
      if (i+1)%6 == 0:
        r+= "\n"
    r += "\n\n"
    return r

class FSuperCell(object):
  def __init__(self, unitcell, size):
    self.unitcell = unitcell
    self.dim = unitcell.dim
    self.csize = np.array(size)
    self.size = np.dot(np.diag(self.csize), unitcell.size)
    self.ncells = np.product(self.csize)
    self.nsites = unitcell.nsites * self.ncells

    self.sites = []
    self.names = []
    self.unitcell_list = []

    for p in it.product(*tuple([range(a) for a in self.csize])):
      self.unitcell_list.append(np.array(p))
      for i in range(len(unitcell.sites)):
        self.sites.append(np.dot(np.array(p), unitcell.size)  + unitcell.sites[i])
        self.names.append(unitcell.names[i])
    
    self.unitcell_dict = dict(zip([tuple(x) for x in self.unitcell_list], range(self.ncells)))
    self.site_dict = dict(zip([tuple(x) for x in self.sites], range(self.nsites)))
    self.fragments = None

  def __str__(self):
    r = self.unitcell.__str__()
    r += "SuperCell Shape\n"
    r += self.size.__str__()
    r += "\nNumber of Sites:%d\n" % self.nsites
    for i, f in enumerate(self.fragments):
      r += "Fragment %3d: %s\n" % (i, f)
    r += "\n"
    return r

  def set_fragments(self, frag):
    if frag is None:
      self.fragments = [range(self.nsites)]
    else:
      sites = []
      for f in frag:
        sites += f.tolist()

      if len(sites) != len(set(sites)): # check duplicate entries
        raise Exception("Fragments have overlaps")
      for i in range(len(sites)):
        if not sites[i] in range(self.nsites):
          raise Exception("SiteId %d is not a valid site" % sites[i])
      self.fragments = frag

class FLattice(object):
  def __init__(self, size, sc, bc, OrbType = "RHFB"):
    self.supercell = sc
    self.dim = sc.dim

    self.scsize = np.array(size)
    self.size = np.dot(np.diag(self.scsize), sc.size)
    self.nscells = np.product(self.scsize)
    self.nsites = sc.nsites * self.nscells

    if bc == "pbc":
      self.bc = 1
    else:
      raise Exception("Unsupported Boudary condition")

    self.sites = []
    self.names = []
    self.supercell_list = []
    for p in it.product(*tuple([range(a) for a in self.scsize])):
      self.supercell_list.append(np.array(p))
      for i in range(len(sc.sites)):
        self.sites.append(np.dot(np.array(p), sc.size)  + sc.sites[i])
        self.names.append(sc.names[i])
    self.supercell_dict = dict(zip([tuple(x) for x in self.supercell_list], range(self.nscells)))
    self.site_dict = dict(zip([tuple(x) for x in self.sites], range(self.nsites)))
    self.OrbType = OrbType
    self.h0 = None
    self.h0_kspace = None
    self.fock = None
    self.fock_kspace = None
    self.neighbor1 = None
    self.neighbor2 = None

  def __str__(self):
    r = self.supercell.__str__()
    r += "Lattice Shape\n%s\n" % self.scsize
    r += "Number of SuperCells: %4d\n" % self.nscells
    r += "Number of Sites:      %4d\n" % self.nsites
    r += "\n"
    return r

  def sc_idx2pos(self, i):
    return self.supercell_list[i % self.nscells]

  def sc_pos2idx(self, p):
    return self.supercell_dict[tuple(p % self.scsize)]

  def add(self, i, j):
    return self.sc_pos2idx(self.sc_idx2pos(i) + self.sc_idx2pos(j))

  def set_Hamiltonian(self, Ham):
    self.Ham = Ham
  
  def get_h0(self, kspace = False, SpinOrb = True):
    if kspace:
      if self.h0_kspace is None:
        self.h0_kspace = self.FFTtoK(self.get_h0(SpinOrb = False))
      if self.OrbType == "UHFB" and SpinOrb:
        return np.array([ToSpinOrb(self.h0_kspace[i]) for i in range(self.nscells)])
      else:
        return self.h0_kspace
    elif self.h0 is None:
        self.h0 = self.Ham.get_h0()
    if self.OrbType == "UHFB" and SpinOrb:
      return np.array([ToSpinOrb(self.h0[i]) for i in range(self.nscells)])
    else:
      return self.h0

  def get_fock(self, kspace = False, SpinOrb = True):
    if kspace:
      if self.fock_kspace is None:
        self.fock_kspace = self.FFTtoK(self.get_fock(SpinOrb = False))
      if self.OrbType == "UHFB" and SpinOrb:
        return np.array([ToSpinOrb(self.fock_kspace[i]) for i in range(self.nscells)])
      else:
        return self.fock_kspace
    elif self.fock is None:
        self.fock = self.Ham.get_fock()
    if self.OrbType == "UHFB" and SpinOrb:
      return np.array([ToSpinOrb(self.fock[i]) for i in range(self.nscells)])
    else:
      return self.fock

  def FFTtoK(self, A):
    # currently only for pbc
    assert(self.bc == 1)
    B = A.reshape(tuple(self.scsize) + A.shape[-2:])
    return np.fft.fftn(B, axes = range(self.dim)).reshape(A.shape)

  def FFTtoT(self, A):
    assert(self.bc == 1)
    B = A.reshape(tuple(self.scsize) + A.shape[-2:])
    C = np.fft.ifftn(B, axes = range(self.dim)).reshape(A.shape)
    if np.allclose(C.imag, 0.):
      return C.real
    else:
      return C

  def get_kpoints(self):
    kpoints = [np.fft.fftfreq(self.scsize[d], 1/(2*np.pi)) for d in range(self.dim)]
    return kpoints

  def expand(self, A, dense = False):
    # expand reduced matrices, eg. Hopping matrix
    assert(self.bc == 1)
    B = np.zeros((self.nsites, self.nsites))
    scnsites = self.supercell.nsites
    if dense:
      for i, j in it.product(range(self.nscells), repeat = 2):
        idx = self.sc_pos2idx(self.sc_idx2pos(i) + self.sc_idx2pos(j))
        B[i*scnsites:(i+1)*scnsites, idx*scnsites:(idx+1)*scnsites] = A[j]
    else:      
      nonzero = [i for i in range(A.shape[0]) if not np.allclose(A[i], 0.)]
      for i in range(self.nscells):
        for j in nonzero:
          idx = self.sc_pos2idx(self.sc_idx2pos(i) + self.sc_idx2pos(j))          
          B[i*scnsites:(i+1)*scnsites, idx*scnsites:(idx+1)*scnsites] = A[j]
    return B

  def transpose_reduced(self, A):
    assert(self.bc == 1)
    B = np.zeros_like(A)
    for n in range(self.nscells):
      B[n] = A[self.sc_pos2idx(-self.sc_idx2pos(n))].T
    return B

  def get_NearNeighbor(self, sites = None, sites2 = None):
    # return nearest neighbors
    # two lists are returned, first is within the lattice, second is along boundary
      
    if sites == None:
      sites = [s for s in range(self.nsites)]
    if sites2 == None:
      sites2 = [s for s in range(self.nsites)]

    neighbor1 = []
    neighbor2 = []
    shifts = [np.array(x) for x in it.product([-1, 0, 1], repeat = self.dim) if x != (0,) * self.dim]
    # first find supercell neighbors
    sc = self.supercell
    for s1 in sites:
      sc1 = s1 / sc.nsites
      sc2 = [sc1] + [self.sc_pos2idx(self.sc_idx2pos(sc1) + shift) for shift in shifts]
      for s2 in list(set(sites2) & set(it.chain.from_iterable([range(s*sc.nsites, (s+1)*sc.nsites)for s in sc2]))):
        if abs(la.norm(self.sites[s2] - self.sites[s1]) - 1.) < 1e-5:
            neighbor1.append((s1, s2))
        else:
          for shift in shifts:
            if abs(la.norm(self.sites[s2]-self.sites[s1] - np.dot(shift, self.size)) - 1.) < 1e-5:
              neighbor2.append((s1, s2))
              break
    return neighbor1, neighbor2

  def get_2ndNearNeighbor(self, sites = None, sites2 = None):
    if sites == None:
      sites = [s for s in range(self.nsites)]
    if sites2 == None:
      sites2 = [s for s in range(self.nsites)]
    neighbor1 = []
    neighbor2 = []
    shifts = [np.array(x) for x in it.product([-1, 0, 1], repeat = self.dim) if x != (0,) * self.dim]
    # first find supercell neighbors
    sc = self.supercell
    for s1 in sites:
      sc1 = s1 / sc.nsites
      sc2 = [sc1] + [self.sc_pos2idx(self.sc_idx2pos(sc1) + shift) for shift in shifts]
      for s2 in list(set(sites2) & set(it.chain.from_iterable([range(s*sc.nsites, (s+1)*sc.nsites)for s in sc2]))):
        if abs(la.norm(self.sites[s2] - self.sites[s1]) - sqrt(2.).real) < 1e-5:
          neighbor1.append((s1, s2))
        else:
          for shift in shifts:
            if abs(la.norm(self.sites[s2]-self.sites[s1] - np.dot(shift, self.size)) - sqrt(2.).real) < 1e-5:
              neighbor2.append((s1, s2))
              break
    return neighbor1, neighbor2

def BuildLatticeFromInput(inp_geom, OrbType = "RHFB", verbose = 5):
  unit = FUnitCell(inp_geom.UnitCell["Shape"],
                   inp_geom.UnitCell["Sites"])
  sc = FSuperCell(unit, np.array(inp_geom.ClusterSize))
  sc.set_fragments(inp_geom.Fragments)

  assert(np.allclose(np.array(inp_geom.LatticeSize) % np.array(inp_geom.ClusterSize), 0.))
  lattice = FLattice(np.array(inp_geom.LatticeSize)/np.array(inp_geom.ClusterSize),
                     sc, inp_geom.BoundaryCondition, OrbType)

  if verbose > 4:
    print "\nGeometry Summary"
    print lattice
  
  return lattice

def Topology(lattice):
  if lattice.dim == 2:
    return Topology2D(lattice)
  else:
    return None

if __name__ == "__main__":
  pass
