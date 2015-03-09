import numpy as np
import ctypes
from settings import CheMPS2_path
import sys
import os
import itertools as it
sys.path.append(CheMPS2_path)
import PyCheMPS2

Initializer = PyCheMPS2.PyInitialize()
Initializer.Init()

def computeGF(h0, Mu, V, eps, Int2e, nelecA, nelecB, omega, real, fout, verbose):
  nImp, nBath = V.shape
  Ham = PyCheMPS2.PyHamiltonian(nImp+nBath, 0, np.zeros([nImp+nBath], dtype=ctypes.c_int))
  Ham.setEconst(0.)
  H1 = np.zeros((nImp+nBath, nImp+nBath))

  H1[:nImp, :nImp] = h0
  H1[nImp:, nImp:] = np.diag(eps)
  H1[:nImp, nImp:] = V
  H1[nImp:, :nImp] = V.T

  for c1 in range(nImp+nBath):
    for c2 in range(nImp+nBath):
      Ham.setTmat(c1, c2, H1[c1, c2])
  
  for (c1,c2,c3,c4) in it.product(range(nImp+nBath), repeat = 4):
    Ham.setVmat(c1, c2, c3, c4, 0.)
  for (c1,c2,c3,c4) in it.product(range(nImp), repeat = 4):
    Ham.setVmat(c1, c2, c3, c4, Int2e[c1, c2, c3, c4])
  FCI = PyCheMPS2.PyFCI(Ham, nelecA, nelecB, 0, 100., verbose)
  CIvector = np.zeros([FCI.getVecLength()], dtype = ctypes.c_double)
  FCI.FillRandom(FCI.getVecLength(), CIvector)

  E = FCI.GSDavidson(CIvector)
  fout.write("FCI energy E = %20.12f\n" % E)
  
  crea_sites = np.array(range(nImp), dtype=ctypes.c_int)
  anni_sites = np.array(range(nImp), dtype=ctypes.c_int)
  spin_up = True
  
  GFarray = []
  for o in omega:
    alpha = Mu + E
    beta = -1
    if real:
      alpha += o
      eta = 0.05
    else:
      eta = o

    ReAdd, ImAdd = FCI.GFmatrix_add(alpha, beta, eta, anni_sites, crea_sites, spin_up, CIvector, Ham)

    GFadd = (ReAdd + 1.j * ImAdd).reshape((nImp, nImp), order = 'F')
    
    alpha = Mu - E
    beta = 1
    #eta = -0.1
    if real:
      alpha += o
      eta = -0.05
    else:
      eta = o

    ReRem, ImRem = FCI.GFmatrix_rem(alpha, beta, eta, crea_sites, anni_sites, spin_up, CIvector, Ham)
    GFrem = (ReRem + 1.j * ImRem).reshape((nImp, nImp), order = 'F').T

    GFtotal = GFadd + GFrem
    GFarray.append(GFtotal)
  
  fout.write("Green's function computed\n")
  return E, GFarray
