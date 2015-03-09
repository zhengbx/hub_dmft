import numpy as np
import numpy.linalg as la
import itertools as it
from diis import SCF_DIIS, DIIS

from utils import ToSpinOrb, ToClass

def get_c(V, n):
  if len(V.shape) == 3:
    return V[n]
  else:
    return V


class RMfdSolver(object):
  def __init__(self, lattice, inp_ctrl):
    self.H0 = lattice.get_h0(kspace = True, SpinOrb = False)
    if inp_ctrl.Fock == "Store":
      Fock = lattice.get_fock(kspace = True, SpinOrb = False)
      self.Fock = np.array([Fock, Fock, np.zeros_like(Fock)])
    else:
      self.Fock = np.array([self.H0, self.H0, np.zeros_like(self.H0)]) # Ha, Hb and Delta parts
    self.lattice = lattice
    self.nscsites = lattice.supercell.nsites
    self.ncells = lattice.nscells
    self.rho = None
    self.kappa = None

  def header(self):
    print "******* Hartree-Fock-Bogoliubov Mean-Field in k-space *******\n"
    print "Sprin Restricted Calculation"
  
  def info(self, mu, vcor):
    print "%1dD Lattice: %6d sites in %6d cells\n" % (self.lattice.dim, \
        self.nscsites*self.ncells, self.ncells)
    print "Mu  = %20.12f\n" % mu
    if vcor[1].shape[0] < 20:
      print "Local Potential (V):\n%s\n" % vcor[0]
      print "Local Pairing Matrix (Delta):\n%s\n" % vcor[1]
    else:
      print "Diagonal of Local Potential (V):\n%s\n" % np.diag(vcor[0])
      print "Diagonal Local Pairing Matrix (Delta):\n%s\n" % np.diag(vcor[1])


  def buildBdG(self, VA, VB, D):
    nsites = self.nscsites
    BdG = np.zeros((self.ncells, nsites*2, nsites*2), dtype = complex)
    for n in range(self.ncells):
      BdG[n, :nsites, :nsites] = self.Fock[0][n] + get_c(VA, n)
      BdG[n, nsites:, nsites:] = -(self.Fock[1][n] + get_c(VB, n))
      BdG[n, :nsites, nsites:] =  self.Fock[2][n] + get_c(D, n)
      BdG[n, nsites:, :nsites] = BdG[n, :nsites, nsites:].T.conj()
    return BdG

  def diagH(self, BdG):
    nsites = self.nscsites

    rho = np.zeros_like(self.H0, dtype = complex)
    kappa = np.zeros_like(self.H0, dtype = complex)
    ews = []
    for n in range(self.ncells):
      ew, ev = la.eigh(BdG[n])
      occ = 1. - 1./(1+exp(1000.*ew))
      sol = np.sum(occ > 0.5)
      ews.append(ew[2*nsites-sol:])
      u = ev[:nsites, 2*nsites-sol:]
      v = ev[nsites:, 2*nsites-sol:]
      rho[n] = np.dot(ev[nsites:], np.dot(occ, ev[nsites:].T.conj()))
      kappa[n] = np.dot(ev[:nsites], np.dot(occ, ev[nsites:].T.conj()))
      #sol = np.sum(ew > 1e-4)

      #ews.append(ew[2*nsites-sol:])
      #u = ev[:nsites, 2*nsites-sol:]
      #v = ev[nsites:, 2*nsites-sol:]

      #rho[n] = np.dot(v, v.T.conj())
      #kappa[n] = np.dot(u, v.T.conj())
    
    return ews, rho, kappa

  def diagH_k(self, BdG): # return u_k, v_k, e_k
    nsites = self.nscsites

    u = []
    v = []
    ews = []
    for n in range(self.ncells):
      ew, ev = la.eigh(BdG[n])
      sol = np.sum(ew > 0)
      ews.append(ew[2*nsites-sol:])
      u.append(ev[:nsites, 2*nsites-sol:])
      v.append(ev[nsites:, 2*nsites-sol:])
    return ews, u, v

  def compute_n(self, rho):
    return np.trace(rho[0])
  
  def compute_gap(self, ews):
    ews = sorted(list(it.chain.from_iterable(ews)))
    return ews[0]

  def compute_E(self, rho, kappa, Vloc, Delta):
    H0T = self.lattice.get_h0(kspace = False, SpinOrb = False)
    E = np.sum((H0T+self.FockT[0])*rho) - np.sum(self.FockT[2]*kappa.conj())
    E += np.sum(Vloc*rho[0]) - np.sum(Delta*kappa[0].conj())
    return E

  def run(self, vcor, mu, verbose):
    if verbose > 2:
      self.header()
      self.info(mu, vcor)
    
    Vloc, Delta = vcor
    BdG = self.buildBdG(Vloc-np.eye(self.nscsites)*mu, \
        Vloc-np.eye(self.nscsites)*mu, Delta)
    ews, rho_k, kappa_k = self.diagH(BdG)
    
    rho = self.lattice.FFTtoT(rho_k)
    kappa = self.lattice.FFTtoT(kappa_k)

    self.FockT = map(self.lattice.FFTtoT, self.Fock)
    n = self.compute_n(rho)
    gap = self.compute_gap(ews)
    Energy = self.compute_E(rho, kappa, Vloc, Delta)

    self.rho = rho
    self.kappa = kappa
    if verbose > 2:
      print "<Nelec> (per spin) = %20.12f" % n
      print "           Energy  = %20.12f" % Energy
      print "              Gap  = %20.12f" % gap
      print "               mu  = %20.12f" % mu
      print

    return ToClass({"n": n, "energy": Energy, "rho": rho, "kappa": kappa, \
                     "mu": mu, "gap":gap})
  
  def run_k(self, vcor, mu, verbose):
    if verbose > 2:
      self.header()
      self.info(mu, vcor)

    Vloc, Delta = vcor
    BdG = self.buildBdG(Vloc-np.eye(self.nscsites)*mu, \
        Vloc-np.eye(self.nscsites)*mu, Delta)
    ews, u_k, v_k = self.diagH_k(BdG)
    return ToClass({"e": ews, "u": u_k, "v": v_k})

class RScfMfdSolver(RMfdSolver):
  def __init__(self, lattice, ham, lattice_int2e, inp_mfd, inp_ctrl):
    RMfdSolver.__init__(self, lattice, inp_ctrl)

    self.sInt2e, self.maskInt2e = lattice_int2e.get_sparse_Int2e()
    self.max_iter = inp_mfd.MaxIter
    self.ThrRdm = inp_mfd.ThrRdm
    self.rho_k = None
    self.kappa_k = None
    self.DiisDim = inp_mfd.DiisDim
    self.DiisStart = inp_mfd.DiisStart

  def header(self):
    print "*** Hartree-Fock-Bogoliubov Self-Consistent Mean-Field in k-space ***\n"
    print "Sprin Restricted Calculation"

  def init_guess(self, Vloc, Delta, mu):
    _, rho_k, kappa_k = self.diagH(self.buildBdG( \
        Vloc-np.eye(self.nscsites)*mu, Vloc-np.eye(self.nscsites)*mu, Delta))
    return self.lattice.FFTtoT(rho_k), self.lattice.FFTtoT(kappa_k)

  def init_diis(self):
    diis = SCF_DIIS()
    diis.diis_space = self.DiisDim
    def scf_diis(cycle, d, f):
      if cycle >= self.DiisStart:
        errvec = np.einsum("ijk,ikl->ijl", d, f) - np.einsum("ijk,ikl->ijl", f, d)
        diis.err_vec_stack.append(errvec)
        if diis.err_vec_stack.__len__() > diis.diis_space:
          diis.err_vec_stack.pop(0)
        f = DIIS.update(diis, f)
        return f
      else:
        return f
    return scf_diis
  
  def compute_imp_corr(self):
    found = False
    for i in range(self.maskInt2e.shape[0]):
      if np.all(self.maskInt2e[i] == 0):
        found = True
        break
    if found:
      Int2eImp = self.sInt2e[i]
      Jimp = np.einsum("ijkl,kl->ij", Int2eImp, self.rho[0])
      Kimp = np.einsum("ilkj,kl->ij", Int2eImp, self.rho[0])
      Limp = np.einsum("ikjl,kl->ij", Int2eImp, self.kappa[0])
      return (Jimp*2-Kimp, -Limp)
    else:
      return (0., 0.)

  def translate(self, i, j):
    return self.lattice.sc_pos2idx(self.lattice.sc_idx2pos(j) - self.lattice.sc_idx2pos(i))

  def build_J(self, rho):
    J = np.zeros((self.ncells, self.nscsites, self.nscsites), dtype = complex)
    for idx, jkl in enumerate(self.maskInt2e):
      j,k,l = jkl
      J[j] += np.einsum("ijkl,kl->ij", self.sInt2e[idx], rho[self.translate(k, l)])
    return self.lattice.FFTtoK(J)
      
  def build_K(self, rho):
    K = np.zeros((self.ncells, self.nscsites, self.nscsites), dtype = complex)
    for idx, lkj in enumerate(self.maskInt2e):
      l,k,j = lkj
      K[j] += np.einsum("ilkj,kl->ij", self.sInt2e[idx], rho[self.translate(k, l)])
    return self.lattice.FFTtoK(K)
      
  def build_L(self, kappa):
    L = np.zeros((self.ncells, self.nscsites, self.nscsites), dtype = complex)
    for idx, kjl in enumerate(self.maskInt2e):
      k,j,l = kjl
      L[j] += np.einsum("ikjl,kl->ij", self.sInt2e[idx], kappa[self.translate(k, l)])
    return self.lattice.FFTtoK(L)
  
  def build_grdm(self, rho, kappa):
    nsites = self.nscsites
    grdm = np.zeros((self.ncells, nsites*2, nsites*2), dtype = complex)
    for n in range(self.ncells):
      grdm[n, :nsites, :nsites] = np.eye(nsites) - rho[n]
      grdm[n, nsites:, nsites:] = rho[n]
      grdm[n, :nsites, nsites:] = kappa[n]
      grdm[n, nsites:, :nsites] = kappa[n].T.conj()
    return grdm

  def run(self, vcor, mu, verbose):
    if verbose > 2:
      self.header()
      self.info(mu, vcor)
    if verbose > 3:
      print "Iter    ErrRdm"
    
    Vloc, Delta = vcor
    if self.rho is None:
      self.rho, self.kappa = self.init_guess(Vloc, Delta, mu)
    
    diis = self.init_diis()
    Conv = False
    grdm = None
    for iter in range(self.max_iter):
      J = self.build_J(self.rho)
      K = self.build_K(self.rho)
      L = self.build_L(self.kappa)

      self.Fock = [self.H0+J*2-K, self.H0+J*2-K, -L]
      BdG = self.buildBdG(Vloc-np.eye(self.nscsites)*mu, Vloc-np.eye(self.nscsites)*mu, Delta)
      
      if self.rho_k is not None:
        grdm = self.build_grdm(self.rho_k, self.kappa_k)
        BdG = diis(iter, grdm, BdG)        
      ews, rho_k, kappa_k = self.diagH(BdG)
    
      rho = self.lattice.FFTtoT(rho_k)
      kappa = self.lattice.FFTtoT(kappa_k)
      
      err_rdm = max(la.norm((rho-self.rho).flatten(), np.inf), \
          la.norm((kappa-self.kappa).flatten(), np.inf))
      
      self.rho_k, self.kappa_k = rho_k, kappa_k     
      self.rho, self.kappa = rho, kappa

      if verbose > 3:
        print "%3d%20.12f" % (iter, err_rdm)
      if err_rdm < self.ThrRdm:
        Conv = True
        break

    if Conv and verbose > 2:
      print "SCF converged at iteration %d\n" % iter
    if not Conv and verbose > 0:
      print "Warning: SCF failed to converge\n"

    self.FockT = map(self.lattice.FFTtoT, self.Fock)
    n = self.compute_n(rho)
    gap = self.compute_gap(ews)
    Energy = self.compute_E(rho, kappa, Vloc, Delta)
    
    if verbose > 2:
      print "<Nelec> (per spin) = %20.12f" % n
      print "           Energy  = %20.12f" % Energy
      print "              Gap  = %20.12f" % gap
      print "               mu  = %20.12f" % mu          
      print

    return ToClass({"n": n, "energy": Energy, "rho": rho, "kappa": kappa, \
                     "mu": mu, "gap":gap})

  def run_k(self, vcor, mu, verbose):
    if verbose > 2:
      self.header()
      self.info(mu, vcor)
    if verbose > 3:
      print "Iter    ErrRdm"

    Vloc, Delta = vcor
    if self.rho is None:
      self.rho, self.kappa = self.init_guess(Vloc, Delta, mu)

    diis = self.init_diis()
    Conv = False
    grdm = None
    for iter in range(self.max_iter):
      J = self.build_J(self.rho)
      K = self.build_K(self.rho)
      L = self.build_L(self.kappa)

      self.Fock = [self.H0+J*2-K, self.H0+J*2-K, -L]
      BdG = self.buildBdG(Vloc-np.eye(self.nscsites)*mu, Vloc-np.eye(self.nscsites)*mu, Delta)
      
      if self.rho_k is not None:
        grdm = self.build_grdm(self.rho_k, self.kappa_k)
        BdG = diis(iter, grdm, BdG)
      ews, rho_k, kappa_k = self.diagH(BdG)
    
      rho = self.lattice.FFTtoT(rho_k)
      kappa = self.lattice.FFTtoT(kappa_k)
      
      err_rdm = max(la.norm((rho-self.rho).flatten(), np.inf), \
          la.norm((kappa-self.kappa).flatten(), np.inf))
      
      self.rho_k, self.kappa_k = rho_k, kappa_k     
      self.rho, self.kappa = rho, kappa

      if verbose > 3:
        print "%3d%20.12f" % (iter, err_rdm)
      if err_rdm < self.ThrRdm:
        Conv = True
        break

    if Conv and verbose > 2:
      print "SCF converged at iteration %d\n" % iter
    if not Conv and verbose > 0:
      print "Warning: SCF failed to converge\n"

    J = self.build_J(self.rho)
    K = self.build_K(self.rho)
    L = self.build_L(self.kappa)

    self.Fock = [self.H0+J*2-K, self.H0+J*2-K, -L]
    BdG = self.buildBdG(Vloc-np.eye(self.nscsites)*mu, Vloc-np.eye(self.nscsites)*mu, Delta)
    ews, u_k, v_k = self.diagH_k(BdG)

    return ToClass({"e": ews, "u": u_k, "v": v_k})


class UMfdSolver(RMfdSolver):
  def __init__(self, lattice, inp_ctrl):
    RMfdSolver.__init__(self, lattice, inp_ctrl)

  def header(self):
    print "******* Hartree-Fock-Bogoliubov Mean-Field in k-space *******\n"
    print "Spin Unrestricted Calculation"
  
  def diagH(self, BdG):
    nsites = self.nscsites

    rho_a = np.zeros_like(self.H0, dtype = complex)
    rho_b = np.zeros_like(self.H0, dtype = complex)    
    kappa = np.zeros_like(self.H0, dtype = complex)
    ews1 = []
    ews2 = []
    for n in range(self.ncells):
      ew, ev = la.eigh(BdG[n])
      sol = np.sum(ew > 0)
      ews1.append(ew[2*nsites-sol:])
      ews2.append(-ew[:2*nsites-sol][::-1])
      ua = ev[:nsites, 2*nsites-sol:]
      vb = ev[nsites:, 2*nsites-sol:]
      rho_a[n] = np.eye(nsites) - np.dot(ua, ua.T.conj())
      rho_b[n] = np.dot(vb, vb.T.conj())
      kappa[n] = np.dot(ua, vb.T.conj())
    
    return ews1, ews2, rho_a, rho_b, kappa

  def diagH_k(self, BdG):
    nsites = self.nscsites
    u = []
    v = []
    ews = []
    for n in range(self.ncells):
      ew, ev = la.eigh(BdG[n])
      sol = np.sum(ew > 0)
      ews.append([ew[2*nsites-sol:], -ew[:2*nsites-sol][::-1]])
      u.append([ev[:nsites, 2*nsites-sol:], ev[nsites:, :2*nsites-sol][:, ::-1].conj()])
      v.append([ev[:nsites, :2*nsites-sol][:, ::-1].conj(), ev[nsites:, 2*nsites-sol:]])
    return ews, u, v

  def compute_n(self, rho):
    return np.trace(rho[0][0]), np.trace(rho[1][0])

  def compute_gap(self, ews):
    return map(lambda x: RMfdSolver.compute_gap(self, x), ews)
  
  def compute_E(self, rho, kappa, Vloc, Delta):
    H0T = self.lattice.get_h0(kspace = False, SpinOrb = False)
    E = 0.5 * (np.sum((H0T+self.FockT[0])*rho[0]) + np.sum((H0T+self.FockT[1])*rho[1])) \
        - np.sum(self.FockT[2]*kappa.conj())
    E += 0.5*(np.sum(Vloc[::2, ::2]*rho[0][0]) + np.sum(Vloc[1::2, 1::2]*rho[1][0])) \
        - np.sum(Delta*kappa[0].conj())
    return E

  def SpinOrbRibbon(self, rho_spat):
    rho = np.array([ToSpinOrb([rho_spat[0][n], rho_spat[1][n]]) for n in range(self.ncells)])
    return rho

  def run(self, vcor, mu, verbose):
    if verbose > 2:
      self.header()
      self.info(mu, vcor)

    Vloc, Delta = vcor
    BdG = self.buildBdG(Vloc[::2, ::2]-np.eye(self.nscsites)*mu, \
        Vloc[1::2, 1::2]-np.eye(self.nscsites)*mu, Delta)
    ews1, ews2, rho_k_a, rho_k_b, kappa_k = self.diagH(BdG)
    
    rho = map(self.lattice.FFTtoT, [rho_k_a, rho_k_b])
    kappa = self.lattice.FFTtoT(kappa_k)

    self.FockT = map(self.lattice.FFTtoT, self.Fock)
    n = self.compute_n(rho)
    gap = self.compute_gap([ews1, ews2])
    Energy = self.compute_E(rho, kappa, Vloc, Delta)
    
    self.rho = rho
    self.kappa = kappa
    
    rho = self.SpinOrbRibbon(rho)
    if verbose > 2:
      print "<Nelec>_a  = %20.12f <Nelec>_b = %20.12f" % tuple(n)
      print "    Gap_a  = %20.12f     Gap_b = %20.12f" % tuple(gap)
      print "   Energy  = %20.12f" % Energy
      print "       mu  = %20.12f" % mu
      print
    return ToClass({"n": 0.5*np.sum(n), "energy": Energy, "rho": rho, "kappa": kappa, \
                   "mu": mu, "gap": min(gap)})
  
  def run_k(self, vcor, mu, verbose):
    if verbose > 2:
      self.header()
      self.info(mu, vcor)

    Vloc, Delta = vcor
    BdG = self.buildBdG(Vloc[::2, ::2]-np.eye(self.nscsites)*mu, \
        Vloc[1::2, 1::2]-np.eye(self.nscsites)*mu, Delta)
    ews, u_k, v_k = self.diagH_k(BdG)
    
    return ToClass({"e": ews, "u": u_k, "v": v_k})


class UScfMfdSolver(UMfdSolver, RScfMfdSolver):
  def __init__(self, lattice, ham, lattice_int2e, inp_mfd, inp_ctrl):
    RScfMfdSolver.__init__(self, lattice, ham, lattice_int2e, inp_mfd, inp_ctrl)

  def header(self):
    print "*** Hartree-Fock-Bogoliubov Self-Consistent Mean-Field in k-space ***\n"
    print "Sprin Restricted Calculation"

  def init_guess(self, Vloc, Delta, mu):
    _, _, rho_k_a, rho_k_b, kappa_k = self.diagH(self.buildBdG( \
        Vloc[::2, ::2]-np.eye(self.nscsites)*mu, Vloc[1::2, 1::2]-np.eye(self.nscsites)*mu, Delta))
    return [self.lattice.FFTtoT(rho_k_a), self.lattice.FFTtoT(rho_k_b)], self.lattice.FFTtoT(kappa_k)

  def compute_imp_corr(self):
    found = False
    for i in range(self.maskInt2e.shape[0]):
      if np.all(self.maskInt2e[i] == 0):
        found = True
        break
    if found:
      Int2eImp = self.sInt2e[i]
      Jimp = 0.5*np.einsum("ijkl,kl->ij", Int2eImp, self.rho[0][0]+self.rho[1][0])
      Kaimp = np.einsum("ilkj,kl->ij", Int2eImp, self.rho[0][0])
      Kbimp = np.einsum("ilkj,kl->ij", Int2eImp, self.rho[1][0])
      Limp = np.einsum("ikjl,kl->ij", Int2eImp, self.kappa[0])
      return (Jimp*2-Kaimp, Jimp*2-Kbimp, -Limp)
    else:
      return (0., 0., 0.)

  def build_grdm(self, rho, kappa):
    nsites = self.nscsites
    grdm = np.zeros((self.ncells, nsites*2, nsites*2), dtype = complex)
    for n in range(self.ncells):
      grdm[n, :nsites, :nsites] = np.eye(nsites) - rho[0][n]
      grdm[n, nsites:, nsites:] = rho[1][n]
      grdm[n, :nsites, nsites:] = kappa[n]
      grdm[n, nsites:, :nsites] = kappa[n].T.conj()
    return grdm

  def run(self, vcor, mu, verbose):
    if verbose > 2:
      self.header()
      self.info(mu, vcor)
    if verbose > 3:
      print "Iter    ErrRdm"

    Vloc, Delta = vcor
    if self.rho is None:
      self.rho, self.kappa = self.init_guess(Vloc, Delta, mu)
    
    diis = self.init_diis()
    Conv = False
    grdm = None
    for iter in range(self.max_iter):
      J = self.build_J(0.5*(self.rho[0]+self.rho[1]))
      Ka = self.build_K(self.rho[0])
      Kb = self.build_K(self.rho[1])
      L = self.build_L(self.kappa)

      self.Fock = [self.H0+J*2-Ka, self.H0+J*2-Kb, -L]
      BdG = self.buildBdG(Vloc[::2, ::2]-np.eye(self.nscsites)*mu, \
          Vloc[1::2, 1::2]-np.eye(self.nscsites)*mu, Delta)

      if self.rho_k is not None:
        grdm = self.build_grdm(self.rho_k, self.kappa_k)
        BdG = diis(iter, grdm, BdG)
      ews1, ews2, rho_k_a, rho_k_b, kappa_k = self.diagH(BdG)
      
      rho = map(self.lattice.FFTtoT, [rho_k_a, rho_k_b])
      kappa = self.lattice.FFTtoT(kappa_k)

      err_rdm = max(la.norm((rho[0]-self.rho[0]).flatten(), np.inf), \
          la.norm((rho[1]-self.rho[1]).flatten(), np.inf), la.norm((kappa-self.kappa).flatten(), np.inf))
      
      self.rho_k, self.kappa_k = [rho_k_a, rho_k_b], kappa_k      
      self.rho, self.kappa = rho, kappa

      if verbose > 3:
        print "%3d%20.12f" % (iter, err_rdm)
      if err_rdm < self.ThrRdm:
        Conv = True
        break
    
    if Conv and verbose > 2:
      print "SCF converged at iteration %d\n" % iter
    if not Conv and verbose > 0:
      print "Warning: SCF failed to converge\n"

    self.FockT = map(self.lattice.FFTtoT, self.Fock)
    n = self.compute_n(rho)
    gap = self.compute_gap([ews1, ews2])
    Energy = self.compute_E(rho, kappa, Vloc, Delta) # FIXME which function does it use
    rho = self.SpinOrbRibbon(rho)

    if verbose > 2:
      print "<Nelec>_a  = %20.12f <Nelec>_b = %20.12f" % tuple(n)
      print "    Gap_a  = %20.12f     Gap_b = %20.12f" % tuple(gap)
      print "   Energy  = %20.12f" % Energy
      print "       mu  = %20.12f" % mu
      print
    return ToClass({"n": 0.5*np.sum(n), "energy": Energy, "rho": rho, "kappa": kappa, \
                   "mu": mu, "gap": min(gap)})

  def run_k(self, vcor, mu, verbose):
    if verbose > 2:
      self.header()
      self.info(mu, vcor)
    if verbose > 3:
      print "Iter    ErrRdm"

    Vloc, Delta = vcor
    if self.rho is None:
      self.rho, self.kappa = self.init_guess(Vloc, Delta, mu)
    
    diis = self.init_diis()
    Conv = False
    grdm = None
    for iter in range(self.max_iter):
      J = self.build_J(0.5*(self.rho[0]+self.rho[1]))
      Ka = self.build_K(self.rho[0])
      Kb = self.build_K(self.rho[1])
      L = self.build_L(self.kappa)

      self.Fock = [self.H0+J*2-Ka, self.H0+J*2-Kb, -L]
      BdG = self.buildBdG(Vloc[::2, ::2]-np.eye(self.nscsites)*mu, \
          Vloc[1::2, 1::2]-np.eye(self.nscsites)*mu, Delta)

      if self.rho_k is not None:
        grdm = self.build_grdm(self.rho_k, self.kappa_k)
        BdG = diis(iter, grdm, BdG)
      ews1, ews2, rho_k_a, rho_k_b, kappa_k = self.diagH(BdG)
      
      rho = map(self.lattice.FFTtoT, [rho_k_a, rho_k_b])
      kappa = self.lattice.FFTtoT(kappa_k)

      err_rdm = max(la.norm((rho[0]-self.rho[0]).flatten(), np.inf), \
          la.norm((rho[1]-self.rho[1]).flatten(), np.inf), la.norm((kappa-self.kappa).flatten(), np.inf))
      
      self.rho_k, self.kappa_k = [rho_k_a, rho_k_b], kappa_k      
      self.rho, self.kappa = rho, kappa

      if verbose > 3:
        print "%3d%20.12f" % (iter, err_rdm)
      if err_rdm < self.ThrRdm:
        Conv = True
        break
    
    if Conv and verbose > 2:
      print "SCF converged at iteration %d\n" % iter
    if not Conv and verbose > 0:
      print "Warning: SCF failed to converge\n"

    J = self.build_J(0.5*(self.rho[0]+self.rho[1]))
    Ka = self.build_K(self.rho[0])
    Kb = self.build_K(self.rho[1])
    L = self.build_L(self.kappa)
    
    self.Fock = [self.H0+J*2-Ka, self.H0+J*2-Kb, -L]
    BdG = self.buildBdG(Vloc[::2, ::2]-np.eye(self.nscsites)*mu, \
        Vloc[1::2, 1::2]-np.eye(self.nscsites)*mu, Delta)

    ews, u_k, v_k = self.diagH_k(BdG)
    return ToClass({"e": ews, "u": u_k, "v": v_k})

class LatticeIntegral(object):
  def __init__(self, lattice, Int2e, Int2eShape, UHFB, thr_rdm = 0., thr_int = 0.):
    self.Int2e = Int2e
    self.Int2eShape = Int2eShape
    self.maskInt2e = None
    self.sInt2e = None
    self.UHFB = UHFB
    self.thr_rdm = thr_rdm
    self.thr_int = thr_int
    self.lattice = lattice
    
  def __Int2e_to_cell(self, Int2e):
    nscsites = self.lattice.supercell.nsites
    ncells = self.lattice.nscells
    Int2eCell = np.zeros((ncells, ncells, ncells, nscsites, \
        nscsites, nscsites, nscsites))
    for j,k,l in it.product(range(ncells), repeat = 3):
      Int2eCell[j,k,l] = Int2e[:, j*nscsites: (j+1)*nscsites, k*nscsites: (k+1)*nscsites, \
          l*nscsites: (l+1)*nscsites]
    return Int2eCell

  def __to_sparse_Int2e(self):
    nscsites = self.lattice.supercell.nsites
    ncells = self.lattice.nscells
    
    if self.Int2eShape == "Lat":
      Int2eCell = self.__Int2e_to_cell(self.Int2e)
      self.maskInt2e = np.array(np.nonzero(la.norm(Int2eCell.reshape(ncells, \
          ncells, ncells, nscsites**4), axis = 3) > self.thr_int*nscsites)).T
      sInt2e = [Int2eCell[tuple(idx)] for idx in self.maskInt2e]
    else:
      self.maskInt2e = np.zeros((1,3), dtype = int)
      sInt2e = [self.Int2e]
    self.sInt2e = np.array(sInt2e)

  def get_sparse_Int2e(self):
    if self.sInt2e is None:
      self.__to_sparse_Int2e()
    return self.sInt2e, self.maskInt2e

  def __to_sparse_rdm(self, rdm):
    nscsites = self.lattice.supercell.nsites
    ncells = self.lattice.nscells

    rdm1 = np.zeros((ncells, ncells, nscsites, nscsites))
    for i, j in it.product(range(ncells), repeat = 2):
      rdm1[i, j] = rdm[i*nscsites: (i+1)*nscsites, j*nscsites: (j+1)*nscsites]
    return rdm1

  def __call__(self, basis):
    nscsites = self.lattice.supercell.nsites
    ncells = self.lattice.nscells

    u, v = basis.u, basis.v

    if self.sInt2e is None:
      self.__to_sparse_Int2e()

    if self.UHFB:
      rdms = map(self.__to_sparse_rdm, [np.dot(v[0], v[0].T), np.dot(v[1], v[1].T), np.dot(u[0], v[1].T)])
      mask_rdm = reduce(np.ndarray.__add__, \
          map(lambda x: la.norm(x, axis = (2,3)) > self.thr_rdm*nscsites, rdms))
      J = np.zeros((nscsites*ncells, nscsites*ncells))
      Ka = np.zeros((nscsites*ncells, nscsites*ncells))
      Kb = np.zeros((nscsites*ncells, nscsites*ncells))
      L = np.zeros((nscsites*ncells, nscsites*ncells))
      
      for idx, jkl in enumerate(self.maskInt2e):
        for i in range(ncells):
          j, k, l = map(lambda x: self.lattice.add(x, i), jkl)
          if mask_rdm[k, l]:
            J[i*nscsites:(i+1)*nscsites, j*nscsites:(j+1)*nscsites] += \
                0.5 * np.einsum("ijkl,kl->ij", self.sInt2e[idx], rdms[0][k,l]+rdms[1][k,l])
          if mask_rdm[k,j]:
            Ka[i*nscsites:(i+1)*nscsites, l*nscsites:(l+1)*nscsites] += \
                np.einsum("ijkl,kj->il", self.sInt2e[idx], rdms[0][k,j])
            Kb[i*nscsites:(i+1)*nscsites, l*nscsites:(l+1)*nscsites] += \
                np.einsum("ijkl,kj->il", self.sInt2e[idx], rdms[1][k,j])
          if mask_rdm[j,l]:
            L[i*nscsites:(i+1)*nscsites, k*nscsites:(k+1)*nscsites] += \
                np.einsum("ijkl,jl->ik", self.sInt2e[idx], rdms[2][j,l])
      return J*2-Ka, J*2-Kb, -L
    else:
      rdms = map(to_sparse_rdm, [np.dot(v, v.T), np.dot(u, v.T)])
      mask_rdm = np.nonzero(reduce(np.ndarray.__add__, \
          map(lambda x: la.norm(x, axis = (2,3)) > self.thr_rdm*nscsites, rdms)))
      J = np.zeros((nscsites*ncells, nscsites*ncells))
      K = np.zeros((nscsites*ncells, nscsites*ncells))
      L = np.zeros((nscsites*ncells, nscsites*ncells))

      for idx, jkl in enumerate(self.maskInt2e):
        for i in range(ncells):
          j, k, l = map(lambda x: self.lattice.add(x, i), jkl)
          if mask_rdm[k, l]:
            J[i*nscsites:(i+1)*nscsites, j*nscsites:(j+1)*nscsites] += \
                np.einsum("ijkl,kl->ij", self.sInt2e[idx], rdms[0][k,l])
          if mask_rdm[k,j]:
            K[i*nscsites:(i+1)*nscsites, l*nscsites:(l+1)*nscsites] += \
                np.einsum("ijkl,kj->il", self.sInt2e[idx], rdms[0][k,j])
          if mask_rdm[j,l]:
            L[i*nscsites:(i+1)*nscsites, k*nscsites:(k+1)*nscsites] += \
                np.einsum("ijkl,jl->ik", self.sInt2e[idx], rdms[1][j,l])
      return J*2-K, None, L

def init_mfd(inp_mfd, lattice, ham, orbtype):
  inp_ctrl = ToClass({'Fock': 'Comp'})
  restricted = (orbtype == "R")  
  lattice_int2e = LatticeIntegral(lattice, ham.Int2e, 'SC', not restricted, 0., 0.)

  if inp_mfd.SCF and restricted:
    return RScfMfdSolver(lattice, ham, lattice_int2e, inp_mfd, inp_ctrl)
  elif  inp_mfd.SCF and not restricted:
    return UScfMfdSolver(lattice, ham, lattice_int2e, inp_mfd, inp_ctrl)
  elif (not inp_mfd.SCF) and restricted:
    return RMfdSolver(lattice, inp_ctrl)
  else:
    return UMfdSolver(lattice, inp_ctrl)
