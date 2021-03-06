import numpy as np
import sys

Common = {}
Common["Hamiltonian"] = {
  "Type": "Hubbard",
  "Params": {
    'U': 2.,
    't': -1,
  },
}

Common["Geometry"] = {
  "UnitCell": {
    'Sites': [(np.array([0,0]), 'X')],
    'Shape': np.eye(2)
  },
  'ClusterSize': np.array([1,1]),
  'LatticeSize': np.array([50,50]), # defines k-points
}

Common["DMFT"] = {
  "Filling": 0.5,
  "OrbType": "R",
  "InitGuessType": "RAND",
  "nbath": 4,
  "ThrNConv": 1e-4,
  "freq_sample": np.linspace(0.5, 3, 6)
}

Common["MFD"] = {
  "MaxIterMu": 20,
  "MaxIter": 10,
}

Common['FORMAT'] = {
  "Verbose": 3,
  "output": sys.stdout,
}
