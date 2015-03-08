import numpy as np

def ToClass(kw):
   class _Struct:
      def __init__(self, kw):
         self.__dict__.update(kw)
   return _Struct(kw)

def heatmap(x):
  import matplotlib.pyplot as plt
  xlim = [max(x.shape[1]/2-5, 0), min(x.shape[1]/2+10, x.shape[1])]
  ylim = [max(x.shape[2]/2-5, 0), min(x.shape[2]/2+10, x.shape[2])]
  for i in range(x.shape[0]):
    fig, ax = plt.subplots(1)
    p = ax.pcolormesh(x[i, xlim[0]:xlim[1], ylim[0]:ylim[1]], vmin = 0, vmax = 1)
    fig.colorbar(p)
    fig.savefig('x%02d.png' % i)
    fig.clf()
    ax.cla()

def ReadFile(FileName):
   File = open(FileName, "r")
   Text = File.read()
   File.close()
   return Text

def WriteFile(FileName, Text):
   File = open(FileName, "w")
   File.write(Text)
   File.close()


def mdot(*args):
   """chained matrix product."""
   r = args[0]
   for a in args[1:]:
      r = np.dot(r,a)
   return r

def ToSpinOrb(A):
  if isinstance(A, list):
    assert(len(A) == 2 and A[0].shape == A[1].shape)
    A1 = np.zeros((A[0].shape[0]*2, A[0].shape[1]*2), dtype = A[0].dtype)
    A1[::2,::2] = A[0]
    A1[1::2,1::2] = A[1]
  elif isinstance(A, np.ndarray):
    A1 = np.zeros((A.shape[0]*2, A.shape[1]*2), dtype = A.dtype)
    A1[::2,::2] = A
    A1[1::2,1::2] = A

  return A1

def ToSpatOrb(A):
  A1 = [None, None]
  A1[0] = A[::2,::2]
  A1[1] = A[1::2,1::2]

  return A1

