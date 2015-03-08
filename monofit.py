import numpy as np
from cmath import sqrt

phi = [(3.-sqrt(5.).real)*0.5, (sqrt(5.).real-1.)*0.5]

class MonoSearch(object):
  def __init__(self, target, x0 = 0):
    self.x = [x0]
    self.y = []
    self.x0 = x0
    self.stage = 0
    self.target = target
    self.sec_x = None
    self.sec_y = None

  def update(self, val):
    self.y.append(val)
    if self.sec_y is not None:
      if np.sign(self.sec_y[0]-self.target) == np.sign(val-self.target):
        self.sec_x[0] = self.x[-1]
        self.sec_y[0] = val
      else:
        self.sec_x[1] = self.x[-1]
        self.sec_y[1] = val
    self.x.append(self.get_next())
  
  def get_next(self):
    assert(len(self.x) == len(self.y))
    n = len(self.x)
    if self.stage == 0:
      if n == 1:
        return self.x[-1] + 1
      elif (self.y[-1]-self.target) * (self.y[-2]-self.target) < 0:
        self.stage = 1
        self.sec_x = self.x[-2:]
        self.sec_y = self.y[-2:]
        return self.get_next_bisec()
      else:
        deri = (self.y[-1]-self.y[-2]) / (self.x[-1]-self.x[-2])
        if abs(deri) < 0.1:
          deri = np.sign(deri) * 0.1
        return self.x[-1] + (self.target-self.y[-1]) / (deri)
    else:
      return self.get_next_bisec()

  def get_next_bisec(self):
    # golden section search
    if abs(self.sec_y[0] - self.target) > abs(self.sec_y[1] - self.target):
      return phi[0] * self.sec_x[0] + phi[1] * self.sec_x[1]
    else:
      return phi[1] * self.sec_x[0] + phi[0] * self.sec_x[1]

  def __iter__(self):
    while True:
      yield self.x[-1]


if __name__ == "__main__":
  y0 = 12
  lins = MonoSearch(y0, x0 = 0)
  fn = lambda x: x**3+0.2*x**2
  a = 0
  for x in lins:
    a += 1
    print x, fn(x)
    lins.update(fn(x))
    if abs(fn(x)- y0) < 1e-5:
      break
