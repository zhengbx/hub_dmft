from time import time

class Timer:
  def __init__(self):
    self.accu_t = 0
    self.t0 = None
  
  def start(self):
    if self.t0 is None:
      self.t0 = time()
    else:
      raise Exception("Timer already started")

  def end(self):
    if self.t0 is not None:
      self.accu_t += time() - self.t0
      self.t0 = None
    else:
      raise Exception("Timer not started yet")

  def get_time(self):
    if self.t0 is None:
      return self.accu_t
    else:
      return self.accu_t + time() - self.t0

class Timers:
  def __init__(self):
    self.timers = {}
  
  def start(self, name):
    if not name in self.timers.keys():
      self.timers[name] = Timer()
    self.timers[name].start()
  
  def end(self, name):
    if not name in self.timers.keys():
      raise Exception("Timer not created yet")
    self.timers[name].end()

  def __call__(self, name):
    if not name in self.timers.keys():
      raise Exception("Timer not created yet")
    return self.timers[name].accu_t

