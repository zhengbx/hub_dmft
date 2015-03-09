#
# File: inputs.py
# Author: Qiming Sun <qimings@princeton.edu>
#

import os, sys
import numpy as np
import settings

__all__ = ['Input', 'dump_input']
__doc__ = 'Handle the input keywords.'

class Input(object):
# To add more input keywords, create new entry in __doc__ and write the
# keyname, default value, etc. in the *dict format* as follows.  `value` is a
# function defined in _parse_strdict. Its argument `default` will be assigned
# to the default input dict; `allow` can be either a list of allowd values or
# a function to check the given value;  `limits` should be a tuple of
# (lower_bound, upper_bound) for the given value.
  ''' Parse inputs and do very simple check for inputs

  The keys and values for the input-dict are case insensitive.  But they are
  case sensitive when accesse via inputobj.xxx.yyy.
  Available functions:
      get_default_dict
      self_check
      sanity_check
      dump_keys

  Keys and default values
  -----------------------
  'HAMILTONIAN': {
      'Name': None,
      'Type': value(allow=("Hubbard",)),
          # type of hamiltonian that is to be solved
      "Params": {
        'U': 0.,
            # Hubbard
        't': 1.,
            # nn hopping
        't1': 0.,
            # nnn hoping
        'Integral': None,          
      },
  },
  'GEOMETRY': {
      'UnitCell': {
          'Sites': [(np.array([0., 0.]), 'X')],
          'Shape': np.eye(2)
      },
      'ClusterSize': value(allow=ndarrayType),
      'LatticeSize': value(allow=ndarrayType),
      'Fragments': None,
      'BoundaryCondition': value(default='pbc', alllow=('pbc')),
  },
  'DMFT': {
      'Filling': value(limits=(0., 1.)),
      'OrbType': value(allow=("R", "U")),
      'MaxIter': value(default=40, allow=intType, limits=(1,100)),
          # maximum number of DMET self-consistency cycles
      'MinIter': value(default=1, allow=intType, limits=(1, 100)),
      'InitGuessType': value(default='RAND', allow=('RAND', 'MAN', 'ZERO')),
          # initial guess for DMFT calculations
          # including N_{imp+bath}, chemical potential and \Sigma
      'InitGuess': None,
          # initial guess for dmet potential
          # when 'init_guess_type' keyword is set to 'MAN', this inputs the initial guess
          # otherwise may provide the scale of the initial guess
      'nbath': value(allow=intType),
      'ThrNConv': 1e-4,
      'ThrBathConv': 1e-5,
      'freq_sample': np.linspace(0.5,6,12),
      'MaxInnerIter': 30,
  },
  'FITTING': {
  },
  'IMPSOLVER': {
      'ImpSolver': value(default="CheMPS2", allow=("CheMPS2",)),
      'TmpDir': settings.TmpDir,
      'nproc': None,
  },
  'MFD': {
      'SCF': True,
      'MaxIter': 20,
      'MaxIterMu': 10,
      'ThrRdm': 1e-6,
      'DiisDim': 8,
      'DiisStart': 2,
  },
  'FORMAT': {
      'Verbose': value(default=5, allow=range(6)),
          # print level, big value outputs more debug messages
      'output': None,
  },
  'CTRL': {
  }
  '''

  def __init__(self, input_dict={}, ofname=None):
    if ofname == None:
      self.output = sys.stdout
    else:
      self.output = open(ofname, 'w')

    # protect the input key.
    self._readonly = True
    self._input_dict = self.get_default_dict()
    self._checker = _parse_strdict(self._inpdict_in_doc(), 'checker')
    self._phony = ['_checker', '_input_dict'] + self.__dict__.keys()

    fstdval = _parse_strdict(self._inpdict_in_doc(), 'standardize')
    # merge input keywords with the default keywords
    # input keywords are case insensitive
    all_mods = self._input_dict.keys()
    all_MODS = [i.upper() for i in all_mods]
    MOD2mod = dict(zip(all_MODS, all_mods))
    for mod, subdict in input_dict.items():
      if mod.upper() not in all_MODS:
        raise KeyError('module %s is not found' % mod)
      std_mod = MOD2mod[mod.upper()]
      all_keys = self._input_dict[std_mod].keys()
      all_KEYS = [i.upper() for i in all_keys]
      KEY2key = dict(zip(all_KEYS, all_keys))
      for k,v in subdict.items():
        if k.upper() in all_KEYS:
          std_key = KEY2key[k.upper()]
          if callable(fstdval[std_mod][std_key]):
            self._input_dict[std_mod][std_key] \
                = fstdval[std_mod][std_key](v)
          elif isinstance(v, dict) and isinstance(self._input_dict[std_mod][std_key], dict):
            for k1, v1 in v.items():
              self._input_dict[std_mod][std_key][k1] = v1
          else:
            self._input_dict[std_mod][std_key] = v
        else:
          raise KeyError('key %s.%s is not found' % (mod, k))
    # phony keywords must be saved before _merge_input2self
    self._merge_input2self(self._input_dict, self._readonly)

  def __str__(self):
    def dict_str(d,level = 0):
      str = ""
      for key in sorted(d.keys()):
        if isinstance(d[key], dict):
          str += "    " * level + "%-20s\n" % (key.__str__()+":") + dict_str(d[key], level+1) + "\n"
        elif isinstance(d[key], np.ndarray):
          array_str = d[key].__repr__().replace("\n", "\n"+"    "*(level+6))
          str += "    " * level + "%-20s" % key.__str__() + "=    " + array_str + "\n"
        else:
          str += "    " * level + "%-20s" % key.__str__() + "=    " + d[key].__str__() + "\n"
      return str
    return dict_str(dump_input(self), 1)
  
  def _merge_input2self(self, input_dict, read_only):
    # merge input_dict with the class attributes, so that the keyword xxx
    # can be accessed directly by self.xxx
    if read_only:
      # @property are added by function set_prop to RdOnlyClass instead
      # of _InlineClass
      class RdOnlyClass(_InlineClass): pass
      # use closure set_prop to hold key for get_x
      def set_prop(modname, key):
        def get_x(obj):
          return getattr(obj, '__'+key)
        def set_x(obj, v):
          raise SyntaxError('Overwriting  %s.%s  is not allowed'
                                % (modname, key))
        setattr(RdOnlyClass, k, property(get_x, set_x))
      for mod, subdict in input_dict.items():
        self_mod = RdOnlyClass()
        setattr(self, mod, self_mod)
        for k,v in subdict.items():
          setattr(self_mod, '__'+k, v)
          set_prop(mod, k)
    else:
        for mod, subdict in input_dict.items():
          self_mod = _InlineClass()
          setattr(self, mod, self_mod)
          for k,v in subdict.items():
              setattr(self_mod, k, v)

  def _inpdict_in_doc(self):
    doc = _find_between(self.__doc__, \
                        '  -----------------------', '  Examples')
    return '{' + doc + '}'

  def _remove_phony(self):
    # Since the input keys have been merged to the class, remove the
    # intrinic member and functions to get input keys
    return filter(lambda s: not (s.startswith('_') or s in self._phony), \
                  self.__dict__.keys())

  def __copy__(self):
    res = Input.__new__(Input)
    inpkeys = self._remove_phony()
    for m in dir(self):
      self_mod = getattr(self, m)
      if m in inpkeys and isinstance(self_mod, _InlineClass):
        res_mod = _InlineClass()
        setattr(res, m, res_mod)
        for k in self_mod.__dict__.keys():
          pk = k.strip('_')
          v = getattr(self_mod, k)
          res_mod.__dict__.update(((pk, v),))
      else:
        res.__dict__.update(((m, self_mod),))
    return res
  def __deepcopy__(self, memo):
    import copy
    shallow = self.__copy__().__dict__
    res = Input.__new__(Input)
    memo[id(self)] = res
    for k in self.__dict__.keys():
      setattr(res, k, copy.deepcopy(shallow[k], memo))
    return res

  def get_default_dict(self):
    '''Return the default input dict generated from __doc__'''
    return _parse_strdict(self._inpdict_in_doc())

  def self_check(self):
    '''Check sanity for all inputs'''
    for mod in self._remove_phony():
      if self._checker.has_key(mod):
        self_mod = self.__getattribute__(mod)
        for k in dir(self_mod):
          if self._checker[mod].has_key(k):
            if callable(self._checker[mod][k]):
              self._checker[mod][k](getattr(self_mod, k))
          else:
            raise KeyError('key %s.%s is not found' % (mod, k))
      else:
        raise KeyError('module %s is not found' % mod)
    return True

  def sanity_check(self, modname, keyname, val):
    '''Check the given modname.keyname and its value.
    This check is case sensitive for modname, and keyname'''
    if self._checker.has_key(modname):
      if self._checker[modname].has_key(keyname):
        if callable(self._checker[modname][keyname]):
          return self._checker[modname][keyname](val)
        else:
          return True
      else:
        raise KeyError('key %s.%s is not found' % (modname, keyname))
    else:
      raise KeyError('module %s is not found' % modname)
    return True

  def dump_keys(self):
    self.output.write('**** modules and keywords ****\n')
    all_mods = self._remove_phony()
    for mod in sorted(all_mods):
      self_mod = getattr(self, mod)
      if isinstance(self_mod, _InlineClass):
        self.output.write('module: %s\n' % mod)
        for k in dir(self_mod):
          v = getattr(self_mod, k)
          self.output.write('    key: %s = %s\n' % (k,v))
      else:
        self.output.write('key: %s = %s\n' % (mod, self_mod))
  
  def gen_control_options(self):
    pass
    #def match(tup1, tup2):
    #  sum = 0
    #  for x,y in zip(tup1, tup2):
    #    if x == y:
    #      sum += 1
    #  return sum

    #ValidComb = {
    #  # Local, MfdHam, MfdSCF, BathInt, BathVcor, Fitting: Int2eShape, Fock, EffCore, EffEnv
    #  ( True,   "H1", False, False,  True,   "EmbH1"): ( "SC",  "None", False,  "None"),
    #  ( True,   "H1", False,  True, False,   "EmbH1"): ( "SC",  "None", False,  "None"),
    #  
    #  ( True, "Fock", False, False,  True, "EmbFock"): ( "SC", "Store", False, "Store"),
    #  ( True, "Fock", False,  True,  True, "EmbFock"): ( "SC", "Store",  True,  "None"),
    #  ( True, "Fock", False,  True, False, "EmbFock"): ( "SC", "Store",  True,  "None"),
    # 
    #  ( True, "Fock",  True, False,  True, "EmbFock"): ( "SC",  "Comp", False,  "Comp"),
    #  ( True, "Fock",  True, False,  True,  "EmbSCF"): ( "SC",  "Comp", False,  "Comp"),
    #  ( True, "Fock",  True, False,  True,  "LatSCF"): ( "SC",  "Comp", False,  "Comp"),
    #
    #  ( True, "Fock",  True,  True,  True, "EmbFock"): ( "SC",  "Comp",  True,  "None"),
    #  ( True, "Fock",  True,  True,  True,  "EmbSCF"): ( "SC",  "Comp",  True,  "None"),
    #  ( True, "Fock",  True,  True,  True,  "LatSCF"): ( "SC",  "Comp",  True,  "None"),
    #  ( True, "Fock",  True,  True, False, "EmbFock"): ( "SC",  "Comp",  True,  "None"),
    #  ( True, "Fock",  True,  True, False,  "EmbSCF"): ( "SC",  "Comp",  True,  "None"),
    #  ( True, "Fock",  True,  True, False,  "LatSCF"): ( "SC",  "Comp",  True,  "None"),

    #  (False, "Fock", False, False,  True, "EmbFock"): ( "SC", "Store", False, "Store"),

    #  (False, "Fock",  True, False,  True, "EmbFock"): ("Lat",  "Comp", False,  "Comp"),
    #  (False, "Fock",  True, False,  True,  "EmbSCF"): ("Lat",  "Comp", False,  "Comp"),
    #  (False, "Fock",  True, False,  True,  "LatSCF"): ("Lat",  "Comp", False,  "Comp"),
   
    #  (False, "Fock",  True,  True,  True, "EmbFock"): ("Lat",  "Comp",  True,  "None"),
    #  (False, "Fock",  True,  True,  True,  "EmbSCF"): ("Lat",  "Comp",  True,  "None"),
    #  (False, "Fock",  True,  True,  True,  "LatSCF"): ("Lat",  "Comp",  True,  "None"),
    #  (False, "Fock",  True,  True, False, "EmbFock"): ("Lat",  "Comp",  True,  "None"),
    #  (False, "Fock",  True,  True, False,  "EmbSCF"): ("Lat",  "Comp",  True,  "None"),
    #  (False, "Fock",  True,  True, False,  "LatSCF"): ("Lat",  "Comp",  True,  "None"),
    #}

    #input_comb = (self.HAMILTONIAN.Local, self.MFD.Ham, self.MFD.SCF, self.DMET.BathInt2e, \
    #    self.DMET.BathVcor, self.FITTING.Level)
    #max_match = max(map(lambda x: match(input_comb, x), ValidComb.keys()))
    #if max_match != 6:
    #  def string(v):
    #    return "Local:%5s, MfdHam:%5s, MfdSCF:%5s, BathInt:%5s, BathVcor:%5s, FitLevel:%8s" % v
    #  possible_matches = [string(vc) for vc in ValidComb if match(input_comb, vc) == max_match]
    #  info = "Current Combination:\n" + string(input_comb) + "\nPossible Combinations:\n" + "\n".join(possible_matches)
    #  raise Exception("Input options not compatible\n%s" % info)
    #
    #setattr(self.CTRL, "__Int2eShape", ValidComb[input_comb][0])
    #setattr(self.CTRL, "__Fock", ValidComb[input_comb][1])
    #setattr(self.CTRL, "__EffCore", ValidComb[input_comb][2])
    #setattr(self.CTRL, "__EffEnv", ValidComb[input_comb][3])
    #
    #setattr(self.CTRL, "__BathInt2e", self.DMET.BathInt2e)
    #setattr(self.CTRL, "__BathVcor", self.DMET.BathVcor)
    #setattr(self.CTRL, "__FitLevel", self.FITTING.Level)

def dump_input(inp_obj):
  inp_dict = {}
  for mod in inp_obj._remove_phony():
    mod_dict = {}
    for key in inp_obj.__dict__[mod].__dict__.keys():
      mod_dict[key[2:]] = inp_obj.__dict__[mod].__dict__[key]
    inp_dict[mod] = mod_dict
  return inp_dict

def dump_inputfile(fileobj):
  try:
    filename = os.path.join(os.getcwd(), sys.argv[0])
    contents = open(filename, 'r').read()
    fileobj.write('**** input file is %s ****\n' % filename)
    fileobj.write(contents)
    fileobj.write('******************** input file end ********************\n')
  except:
    pass


class _InlineClass(object):
  def __dir__(self):
    return filter(lambda s: not s.startswith('_'), self.__dict__.keys())

def _find_between(s, start, end):
  ''' sub-strings between the start string and end string
  >>> _find_between('abcdefg', 'b', 'ef')
  'cd'
  '''
  s0 = s.find(start) + len(start)
  s1 = s.find(end)
  return s[s0:s1]

def _member(v, lst):
  # if v is the member of lst (case insensitive), return v, other wise
  # return False
  def memberf(v, lst, test):
    for m in lst:
      if test(v, m):
        return m # so the input value will become standard value
    return False
  if isinstance(v, str):
    return memberf(v.upper(), [i for i in lst if isinstance(i, str)],
                   lambda x,y: x == y.upper())
  else:
    return memberf(v, lst, lambda x,y: x == y)

def _parse_strdict(docdict, checker=False):
  '''parse the __doc__ of Input class'''
  if checker == 'standardize':
    # standardize the input values, let it be the one provided by allow
    def value(default=None, allow=None, limits=None, **keywords):
      if allow and not callable(allow):
        return lambda v: _member(v, allow)
      else:
        return lambda v: v
  elif checker == 'checker':
    # require the function 'value' in the __doc__ to generate the function
    # which can check the sanity for each key
    def value(default=None, allow=None, limits=None, **keywords):
      def check_sanity(v):
        if allow and callable(allow):
          return allow(v)
        elif allow:
          if not _member(v, allow):
            raise ValueError('%s is not one of %s' % (v, allow))
        if limits:
          if not (limits[0] <= v <= limits[1]):
            raise ValueError('%s is not in %s' % (v, limits))
        return True
      return check_sanity
  else:
    def value(default=None, **keywords):
      return default

  def stringType(s): return isinstance(s, str)
  def intType(s): return isinstance(s, int)
  def floatType(s): return isinstance(s, float)
  def ndarrayType(s): return isinstance(s, np.ndarray)
  dic = eval(docdict)
  return dic

if __name__ == '__main__':
  import doctest
  doctest.testmod()
