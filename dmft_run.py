#!/usr/bin/env python2.7

#
# File: dmft_run.py
# Author: Bo-Xiao Zheng <boxiao@princeton.edu>
#

import numpy as np
import sys
import itertools as it
from copy import deepcopy
import pickle as p
import inspect

from main import main as dmft
from tempfile import mkdtemp
from inputs import dump_input
from utils import ToClass

#def ResultTable(results, id, dmet_result, additional_result = None):
#  data_dict = [
#    ("ID",     " Job",               "%3d "   ),
#    ("U",      "     U  ",           "%8.4f"  ),
#    ("V",      "     V  ",           "%8.4f"  ),
#    ("J",      "     J  ",           "%8.4f"  ),
#    ("t",      "     t  ",           "%8.4f"  ),
#    ("V1",     "     V1 ",           "%8.4f"  ),
#    ("J1",     "     J1 ",           "%8.4f"  ),
#    ("t1",     "     t1 ",           "%8.4f"  ),
#    ("System", "           Name   ", "%18s"   ),    
#    ("Nelec",  "        Nelec     ", "%18.12f"),
#    ("Energy", "          E       ", "%18.12f"),
#    ("Mu",     "    Chemical.Pot. ", "%18.12f"),
#    ("Conv",   "   Conv"            , "%8s"   ),
#  ]
#  data = []
#  names = []
#  fmt = []
#  value = []
#  for term in data_dict:
#    if term[0] == "ID":
#      data.append(term[0])
#      names.append(term[1])
#      fmt.append(term[2])
#      value.append(id)
#    elif term[0] in dmet_result.__dict__:
#      data.append(term[0])
#      names.append(term[1])
#      fmt.append(term[2])
#      value.append(getattr(dmet_result, term[0]))
#    elif additional_result is not None and term[0] in additional_result.__dict__:
#      data.append(term[0])
#      names.append(term[1])
#      fmt.append(term[2])
#      value.append(getattr(additional_result, term[0]))
#
#  names = "".join(names)
#  fmt = "".join(fmt)
#  
#  results.append(fmt % tuple(value))
#  fout.write("\n\nResult Table After Finishing %3d Jobs:\n\n" % len(results))
#  fout.write(names + '\n')
#  fout.write('-' * (len(names) + 3) + '\n')
#  for r in results:
#    fout.write(r)
#    fout.write('\n')
#  fout.write('\n')
#  return results

def banner():
  fout.write("***********************************************************************\n\n")
  fout.write("                        D M F T   P r o g r a m                        \n\n")
  fout.write("                             08/2014                                   \n\n")
  fout.write("***********************************************************************\n\n")

def default_keywords(g):
  if not "Common" in g.__dict__.keys():
    setattr(g, "Common", {})
  keywords = ["IterOver", "First", "FromPrevious"]
  for key in keywords:
    if not key in g.__dict__.keys():
      setattr(g, key, [])

def summary(g):
  fout.write("\nDMFT MODEL SUMMARY\n\n")
  fout.write("Common Settings:\n\n")
  for item in g.Common.items():
    fout.write("%-8s\n" % item[0])
    for subitem in item[1].items():
      fout.write("    %-12s = %s\n" % (subitem[0], subitem[1]))
    fout.write('\n')

  fout.write("\nIteration over Settings:\n")
  for item in g.IterOver:
    fout.write("    %-12s = %s\n" % (item[1], item[2]))
  if len(g.IterOver) == 0:
    fout.write("  None\n")
  fout.write("Total Number of Jobs: %d\n\n" % np.product([len(v[2]) for v in g.IterOver]))

  fout.write("Special Settings for First Iteration:\n")
  for item in g.First:
    fout.write("    %-12s = %s\n" % (item[1], item[2]))
  if len(g.First) == 0:
    fout.write("  None\n")
  print

  fout.write("Settings From Previous Converged Calculation:\n")
  for item in g.FromPrevious:
    fout.write("    %-12s = %s\n" % (item[1], item[2]))
  if len(g.FromPrevious ) == 0:
    fout.write("  None\n")
  fout.write('\n')

def expand_iter(iterover):
  keys = []
  vals = []
  for item in iterover:
    assert(len(item) == 3)
    keys.append((item[0], item[1]))
    vals.append(item[2])

  val_expand = list(it.product(*vals))
  return keys, val_expand, vals

if __name__ == "__main__":
  if len(sys.argv) < 2:
    raise Exception("No input file.")
  np.set_printoptions(precision = 12, suppress = True, threshold = 10000, linewidth = np.inf)

  filename = sys.argv[1]
  if filename.endswith(".py"):
    filename = filename[:-3].replace("/", ".")
  
  exec("import %s as g" % filename)
  fout = g.Common['FORMAT']['output'] # FIXME a hacker
  banner()
  fout.write("Input File\n")
  fout.write(">" * 60 + '\n')
  fout.write(inspect.getsource(g) + '\n')
  fout.write("<" * 60 + '\n\n')
  
  default_keywords(g)
  summary(g)
  
  iter_key, iter_val, compact_val = expand_iter(g.IterOver)
  History = []
  
  for i, val in enumerate(iter_val):
    # add common options
    input = deepcopy(g.Common)
    # for each job, add their own special options
    for j, key in enumerate(iter_key):
      if not key[0] in input.keys():
        input[key[0]] = {}
      input[key[0]][key[1]] = val[j]
    # special option for the first run
    if i == 0:
      for item in g.First:
        if not item[0] in input.keys():
          input[item[0]] = {}
        input[item[0]][item[1]] = item[2]
    else: # special option from previous run
      for item in g.FromPrevious:
        if not item[0] in input.keys():
          input[item[0]] = {}
        input[item[0]][item[1]] = getattr(out, item[2])
    
    sys.stdout.flush()
    fout.write("\n---------- Entering Job %03d ----------\n" % i)
    inp, out = dmft(input, fout)
    # FIXME this is a hack, should come up with better ways to deal with it
    #if inp.HAMILTONIAN.Type == "Hubbard":
    #  empty = {}
    #  for k,v in inp.HAMILTONIAN.Params.items():
    #    if v != 0. or k in ["U", "V", "J", "t"]:
    #      empty[k] = v
    #  History = ResultTable(History, i, out, ToClass(empty))
    #else:
    #  History = ResultTable(History, i, out, ToClass({"System":inp.HAMILTONIAN.Name}))

    sys.stdout.flush()
