#!/usr/bin/env python3
from argparse import ArgumentParser
from enum     import Enum
import numpy  as np
import random as randm
import Canvas
import time
import sys
from os import system as systm
# wtf-8
# z-z-z
################################################################
# poom
def gerop(): ############################################# GEROP
  ''' Ger Options '''
  pas = ArgumentParser()
  pas.add_argument(
    '-d',
    '--debug',
    help='Turn on debuging flag.',
    action='store_true')
  pas.add_argument(
    'm',
    type=int,
    default=6,
    help='Number of horizontal squares.',
    nargs='?')
  pas.add_argument(
    'n',
    type=int,
    default=6,
    help='Number of vertical squares.',
    nargs='?')
  pas.add_argument(
    '-j',
    '--entry',
    dest='j',
    type=int,
    default=3,
    help='Entry column, 1 <= J <= n.')
  pas.add_argument(
    '-N',
    '--max-no-jump',
    dest='N',
    type=int,
    default=10,
    help='Number of maximum "no-jump" squares.')
  return pas.parse_args()  
################################################################
#
class Sqr(Enum): ########################################### SQR
   Fog   = (':', '#91a3b2', '#031354', Canvas.Brush(':'))
   Wall  = ('@', '#821520', '#c32135', Canvas.Brush('@'))
   Space = ('-', '#354120', '#abcdef', Canvas.Brush('-'))
################################################################
#
def frame(ryShape, ryVal, fSize, fVal): ################## FRAME
  ''' framing an array
  ryShape - array shape
  ryVal   - array fill value
  fSize   - frame size
  fVal    - frame fill value
  '''
  ry = np.full(ryShape, ryVal)    
  fr = np.full((fSize, 1), fVal)
  ry = np.insert(ry, ry.shape[1], fr, axis=1) 
  ry = np.insert(ry, ry.shape[0], fr, axis=0) 
  ry = np.insert(ry, 0, fr, axis=1) 
  ry = np.insert(ry, 0, fr, axis=0) 
  return ry
################################################################
#
class Coor(tuple): ######################################## COOR
  def __add__(self, othr): ############################# __ADD__
    return Coor([sum(j) for j in zip(self, othr)])
  ##############################################################
  #
################################################################
#
class Globus(Enum): ##################################### GLOBUS
  North = Coor((-1,  0))
  East  = Coor(( 0,  1))
  South = Coor(( 1,  0))
  West  = Coor(( 0, -1))
  @staticmethod
  def ls(): return [x.value for x in Globus] # stackoverflow.com
################################################################
#
def probbty(cont): ##################################### PROBBTY
  ''' It could be linear (or any) function of cont '''
  global args
  cont /= args.N
  return cont**2
################################################################
#
def jump_(p): return (randm.uniform(0, 1) < p) ########### JUMP_
#
################################################################
#
class Maze: ############################################### MAZE
  def __init__(self, m, n, j, offset=(10, 10)): ####### __INIT__
    self.ry = frame((m, n), Sqr.Fog, 1, Sqr.Wall)
    self.j  = j
    self.ry[0, j] = self.ry[1, j] = Sqr.Space
    self.offset = offset
    self.path = np.zeros(self.ry.shape, dtype=int)
    self.path[1, j] = 1 # path counter for treasure location
  ##############################################################
  #
  def __repr__(self): ################################# __REPR__
    bf = []
    for row in self.ry:
      bf.append(' '.join([x.value[0] for x in row]))
    return '\n'.join(bf)
  ##############################################################
  # nous - Hue
  def _cycle(self, r): ################################## _CYCLE
    ''' ck if clearing the sqr at r will create a cycle '''
    cont = 0
    Mz = self.ry
    for dr in Globus.ls():
      cont += (Mz[r + dr] == Sqr.Space)
      if cont > 1: return True
    return False
  ##############################################################
  #
  def getvar(self, r): ################################## GETVAR
    ''' get all possible directions '''
    bf = []
    Mz = self.ry
    for dr in Globus.ls():
      r1 = r + dr
      if Mz[r1] != Sqr.Fog: continue
      if self._cycle(r1):
        Mz[r1] = Sqr.Wall
        sys.stdout.flush()
        time.sleep(0.5)
        self.DrawSqr(r1)
      else:
        bf.append(dr)
    return bf
  ##############################################################
  #
  def Build(self): ####################################### BUILD
    global args
    if args.debug: import pdb; pdb.set_trace()
    # [0 Init.]
    r = Coor((1, self.j))
    föok = {r} # fork set
    cont = 0 # path counter
    Mz = self.ry
    while True:
      # [1 Rien ne va plus.]
      bf = self.getvar(r)
      randm.shuffle(bf)
      # [2 TP(TelePort)]
      if bf:
        föok.add(r) # add r to fork set
        P = probbty(cont)
      else:
        P = 1
      # [3 JMP]
      if jump_(P):
        if not bf: föok.discard(r)
        if not föok: return # Exit
        r = randm.choice(tuple(föok.difference({r})))
        cont = 0
        continue
      # [4 Just gou]
      dr = bf.pop()
      if not bf: föok.discard(r)
      r1 = r + dr
      self.path[r1] = self.path[r] + 1
      r = r1
      Mz[r] = Sqr.Space
      self.DrawSqr(r)
      sys.stdout.flush()
      time.sleep(0.5)
      cont += 1      
  ##############################################################
  #
  def DrawSqr(self, r): ################################ DRAWSQR
    ''' r = i, j (numpy index) corresponds to y, x terminal
    coordinates, that is i is y, j is x.
    '''
    # Unpack index and get square data.
    i, j = r
    sq = self.ry[i, j]
    # Calculate y and x terminal coordinates.
    y = self.offset[1] + i
    x = self.offset[0] + j
    # Get square paint tools.
    dmmy, fgr, bgr, brush = sq.value
    # Set drawing attributes
    fgr = Canvas.hex2rgb(fgr)
    bgr = Canvas.hex2rgb(bgr)
    sgr = [Canvas.SGR.Rndm()]
    fgr = [Canvas.RndmClr(fgr)]
    bgr = [Canvas.RndmClr(bgr)]
    brush.Draw((x, y), sgr, fgr, bgr, True)
  ##############################################################
  # set offset field
  def Draw(self): ######################################### DRAW
    ''' For drawing the initial Maze '''
    for r in np.ndindex(*self.ry.shape):
      self.DrawSqr(r)
  ##############################################################
  #
################################################################
#
if __name__ == '__main__': ################################ VROO
  args = gerop()
  if args.debug:
    print('Debuging: On')
  else:
    print('Debuging: Off')
  print('m = ', args.m, '\n',
        'n = ', args.n, '\n',
        'j = ', args.j, '\n',
        'N = ', args.N, sep='')
  Mz = Maze(args.m, args.n, args.j, offset=(1, 1))
  systm("clear")
  Mz.Draw()
  print(Canvas.CSI('?25'), end = 'l')
  Mz.Build()
  trsr = np.unravel_index(Mz.path.argmax(), Mz.path.shape)
  y = Mz.offset[1] + trsr[0]
  x = Mz.offset[0] + trsr[1]
  Canvas.Display(x, y, [5], [100,100,0], [200,200,0], 'T')
  Mz.DrawSqr((args.m + 1, args.n + 1))
  print()
  print(Canvas.CSI('?25'), end = 'h')
  print(trsr)
################################################################
# * add delay option
# * add path counter (ck)
# * args.m and n are reversed
# * add to Canvas show and hide cursor functions
# * after finishing building maze convyort fog to wall
# and draw ze treasure
