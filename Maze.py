#!/usr/bin/env python3
from argparse import ArgumentParser
from enum     import Enum
import numpy  as np
import random as randm
import Canvas
import time
import sys
from os       import system as systm
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
    help='Number of vertical squares.',
    nargs='?')
  pas.add_argument(
    'n',
    type=int,
    default=6,
    help='Number of horizontal squares.',
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
  pas.add_argument(
    '-s',
    '--sleep',
    dest='s',
    type=float,
    default=0.1,
    help='Time between two brush strokes.')
  return pas.parse_args()
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
class Sqr(Enum): ########################################### SQR
   Fog   = (':', '#91a3b2', '#031354', Canvas.Brush(':'))
   Wall  = ('@', '#821520', '#c32135', Canvas.Brush('@'))
   Space = ('-', '#354120', '#abcdef', Canvas.Brush('-'))
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
  ''' It could be linear(or any) function of cont '''
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
  global args
  def __init__(self, offset=(10, 10)): ################ __INIT__
    self.ry = frame((args.m, args.n), Sqr.Fog, 1, Sqr.Wall)
    j = args.j
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
        time.sleep(args.s)
        self.DrawSqr(r1)
      else:
        bf.append(dr)
    return bf
  ##############################################################
  #
  def Build(self): ####################################### BUILD
    if args.debug: import pdb; pdb.set_trace()
    # [0 Init.]
    r = Coor((1, args.j))
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
      time.sleep(args.s)
      cont += 1      
  ##############################################################
  #
  def DrawSqr(self, r): ################################ DRAWSQR
    ''' r = i, j corresponds to y, x terminal coordinates. '''
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
    ''' Draw all Maze squares. '''
    for r in np.ndindex(*self.ry.shape):
      self.DrawSqr(r)
  ##############################################################
  #
  def GetTrsr(self): ################################### GETTRSR
    ''' This class won't win a Design Award! '''
    Mz = self.ry
    # convyort left fog squares to wall
    for r in np.ndindex(Mz.shape):
      if Mz[r] == Sqr.Fog:
        Mz[r] = Sqr.Wall
        self.DrawSqr(r)
    trsr = np.unravel_index(self.path.argmax(), self.path.shape)
    y = self.offset[1] + trsr[0]
    x = self.offset[0] + trsr[1]
    Canvas.Display(x, y, [5], [100,100,0], [200,200,0], 'T')
    r = Coor(trsr)
    while self.path[r] > 0:
      for dr in Globus.ls():
        r1 = r + dr
        if self.path[r1] == self.path[r] - 1:
          y = self.offset[1] + r1[0]
          x = self.offset[0] + r1[1]
          Canvas.Display(x,y,[5],[10,100,240],[50,30,80],'»')
          sys.stdout.flush()
          time.sleep(args.s)
          r = r1
          break
    return trsr
  ##############################################################
  #
################################################################
#
if __name__ == '__main__': ################################ VROO
  args = gerop()
  Mz = Maze(offset=(1, 1))
  systm("clear")
  Mz.Draw()
  Canvas.HideCursor()
  Mz.Build()
  trsr = Mz.GetTrsr()
  # Position the cursor below the Maze.
  print(Canvas.CSI(f'{args.m + 4};0'), end='H')
  print(trsr)
  Canvas.ShowCursor()
################################################################
# 
