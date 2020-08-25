#!/usr/bin/env python3
########Ö#######ö#######O#######ò#######o#######6#######0#######
#       |       |       |       |       |       |       |       
def CSI(s): return '\033[' + s #|####Control#Sequence#In|roducer
#       |       |       |       |       |       |       |
def SgrStr(sgr): #######|####get#Select#Graphic#Rendition#string
  """ ##|#######|#######'#######,#######¦#######¦#######'#######
  infut: sgr: list
  oufut: formated string 4 colored oufut
  xmple: ';5' <- SgrStr([5]) (Slow Blink)
  notes: if passing single tuple add trailing comma e.g.: (1,)
  """ ##|#######,#######|#######¦#######,#######'#######¦#######
  return (';{}'*len(sgr)).format(*sgr) #|       |       |
#       ¦       |       |       |       |       |       |
def ClrStr(type, clr): #|###############|#######get#color#string
  """ ##'#######¦#######|###############¦#######¦#######:#######
  infut: type: 0 - foreground
               1 - background
          clr: rgb list
  oufut: formated string for colored oufut
  xmple: ';48;2;120;30;40' <- ClrStr(1, (120, 30, 40))
  notes: clr can be empty (for transparent background)
  """ ##|#######|#######¦#######'#######|#######;#######|#######
  if len(clr): #¦       |       |       '       ¦       |
    return f";{type + 3}8;2" + SgrStr(clr) #    |    do the math
  else: ########,#######|#######¦#######¦#######|#######|#######
    return '' # |       ¦       :       |no colo'r (tran|parent)
#       |       '       |       ¦       |       |       |
def Display(x, y, sgr, fgr, bgr, t): ###|#######¦#######'#######
  """ ##'#######|#######¦#######,#######`#######|#######|#######
  infut: (x, y): coorz
         (fgr, bgr): RGB colorz
         sgr: list
         t: text
  oufut: void
  xmple: Display(1,2,[1,3],[10,150,230],[30,15,70],'xa-xa')
  notes: void
  """ ##|#######¦#######'#######|#######,#######¦#######'#######
  fgr = ClrStr(0, fgr) #,       '       ¦       |       ¦
  bgr = ClrStr(1, bgr) #'       `       .       |       '
  sgr = SgrStr(sgr) #   |       |       |       |       ¦
  print(CSI(f'{y};{x}H'), #     '       ,       ;       :
        CSI(f'{fgr}{bgr}{sgr}m'), #     ¦       |       '
        t, #    |       "       |       |       |       |
        sep='', #       |       ,       |       "       |
        end=CSI('m')) # ¦       '       "       ¦       |
#       |       '       ¦       '       ¦       |       |
class Texture: #|#######¦#######|#######|#######;#######¦#######
  font = {} #   |       "       |       |       '       "
  font['W'] = 'wŵώW' #  |       ¦       ¦       |       ,
  font['G'] = 'gGģĝ' #  ¦       ,       |       |       |
  font['Ø'] = 'øØöò' #  |       ¦       ¦       '       ¦
  font['P'] = 'ƿPþp' #  ¦       |       |       `       "
  font['B'] = 'ɓbBƂ' #  `       ¦       ;       ,       ¦
  font['L'] = 'LĹlľ' #  |       |       |       ;       |
  font['M'] = 'mMЖµ' #  ¦       |       '       .       |
  font['C'] = 'cĆčC' #  |       '       ¦       ;       "
  font['?'] = '?%"&' #  '       ¦       |       ,       `
  font[':'] = ':;„⌄' #  ¦       '       ;       ¦       |
  font['@'] = '@åæ¾' #  |       `       ¦       ;       ,
  font['-'] = '-_‾=' #  ¦       "       ,       ;       `
#       |       '       ¦       ¦       .       |       :
from numpy.random import choice ########|#######'#######|#######
#       ¦       |       |       '       ¦       |       |
def GetRndmTxtr(patron): #######¦#######|#######'#######,#######
  """ BØØW! -> böøώ? """ #######|#######¦#######"#######|#######
  bufr = [] # buffer as textures container      ¦       |
  for key in patron: # patron loop      '       |       ¦
    if key not in Texture.font: key = '?' # unknown key |
    txtr = Texture.font[key] #  |       |       ¦       ¦
    bufr.append(txtr[choice(len(txtr))]) #      ,       `
  return ''.join(bufr) #,       ¦       '       |       ¦
#       |       '       ¦       |       `       .       :    
from numpy import add, resize, array ###,#######|#######'#######
#       |       ¦       '       :       |       ¦       "
def resiz(ry, newsiz): #¦#######|#######¦#######,#######¦#######
  """ ##|#######|#######,#######'#######;#######:#######|#######
  Innput: ry     - array
          newsiz - what is this?
  Exmple: [[1,2],[3,4],[1,2]] <- resiz([[1,2],[3,4]], 3)
  """ ##,#######:#######;#######|#######|#######¦#######¦#######
  ry = array(ry, dtype=object) #|    sgr|((5,), ¦1, 2)) |       
  newshape = list(ry.shape) #   ¦       |       '       ¦
  newshape[0] = newsiz #¦       '       "       |       |
  return list(resize(ry, newshape)) #   ¦       `       ;
#       ¦       ¦       |       ,       ;       ¦       |
from numpy.random import normal ########|#######'#######"#######
#       '       "       ,       |       ¦       ¦       '
def RndmRGB(m, s): #####|#######"#######"#######,#######|#######
  """ normal rv with [0, 255] cut """
  clr = int(normal(m, s)) # take the floor value|       '
  if clr < 0:   clr = 0   # cut below values    '       |
  if clr > 255: clr = 255 # cut above values    ¦       ¦
  return clr #  |       "       '       |       `       ¦
#       ¦       |       '       ¦       "       ¦       |
def RndmClr(mean, sigma=(5, 6, 4)): ####|#######|#######"#######
  return list(map(RndmRGB, mean, sigma)) #      '       |
#       |       |       |       |       |       '       |
class SGR(): #  ¦       ,       ¦       |       ;       .
  Bold, Ital, Norm = 1, 3, 22 # |       '       ¦       ¦
  List = array(((Norm,), (Bold,), (Ital,), #    |       ¦
                (Bold, Ital)), dtype=object) #  ¦       '
  P, Q = 1/3, 2/9 # List probabilities  |       ,       :
  Prob = (P, Q, Q, Q) # |       `       ;       `       |
  @staticmethod #       ¦       |       |       ,       .
  def Rndm(): return choice(SGR.List, p=SGR.Prob) #     |
#       |       ¦       ¦       '       ¦       |       '
class Brush: ###|#######|#######|#######"#######,#######|#######
  def __init__(self, patron): ##¦#######¦#######|######string###
    self.patron = patron #      '       |       "       |
    self.size   = len(patron) # `       ¦       ,       `
    self.dr     = (1, 0) # direction    |       '       '
  #     |       |       |       |       ¦       ¦       |
  def Draw(self, r, sgr, fgr, bgr, rndm=False): ########|#######
    sgr = resiz(sgr, self.size) #       '       `       "
    fgr = resiz(fgr, self.size) #       "       ¦       `       
    bgr = resiz(bgr, self.size) #       |       |       ¦
    if rndm: #  ¦       ¦       |       ¦       '       "
      patron = GetRndmTxtr(self.patron) #       ,       ;
    else: #     |       |       |       '       |       ¦
      patron = self.patron #    ¦       ¦       ¦       |
    for j, p in enumerate(patron): #    |       |       '
      Display(*r, sgr[j], fgr[j], bgr[j], p) #  ¦       "
      r = add(r, self.dr) #     |       ,       |       ¦
    return r #  ¦       ¦       :       ;       }       }
#       |       '       |       ;       ,       ¦       ¦
def hex2rgb(hx): #######|#######|#######:#######|#######'#######
  # yee "       |       ¦       ¦       ¦       ¦       "
  hx = hx[1:] # ¦       '       ;       ,       ¦       ¦
  hx = [''.join(p) for p in zip(*[iter(hx)]*2)] # o_o   |
  return tuple(map(lambda x: int(x, base=16), hx)) #    ¦
#       ¦       |       '       ¦       |       ,       :
def HideCursor(): print(CSI('?25'), end = 'l') #|#######¦#######
#       |       ¦       ¦       |       '       |       ' 
def ShowCursor(): print(CSI('?25'), end = 'h') #¦#######|#######
if __name__ == "__main__":
#  import pdb; pdb.set_trace()
  b = Brush('W')
  sgr = [(SGR.Norm,)]
  fgr = [(10, 20, 150)]
  bgr = [(20, 30, 50)]
  r   = (30, 10)
  for j in range(10):
    r = b.Draw(r, sgr, fgr, bgr, rndm=True)
    sgr[0] = SGR.Rndm()
    fgr[0] = RndmClr(fgr[0])
    bgr[0] = RndmClr(bgr[0])
  print()
################################################################
#
