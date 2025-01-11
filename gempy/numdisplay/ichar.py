import numpy as np
import os

def read_inc():
    """Read the iraf include file that defines the bit patterns for characters used
    for image display"""
    f = open('pixelfont.inc')
    lines = f.readlines()
    cdict = {}
    while len(lines) > 6:
        ic, bits, lines = nextchar(lines)
        cdict[ic] = bits
    return cdict

def nextchar(lines):
    """read next character data in file, and return numeric array 5x7 pixels"""
    if not lines[0].startswith('data'):
        raise ValueError
    else:
        clist = [lines[0][lines[0].find('/')+2:-3]]
        for i in range(1,7):
            clist.append(lines[i][:lines[i].find('B')].strip())
        ic = lines[6][lines[6].find('#')+1:].strip()
    carr
    return ic, clist, lines[8:]

def initichar_old():
    '''read the data file (old form) and generate the dict for numdisplay to use'''
    f = open('ichar_old.dat')
    fstr = f.read()
    cdict = eval(fstr)
    for key in cdict:
        clist = cdict[key]
        alist = []
        for item in clist:
            alist.append([int(c) for c in item])
        alist.reverse()
        cdict[key] = np.where(np.array(alist, dtype=np.bool_))
    return cdict

def initichar():
    '''read the data file (old form) and generate the dict for numdisplay to use'''
    filepath = os.path.split(__file__)[0]
    f = open(filepath+'/ichar.dat')
    fstr = f.read()
    cdict = eval(fstr)
    return cdict

def expandchar(indices, size):
    '''block replicate a character. This implementation is inefficient
    but probably won't matter unless many large characters are used'''
    iy, ix = indices
    ox = np.zeros((size, size, len(indices[0])))
    oy = ox.copy()
    for i in range(size):
        for j in range(size):
            ox[i, j, :] = size*ix + i
            oy[i, j, :] = size*iy + j
    return (np.array(oy.flat, dtype=np.int32), np.array(ox.flat, dtype=np.int32))
