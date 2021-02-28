from qutiepy import *

WORLD_DIM = 5
BLOCK_BITS = 2
BLOCK_STATES = 2**BLOCK_BITS
WORLD_LEN = (WORLD_DIM ** 2) * BLOCK_BITS
WORLD_STATES = 2 ** WORLD_LEN


#world = register(WORLD_LEN)

#world = hw(world)

## states 0= edge case, 1=water, 2=grass, 3=forest

"""
RULES:

- edges must be state 0
- water must be next to at >1 water    
- grass must be next to water
- forest must be next to grass and >1 forest

- [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15] 

"""

class stateSpace():
    def __init__(self, NBits):
        self.bits = [None]*NBits
    
    def excludeBitPatterns(patterns):
        pass


def getBadStates(x0, y0, x1, y1, s0, s1):
    i0 = 2*(x0 + y0*WORLD_DIM)   #first bit of pair
    i1 = 2*(x1 + y1*WORLD_DIM) 
    badStates = []
    
    
    soi1 = range(i0-1,i0+3)
    soi2 = range(i1-1,i1+3)
    
    for i in soi:
        print()
        if bin(i).lstrip("0b")[i0:i0+2] == bin(s0).lstrip("0b") and bin(i).lstrip("0b")[i1:i1+2] == bin(s1).lstrip("0b"):
            badStates.append(i)
            print(bin(i))
    
    return badStates

def removeStatesWhere(world, states):
    for state in states:
        world.amps[state] = 0
    

def genSkipBits(dim, yesBits):
    allBits = set(range(dim))
    yesBits = set(yesBits)
    return list(allBits - yesBits)

def offsetsToIndices(i, yoff, xoff):
    target = i + xoff + (yoff*WORLD_DIM)
    return [target, target+1, i]     ## for use with CCNOT o.e. on two target bits representing the target block.

def constructGateSet(gate, inputOffsetX, inputOffsetY):
    i = 0
    gateSet = gate(WORLD_LEN, skipBits=genSkipBits(offsetsToIndices(i, inputOffsetX, inputOffsetY)))
    
    for i in range(1,((WORLD_DIM-2)**2) - 1):
        gateSet = gateSet(gate(WORLD_LEN, skipBits=genSkipBits(offsetsToIndices(i, inputOffsetX, inputOffsetY))))

print(getBadStates(2,3,3,3,1,2))