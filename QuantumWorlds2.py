import numpy as np
from random import choices
from matplotlib import pyplot as plt
import PIL
from copy import deepcopy

""" TODO:
- image input??
- make directions:
    1 2 3
    4 s 5
    6 7 8
- generalise/comment
- make entropy only a function
- add arbitrary distancing of positions

"""
N = 1

offsetRange = list(range(-N,N+1))
directions = list(range((((2*N)+1)**2)-1))

#directionOffsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
directionOffsets = [[x1, y1] for x1 in offsetRange for y1 in offsetRange]
directionOffsets.remove([0,0])


inp = [[ 0, 0, 0, 0, 0],
       [ 0, 3, 3, 3, 0],
       [ 0, 3, 3, 3, 0],
       [ 0, 0, 1, 0, 0],
       [ 0, 0, 1, 0, 0],
       [ 0, 0, 0, 0, 0]]
       
def loadImageInp(path):
    an_image = PIL.Image.open(path).convert("RGB")
    image_sequence = an_image.getdata()
    image_array = list(image_sequence)
    
    new_array = []
    colours = []
    
    for pix in image_array:
        if pix not in colours:
            colours.append(pix)
        
        new_array.append(colours.index(pix))      
    
    width, height = an_image.size
    #print(width, height)
    pixels = [new_array[i * width:(i + 1) * width] for i in list(range(height))]
    
    return pixels, colours
    
def subtile(pixels, tiledim):
    tiles = []
    out = []
    pixels = np.array(pixels)
    for x in range(0,len(pixels), tiledim):
        row = []
        for y in range(0,len(pixels[0]),tiledim):
            pixs = pixels[x:x+tiledim, y:y+tiledim].tolist()
            tileStrings = [str(t) for t in tiles]
            
            if str(pixs) not in tileStrings:
                tiles.append(pixs)
            
            tileStrings = [str(t) for t in tiles]
            row.append(tileStrings.index(str(pixs)))
        
        out.append(row)
    
    return out, tiles
            
def mosaic(inp, tiles):
    tiledim = len(tiles[0])
    out = np.zeros((len(inp)*tiledim,len(inp)*tiledim))
    
    for x in range(len(inp)):
        for y in range(len(inp[0])):
            out[x*tiledim:(x+1)*tiledim, y*tiledim:(y+1)*tiledim] = np.array(tiles[inp[x][y]])
            
    return out.astype(int).tolist()
            
    

def countStates(inp):
    NStates = max([item for sublist in inp for item in sublist]) + 1
    
    return NStates

def translate(world):
    dic = [[255,0,0],[0,0,255],[0,255,0]]
    
    out = [[0]*len(world)]*len(world)
    for i in range(len(world)):
        for j in range(len(world)):
            if world[i][j] == None:
                out[i][j] = [0,0,0]
            else:
                out[i][j] = dic[world[i][j]]
    
    return out

def extractRules(inp): 
    global N
    """
         [ self, [direction states], w]
        
    """
    
    xdim = len(inp)
    ydim = len(inp[0])
    
    rules = []
    
    for x in range(N, xdim-N):
        
        for y in range(N, ydim-N):
            rule = []
            rule.append(inp[x][y])
            print(inp[x][y])
            rule.append([inp[x+xo][y+yo] for xo, yo in directionOffsets])
            rules.append(rule)
            
    #print("Rules are: ", rules)
    rules = [i + [rules.count(i)] for i in [x for i, x in enumerate(rules) if i == rules.index(x)]]

    return rules

class world:
    def __init__(self, dim, states):
        self.dim = dim
        self.NStates = states
        self.tiles = [[cell(states, self,j,i) for i in range(dim)] for j in range(dim)]
        self.iterations = 0
        self.backups = [deepcopy(self.tiles)]
        
    
    def generate(self, rules):
        self.iterations += 1
        leftToCollapse = []
        
        for x in range(N,self.dim-N):
            leftToCollapse = leftToCollapse + [i for i in self.tiles[x][N:self.dim-N] if i.iterations < self.iterations]
        
        leftToCollapse.sort(key=getEntropy)
        
        x = leftToCollapse[0].x
        y = leftToCollapse[0].y
        
        justBackedUp = False
        backedUpAt = 0.0
        
        while len(leftToCollapse) > 0:
            ltcPercent = round(100 - (len(leftToCollapse)*100/((self.dim-(2*N))**2)),2)
            
            if justBackedUp:
                backedUpAt = ltcPercent
        
            if ltcPercent > backedUpAt + 10 and not justBackedUp:
                print("BACKING UP...")
                backedUpAt = ltcPercent
                self.backups.append(deepcopy(self.tiles))
            
            if int(ltcPercent*100) % 4 == 0:
                print("Generating: ", ltcPercent, "%", " Iteration: ", self.iterations)
                
            x = leftToCollapse[0].x
            y = leftToCollapse[0].y
            
            try:
                self.tiles[x][y].collapse(rules)
                justBackedUp = False
                
            except IndexError: 
                print("GOING BACK...")
                try:
                    self.tiles = self.backups.pop()
                except IndexError:
                    self.tiles = [[cell(self.NStates, self,j,i) for i in range(self.dim)] for j in range(self.dim)]
                justBackedUp = True
       
            leftToCollapse = []
            for x in range(N,self.dim-N):
                leftToCollapse = leftToCollapse + [i for i in self.tiles[x][N:self.dim-N] if i.iterations < self.iterations]
        
            leftToCollapse.sort(key=getEntropy)
            
   
    
    def draw(self):
        out = []
        for x in range(N,self.dim-N):
            out = out + [[i.isCollapsed for i in self.tiles[x][N:self.dim-N]]]
        return out
        
    def fullyConstrained(self):
        return all([len(self.tiles[i][j].states) == 1 for i in range(N,self.dim-N) for j in range(N, self.dim-N)])
        
    def levelOfConstraint(self):
        stateLengths = [len(self.tiles[i][j].states) for i in range(N,self.dim-N) for j in range(N, self.dim-N)]
        #print(stateLengths)
        return 1/(sum(stateLengths)/float((self.dim-(2*N))**2))
        
def getEntropy(i):
    global rules
    return i.entropy(rules)
    
def mapColours(inp, cMap):
    out = []
    for x in range(len(inp)):
        row = []
        for y in range(len(inp[0])):
            row.append(cMap[inp[x][y]])
        
        out.append(row)
    
    return out
    

class cell:
    def __init__(self, states, world, x, y):
        self.states = list(range(states))
        self.NStates = states
        self.isCollapsed = None
        self.world = world
        self.iterations = 0
        self.x = x
        self.y = y
    
    def entropy(self, rules):
        myStateRules = [r for r in rules if r[0] in self.states]
        weights = [0]*self.NStates
       
        for r in myStateRules:
            weights[r[0]] += r[2]
            
        
        
        weights = [weights[i] for i in self.states]
        #print(weights)
        
        
       
        try:
            weights = [i/sum(weights) for i in weights]
        except:
            print("myStateRules: ", myStateRules)
            print("weights: ", weights)
            print("self.states: ", self.states)
        
        entropy = -sum([np.log2(w+1e-10)*w for w in weights])
        
        return entropy
    
    def collapse(self, rules):
        
        ## choose a state
        #print(rules)
        myStateRules = [r for r in rules if r[0] in self.states]
        weights = [0]*self.NStates
       
        for r in myStateRules:
            weights[r[0]] += r[2]
        
        weights = [weights[i] for i in self.states]
        
        #print(weights)
        #print(self.states)
         
        
        self.isCollapsed = choices(self.states, weights=weights)[0]
        self.iterations += 1
        
        neis = [self.world.tiles[self.x+xo][self.y+yo] for xo, yo in directionOffsets]
        
        ## propagate restrictions to neighbours
        for direc in directions:
            xoff, yoff = directionOffsets[direc]
            myRules = [rule for rule in rules if rule[0] == self.isCollapsed]
            relRules = [rule for rule in myRules if all([neis[d].isCollapsed == rule[1][d] or neis[d].isCollapsed == None for d in directions])]
            canBe = set([rule[1][direc] for rule in relRules])
            allStates = set(list(range(self.NStates)))
            cantBe = list(allStates - canBe)
            nei = self.world.tiles[self.x+xoff][self.y+yoff]
            for ele in cantBe:
                if ele in nei.states:
                    nei.states.remove(ele)
        
        
      

inp, colourMap = loadImageInp("flowers.png")
inp, tilemap = subtile(inp, 1)
# print(inp)
plt.imshow(inp)
plt.show()
# plt.imshow(mosaic(inp, tilemap))
# plt.show()
# print(tilemap)
# plt.imshow(inp)
# plt.show()

#quit()
#print(inp)

N_STATES = countStates(inp)
rules = extractRules(inp)
print(rules)
WORLD_DIM = 20

done = False
tries = 0

wrld = world(WORLD_DIM, N_STATES)
# print(tries, " Tries")
# tries += 1

N_ITERATIONS = 1
n=0

wrld.generate(rules)
print("Level of Constraint: ", wrld.levelOfConstraint())

done = True



# while not done:
    # try:
        # wrld = world(WORLD_DIM, N_STATES)
        # print(tries, " Tries")
        # tries += 1
        # for i in range(N_ITERATIONS):
            # wrld.generate(rules)
        # done = True
    # except IndexError:
        # pass
#print(translate(wrld.draw()))
plt.imshow(mapColours(mosaic(wrld.draw(), tilemap), colourMap))
plt.show()


