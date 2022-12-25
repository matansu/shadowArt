import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.path as matpath
import svgpathtools as svg
import importlib
from imageUtils import *
import scipy
from skimage import measure
from timeit import timeit
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import svgpathtools as svg
from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
import re
import math

from botCoord import *

#image to vector functions

class MyPath(object):
    def __init__(self, path, straightBot, imBotCoord):
        self.path = path
        self.straightBot = straightBot
        
        #calculate directions
        self.dirs = np.diff(path,axis=0)
        self.nDirs = len(self.dirs)
        #calculate sums
        x = np.append(0,path[:,0] - path[0,0])
        y = np.append(0,path[:,1] - path[0,1])
        self.sumx = np.cumsum(x)
        self.sumy = np.cumsum(y)
        self.sumxy = np.cumsum(np.multiply(x,y))
        self.sumx2 = np.cumsum(np.square(x))
        self.sumy2 = np.cumsum(np.square(y))
        
        if(straightBot):
            xs = path[:-1,0]
            ys = path[:-1,1]
            minY = np.min(ys)
            
            botXs = xs[ys==minY]
            maxBotInd = np.max(np.where(ys==minY))
            minBotX = np.min(botXs)
            maxBotX = np.max(botXs)

            addHalfs = False
            pathBotCoord = calcBotCoord(xs,ys,minY,addHalfs)

            botCoordsEqual = np.array_equal(imBotCoord.botSegments - imBotCoord.botSegments[0,0],pathBotCoord.botSegments - pathBotCoord.botSegments[0,0])

            if not botCoordsEqual:
                print('issue of botCoords euqality - before and after warp')
                assert(0)

            self.minY = minY
            self.minBotX = minBotX
            self.maxBotX = maxBotX
            self.botCoord = pathBotCoord
            corInds = []
            for pair in self.botCoord.botSegments:
                for x in pair:
                    ind = np.argwhere(np.logical_and(path[:,0] == x,path[:,1] == minY))
                    corInds.append(ind[0][0])

            assert(len(corInds) % 2 == 0) #number of bot corners must be EVEN
            self.botCoord.corInds = corInds

#define 4 steps
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def areaFour(img,coord):
    y = int(coord[0,1]-0.5)
    x = int(coord[0,0]-0.5)
    return img[y:y+2,x:x+2]

def nextDir(area):
    assert(area[0,0]==0)
    assert(area[0,1]==1)
    if(area[1,0]==1):
        return LEFT
    if(area[1,1]==1):
        return UP
    return RIGHT

#find one path in image
def getPathInImg(imgIn):    
    [ys,xs] = np.asarray(imgIn==0).nonzero()
    topC = np.min(ys)
    leftC = np.min(xs[ys==topC])
    dirs = np.array([[0,1],[1,0],[0,-1],[-1,0]])

    #start in our pixel
    coord = np.array([[leftC-0.5,topC-0.5]])
    path = copy.deepcopy(coord)
    direction = RIGHT

    it = 0
    maxIters = 200000
    while it<maxIters and (path.shape[0]<3 or coord[0,0]!=path[0,0] or coord[0,1]!=path[0,1]):
        it = it + 1
        step = dirs[direction,:]
        coord = coord + step
        path = np.append(path, coord, axis = 0)
        currSquare = areaFour(imgIn,coord)
        rotSquare = np.rot90(currSquare,-direction)
        direction = (nextDir(rotSquare)+direction) % 4
    
    assert(it<maxIters)
    return path

#iterate on an image until we find and return all the paths
def getAllPathsInImg(imgIn):
    currImg = copy.deepcopy(imgIn)
    currImg = np.where(currImg < 0.5, 0, 1)
    paths = []

    while((currImg==0).sum()!=0):
        path = getPathInImg(currImg)
        paths.append(path)
        mpath = matpath.Path(path)
        # select points included in the path
        x, y = np.mgrid[:currImg.shape[1], :currImg.shape[0]]
        points = np.vstack((x.ravel(), y.ravel())).T
        maskIn = mpath.contains_points(points).reshape(x.shape).T
        mask1 = currImg==1
        mask0 = currImg==0
        #flip color of points in the path
        currImg[np.bitwise_and(maskIn,mask0)] = 1
        currImg[np.bitwise_and(maskIn,mask1)] = 0
    
    return paths

#return l1 max dist between line and point
def l1dist(a,b,c,x,y):
    if(a==0):
        assert(b==1)
        return abs(y+c)
    if(b==0):
        assert(a==1)
        return abs(x+c)
    assert(b==1)
    c1 = y - x
    c2 = y + x
    if(a!=-1):
        x1 = -(c1 + c)/(a + 1)
        y1 = x1 + c1
        dist1 = max(abs(x1-x),abs(y1-y))
    if(a!=1):
        x2 = -(c2 + c)/(a - 1)
        y2 = -x2 + c2
        dist2 = max(abs(x2-x),abs(y2-y))
    if(a==1):
        return dist1
    if(a==-1):
        return dist2
    return min(dist1,dist2)

#convert 2 points to a line of the form "a*x + b*y + c = 0"
def points2line(p1,p2):
    assert((p1!=p2).any())
    #if x is constant
    if(p1[0]-p2[0]==0):
        return 1,0,-p1[0]
    #if y is constant
    if(p1[1]-p2[1]==0):
        return 0,1,-p1[1]
    #else
    a = -(p1[1]-p2[1])/(p1[0]-p2[0])
    c = -a*p1[0]-p1[1]
    return a,1,c

code2dirs = np.array([[LEFT],[DOWN],[UP],[RIGHT]])

def calc3Dirs(pathObj):
    #ugly but fast code
    #computes lastDirs[i], is the farthest index from i that we didn't see all directions
    dirsCoded = np.array([[(pathObj.dirs[:,0]*3 + pathObj.dirs[:,1] + 3)/2]]).reshape(1, -1)
    dirsMat = np.concatenate([dirsCoded==0,dirsCoded==1,dirsCoded==2,dirsCoded==3],axis=0).astype(int)
    dirsCum = np.cumsum(dirsMat,axis=1)
    dirsCumShift = np.concatenate([dirsCum[:,0:1],dirsCum[:,0:-1]],axis=1).reshape(-1,1,order='F')

    dirsCumM = np.tile(dirsCum,(pathObj.nDirs,1))
    dirsCumM = dirsCumM - dirsCumShift;
    dirsCumM[dirsCumM<0] = 0

    first = np.argmax(dirsCumM!=0,axis=1)
    first[first==0] = pathObj.nDirs
    
    firstM = first.reshape(-1,4)
    max3DirsInds = np.max(firstM,axis=1).reshape(1,-1)
    lastMissingDirsCoded = np.argmax(firstM,axis=1)
    
    missingDirs = code2dirs[lastMissingDirsCoded,:]
    missingDirs = np.concatenate((missingDirs,missingDirs[0,:].reshape(-1,1)),axis=0)
    
    return max3DirsInds,missingDirs

def getLastStraight(pathObj):
    #ugly but fast code
    nDirs = pathObj.nDirs

    last = np.zeros((1,nDirs), dtype=int)
    pathXs = np.tile(pathObj.path[:,0].T,(nDirs+1,1))
    pathYs = np.tile(pathObj.path[:,1].T,(nDirs+1,1))
    
    sbtrctXs = pathXs - pathXs.T
    sbtrctYs = pathYs - pathYs.T
    
    maxXs = copy.deepcopy(sbtrctXs)
    minXs = copy.deepcopy(sbtrctXs)
    maxYs = copy.deepcopy(sbtrctYs)
    minYs = copy.deepcopy(sbtrctYs)
    
    maxAddX1 = np.bitwise_and(sbtrctYs>=0,np.bitwise_or(sbtrctYs>0,sbtrctXs<0))
    maxAddY1 = np.bitwise_and(sbtrctXs<=0,np.bitwise_or(sbtrctXs<0,sbtrctYs<0))
    
    minAddX1 = np.bitwise_and(sbtrctYs<=0,np.bitwise_or(sbtrctYs<0,sbtrctXs<0))
    minAddY1 = np.bitwise_and(sbtrctXs>=0,np.bitwise_or(sbtrctXs>0,sbtrctYs<0))
    
    maxXs = np.where(maxAddX1, maxXs + 1, maxXs - 1)
    maxYs = np.where(maxAddY1, maxYs + 1, maxYs - 1)
    
    minXs = np.where(minAddX1, minXs + 1, minXs - 1)
    minYs = np.where(minAddY1, minYs + 1, minYs - 1)
    
    angles = np.arctan2(sbtrctYs, sbtrctXs) * 180 / np.pi
    anglesMax = np.arctan2(maxYs, maxXs) * 180 / np.pi
    anglesMin = np.arctan2(minYs, minXs) * 180 / np.pi
    
    numRots = 90*(pathObj.missingDirs - LEFT)

    angles    = angles + numRots+360
    anglesMax = anglesMax + numRots+360
    anglesMin = anglesMin + numRots+360
    
    angles[angles>180] = angles[angles>180] - 360
    anglesMax[anglesMax>180] = anglesMax[anglesMax>180] - 360
    anglesMin[anglesMin>180] = anglesMin[anglesMin>180] - 360
    
    #need to erase distx <= 1 and disty <= 1, or before index
    toErase = np.bitwise_and(np.fabs(sbtrctXs)<=1,np.fabs(sbtrctYs)<=1)
    toErase = np.bitwise_or(toErase,np.tril(np.ones(angles.shape, dtype=bool)))
    anglesMax[toErase] = -999
    anglesMin[toErase] =  999
    
    lessThanMax = angles < np.maximum.accumulate(anglesMax,axis=1)
    greaterThanMin = angles > np.minimum.accumulate(anglesMin,axis=1)
    
    firstLess = np.argmax(lessThanMax,axis=1)-1
    firstGreater = np.argmax(greaterThanMin,axis=1)-1
    
    #if all the row is False -> assign nDirs
    firstLess[np.all(lessThanMax==False,axis=1)] = nDirs
    firstGreater[np.all(greaterThanMin==False,axis=1)] = nDirs
    
    last = np.minimum(firstLess,firstGreater)[0:-1].reshape(1,-1)
    last = last[0,::-1].reshape(1,-1)
    last = np.minimum.accumulate(last,axis=1)
    last = last[0,::-1].reshape(1,-1)
    
    #only if we want straight bottom
    if(pathObj.straightBot):
        for ind in pathObj.botCoord.corInds:
            last[0,0:ind] = np.minimum(last[0,0:ind],ind)

    return last

def penalty(pathObj,i,j):
    path = pathObj.path
    x = pathObj.sumx[j+1] - pathObj.sumx[i]
    y = pathObj.sumy[j+1] - pathObj.sumy[i]
    xy = pathObj.sumxy[j+1] - pathObj.sumxy[i]
    x2 = pathObj.sumx2[j+1] - pathObj.sumx2[i]
    y2 = pathObj.sumy2[j+1] - pathObj.sumy2[i]
    k = j+1 - i
    
    xi = path[i,0] - path[0,0]
    yi = path[i,1] - path[0,1]
    xj = path[j,0] - path[0,0]
    yj = path[j,1] - path[0,1]
    
    px = (xi + xj)/2
    py = (yi + yj)/2
    ey = xj - xi
    ex = -(yj - yi) #todo: figure out why negative

    a = ((x2 - 2*x*px) / k + px*px)
    b = ((xy - x*py - y*px) / k + px*py)
    c = ((y2 - 2*y*py) / k + py*py)

    s = ex*ex*a + 2*ex*ey*b + ey*ey*c

    return np.sqrt(s)

def calcOptimalPolygon(pathObj):
    nDirs = pathObj.nDirs
    #now calculate optimal polygon
    #clip0
    clip0 = copy.deepcopy(pathObj.last)
    #following lines are probably wrong...
    #clip0[0] = pathObj.last[0] - 1
    #clip0[1:] = pathObj.last[:-1] - 1
    #clip0[clip0==nDirs-1] = nDirs

    #clip1
    clip1 = np.zeros(nDirs+1, dtype=int) 
    j = 1
    for i in range(0,nDirs):
        while (j <= clip0[i]):
            clip1[j] = i
            j = j + 1
    
    #seg0
    seg0 = np.zeros(nDirs+1, dtype=int)
    i = 0;
    j = 0;
    while i<nDirs:
        seg0[j] = i
        i = clip0[i]
        j = j+1
        
    seg0[j] = nDirs
    m = j
    
    pathObj.nLines = m
    
    #seg1
    seg1 = np.zeros(nDirs+1, dtype=int)
    i = nDirs;
    for j in range(m,0,-1):
        seg1[j] = i
        i = clip1[i]
        
    seg1[0] = 0
    
    #pen
    pen = np.zeros(nDirs+1)
    prev = np.zeros(nDirs+1, dtype=int)
    pen[0]=0;
    for j in range(1,m+1):
        for i in range(seg1[j],seg0[j]+1):
            best = -1;
            for k in range(seg0[j-1],clip1[i]-1,-1):
                thispen = penalty(pathObj, k, i) + pen[k]
                if (best < 0 or thispen < best):
                    prev[i] = k
                    best = thispen
            pen[i] = best;
    
    #optPoly
    optPoly = np.zeros(m, dtype=int)
    i = nDirs
    j = m - 1
    while(i > 0):
        i = prev[i]
        optPoly[j] = i
        j = j - 1
    
    optPoly = np.append(optPoly,nDirs).astype(np.int)
    return optPoly

def fitLines(pathObj):
    centers = np.zeros((pathObj.nLines,2))
    dirs = np.zeros((pathObj.nLines,2))
    
    for f in range(0,pathObj.nLines):
        i = pathObj.optPoly[f]
        j = pathObj.optPoly[f+1]
        
        x = pathObj.sumx[j+1] - pathObj.sumx[i]
        y = pathObj.sumy[j+1] - pathObj.sumy[i]
        x2 = pathObj.sumx2[j+1] - pathObj.sumx2[i]
        xy = pathObj.sumxy[j+1] - pathObj.sumxy[i]
        y2 = pathObj.sumy2[j+1] - pathObj.sumy2[i]
        
        k = j+1-i;
        
        d1 = np.array([x/k,y/k])
        centers[f,:] = d1 + pathObj.path[0,:]
        dirs[f,:] = d1/np.linalg.norm(d1)
        
        a = (x2-x*x/k)/k;
        b = (xy-x*y/k)/k;
        c = (y2-y*y/k)/k;
        
        lambda2 = (a+c+np.sqrt((a-c)*(a-c)+4*b*b))/2;
        a = a - lambda2;
        c = c - lambda2;
        
        if(abs(a) >= abs(c)):
            l = np.sqrt(a*a+b*b)
            if(l!=0):
                dirs[f,0] = -b/l
                dirs[f,1] = a/l
        else:
            l = np.sqrt(c*c+b*b)
            if(l!=0):
                dirs[f,0] = -c/l;
                dirs[f,1] = b/l;
        
    centers = np.concatenate((centers[-1,:].reshape(-1,2),centers))
    dirs = np.concatenate((dirs[-1,:].reshape(-1,2),dirs))
    return centers,dirs

def findIntersections(pathObj):
    intersections = np.zeros((pathObj.nLines,2))
    for k in range(0,pathObj.nLines):
        ind = pathObj.optPoly[k]
        p = pathObj.path[ind,:]
        
        
        #try to fit
        A = np.array([[pathObj.fitDirs[k,0],-pathObj.fitDirs[k+1,0]],[pathObj.fitDirs[k,1],-pathObj.fitDirs[k+1,1]]])
        b = np.array([pathObj.centers[k+1,0]-pathObj.centers[k,0],pathObj.centers[k+1,1]-pathObj.centers[k,1]])
        detA = np.linalg.det(A)
        if(detA != 0): #if fit succeeded
            z = np.linalg.solve(A,b)
            intrP = pathObj.centers[k,:] + z[0]*pathObj.fitDirs[k,:]
        else: #if fit failed
            intrP = p
            
        #if this is the exception that we want a flat bottom
        if(pathObj.straightBot and p[1]==pathObj.minY):
            intrP = p
                
        
              
        diff = intrP - p;
        
        #if there is a need to fix the point (L1 distance > 0.5)
        if(abs(diff[0])> 0.5 or abs(diff[1])>0.5):
            correctDiff = np.zeros(2)
            #correct the point
            gx = diff[1]>diff[0] # y > x
            gnx = diff[1]>-diff[0] # y > -x
            if((gx != gnx)): # xor(ngx,gx)
                #left or right
                correctDiff[0] = np.sign(diff[0])*0.5
                correctDiff[1] = np.sign(diff[0])*diff[1]/(2*diff[0])
            else:
                #up or down
                correctDiff[0] = np.sign(diff[1])*diff[0]/(2*diff[1])
                correctDiff[1] = np.sign(diff[1])*0.5
            intrP = p + correctDiff
        
        #output
        intersections[k,:] = intrP
    
    intersections = np.concatenate((intersections,intersections[0,:].reshape(-1,2)))
    midPoints = (intersections[:-1,:] + intersections[1:,:])/2
    midPoints = np.concatenate((midPoints[-1,:].reshape(-1,2),midPoints))
    return intersections,midPoints

def findAlpha(pathObj):
    p0 = pathObj.midPoints[:-1,:]
    p1 = pathObj.intersections[:-1,:]
    p2 = pathObj.midPoints[1:,:]
    
    bb = (p2 - p0).reshape(-1,2)
    prpnd = np.zeros(p0.shape)
    prpnd[:,0] = -bb[:,1]
    prpnd[:,1] = bb[:,0]
    #normalize
    prpnd = np.divide(prpnd,np.linalg.norm(prpnd,axis=1).reshape(-1,1))
    
    p21 = (p2 - p1).reshape(-1,2)
    p01 = (p0 - p1).reshape(-1,2)
    
    projs = np.fabs(p21[:,0]*prpnd[:,0]+p21[:,1]*prpnd[:,1])
    gamma = (projs - 0.5)/projs
    alpha = (4/3)*gamma
    alpha[alpha<0.55] = 0.55
    
    #if this is the exception that we want a flat bottom
    if(pathObj.straightBot):
        alpha[p1[:,1]==pathObj.minY] = 2.0
    
    return alpha

def p2c(p):
    return p[0] + 1j*p[1]

def saveSVG(svgPaths, fname):
    (l,r,b,t) = svg.paths2svg.big_bounding_box(svgPaths)
    
    l = 0
    b = 0
    r = math.ceil(r)
    t = math.ceil(t)
    
    vb = (l,b,r,t)
    dims = (r-l,t-b)

    svg.paths2svg.disvg(paths=svgPaths,margin_size=0.0, viewbox = vb, stroke_widths=[0.1]*len(svgPaths), colors=None, filename=fname, openinbrowser=True)
    
    #change units to mm instead of pixels
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(fname) as old_file:
            for line in old_file:
                if "viewBox" in line:
                    p = re.compile('viewBox=\"\(([-\d\s,.]*)\)')
                    x = p.findall(line)
                    print(x)
                    print(x[0])
                    [l, b, r, t] = x[0].replace(',','').split()

                    w = float(r)-float(l)
                    h = float(t)-float(b)

                    line = re.sub(r'height="[-\d.]*"', 'height="' + str(h) + 'mm"', line)
                    line = re.sub(r'width="[-\d.]*"', 'width="' + str(w) + 'mm"', line)
                    
                    if('px' in line):
                        print('should not happen - some units are still in pixels')
                        assert(0)

                new_file.write(line)
    #Remove original file
    remove(fname)
    #Move new file
    move(abs_path, fname)

def saveSVG2(svgPaths, fname):
    svg.paths2svg.disvg(paths=svgPaths,margin_size=0.0,stroke_widths=[0.0]*len(svgPaths),colors=None,filename=fname,openinbrowser=True)

    #change units to mm instead of pixels
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(fname) as old_file:
            for line in old_file:
                if "viewBox" in line:
                    p = re.compile('viewBox=\"([-\d\s.]*)')
                    x = p.findall(line)
                    [l, b, r, t] = x[0].split()

                    w = float(r)-float(l)
                    h = float(t)-float(b)

                    line = re.sub(r'height="[-\d.]*px"', 'height="' + str(h) + 'mm"', line)
                    line = re.sub(r'width="[-\d.]*px"', 'width="' + str(w) + 'mm"', line)
                    
                    if('px' in line):
                        print('should not happen - some units are still in pixels')
                        assert(0)

                new_file.write(line)
    #Remove original file
    remove(fname)
    #Move new file
    move(abs_path, fname)

#object to svgPath
def obj2svgPath(pathObj,CORNER_THRESH,ALPHA_MIN,i,boardThckns,scalePix2mm,partsMargin):
    svgpaths = []
    odd = True
    
    if(pathObj.straightBot):
        #check that we managed to keep corners
        b1 = pathObj.botCoord.botSegments.reshape(1,-1)
        inds = pathObj.intersections[:,1] == pathObj.minY
        b2 = pathObj.intersections[inds,0]
        b2 = b2[:-1].reshape(1,-1)
        areEqual = np.array_equal(b1,b2)
        assert(areEqual)
    
    for k in range(0,pathObj.nLines):
        p0 = scalePix2mm*pathObj.midPoints[k,:]
        interY = pathObj.intersections[k,:][1]
        p1 = scalePix2mm*pathObj.intersections[k,:]
        p2 = scalePix2mm*pathObj.midPoints[k+1,:]
        alpha = pathObj.alpha[k]
        
        if(pathObj.straightBot and interY == pathObj.minY):
            partMrgnVec = np.array([partsMargin, 0])
            boardThckVec = np.array([0, -boardThckns])
            if(odd):
                seg1 = svg.Line(p2c(p0), p2c(p1))
                seg2 = svg.Line(p2c(p1), p2c(p1+partMrgnVec))
                seg3 = svg.Line(p2c(p1+partMrgnVec), p2c(p1+partMrgnVec+boardThckVec))
                seg4 = svg.Line(p2c(p1+partMrgnVec+boardThckVec), p2c(p2+boardThckVec))
                svgpaths.append(seg1)
                svgpaths.append(seg2)
                svgpaths.append(seg3)
                svgpaths.append(seg4)
            else:
                seg1 = svg.Line(p2c(p0+boardThckVec), p2c(p1-partMrgnVec+boardThckVec))
                seg2 = svg.Line(p2c(p1-partMrgnVec+boardThckVec), p2c(p1-partMrgnVec))
                seg3 = svg.Line(p2c(p1-partMrgnVec), p2c(p1))
                seg4 = svg.Line(p2c(p1), p2c(p2))
                svgpaths.append(seg1)
                svgpaths.append(seg2)
                svgpaths.append(seg3)
                svgpaths.append(seg4)
            
            odd = not odd
            
        else:
            #if alpha is small enough - fit a Bezier curve
            if(alpha <= CORNER_THRESH):
                alpha = max(alpha,ALPHA_MIN)
                c1 = p0 + alpha*(p1-p0)
                c2 = p2 + alpha*(p1-p2)
                seg = svg.CubicBezier(p2c(p0), p2c(c1), p2c(c2), p2c(p2))
                svgpaths.append(seg)
            #else - draw two lines
            else:
                seg1 = svg.Line(p2c(p0), p2c(p1))
                seg2 = svg.Line(p2c(p1), p2c(p2))
                svgpaths.append(seg1)
                svgpaths.append(seg2)
    
    return svg.Path(*svgpaths)

def getPathObjs(paths,imBotCoord,drawVerbose):
    pathObjs = []
    
    #flat bottom
    straightBot = False;
    
    nPaths = len(paths)
    #run on all paths
    for i in range(0,nPaths):
        path = paths[i]
        if(len(path) < 10):
            continue
            
        if(i==0):
            straightBot = True
        else:
            straightBot = False
        
        pathObj = MyPath(path,straightBot,imBotCoord)    
        nDirs = pathObj.nDirs

        drawPath(path,drawVerbose)

        # last 3 DIRS
        max3DirsInds,pathObj.missingDirs = calc3Dirs(pathObj)

        # last straight lines
        last = getLastStraight(pathObj).astype(np.int)
        
        max3DirsInds = max3DirsInds.astype(np.int)
        pathObj.last = np.minimum(last,max3DirsInds).reshape(-1)
        
        #drawLastStraight(pathObj,drawVerbose)
        #drawVerbose = False

        # optimal poligon
        pathObj.optPoly = calcOptimalPolygon(pathObj)
        drawPolygon(pathObj,drawVerbose)

        # fit lines
        pathObj.centers,pathObj.fitDirs = fitLines(pathObj)
        drawCenterDirs(pathObj,drawVerbose)
        #continue

        # find intersections
        pathObj.intersections,pathObj.midPoints = findIntersections(pathObj)
        drawIntersections(pathObj,drawVerbose)

        # find alpha
        pathObj.alpha = findAlpha(pathObj)
        
        pathObjs.append(pathObj)
        
    return pathObjs