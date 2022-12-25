import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.path as matpath
import scipy

from botCoord import *

#how to install packages:
#import sys
#!{sys.executable} -m pip install svgwrite
#!{sys.executable} -m pip install svgpathtools

#image and drawing functions
def binImg(img):
    return np.where(img < 0.5, 0, 1)

def readImg(imName):
    img=mpimg.imread(imName)
    if len(img.shape) > 2:
        img=img[:,:,0]
    if(img.max() > 125):
        img=img/255.0
    img=binImg(img)

    img = np.flipud(img)
    assert((img==1).sum() + (img==0).sum() == img.size)
    return img

def drawBW(img):
    plt.figure()
    imgplot = plt.imshow(img, cmap="gray")

def subDrawBW(img,r,c,i):
    plt.subplot(r,c,i)
    imgplot = plt.imshow(img, cmap="gray")

def drawBWf(img):
    plt.figure()
    imgplot = plt.imshow(img, cmap="gray")
    plt.gca().invert_yaxis()

def drawUnique(img):
    plt.figure()
    imgplot = plt.imshow(img, cmap="nipy_spectral")
    plt.gca().invert_yaxis()
    
def drawPath(path,drawStuff):
    if not drawStuff:
        return
     #draw path
    x = []
    y = []
    for i in range(0,len(path)-1):# range(ind,ind+1):#
        j = i+1
        p1 = path[i,:]
        p2 = path[j,:]
        x.append(p1[0])
        x.append(p2[0])
        x.append(None)
        y.append(p1[1])
        y.append(p2[1])
        y.append(None)
    plt.plot(x, y)
    
def drawLastStraight(pathObj,drawStuff):
    if not drawStuff:
        return
    #draw last straight.
    x = []
    y = []
    for i in range(0,pathObj.nDirs):
        j = pathObj.last[i]
        p1 = pathObj.path[i,:]
        p2 = pathObj.path[j,:]
        x.append(p1[0])
        x.append(p2[0])
        x.append(None)
        y.append(p1[1])
        y.append(p2[1])
        y.append(None)
    plt.plot(x, y)
    
def drawPolygon(pathObj,drawStuff):
    if not drawStuff:
        return
    #draw polygon
    x = []
    y = []
    for k in range(0,pathObj.nLines+1):# range(ind,ind+1):#
        i = pathObj.optPoly[k]
        p = pathObj.path[i,:]
        x.append(p[0])
        y.append(p[1])
    plt.plot(x, y)

def drawCenterDirs(pathObj,drawStuff):
    if not drawStuff:
        return
    for k in range(0,pathObj.nLines):
        x = []
        y = []
        p1 = pathObj.centers[k,:] + pathObj.fitDirs[k,:]*5
        p2 = pathObj.centers[k,:] - pathObj.fitDirs[k,:]*5
        plt.plot([p1[0],p2[0]], [p1[1],p2[1]])
        
def drawIntersections(pathObj,drawStuff):
    if not drawStuff:
        return
    #draw fixed intersections
    x = []
    y = []     
    x1 = []
    y1 = []
    for k in range(0,pathObj.nLines):
        p = pathObj.intersections[k,:]
        q = pathObj.midPoints[k,:]
        x.append(p[0])
        y.append(p[1])
        x1.append(q[0])
        y1.append(q[1])
    plt.plot(x,y,'ro')
    plt.plot(x1,y1,'bo')
    
def drawSvgPath(svgPath,nPnts):
    x = []
    y = []
    NUM_SAMPLES = nPnts
    for i in range(NUM_SAMPLES):
        point = svgPath.point(i/(NUM_SAMPLES-1))
        x.append(point.real)
        y.append(point.imag)
    
    plt.plot(x, y, 'r', linewidth=3)  
    
def drawSvgPathAxis(ax,svgPath,nPnts):
    x = []
    y = []
    NUM_SAMPLES = nPnts
    for i in range(NUM_SAMPLES):
        point = svgPath.point(i/(NUM_SAMPLES-1))
        x.append(point.real)
        y.append(point.imag)
    
    ax.plot(x, y, 'r', linewidth=2)  
    
def drawSvgPathsAxis(ax,svgPaths,nPnts):
    for svgPath in svgPaths:
        drawSvgPathAxis(ax,svgPath,nPnts)
    
def forwardProj(x,y,yt,xLight,yLight,zLight):
    beta = np.divide(y-yt,y-yLight);
    z = np.dot(beta,zLight);
    xt = x - np.dot(beta,x-xLight);
    return z,xt

def backwardProj(xt,z,yt,xLight,yLight,zLight):
    alpha = np.divide(z,zLight);
    y = np.divide(yt - np.dot(alpha,yLight),1-alpha);
    beta = np.divide(y-yt,y-yLight);
    x = np.divide(xt - np.dot(xLight,beta),1-beta);
    return x,y

# note that you'll need to reshape afterwards
def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    ooi = np.logical_or(np.logical_or(np.logical_or(x0 == 0 , x0 == im.shape[1]-1 ), np.logical_or(x1 == 0 , x1 == im.shape[1]-1)),
    np.logical_or(np.logical_or(y0 == 0,  y0 == im.shape[0]-1), np.logical_or(y1 == 0, y1 == im.shape[0]-1)))

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    interpImg = wa*Ia + wb*Ib + wc*Ic + wd*Id
    interpImg[ooi] = 1
    return interpImg

class ImLight(object):
    pass

def getImLightStruct(img,scalePix2mm,maxDistZLightMM,maxDistYLightMM):
    imLObj = ImLight()
    [ys,xs] = np.asarray(img==0).nonzero()
    #get some necessary measurements
    imLObj.origHieght = img.shape[0]
    imLObj.xs = xs.astype(np.float64)
    imLObj.ys = ys.astype(np.float64)
    imLObj.imLeft = np.min(imLObj.xs)
    imLObj.imRight = np.max(imLObj.xs)
    imLObj.imTop = np.max(imLObj.ys)
    imLObj.imBot = np.min(imLObj.ys)
    imLObj.imCenter = (imLObj.imTop + imLObj.imBot)/2
    imLObj.actualHeight =  imLObj.imTop - imLObj.imBot
    imLObj.actualWidth = imLObj.imRight - imLObj.imLeft
    
    scaleMM2Pix = 1/scalePix2mm
    
    #set the light position
    imLObj.yLight = min(imLObj.actualHeight + imLObj.imCenter,imLObj.origHieght + maxDistYLightMM*scaleMM2Pix)
    imLObj.xLight = (imLObj.imLeft + imLObj.imRight) / 2
    imLObj.zLight = min(imLObj.actualHeight,maxDistZLightMM*scaleMM2Pix)
    
    print('light (px):',imLObj.xLight,imLObj.yLight,imLObj.zLight)
    print('light (mm):',imLObj.xLight*scalePix2mm,imLObj.yLight*scalePix2mm,imLObj.zLight*scalePix2mm)
    return imLObj



def warpComponents(components,img,imLObj,drawStuff):
    if(drawStuff):
        plt.figure()
    #for each component
    allWarped = []
    botCoords = []
    numCompo = np.max(np.unique(components))
    for compo in range(1,numCompo+1):
        #calc part
        margin = 5
        img1 = copy.deepcopy(img)
        img1[components!=compo] = 1
        #drawBW(img1)
        [ys,xs] = np.asarray(components==compo).nonzero()
        xs = xs.astype(np.float64)
        ys = ys.astype(np.float64)
        yt = np.max(ys)
        yb = np.min(ys)
        xr = np.max(xs)
        xl = np.min(xs)
        
        
        ##### bot coords
        addHalfs = True
        botCoord = calcBotCoord(xs,ys,yt,addHalfs)
        #append to all bot coords
        botCoords.append(botCoord)

        #find limits to the warped image
        [Zrb,Xrb] = forwardProj(xr,yb,yt,imLObj.xLight,imLObj.yLight,imLObj.zLight);
        [Zlb,Xlb] = forwardProj(xl,yb,yt,imLObj.xLight,imLObj.yLight,imLObj.zLight);
        
        #prepare warp coordinates
        Xrb = np.ceil(np.maximum(Xrb,xr) + margin);
        Xlb = np.floor(np.minimum(Xlb,xl) - margin);
        Zb = np.ceil(np.maximum(Zrb,Zlb) + margin);

        XS = np.arange(Xlb, Xrb+1, 1)
        ZS = np.arange(-margin,Zb+1,1)
           
        #create mesh and calc U,V
        X,Z = np.meshgrid(XS, ZS)
        U,V = backwardProj(X,Z,yt,imLObj.xLight,imLObj.yLight,imLObj.zLight);

        #Warp!
        #warped = scipy.ndimage.map_coordinates(img1, [V.ravel(), U.ravel()], order=5, mode='nearest').reshape(U.shape)
        warped = bilinear_interpolate(img1, U, V).reshape(U.shape)

        #erase botr margin
        warped[0:margin,:] = 1
        
        #add the warped image
        allWarped.append(warped)

        assert(np.max(warped) <= 1.0)
        assert(np.min(warped) >= 0.0)

        if(drawStuff):
            subDrawBW(warped,2,np.ceil(numCompo/2),compo)
        
    return allWarped,botCoords