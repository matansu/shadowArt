import numpy as np

class BotCoord(object):
    pass

def calcBotCoord(xs,ys,yt,addHalfs):
    toAdd = 0
    if(addHalfs):
        toAdd = 0.5
    
    bottomXs = xs[ys==yt];
    botCoord = BotCoord()
    botCoord.y = yt + 0.5;
    botCoord.l = np.min(bottomXs) - toAdd
    botCoord.r = np.max(bottomXs) + toAdd
    diffs = np.diff(bottomXs)
    jumpInds = np.argwhere(diffs!=1)

    #create 2d array of bot segments
    
    botSegments = np.empty([0, 1])        
    botSegments = np.append(botSegments, np.min(bottomXs) - toAdd)
    for ind in jumpInds:
        botSegments = np.append(botSegments, bottomXs[ind[0]] + toAdd)
        botSegments = np.append(botSegments, bottomXs[ind[0]+1] - toAdd)
    botSegments = np.append(botSegments, np.max(bottomXs) + toAdd)

    botSegments = np.reshape(botSegments,[-1,2])
    #add the botSegments to botCoord
    botCoord.botSegments = botSegments
    return botCoord