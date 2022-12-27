# ShadowArt

An art project using laser cutting.
I used my own python implementation of Potrace for tracing the image into svg.
My Potrace implementation is numpy vectorized, currently the fastest python implementation of Potrace that doesn't wrap the potrace C executable.

![](made/caveman.gif) <img src="made/cavemanDone.jpg" height="569">

# Crux of the code
The functions getLastStraight() and calc3Dirs() in image2vec.py are where the magic happens in terms of numpy vectorization.
It's also the first time I got to use np.maximum.accumulate():

```python
    lessThanMax = angles < np.maximum.accumulate(anglesMax,axis=1)
    greaterThanMin = angles > np.minimum.accumulate(anglesMin,axis=1)
    
    firstLess = np.argmax(lessThanMax,axis=1)-1
    firstGreater = np.argmax(greaterThanMin,axis=1)-1
```
