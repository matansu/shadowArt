# ShadowArt

A fun art project using laser cutting.
I used my own python implementation of Potrace for tracing the image into svg.
Currently the fastest pure python implementation of potrace as far as I know (numpy vectorized).

![](made/caveman.gif) <img src="made/cavemanDone.jpg" height="569">

# Crux of the code
```python
    lessThanMax = angles < np.maximum.accumulate(anglesMax,axis=1)
    greaterThanMin = angles > np.minimum.accumulate(anglesMin,axis=1)
    
    firstLess = np.argmax(lessThanMax,axis=1)-1
    firstGreater = np.argmax(greaterThanMin,axis=1)-1
```
