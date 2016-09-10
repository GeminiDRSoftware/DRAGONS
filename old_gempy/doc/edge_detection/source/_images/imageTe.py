import matplotlib.pyplot as pl
import matplotlib.image as mpimg
from pylab import rcParams
import matplotlib.cm as cm
import numpy as np

def imageLabel(png_file,label,out_pngname):
    """
      img is a numpy array from an pgn file:

      img = mpimg.imread('file.png')
    """
    img = mpimg.imread(png_file)
    dpi = rcParams['figure.dpi']
    sz=img.shape
    figsize = sz[1]/dpi, sz[0]/dpi+1.

    fig=pl.figure(figsize=figsize)
    ax = pl.axes([0,0,1,1], frameon=False)
    ax.text(100,sz[0]+20.,label,fontsize=26,
        weight='bold',bbox=dict(facecolor='white', alpha=0.5))
    ax.set_axis_off()
    ax.imshow(img,cmap=cm.Greys_r)
    fig.savefig(out_pngname)
    pl.show()

#ax.set_ylim([-1,20])
#ax.grid(False)
#ax.set_xlabel('Model complexity --->')
#ax.set_ylabel('Message length --->')
#ax.set_title('Minimum Message Length')
#
#ax.set_yticklabels([])
#ax.set_xticklabels([])


#import matplotlib.cm as cm
#import Image

#fname='cartoon.png'
#image=Image.open(fname).convert("L")
#arr=np.asarray(image)
#pylab.imshow(arr,cmap=cm.Greys_r)



