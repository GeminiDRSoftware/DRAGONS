from pyraf import iraf
import getiq

def __abc(image, outFile, function, verbose, residuals, display, interactive, rawpath, prefix, observatory, clip, sigma, pymark, niters, boxSize, debug):
    iq.iq.gemiq(image=image, outFile=outFile, function=function, \
          verbose=verbose, residuals=residuals, display=display, \
          interactive=interactive, rawpath=rawpath, prefix=prefix, \
          observatory=observatory, clip=clip, sigma=sigma, \
          pymark=pymark, niters=niters, boxSize=boxSize, debug=debug)

parfile = iraf.osfn("/data2/jholt/iqtool/IQTool/IQToolNov/pygem/iq/gemiq.par")
t = iraf.IrafTaskFactory(taskname="gemiq", value=parfile, function=__abc)
