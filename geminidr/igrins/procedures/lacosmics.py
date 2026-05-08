import astroscrappy

def get_cr_mask(d, gain=2.2, readnoise=10.0,
                sigclip=5, sigfrac = 0.3, objlim = 5.0):

    c = astroscrappy.detect_cosmics(d, gain=gain, readnoise=readnoise,
                                    sigclip=sigclip, sigfrac=sigfrac,
                                    objlim=objlim,
                                    cleantype='medmask',
                                    psfmodel="gaussx")

    return c[0]

