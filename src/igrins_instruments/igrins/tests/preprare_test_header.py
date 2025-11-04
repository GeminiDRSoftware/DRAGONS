from pathlib import Path
from astropy.io import fits

import astrodata
import igrins_instruments


def prepare_mef():
    srcdir = Path("mef_20240429")

    samples = [('N20240429S0122.fits', 'TGT'),
               ('N20240429S0125.fits', 'STD'),
               ('N20240429S0204.fits', 'SKY'),
               ('N20240429S0365.fits', 'FLATOFF'),
               ('N20240429S0375.fits', 'FLATON')]

    outdir = Path("sample_mef")

    for fn, kind in samples:
        hdul = fits.open(srcdir / fn)
        hdul = fits.HDUList(hdul[:3]) # save only H & K.
        for ext in hdul:
            if isinstance(ext, (fits.ImageHDU)):
                ext.data = None

        outdir.mkdir(exist_ok=True)
        hdul.writeto(outdir / fn, overwrite=True, output_verify="fix")



def prepare_ubundled():
    srcdir = Path("unbundled_20240429")

    samples = [('N20240429S0122_H.fits', 'TGT'),
               ('N20240429S0125_H.fits', 'STD'),
               ('N20240429S0204_H.fits', 'SKY'),
               ('N20240429S0365_H.fits', 'FLATOFF'),
               ('N20240429S0375_H.fits', 'FLATON')]

    sampledir = dict((kind, Path(f"sample_{kind.lower()}"))
                     for kind in ["TGT", "STD", "SKY", "FLATOFF", "FLATON"])

    for fn, kind in samples:
        hdul = fits.open(srcdir / fn)
        for ext in hdul:
            ext.data = None

        sampledir[kind].mkdir(exist_ok=True)
        hdul.writeto(sampledir[kind] / fn, overwrite=True, output_verify="fix")


if __name__ == '__main__':
    prepare_ubundled()


# srcdir = Path("unbundled_20240721")

# samples = [('N20240721S0116_H.fits', 'TGT'),
#            ('N20240721S0128_H.fits', 'STD'),
#            ('N20240721S0132_H.fits', 'SKY'),
#            ('N20240721S0276_H.fits', 'FLATOFF'),
#            ('N20240721S0295_H.fits', 'FLATON'),
#            ]

# sampledir = dict((kind, Path(f"sample_{kind.lower()}"))
#                  for kind in ["TGT", "STD", "SKY", "FLATOFF", "FLATON"])

# for fn, kind in samples:
#     hdul = fits.open(srcdir / fn)
#     for ext in hdul:
#         ext.data = None

#     outdirs[kind].mkdir(exist_ok=True)
#     hdul.writeto(outdirs[kind] / fn, overwrite=True, output_verify="fix")
