import os
import numpy as np
from astropy.io import fits
from astropy.table import Table

from gempy.eti_core.etifile import ETIFile

# Move this to a more central place
X_IMAGE = 'X_IMAGE'
Y_IMAGE = 'Y_IMAGE'
FLUX = 'FLUX_ISO'
RA = 'ra'
DEC ='dec'

FLAGS_DTYPE = np.int16

PREFIX = "tmp"
SUFFIX = ".fits"
FLAGS_TO_MASK = None

class SExtractorETIFile(ETIFile):
    """This class coordinates the ETI files as it pertains to Sextractor
    tasks in general.
    """
    def __init__(self, input, mask_dq_bits=None):
        """
        input: a single extension from an AstroData object
        """
        def strip_fits(s):
            return s[:-5] if s.endswith('.fits') else s
        super(SExtractorETIFile, self).__init__(
            name=strip_fits(input.filename)+'_{}'.format(input.hdr['EXTVER']))
        # Replace bad pixels with median value of good data, so need to
        # copy the data plane in case we edit it
        self.data = input.data.copy()
        self.mask = input.mask
        self.header = input.hdr
        if mask_dq_bits and self.mask is not None:
            self.data[(self.mask & mask_dq_bits)>0] = np.median(
                self.data[(self.mask & mask_dq_bits)==0])
        self._disk_file = None
        self._catalog_file = None

    def prepare(self):
        # This looks silly, but we're pretending the array data is a "file"
        self._catalog_file = PREFIX + self.name + '_cat' + SUFFIX
        self._objmask_file = PREFIX + self.name + '_obj' + SUFFIX
        filename = PREFIX + self.name + SUFFIX
        hdulist = fits.HDUList()
        hdulist.append(fits.ImageHDU(self.data,
                                     header=self.header, name='SCI'))
        self._sci_image = filename+'[0]'
        if self.mask is not None:
            hdulist.append(fits.ImageHDU(np.array(self.mask, dtype=np.int16),
                                         header=self.header, name='DQ'))
            self._dq_image = filename+'[1]'
        else:
            self._dq_image = None
        #else:
        #    # Need a dummy DQ array because .param file needs a FLAG_IMAGE
        #    hdulist.append(fits.ImageHDU(np.zeros_like(self.data, dtype=np.int16),
        #                                 header=self.header, name='DQ'))
        try:
            hdulist.writeto(filename, overwrite=True)
        except TypeError:
            hdulist.writeto(filename, clobber=True)
        self._disk_file = filename

    def recover(self):
        objcat = Table.read(self._catalog_file)
        return (objcat, self._objmask_file)

    def clean(self, remove_inputs=True):
        if os.path.isfile(self._disk_file):
            os.remove(self._disk_file)
        if os.path.isfile(self._catalog_file):
            os.remove(self._catalog_file)
        if os.path.isfile(self._objmask_file):
            os.remove(self._objmask_file)