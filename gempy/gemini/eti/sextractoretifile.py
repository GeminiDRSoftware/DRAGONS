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

PREFIX = ""
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
        super().__init__(
            name=strip_fits(input.filename)+'_{}'.format(input.hdr['EXTVER']))
        # Replace bad pixels with median value of good data, so need to
        # copy the data plane in case we edit it
        if mask_dq_bits and self.mask is not None:
            self.data = input.data.copy()
            self.data[(self.mask & mask_dq_bits) > 0] = np.median(
                self.data[(self.mask & mask_dq_bits) == 0])
        else:
            self.data = input.data
        self.ext = input
        self._disk_file = None
        self._catalog_file = None

    def prepare(self):
        # This looks silly, but we're pretending the array data is a "file"
        self._catalog_file = os.path.join(self.directory, PREFIX + self.name + '_cat' + SUFFIX)
        self._objmask_file = os.path.join(self.directory, PREFIX + self.name + '_obj' + SUFFIX)
        filename = os.path.join(self.directory, PREFIX + self.name + SUFFIX)
        hdulist = fits.HDUList()

        # By using the to_hdulist() method, we write the current gWCS
        for hdu in self.ext.to_hdulist():
            if hdu.header.get('EXTNAME') in ('SCI', 'DQ'):
                hdulist.append(hdu)

        # Replace SCI with bitmasked data, and DQ as int16
        hdulist[0].data = self.data
        self._sci_image = filename + '[0]'
        if len(hdulist) > 1:
            hdulist[1].data = hdulist[1].data.astype(FLAGS_DTYPE)
            self._dq_image = filename + '[1]'
        else:
            self._dq_image = None

        hdulist.writeto(filename, overwrite=True)
        self._disk_file = filename

    def recover(self):
        objcat = Table.read(self._catalog_file)
        return (objcat, self._objmask_file)

    def clean(self, remove_inputs=True):
        if self._disk_file and os.path.isfile(self._disk_file):
            os.remove(self._disk_file)
        if self._catalog_file and os.path.isfile(self._catalog_file):
            os.remove(self._catalog_file)
        if self._objmask_file and os.path.isfile(self._objmask_file):
            os.remove(self._objmask_file)
