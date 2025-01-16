
#                                                                  gemini_python
#
#                                                     primitives_ghost_bundle.py
# ------------------------------------------------------------------------------
from collections import Counter
import copy
import itertools
from datetime import timedelta

import astrodata, gemini_instruments
from .primitives_ghost import GHOST
from . import parameters_ghost_bundle

from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import parameter_override, capture_provenance

from astropy.io.fits import PrimaryHDU, Header
from astropy.table import Table

# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GHOSTBundle(GHOST):
    """
    Primitives for unpacking GHOST observation bundle files.
    """
    tagset = set(["GEMINI", "GHOST", "BUNDLE"])

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_ghost_bundle)

    def splitBundle(self, adinputs=None, **params):
        """
        Break a GHOST observation bundle into individual exposures.

        This primitive breaks up a GHOST observation bundle into multiple
        files: one for each Red camera exposure, one for each Blue camera
        exposure, and another containing all the Slit Viewer (SV) frames.

        If the observation is not tagged as a CAL (calibration), then a
        Table is attached to the SV file to store the start and end times
        of all the science exposures from the bundle, ensuring that
        appropriate stacks can be made to use as processed_slit
        calibrations for the red/blue reductions.
        """
        log = self.log
        log.debug(gt.log_message('primitive', self.myself(), 'starting'))
        timestamp_key = self.timestamp_keys[self.myself()]

        adoutputs = []
        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by {}".format(
                            ad.filename, self.myself()))
                continue
            log.stdinfo(f"Unbundling {ad.filename}")

            # No SCIEXP table for biases or observations of lamps
            on_sky = 'CAL' not in ad.tags or 'STANDARD' in ad.tags
            sci_exposures = []
            extns = [x for x in ad if (x.arm() == 'slitv' and x.shape)]
            if len(extns) > 0:
                slit_images = True
                ad_slit = _write_newfile(extns, 'slit', ad, log)
                adoutputs.append(ad_slit)
            else:
                log.warning(f"{ad.filename} has no slit viewer images")
                slit_images = False

            # now do non-slitv extensions
            extns = [x for x in ad if x.arm() != 'slitv']
            key = lambda x: f"{x.hdr['CAMERA'].lower()}{x.hdr['EXPID']:03d}"
            extns = sorted(extns, key=key)
            for k, g in itertools.groupby(extns, key=key):
                ad_arm = _write_newfile(list(g), k, ad, log)
                adoutputs.append(ad_arm)

                # We want to attach a Table to the slitviewer file, that
                # provides the start and end times of all science exposures
                # to enable stacking of contemporaneous slit images later
                if on_sky and slit_images:
                    arm_exptime = ad_arm.exposure_time()
                    ut_start = ad_arm.ut_datetime()
                    ut_end = ut_start + timedelta(seconds=arm_exptime)
                    exp_data = [k, arm_exptime, ut_start, ut_end]
                    # If we can use the same set of SLITV images for more than
                    # one arm exposure, do so
                    for xtr in sci_exposures:
                        if (xtr[1] == exp_data[1] and
                                abs((xtr[2] - exp_data[2]).total_seconds()) < 0.01):
                            xtr[0] += f",{k}"
                            break
                    else:
                        sci_exposures.append(exp_data)

            # Format and attach the table
            if sci_exposures and slit_images:
                sci_exposures = [xtr[:2] + [x.isoformat() for x in xtr[2:]]
                                    for xtr in sci_exposures]
                exposure_table = Table(names=("for", "exptime", "UTSTART", "UTEND"),
                                       dtype=(str, float, str, str),
                                       rows=sci_exposures)
                ad_slit.SCIEXP = exposure_table

        return adoutputs

    def validateData(self, adinputs=None, suffix=None):
        """
        GHOSTBundle-specific version of validateData to ignore the invalid WCS
        exception.
        """
        try:
            super().validateData(adinputs, suffix=suffix)
        except ValueError as e:
            if 'valid WCS' not in str(e):
                raise
        return adinputs


##############################################################################
# Below are the helper functions for the primitives in this module           #
##############################################################################
def _get_common_hdr_value(base, extns, key):
    """
    Helper function to get a common header value from a list of extensions.

    Parameters
    ----------
    base : :any:`astrodata.AstroData`
        Original AstroData instance that may contain useful headers
    extns : iterable of :any:`astrodata.Astrodata`
        AstroData extensions to be examined
    key : str
        The FITS header key to find in each extension

    Raises
    ------
    KeyError
        If the ``key`` parameter has multiple values across the extensions
    """

    # Get the keyword from every extension
    vals = [x.hdr.get(key) for x in extns]
    c = Counter(vals)
    # Not all extensions may not contain the keyword,
    # but we don't care about blanks
    del c[None]
    # If the keyword doesn't exist at all in the extensions,
    # then use the base value instead
    if len(c) == 0:
        return base.phu.get(key)
    # Check that every extension has the same value for the keyword
    most_common = c.most_common()
    base = most_common[0]
    if len(most_common) != 1:
        # Ignore single errors in this header, as long as the most common
        # value is more than half
        if base[1] < len(vals) // 2:
          raise KeyError('multiple values for ' + key + " " + str(vals))

        for val in most_common[1:]:
          if val[1] == 1:
            print('Ignoring single error in', key, 'header')
          else:
            raise KeyError('multiple values for ' + key + " " + str(vals))

    return base[0]

def _get_hdr_values(extns, key):
    """
    Helper function to get the all header values from a list of extensions.
    The return value is a dict keyed on the EXPID value.

    Parameters
    ----------
    extns : iterable of :any:`astrodata.Astrodata`
        AstroData extensions to be examined
    key : str
        The FITS header key to find in the list of extensions
    """

    return {x.hdr.get('EXPID'): x.hdr.get(key) for x in extns}


def _write_newfile(extns, suffix, base, log):
    """
    Helper function to write sub-files out from a MEF bundle.

    Parameters
    ----------
    extns : iterable of :any:`astrodata.Astrodata`
        AstroData extensions to be appended to the new file
    suffix : str
        Suffix to be appended to file name
    base : :any:`astrodata.AstroData`
        Original AstroData instance to base the new file on. The file's
        primary header unit will be used, as will the base of the filename.
    log : AstroData logging object
        Log for recording actions. This should be the log in use in the calling
        primitive.

    Raises
    ------
    AssertionError
        If the ``extns`` parameter is :any:`None`, or empty
    """
    assert extns and len(extns) > 0

    # Start with a copy of the base PHU
    # But also look for the extension with an empty data array,
    # because this is the real PHU from the original file before
    # it was mashed into a giant MEF.
    for x in extns:
        if (x.hdr.get('NAXIS') == 0) or (x.data.size == 0):
            phu = PrimaryHDU(data=None, header=copy.deepcopy(x.hdr))
            n = astrodata.create(phu)
            break
    else:
        n = astrodata.create(copy.deepcopy(base.phu))

    # Collate headers into the new PHU
    for kw in ['CAMERA', 'CCDNAME',
               'CCDSUM', 'DETECTOR',
               'OBSTYPE', 'SMPNAME', 'EXPTIME']:
        n.phu.set(kw, _get_common_hdr_value(base, extns, kw))
    vals = _get_hdr_values(extns, 'DATE-OBS')
    n.phu.set('DATE-OBS', vals[min(vals.keys())])
    vals = _get_hdr_values(extns, 'UTSTART')
    n.phu.set('UTSTART', vals[min(vals.keys())])
    vals = _get_hdr_values(extns, 'UTEND')
    n.phu.set('UTEND', vals[max(vals.keys())])

    # Copy keywords into each separate file if they aren't already there
    for kw in base.phu:
        if kw not in n.phu or n.phu.get(kw) == '':
            n.phu[kw] = (base.phu[kw], base.phu.comments[kw])

    # Other stuff
    n.phu['OBSID'] = base.phu.get('OBSID', '')

    # Remove some keywords that are only relevant to the bundle
    for kw in ['NEXTEND', 'NREDEXP', 'NBLUEEXP', 'NSLITEXP', 'UT',
               'EXTNAME', 'EXTVER']:
        try:
            del n.phu[kw]
        except KeyError:
            pass

    # Append the extensions that contain pixel data
    extver = 1
    for x in extns:
        if (x.hdr.get('NAXIS') > 0) and (x.data.size > 0):
            x.hdr['EXTVER'] = extver
            extver += 1
            n.append(x)

    # Construct a filename
    n.filename = base.filename
    n.update_filename(suffix="_"+suffix)

    # MCW 190813 - Update the ORIGNAME of the file
    # Otherwise, every time we do a 'strip' file rename, the base file name
    # will go back to being the MEF bundle file name, and things will
    # quickly start to overlap each other
    n.phu['ORIGNAME'] = n.filename

    # CJS 20221128: to ensure that processed cals from the different arms
    # have different data labels before going in the archive
    if n.phu['CAMERA'] == "SLITV":
        n.phu['DATALAB'] += "-SLITV-001"
    else:
        n.phu['DATALAB'] += f"-{n.phu['CAMERA']}-{suffix[-3:]}"  # sequence number

    # And add GHOSTDR version number
    #n.phu['GHOSTDR'] = (ghost_instruments.__version__, "GHOSTDR version")

    return n
