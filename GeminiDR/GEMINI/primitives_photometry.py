from astrodata.utils import logutils
from gempy.gemini import gemini_tools as gt

from GEMINI.lookups import ColorCorrections

from .primitives_CORE import PrimitivesCORE

# Define the earliest acceptable SExtractor version, currently: 2.8.6
SEXTRACTOR_VERSION = [2,8,6]
# ------------------------------------------------------------------------------
class Photometry(PrimitivesCORE):
    """
    This is the class containing all of the photometry primitives for
    the GEMINI level of the type hierarchy tree. It inherits all the
    primitives from the level above, 'GENERALPrimitives'.
    """
    tagset = set(["GEMINI"])
    
    def addReferenceCatalog(self, adinputs=None, stream='main', **params):
        """
        This primitive calls the gemini_catalog_client module to
        query a catalog server and construct a fits table containing
        the catalog data.

        That module will query either gemini catalog servers or
        vizier. Currently, sdss9 and 2mass (point source catalog)
        are supported.

        For example, with sdss9, the FITS table has the following columns:

        - 'Id'       : Unique ID. Simple running number
        - 'Cat-id'   : SDSS catalog source name
        - 'RAJ2000'  : RA as J2000 decimal degrees
        - 'DEJ2000'  : Dec as J2000 decimal degrees
        - 'umag'     : SDSS u band magnitude
        - 'e_umag'   : SDSS u band magnitude error estimage
        - 'gmag'     : SDSS g band magnitude
        - 'e_gmag'   : SDSS g band magnitude error estimage
        - 'rmag'     : SDSS r band magnitude
        - 'e_rmag'   : SDSS r band magnitude error estimage
        - 'imag'     : SDSS i band magnitude
        - 'e_imag'   : SDSS i band magnitude error estimage
        - 'zmag'     : SDSS z band magnitude
        - 'e_zmag'   : SDSS z band magnitude error estimage

        With 2mass, the first 4 columns are the same, but the photometry
        columns reflect the J H and K bands.

        This primitive then adds the fits table catalog to the Astrodata
        object as 'REFCAT'

        :param source: Source catalog to query, as defined in the
                       gemini_catalog_client module
        :type source: string

        :param radius: The radius of the cone to query in the catalog, 
                       in degrees. Default is 4 arcmin
        :type radius: float
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.addReferenceCatalog
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        timestamp_key = self.timestamp_keys[self.myself()]
        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        return

    def detectSources(self, adinputs=None, stream='main', **params):
        """
        Find x,y positions of all the objects in the input image. Append 
        a FITS table extension with position information plus columns for
        standard objects to be updated with position from addReferenceCatalog
        (if any are found for the field).
    
        :param method: source detection algorithm to use
        :type method: string; options are 'daofind','sextractor'

        :param centroid_function: Function for centroid fitting with daofind
        :type centroid_function: string, can be: 'moffat','gauss'
                                 Default: 'moffat'

        :param sigma: The mean of the background value for daofind. If nothing
                      is passed, it will be automatically determined
        :type sigma: float
        
        :param threshold: Threshold intensity for a point source for daofind;
                      should generally be at least 3 or 4 sigma above
                      background RMS.
        :type threshold: float
        
        :param fwhm: FWHM to be used in the convolve filter for daofind. This
                     ends up playing a factor in determining the size of the
                     kernel put through the gaussian convolve.
        :type fwhm: float
        
        :param mask: Whether to apply the DQ plane as a mask before detecting
                     the sources.
        :type sigma: bool
        """
        # Get the necessary parameters from the RC

        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.detectSources
        sfx = p_pars["suffix"]
        method = p_pars["method"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        timestamp_key = self.timestamp_keys[self.myself()]
        if adinputs:
            self.adinputs = adinputs
        
        # Check soup_parse detection method
        if method not in ["sextractor","daofind"]:
            raise IOError("Unsupported source detection method {}".format(method))

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        return


    def measureCCAndAstrometry(self, adinputs=None, stream='main', **params):

        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.measureCCAndAstrometry
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        if adinputs:
            self.adinputs = adinputs
        
        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        self.addReferenceCatalog()
        self.measureCC()
        self.determineAstrometricSolution()
        return
