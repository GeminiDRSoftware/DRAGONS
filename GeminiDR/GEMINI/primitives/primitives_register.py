import math
import pywcs
import numpy as np

from importlib import import_module

from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import logutils

from gempy.library import astrotools as at
from gempy.gemini import gemini_tools as gt

from GEMINI.lookups import keyword_comments

from primitives_CORE import PrimitivesCORE

# ------------------------------------------------------------------------------
# pkgname = __file__.split('astrodata_')[1].split('/')[0]
keyword_comments = keyword_comments.keyword_comments
# ------------------------------------------------------------------------------
class Register(PrimitivesCORE):
    """
    This is the class containing all of the registration primitives for the
    GEMINI level of the type hierarchy tree. It inherits all the primitives
    from the level above, 'GENERALPrimitives'.
    """
    tag = "GEMINI"
    
    def correctWCSToReferenceFrame(self, adinputs=None, stream='main', **params):
        """ 
        This primitive registers images to a reference image by correcting
        the relative error in their world coordinate systems. The function
        uses points of reference common to the reference image and the
        input images to fit the input WCS to the reference one. The fit
        is done by a least-squares minimization of the difference between
        the reference points in the input image pixel coordinate system.
        This function is intended to be followed by the
        align_to_reference_image function, which applies the relative
        transformation encoded in the WCS to transform input images into the
        reference image pixel coordinate system.
        
        The primary registration method is intended to be by direct mapping
        of sources in the image frame to correlated sources in the reference
        frame. This method fails when there are no correlated sources in the
        field, or when the WCSs are very far off to begin with. As a back-up
        method, the user can try correcting the WCS by the shifts indicated 
        in the POFFSET and QOFFSET header keywords (option fallback='header'), 
        By default, only the direct method is
        attempted, as it is expected that the relative WCS will generally be
        more correct than either indirect method. If the user prefers not to
        attempt direct mapping at all, they may set method to 'header'.
        
        In order to use the direct mapping method, sources must have been
        detected in the frame and attached to the AstroData instance in an 
        OBJCAT extension. This can be accomplished via the detectSources
        primitive. Running time is optimal, and sometimes the solution is 
        more robust, when there are not too many sources in the OBJCAT. Try
        running detectSources with threshold=20. The solution may also be
        more robust if sub-optimal sources are rejected from the set of 
        correlated sources (use option cull_sources=True). This option may
        substantially increase the running time if there are many sources in
        the OBJCAT.
        
        It is expected that the relative difference between the WCSs of 
        images to be combined should be quite small, so it may not be necessary
        to allow rotation and scaling degrees of freedom when fitting the image
        WCS to the reference WCS. However, if it is desired, the options 
        rotate and scale can be used to allow these degrees of freedom. Note
        that these options refer to rotation/scaling of the WCS itself, not the
        images. Significant rotation and scaling of the images themselves 
        will generally already be encoded in the WCS, and will be corrected for
        when the images are aligned.
        
        The WCS keywords in the headers of the output images are updated
        to contain the optimal registration solution.
        
        :param method: method to use to generate reference points. Options
                       are 'sources' to directly map sources from the input
                       image to the reference image,
                       or 'header' to generate reference points from the 
                       POFFSET and QOFFSET keywords in the image headers.
        :type method: string, either 'sources' or 'header'
        
        :param fallback: back-up method for generating reference points.
                         if the primary method fails. The 'sources' option
                         cannot be used as the fallback.
        :type fallback: string, either 'header' or None.
        
        :param use_wcs: the alignment method will use the encoded WCS in the 
                        header for the initial adjustment of the frames. The
                        alternative is that the shifts and rotation from the
                        header will be used.
        :type use_wcs: bool

        :param first_pass: estimated maximum distance between correlated
                           sources. This distance represents the expected
                           mismatch between the WCSs or header shifts of 
                           the input images.
        :type first_pass: float
        
        :param min_sources: minimum number of sources to use for cross-
                            correlation, depending on the instrument used.
        :type min_sources: int                    
                            
        :param cull_sources: flag to indicate whether sub-optimal sources 
                             should be rejected before attempting a direct
                             mapping. If True, sources that are saturated,
                             or otherwise unlikely to be point sources
                             will be eliminated from the list of reference
                             points.
        :type cull_sources: bool
        
        :param rotate: flag to indicate whether the input image WCSs should
                       be allowed to rotate with respect to the reference image
                       WCS
        :type rotate: bool
        
        :param scale: flag to indicate whether the input image WCSs should
                      be allowed to scale with respect to the reference image
                      WCS. The same scale factor is applied to all dimensions.
        :type scale: bool

        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.correctWCSToReferenceFrame
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        timestamp_key = self.timestamp_keys[self.myself()]
        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        for par, val in p_pars.items():
            log.stdinfo("  {}:  {}".format(par, val))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return
    
    def determineAstrometricSolution(self, adinputs=None, stream='main', **params):
        """
        This primitive calculates the average astrometric offset between
        the positions of sources in the reference catalog, and their
        corresponding object in the object catalog.
        It then reports the astrometric correction vector.
        For now, this is limited to a translational offset only.
        
        The solution is stored in a WCS object in the RC.  It can
        be applied to the image headers by calling the updateWCS
        primitive.
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.determineAstrometricSolution
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

        logutils.update_indent(0)
        return

    def updateWCS(self, adinputs=None, stream='main', **params):
        """
        This primitive applies a previously calculated WCS correction.
        The solution should be stored in the RC as a dictionary, with
        astrodata instances as the keys and pywcs.WCS objects as the
        values.

        """

        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.updateWCS
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

        logutils.update_indent(0)
        return
