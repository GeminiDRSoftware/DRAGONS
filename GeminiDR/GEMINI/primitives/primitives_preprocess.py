import os
import math
import datetime
import numpy as np

from copy import deepcopy

from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.gemconstants import SCI, VAR, DQ

from gempy.gemini import gemini_tools as gt

from recipe_system.reduction import reductionContextRecords as RCR

from primitives_CORE import PrimitivesCORE
# ------------------------------------------------------------------------------
# pkgname =  __file__.split('astrodata_')[1].split('/')[0]
# ------------------------------------------------------------------------------
class Preprocess(PrimitivesCORE):
    """
    This is the class containing all of the preprocessing primitives
    for the GEMINI level of the type hierarchy tree. It inherits all
    the primitives from the level above, 'GENERALPrimitives'.
    """
    tag = "GEMINI"

    def ADUToElectrons(self, adinputs=None, stream='main', **params):
        """
        This primitive will convert the units of the pixel data extensions
        of the input AstroData object from ADU to electrons by multiplying
        by the gain.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = self.parameters.ADUToElectrons["suffix"]
        for ad in self.adinputs:
            # Check whether the ADUToElectrons primitive has been run
            if ad.phu_get_key_value(timestamp_key):
                lmsg = "No changes will be made to {}."
                lmsg += "Already processed by " + self.myself()
                log.warning(lmsg.format(ad.filename))
                continue
            
            gain = ad.gain()
            log.status("Converting %s from ADU to electrons by multiplying by "
                       "the gain" % (ad.filename))
            for ext in ad[SCI]:
                extver = ext.extver()
                log.stdinfo("  gain for [%s,%d] = %s" %
                            (SCI, extver, gain.get_value(extver=extver)))

            ad = ad.mult(gain)
            gt.update_key(adinput=ad, keyword="BUNIT", value="electron", comment=None,
                          extname=SCI, keyword_comments=self.keyword_comments)
            if ad[VAR]:
                gt.update_key(adinput=ad, keyword="BUNIT", value="electron*electron",
                              comment=None, extname=VAR, 
                              keyword_comments=self.keyword_comments)
            
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return
    
    def associateSky(self, adinputs=None, stream='main', **params):
        """
        This primitive determines which sky AstroData objects are associated
        with each science AstroData object and adds this information to a
        dictionary (in the form {science1:[sky1,sky2],science2:[sky2,sky3]}),
        where science1 and science2 are the science AstroData objects and sky1,
        sky2 and sky3 are the sky AstroDataRecord objects, which is then added
        to the reduction context.
        
        The input sky AstroData objects can be provided by the user using the
        parameter 'sky'. Otherwise, the science AstroData objects are found in
        the main stream (as normal) and the sky AstroData objects are found in
        the sky stream.
        
        :param adinput: input science AstroData objects
        :type adinput: Astrodata or Python list of AstroData
        
        :param sky: The input sky frame(s) to be subtracted from the input
                    science frame(s). The input sky frame(s) can be a list of
                    sky filenames, a sky filename, a list of AstroData objects
                    or a single AstroData object. Note: If there are multiple
                    input science frames and one input sky frame provided, then
                    the same sky frame will be applied to all inputs; otherwise
                    the number of input sky frames must match the number of
                    input science frames.
        :type sky: string, Python list of string, AstroData or Python list of
                   Astrodata 
        """

    def correctBackgroundToReferenceImage(self, adinputs=None, stream='main', **params):
        """
        This primitive does an additive correction to a set
        of images to put their sky background at the same level
        as the reference image before stacking.
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.correctBackgroundToReferenceImage
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

    def divideByFlat(self, adinputs=None, stream='main', **params):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding flat. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the division on the data.

        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.divideByFlat
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
     
    def nonlinearityCorrect(self, adinputs=None, stream='main', **params):
        """
        Apply a generic non-linearity correction to data.
        At present (based on GSAOI implementation) this assumes/requires that
        the correction is polynomial. The ad.non_linear_coeffs() descriptor
        should return the coefficients in ascending order of power
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.nonlinearityCorrect
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

    def normalizeFlat(self, adinputs=None, stream='main', **params):
        """
        This primitive normalizes each science extension of the input
        AstroData object by its mean
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.normalizeFlat
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

    def separateSky(self, adinputs=None, stream='main', **params):

        """
        Given a set of input exposures, sort them into separate but
        possibly-overlapping streams of on-target and sky frames. This is
        achieved by dividing the data into distinct pointing/dither groups,
        applying a set of rules to classify each group as target(s) or sky
        and optionally overriding those classifications with user guidance
        (up to and including full manual specification of both lists).

        If all exposures are found to be on source then both output streams
        will replicate the input. Where a dataset appears in both lists, a
        separate copy (TBC: copy-on-write?) is made in the sky list to avoid
        subsequent operations on one of the output lists affecting the other.

        The following optional parameters are accepted, in addition to those
        common to other primitives:

        :param frac_FOV: Proportion by which to scale the instrumental field
            of view when determining whether points are considered to be
            within the same field, for tweaking borderline cases (eg. to avoid
            co-adding target positions right at the edge of the field).
        :type frac_FOV: float

        :param ref_obj: Exposure filenames (as read from disk, without any
            additional suffixes appended) to be considered object/on-target
            exposures, as overriding guidance for any automatic classification.
        :type ref_obj: string with comma-separated names

        :param ref_sky: Exposure filenames to be considered sky exposures, as
            overriding guidance for any automatic classification.
        :type ref_obj: string with comma-separated names

        :returns: Separate object and sky streams containing AstroData objects

        Any existing OBJFRAME or SKYFRAME flags in the input meta-data will
        also be respected as input (unless overridden by ref_obj/ref_sky) and
        these same keywords are set in the output, along with a group number
        with which each exposure is associated (EXPGROUP).
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.separateSky
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
    
    def subtractDark(self, adinputs=None, stream='main', **params):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding dark. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data.

        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.subtractDark
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

    def subtractSky(self, adinputs=None, stream='main', **params):
        """
        This function will subtract the science extension of the input sky
        frames from the science extension of the input science frames. The
        variance and data quality extension will be updated, if they exist.
        
        :param adinput: input science AstroData objects
        :type adinput: Astrodata or Python list of AstroData
        
        :param sky: The input sky frame(s) to be subtracted from the input
                    science frame(s). The input sky frame(s) can be a list of
                    sky filenames, a sky filename, a list of AstroData objects
                    or a single AstroData object. Note: If there are multiple
                    input science frames and one input sky frame provided, then
                    the same sky frame will be applied to all inputs; otherwise
                    the number of input sky frames must match the number of
                    input science frames.
        :type sky: string, Python list of string, AstroData or Python list of
                   Astrodata 
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.subtractSky
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

    def subtractSkyBackground(self, adinputs=None, stream='main', **params):
        """
        This primitive is used to subtract the sky background specified by 
        the keyword SKYLEVEL.
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        logutils.update_indent(3)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.subtractSkyBackground
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
