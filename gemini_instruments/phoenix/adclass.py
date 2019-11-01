#
#                                                            Gemini Observatory
#
#                                                                        Dragons
#                                                             gemini_instruments
#                                                             phoenix.adclass.py
# ------------------------------------------------------------------------------
import re
import datetime
import dateutil

from astropy import units as u
from astropy.coordinates import Angle

from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from ..gemini import AstroDataGemini

from .. import gmu

# ------------------------------------------------------------------------------
class AstroDataPhoenix(AstroDataGemini):

    __keyword_dict = dict(focal_plane_mask = 'SLIT_POS')

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'PHOENIX'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['PHOENIX'])

    @astro_data_tag
    def _tag_image(self):
        if "image" in self.phu.get('VIEW_POS', '').lower():
            return TagSet(['IMAGE'])

    @astro_data_tag
    def _tag_spect(self):
        if "open" in self.phu.get('VIEW_POS', '').lower():
            return TagSet(['SPECT'])

    @astro_data_tag
    def _tag_dark(self):
        if "dark" in self.phu.get('VIEW_POS', '').lower():
            return TagSet(['DARK', 'CAL'], blocks=['IMAGE'])

    @astro_data_tag
    def _tag_flat(self):
        if "flat" in self.phu.get('OBJECT', '').lower():
            return TagSet(['FLAT', 'CAL'])

    @astro_data_tag
    def _tag_acquisition(self):
        if "acq" in self.phu.get('OBJECT', '').lower():
            return TagSet(['ACQUISITION'])


    @astro_data_descriptor
    def dec(self):
        """
        Returns the Declination of the center of the field, in degrees.

        Returns
        -------
        float
            declination in degrees
        """
        return Angle(self.phu.get('DEC', 0), unit=u.degree).degree

    @astro_data_descriptor
    def filter_name(self, stripID=False, pretty=False):
        """
        Returns the name of the filter(s) used.  The component ID can be
        removed with either 'stripID' or 'pretty'.  If a combination of filters
        is used, the filter names will be join into a unique string with '&' as
        separator.  If 'pretty' is True, filter positions such as 'Open',
        'Dark', 'blank', and others are removed leaving only the relevant
        filters in the string.

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID and returns only the name of
            the filter.
        pretty : bool
            Same as for stripID.  Pretty here does not do anything more.

        Returns
        -------
        str
            The name of the filter combination with or without the component ID.
        """
        return self._may_remove_component('FILT_POS', stripID, pretty)

    @astro_data_descriptor
    def ra(self):
        """
        Returns the Right Ascension of the center of the field, in degrees.

        Returns
        -------
        float
            right ascension in degrees
        """
        return Angle(self.phu.get('RA', 0), unit=u.hour).degree

    @astro_data_descriptor
    def ut_datetime(self, strict=False, dateonly=False, timeonly=False):
        utd = super(AstroDataPhoenix, self).ut_datetime(strict=strict,
                                                        dateonly=dateonly,
                                                        timeonly=timeonly)
        if utd is None:
            utime = self[0].hdr.get('UT')
            udate = self[0].hdr.get('UTDATE')
        else:
            return utd

        if not utime and not udate:
            return None

        if utime and udate:
            dt_utime = dateutil.parser.parse(utime).time()
            dt_udate = dateutil.parser.parse(udate).date()

        if dateonly:
            return dt_udate
        elif timeonly:
            return dt_utime
        else:
            return datetime.datetime.combine(dt_udate, dt_utime)
