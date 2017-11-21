from astropy import units as u
from astropy.coordinates import Angle

from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from ..gemini import AstroDataGemini
from .. import gmu

class AstroDataPhoenix(AstroDataGemini):

    __keyword_dict = dict(focal_plane_mask = 'SLIT_POS')

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'PHOENIX'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(['PHOENIX'])


    @astro_data_descriptor
    def dec(self):
        """
        Returns the Declination of the center of the field

        Returns
        -------
        float
            right ascension in degrees
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
        Returns the Right Ascension of the center of the field

        Returns
        -------
        float
            right ascension in degrees
        """
        return Angle(self.phu.get('RA', 0), unit=u.hour).degree
