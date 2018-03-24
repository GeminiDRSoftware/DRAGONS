from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from ..gemini import AstroDataGemini
from .. import gmu

class AstroDataNici(AstroDataGemini):
    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'NICI'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(set(['NICI', 'IMAGE']), ())

    @astro_data_tag
    def _tag_dark(self):
        if self.phu.get('OBSTYPE') == 'DARK':
            return TagSet(set(['DARK', 'CAL']), ())

    @astro_data_tag
    def _tag_flat(self):
        # NOTE: This will be set also for old type DARK frames... What should we do?
        if self.phu.get('OBSTYPE') == 'FLAT':
            return TagSet(set(['FLAT', 'CAL']), set(['SDI', 'ASDI']))

    @astro_data_tag
    def _tag_dichroic(self):
        dich = self.phu.get('DICHROIC', '')
        if 'Mirror' in dich:
            return TagSet(set(['ADI_B']), ())
        elif 'Open' in dich:
            return TagSet(set(['ADI_R']), ())
        elif '50/50' in dich:
            crmode = self.phu.get('CRMODE')
            if crmode == 'FOLLOW':
                return TagSet(set(['SDI']), ())
            elif crmode == 'FIXED':
                return TagSet(set(['ASDI']), ())

    @astro_data_descriptor
    def exposure_time(self):
        """
        Returns the exposure time in seconds.

        Returns
        -------
        float/list of floats
            exposure time for each extension
        """
        try:
            exptime_r = self.phu['ITIME_R'] * self.phu['NCOADD_R']
        except KeyError:
            exptime_r = None
        try:
            exptime_b = self.phu['ITIME_B'] * self.phu['NCOADD_B']
        except KeyError:
            exptime_b = None

        # Use the filter header keywords to determine which exptime to use
        # Assume it's a red exposure if FILTER_B is not defined
        filt_b = self.hdr.get('FILTER_B')
        try:
            return [exptime_r if f is None else exptime_b for f in filt_b]
        except TypeError:
            return exptime_r if filt_b is None else exptime_b

    @astro_data_descriptor
    def filter_name(self, stripID=False, pretty=False):
        """
        Returns the name of the filter(s) used.  The component ID can be
        removed with either 'stripID' or 'pretty'.  If a combination of filters
        is used, the filter names will be join into a unique string with '&' as
        separator.  If 'pretty' is True, filter positions such as 'Open',
        'Dark', 'blank', and others are removed leaving only the relevant
        filters in the string.

        NICI has filter names in the extension HDUs, so this can return a list

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID and returns only the name of
            the filter.
        pretty : bool
            Same as for stripID.  Pretty here does not do anything more.

        Returns
        -------
        str/list of str
            The name of the filter combination with or without the component ID.
        """
        filt_r = self.hdr.get('FILTER_R')
        filt_b = self.hdr.get('FILTER_B')

        # This assumes that precisely one of FILTER_R or FILTER_B is defined
        # in each HDU
        try:
            filters = [r if b is None else b for r,b in zip(filt_r, filt_b)]
            return "+".join([gmu.removeComponentID(f) if pretty or stripID else
                        f for f in filters])
        except TypeError:
            filter = filt_r if filt_b is None else filt_b
            return gmu.removeComponentID(filter) if pretty or stripID else filter

    @astro_data_descriptor
    def pixel_scale(self):
        """
        Returns the image scale in arcseconds per pixel

        Returns
        -------
        float
            the pixel scale
        """
        return 0.018
