from astrodata import astro_data_tag, astro_data_descriptor, returns_list, TagSet
from ..gemini import AstroDataGemini
from .. import gmu

class AstroDataGpi(AstroDataGemini):

    __keyword_dict = dict(array_section = 'DATASEC',
                          detector_section = 'DATASEC',
                          exposure_time = 'ITIME',
                          filter = 'IFSFILT',
                          focal_plane_mask = 'OCCULTER',
                          pupil_mask = 'APODIZER')

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'GPI'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet(set(['GPI']), ())

    @astro_data_tag
    def _tag_disperser(self):
        disp = self.phu.get('DISPERSR', '')
        if disp.startswith('DISP_WOLLASTON'):
            return TagSet(set(['POL']), ())
        elif disp.startswith('DISP_PRISM'):
            return TagSet(set(['SPECT', 'IFU']), ())

    @astro_data_descriptor
    def dec(self):
        """
        Returns the Declination of the target, using the target_ra descriptor
        because the WCS is completely bogus.

        Returns
        -------
        float
            right ascension in degrees
        """
        return self.target_dec(offset=True, icrs=True)

    @astro_data_descriptor
    def exposure_time(self):
        """
        Returns the exposure time in seconds.

        Returns
        -------
        float
            Exposure time.
        """
        # The ITIME keyword is in the extension HDUs!
        exposure_time = self.hdr.get('ITIME', -1)[0]
        if exposure_time < 0:
            return None

        if 'PREPARED' in self.tags:
            return exposure_time
        else:
            return exposure_time * self.coadds()

    @astro_data_descriptor
    def filter_name(self, stripID=False, pretty=False):
        """
        Returns the name of the filter(s) used.  The component ID can be
        removed with either 'stripID' or 'pretty'. If 'pretty' is True,
        'IFSFILT' is stripped from the start of the name.

        Parameters
        ----------
        stripID : bool
            If True, removes the component ID and returns only the name of
            the filter.
        pretty : bool
            Strips the component ID and 'IFSFILT_' prefix.

        Returns
        -------
        str
            The name of the filter combination with or without the component ID.
        """
        filter_name = self._may_remove_component('IFSFILT', stripID, pretty)
        if pretty:
            filter_name = filter_name.replace('IFSFILT_', '')
        return filter_name

    @astro_data_descriptor
    def ra(self):
        """
        Returns the Right Ascension of the target, using the target_ra descriptor
        because the WCS is completely bogus.

        Returns
        -------
        float
            right ascension in degrees
        """
        return self.target_ra(offset=True, icrs=True)
