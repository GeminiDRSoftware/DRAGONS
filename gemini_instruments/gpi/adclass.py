from astrodata import astro_data_tag, astro_data_descriptor, TagSet
from ..gemini import AstroDataGemini


class AstroDataGpi(AstroDataGemini):

    __keyword_dict = dict(array_section='DATASEC',
                          detector_section='DATASEC',
                          exposure_time='ITIME',
                          filter='IFSFILT',
                          focal_plane_mask='OCCULTER',
                          pupil_mask='APODIZER')

    @classmethod
    def read(cls, source):
        def gpi_parser(hdu):
            if hdu.header.get('EXTNAME') == 'DQ' and hdu.header.get('EXTVER') == 3:
                hdu.header['EXTNAME'] = ('SCI', 'BPM renamed by AstroData')
                hdu.header['EXTVER'] = (int(2), 'BPM renamed by AstroData')

        return super().read(source, extname_parser=gpi_parser)

    @staticmethod
    def _matches_data(source):
        return source[0].header.get('INSTRUME', '').upper() == 'GPI'

    @astro_data_tag
    def _tag_instrument(self):
        return TagSet({'GPI'}, ())

    @astro_data_tag
    def _tag_disperser(self):
        disp = self.phu.get('DISPERSR', '')
        if disp.startswith('DISP_WOLLASTON'):
            return TagSet({'POL'}, ())
        elif disp.startswith('DISP_PRISM'):
            return TagSet({'SPECT', 'IFU'}, ())

    @astro_data_descriptor
    def dec(self):
        """
        Returns the Declination of the center of the field in degrees.
        It coincides with the position of the target, so that is used since
        the WCS in GPI data is completely bogus. For code re-used, use
        target_dec() if you really want the position of the target rather
        than the center of the field.

        Returns
        -------
        float
            declination in degrees
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
            Strips the component ID and ``IFSFILT_`` prefix.

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
        Returns the Right Ascension of the center of the field in degrees.
        It coincides with the position of the target, so that is used since
        the WCS in GPI data is completely bogus. For code re-used, use
        target_ra() if you really want the position of the target rather
        than the center of the field.

        Returns
        -------
        float
            right ascension in degrees
        """
        return self.target_ra(offset=True, icrs=True)
