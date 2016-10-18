from astrodata import astro_data_tag, astro_data_descriptor
from ..gemini import AstroDataGemini

class AstroDataMichelle(AstroDataGemini):

    __keyword_dict = dict(coadds = 'NUMEXPOS',
                          exposure_time = 'EXPOSURE',
                          filter_name = 'FILTER')

    @staticmethod
    def _matches_data(data_provider):
        return data_provider.phu.get('INSTRUME', '').upper() == 'MICHELLE'

    @astro_data_tag
    def _tag_instrument(self):
        return (set(['MICHELLE']), ())

    @astro_data_tag
    def _tag_mode(self):
        camera = self.phu.get('CAMERA')
        if camera == 'imaging':
            return (set(['IMAGE']), ())
        elif camera == 'spectroscopy':
            return (set(['SPECT', 'LS']), ())

    @astro_data_descriptor
    def exposure_time(self):
        """
        Returns the exposure time in seconds.

        Returns
        -------
        float
            Exposure time.
        """
        exposure_time = self.phu.get(self._keyword_for('exposure_time'), -1)
        num_ext = self.phu.get('NUMEXT', 1)
        if exposure_time < 0:
            raise ValueError("Invalid exposure time: {}".format(exposure_time))

        return exposure_time * num_ext * self.coadds()

    @astro_data_descriptor
    def filter_name(self, stripID=False, pretty=False):
        """
        Returns the name of the filter(s) used. Since MICHELLE was not originally
        a Gemini instrument, its filters don't have componentIDs and so pretty
        and stripID do nothing.

        Parameters
        ----------
        stripID : bool
            Does nothing
        pretty : bool
            Does nothing

        Returns
        -------
        str
            The name of the filter
        """
        filter_name = self.phu.FILTER
        if filter_name == 'NBlock':
            filter_name = 'blank'
        return filter_name
