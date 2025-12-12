.. ex1_f2ls_JHHK_cmdline.rst

.. include:: symbols.txt

.. _f2ls_JHHK_cmdline:

************************************************************************
Example 1 - JH and HK Longslit Point Source - Using the "reduce" command
************************************************************************


::

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="partnerCal" and disperser=="JH"' | showd -d exposure_time,read_mode
    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science" and disperser=="JH"' | showd -d exposure_time,read_mode

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="partnerCal" and disperser=="HK"' | showd -d exposure_time,read_mode
    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science" and disperser=="HK"' | showd -d exposure_time,read_mode

::

    cp ../playdata/example1/S20220617S0041.fits ../playdata/example1/S20220617S0041_fixed.fits
    fixheader ../playdata/example1/S20220617S0041_fixed.fits EXPTIME 25
    fixheader ../playdata/example1/S20220617S0041_fixed.fits LNRS 4

::

    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==6' -o dark6.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==8' -o dark8.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==15' -o dark15.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==18' -o dark18.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==25' -o dark25.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==60' -o dark60.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==90' -o dark90.lis

::

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science" and disperser=="JH"' | showd -d ut_datetime
    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="partnerCal" and disperser=="JH"' | showd -d ut_datetime
    dataselect ../playdata/example1/*.fits --tags FLAT --expr='filter_name=="JH" and ut_time<="01:56:00"' -o flatsciJH.lis
    dataselect ../playdata/example1/*.fits --tags FLAT --expr='filter_name=="JH" and ut_time>="01:56:00"' -o flattelJH.lis

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science" and disperser=="HK"' | showd -d ut_datetime
    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="partnerCal" and disperser=="HK"' | showd -d ut_datetime
    dataselect ../playdata/example1/*.fits --tags FLAT --expr='filter_name=="HK" and ut_time>="00:52:00"' -o flatsciHK.lis
    dataselect ../playdata/example1/*.fits --tags FLAT --expr='filter_name=="HK" and ut_time<="00:52:00"' -o flattelHK.lis

::

    dataselect ../playdata/example1/*.fits --tags ARC -o arc.lis

::

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="partnerCal" and disperser=="JH"' -o telJH.lis
    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science" and disperser=="JH"' -o sciJH.lis

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="partnerCal" and disperser=="HK"' -o telHK.lis
    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science" and disperser=="HK" and exposure_time==25' -o sciHK.lis

::

    reduce @dark6.lis
    reduce @dark8.lis
    reduce @dark15.lis
    reduce @dark18.lis
    reduce @dark25.lis
    reduce @dark60.lis
    reduce @dark90.lis

::

    for file in $(ls dark*.lis); do reduce @$file; done

::

    reduce @flatsciJH.lis -p interactive=True   # spline3 2, no regions
    reduce @flattelJH.lis -p interactive=True

    reduce @flatsciHK.lis -p interactive=True normalizeFlat:regions=289:1820  # spline3 1, need to go beyond 6 and then get flaring. avoid negative fit
    reduce @flattelHK.lis -p interactive=True

::

    reduce @arc.lis -p interactive=True

::

    reduce -r reduceTelluric @telJH.lis @hip83920.param -p interactive=True
    reduce -r reduceTelluric @telHK.lis @hip83920.param -p interactive=True prepare:bad_wcs=new

Fits are bad.  LSF 0.5 limit.
HK spline3 30 with region: 1286:2503

::

    reduce @sciJH.lis -p interactive=True prepare:bad_wcs=new findApertures:max_apertures=1 extractSpectra:write_outputs=True telluricCorrect:write_outputs=True telluricCorrect:telluric=S20220617S0073_telluric.fits fluxCalibrate:standard=S20220617S0073_telluric.fits
    reduce @sciHK.lis -p interactive=True prepare:bad_wcs=new findApertures:max_apertures=1 extractSpectra:write_outputs=True telluricCorrect:write_outputs=True telluricCorrect:telluric=S20220617S0027_telluric.fits fluxCalibrate:standard=S20220617S0027_telluric.fits


JH:  Matches the publish spectrum except below 1.040 um where the continuum starts looking weird and
    likely exaggerated features show up.  This is the section where the fitTelluric fit struggles.

HK:  Right now, using the data rather than the model for telluricCorrect is far better.  The model leaves
    wiggles and the Pacshen alpha lines is very noisy.  Very good match to publish spectrum.  Again
    it's the blue end which is wrong, but not as wrong as for JH.

::

    reduce -r joinrecipe.joinSpectra *1D.fits -p scale=True
