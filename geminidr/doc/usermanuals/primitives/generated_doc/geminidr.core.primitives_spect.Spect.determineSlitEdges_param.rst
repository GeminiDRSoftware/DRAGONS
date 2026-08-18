Parameter defaults and options
------------------------------
::

   suffix               '_slitEdgesDetermined' Filename suffix
   nsum                 6                    Number of lines to sum in spatial profile for edge detection
      	Valid Range = [1,inf)
   min_snr              10.0                 Minimum SNR for edge detection
      	Valid Range = [0.1,inf)
   spectral_order       3                    Fitting order in spectral direction
      	Valid Range = [1,inf)
   edge1                None                 Expected pixel location for left/lower edge of illuminated region. If None, a MDF must be present in the input.
      	Valid Range = [1,inf)
   edge2                None                 Expected pixel location for right/top edge of illuminated region. If None, a MDF must be present in the input.
      	Valid Range = [1,inf)
   search_radius        30.0                 Radius (in pixels) to search for edges
      	Valid Range = [5,inf)
