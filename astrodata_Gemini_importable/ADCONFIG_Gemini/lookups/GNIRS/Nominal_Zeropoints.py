
nominal_zeropoints = {
    # Table of GNIRS nominal zeropoint magnitude by camera and filter.
    # Updated XXXXX
    
    # Check what the units are...
    # NOTE NOTE NOTE
    # The numbers on that web page are for 1 ADU/s not 1 electron/s, so we need to
    # add 2.5*log_10(gain) = 2.5*log_10(12.3) = 2.72 to each of them.

    # (BAND, CAMERA): Nominal zeropoint for airmass=1
    ('J', 'ShortBlue_G5540'): x,
    ('H', 'ShortBlue_G5540'): x,
    ('K', 'ShortBlue_G5540'): x,  
    ('J', 'ShortBlue_G5538'): x,
    ('H', 'ShortBlue_G5538'): x,
    ('K', 'ShortBlue_G5538'): x,  
    ('J', 'LongBlue_G5542'):  x,
    ('H', 'LongBlue_G5542'): x,
    ('K', 'LongBlue_G5542'): x,  
    ('J', 'ShortRed_G5539'): x,
    ('H', 'ShortRed_G5539'): x,
    ('K', 'ShortRed_G5539'): x,  
    ('J', 'LongRed_G5516'):  x,
    ('H', 'LongRed_G5516'): x,
    ('K', 'LongRed_G5516'): x,  
    ('J', 'LongRed_G5543'):  x,
    ('H', 'LongRed_G5543'): x,
    ('K', 'LongRed_G5543'): x,  

}
