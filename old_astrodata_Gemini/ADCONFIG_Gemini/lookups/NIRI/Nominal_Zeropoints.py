
nominal_zeropoints = {
    # Table of NIRI Nominal Zeropoint magnitudes
    # By Camera and Filter ID.
    # From Photometric zero points in "sciops.instruments
    #   .PerformanceMonitoring.DataProducts.NIRI"

    # NOTE NOTE NOTE
    # The numbers on that web page are for 1 ADU/s not 1 electron/s, so we need to
    # add 2.5*log_10(gain) = 2.5*log_10(12.3) = 2.72 to each of them.

        # BAND  CAMERA: Zeropoint (average)
        ('Y',  'f6'):  22.99+2.72,
        ('J',  'f6'):  23.97+2.72,
        ('J',  'f32'): 23.33+2.72,
        ('H',  'f6'):  24.05+2.72,
        ('H',  'f32'): 23.62+2.72,
        ('K',  'f6'):  23.43+2.72,
        ('K',  'f32'): 22.99+2.72,  
        ('K(short)',  'f6'):  23.40+2.72,
        ('K(short)',  'f32'): 22.95+2.72,
        ('K(prime)',  'f6'):  23.68+2.72,
        ('K(prime)',  'f32'): 23.60+2.72,

}
