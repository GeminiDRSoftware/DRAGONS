niriSpecDict = {
        # Database for nprepare.cl
        # Date: 2004 July 6
        # Author: Joe Jensen, Gemini Observatory
        # The long 6-pix and 4-pix centered slits are currently installed
        #
        # Array characteristics 
        "readnoise"  : 70,          # electrons (1 read pair, 1 digital av.)
        "medreadnoise" : 35.,       # electrons (1 read pair, 16 dig av.)
        "lowreadnoise" : 12.3,      # electrons (16 read pairs, 16 dig av.)
        "gain"         : 12.3,      # electrons/ADU
        "shallowwell"  : 200000.,   # electrons full-well
        "deepwell"     : 280000.,   # electrons full-well
        "shallowbias"  : -0.6,      # detector bias (V)
        "deepbias"     : -0.87,     # detector bias (V)
        "linearlimit"  : 0.7,       # non-linear regime (fraction of saturation)
        #
        # Camera+FPmask        SPECSEC1           SPECSEC2          SPECSEC3
        #
        "f6f6-2pix_G5211"   :   (  "[1:1024,276:700]" ,  "none", "none" ),
        "f6f6-4pix_G5212"   :   (  "[1:1024,1:1024]"  ,  "none", "none" ),
        "f6f6-6pix_G5213"   :   (  "[1:1024,1:1024]"  ,  "none", "none" ),
        "f6f6-2pixBl_G5214" :   (  "[1:1024,276:700]" ,  "none", "none" ),
        "f6f6-4pixBl_G5215" :   (  "[1:1024,276:700]" ,  "none", "none" ),
        "f6f6-6pixBl_G5216" :   (  "[1:1024,276:700]" ,  "none", "none" ),
        "f6f6-4pix_G5222"   :   (  "[1:1024,276:700]" ,  "none", "none" ),
        "f6f6-6pix_G5223"   :   (  "[1:1024,276:700]" ,  "none", "none" )
    }
 
