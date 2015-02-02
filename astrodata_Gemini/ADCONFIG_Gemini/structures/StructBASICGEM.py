# Structure Definitions, imported elsewhere.
from astrodata.interface.Structures import *
from astrodata.utils.Requirements import *

import pyfits

class ExtBinTable(ExtID):
    requirement = HDUTYPE(pyfits.core.BinTableHDU) 
    
class ExtImageTable(ExtID):
    requirement = HDUTYPE(pyfits.core.ImageHDU)

class StructPixelDataMembers(Structure):
    parts = { "pixel_exts":{ structure:"StructBASICGEM.ExtImageTable()",\
                            array_by: lambda ext: (ext.extname(), ext.extver()),
                        }
            }

class StructBintableMembers(Structure):
    parts = { "bintables":{ structure:"StructBASICGEM.ExtBinTable()",\
                            array_by: lambda ext: (ext.extname(), ext.extver()),
                        }
            }
class StructDUTypeMembers(Structure):
    parts = { "bintables":{ structure:"StructBASICGEM.ExtBinTable()",
                             array_by: lambda ext: (ext.extname(), ext.extver()),
                             "optional": True
                          },
              "pixel_exts":{ structure:"StructBASICGEM.ExtImageTable()",\
                            array_by: lambda ext: (ext.extname(), ext.extver()),
                        },
            }
      
class StructWithMDF(Structure):
    parts = { "mdf": {structure:"StructBASICGEM.ExtMDF()",
                        "optional":True}
             }


class ExtMDF(ExtID):
    requirement = HU(EXTNAME="MDF")
    
class ExtScience(ExtID):
    requirement = HU({ "EXTNAME":"SCI" })
    
class ExtVariance(ExtID):
    requirement = HU({ "EXTNAME":"VAR" })

class ExtDataQuality(ExtID):
    requirement = HU({ "EXTNAME":"DQ" })
    

class StructGemBundle(Structure):
    """ This structure exists to handle data as array of (SCI,VAR,DQ) triplets
    """
    parts = {
        "sci" : {structure:"StructBASICGEM.ExtScience()"}, 
        "var" : {structure:"StructBASICGEM.ExtVariance()", optional:True},
        "dq"  : {structure:"StructBASICGEM.ExtDataQuality()", optional:True},
         }
    
class StructGemBundleArray(Structure):
    parts = {
        "bundles" : {structure: "StructBASICGEM.StructGemBundle()",
                    array_by :lambda ext: ext.extver()}
        }

class StructSpecArray(Structure):
    parts = {
        "spectra" : {structure: "StructBASICGEM.StructGemBundle()",
                     array_by : lambda ext: ext.extver() },
        "mdf" : {structure: "StructBASICGEM.ExtMDF()", retain:True }
        }
    
#class StructChopArray(Structure):
#    parts = {
#        "chops" : {structure: "StructBASICGEM.StructGemBundle()",
#                  array_by : "CHOP" }
#    }
#
#class StructNodChopTree(Structure):
#    parts = {
#        "nods" : { structure: "StructBASICGEM.StructChopArray()",
#                array_by : "NOD" }
#        }
#        
