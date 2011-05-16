# Structure Definitions, imported elsewhere.
from Structures import *

class ExtMDF(ExtID):
    headReqs= { "EXTNAME":"MDF"}
    
class ExtScience(ExtID):
    headReqs = { "EXTNAME":"SCI" }
    
class ExtVariance(ExtID):
    headReqs = { "EXTNAME":"VAR" }

class ExtDataQuality(ExtID):
    headReqs = { "EXTNAME":"DQ" }

class StructGemBundle(Structure):
    """ This structure exists to handle data as array of (SCI,VAR,DQ) triplets
    """
    parts = {
        "sci" : {structure:"StructBASICGEM.ExtScience()"}, 
        "var" : {structure:"StructBASICGEM.ExtVariance()", optional:True},
        "dq"  : {structure:"StructBASICGEM.ExtDataQuality()", optional:True}
        }
    
class StructGemBundleArray(Structure):
    parts = {
        "bundles" : {structure: "StructBASICGEM.StructGemBundle()",
                    array_by : "EXTVER"}
        }

class StructSpecArray(Structure):
    parts = {
        "spectra" : {structure: "StructBASICGEM.StructGemBundle()",
                     array_by : "EXTVER" },
        "mdf" : {structure: "StructBASICGEM.ExtMDF()", retain:True }
        }
    
class StructChopArray(Structure):
    parts = {
        "chops" : {structure: "StructBASICGEM.StructGemBundle()",
                  array_by : "CHOP" }
    }

class StructNodChopTree(Structure):
    parts = {
        "nods" : { structure: "StructBASICGEM.StructChopArray()",
                array_by : "NOD" }
        }
        
