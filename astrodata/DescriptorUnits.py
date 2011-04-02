# Contains astrodata unit conversion system, 
# however infrastructure is set up to use sympy as well.
#
# *WARNING*
# To use sympy comment out the astrodata unit conversion 
#  system code and uncomment the appropriate sympy code.
#
# from sympy.physics import units 

    
class Unit(object):
    name = "unitless"
    factor = None
    mks = None
    
    def __init__(self, name, factor, mks):
        self.name = name
        self.factor = factor
        self.mks = mks
        
    def __str__(self):
        return self.name
	
    def convert(self, value, newunit):
        if newunit.mks != self.mks:
            print "DU26:", newunit.mks, self.mks
            if newunit.mks is None:
                print "MKS compatability not defined"
            raise "DescriptorUnits.py, line 28: Imcompatible types "
        
        # If using sympy use the alternate return, notice the swap,
        # (this was to gain readability of the iunits dict below)
        #
        #return value*(newunit.factor/self.factor)
        return value*(self.factor/newunit.factor)

# astrodata unit conversion dictionary
# Using meters, kilograms, seconds (mks, SI Units)
# as metadata in the nested dict for compatability testing
iunits = {"m"           :{1.          :"m"},
          "meter"       :{1.          :"m"},
          "meters"      :{1.          :"m"},
          "mile"        :{201168./125.:"m"},
          "miles"       :{201168./125.:"m"},
          "feet"        :{381./1250.  :"m"},
          "foot"        :{381./1250.  :"m"},
          "inch"        :{127./5000.  :"m"},
          "inches"      :{127./5000.  :"m"},
          "km"          :{1e3         :"m"},
          "kilometer"   :{1e3         :"m"},
          "kilometers"  :{1e3         :"m"},
          "cm"          :{1e-2        :"m"},
          "centimeter"  :{1e-2        :"m"},
          "centimeters" :{1e-2        :"m"},
          "mm"          :{1e-3        :"m"},
          "millimeter"  :{1e-3        :"m"},
          "millimeters" :{1e-3        :"m"},
          "um"          :{1e-6        :"m"},
          "micron"      :{1e-6        :"m"},
          "microns"     :{1e-6        :"m"},
          "micrometer"  :{1e-6        :"m"},
          "micrometers" :{1e-6        :"m"},
          "nm"          :{1e-9        :"m"},
          "nanometer"   :{1e-9        :"m"},
          "nanometers"  :{1e-9        :"m"},
          "angstrom"    :{1e-10       :"m"},
          "angstroms"   :{1e-10       :"m"},
          "kilogram"    :{1.          :"k"},
          "kilo"        :{1.          :"k"},
          "kilograms"   :{1.          :"k"},
          "second"      :{1.          :"s"},
          "sec"         :{1.          :"s"},
          "seconds"     :{1.          :"s"},
          "scaler"      :{1.          :"scaler"}
         }

# set all the Unit objects
for name in iunits.keys():
    for factor in iunits[name].keys():
        for mks in iunits[name].values():
            #print "Unit(",name,",",factor,",",mks,")"
            g = globals()
            g[name] = Unit(name,factor,mks)

# sympy implementation 
# NOTE: Need to remove sec and angstrom from dict 
#       because they are not supported in sympy
#for name in iunits.keys():
#    g= globals()
#    print "Unit(",name,",",units.__dict__[utyp],")"
#    g[utyp]= Unit(name, units.__dict__[utyp])
           


