# Descriptor Index Rules
# (1) file must begin with "calculatorIndex." and end in ".py"
# (2) the dictionary set must be named "calculatorIndex"
# (3) both the key and value should be strings
# (4) the key is an AstroDataType, the value is the name of the object
#       to use as the calculator for that type
# (5) the value should have the form <module name>.<calculator class name>
# (6) the "descriptors" subdirectory and all subdirectories of it are added to
#       import path
# (7) when the calculator is needed the Descriptor system execs
#       "import <module name>" and gets the object by
#       "eval <module name>.<calculator class name>"
# (8) indexes may not conflict in the type names, but otherwise can be
#       distributed however one likes in the directory structure. Also,
#       multiple calculators can be defined in the same module ... i.e., we
#       could have a single giant index and single giant calculator .py file,
#       but that would be a mess.

calculatorIndex = {
    "FITS":"FITS_Descriptors.FITS_DescriptorCalc()",
    }
