# This parameter file contains the parameters related to the primitives located
# in the primitives_bookkeeping.py file, in alphabetical order.
from gempy.library import config

class addToListConfig(config.Config):
    purpose = config.Field("Purpose of list", str, None, optional=True)

class getListConfig(config.Config):
    purpose = config.Field("Purpose of list", str, None, optional=True)
    max_frames = config.RangeField("Maximum number of frames", int, None, min=1, optional=True)

class rejectInputsConfig(config.Config):
    at_start = config.RangeField("Number of files to remove from start of list", int, 0, min=0)
    at_end = config.RangeField("Number of files to remove from end of list", int, 0, min=0)

class selectFromInputsConfig(config.Config):
    tags = config.Field("List of tags for selection", str, None, optional=True)

class showInputsConfig(config.Config):
    purpose = config.Field("Purpose of displaying list", str, None, optional=True)

class showListConfig(config.Config):
    purpose = config.Field("Purpose of displaying list", str, 'all')

class sortInputsConfig(config.Config):
    descriptor = config.Field("Name of descriptor for sorting", str, 'filename')
    reverse = config.Field("Reverse order of sort?", bool, False)

class transferAttributeConfig(config.Config):
    source = config.Field("Stream to transfer from", str, None)
    attribute = config.Field("Name of attribute to transfer", str, None)

class writeOutputsConfig(config.Config):
    overwrite = config.Field("Overwrite exsting files?", bool, True)
    outfilename = config.Field("Output filename", str, None, optional=True)
    prefix = config.Field("Prefix for output files", str, '')
    suffix = config.Field("Suffix for output files", str, '')
    strip = config.Field("Strip prefix/suffix from filenames?", bool, False)
