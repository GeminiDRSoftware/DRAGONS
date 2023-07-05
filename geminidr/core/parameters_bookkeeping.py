# This parameter file contains the parameters related to the primitives located
# in the primitives_bookkeeping.py file, in alphabetical order.
from gempy.library import config

class addToListConfig(config.Config):
    purpose = config.Field("Purpose of list", str, None, optional=True)

class appendStreamConfig(config.Config):
    from_stream = config.Field("Stream from which images are to be appended", str, "")
    copy = config.Field("Append deepcopies of frames?", bool, False)

class clearAllStreamsConfig(config.Config):
    pass

class clearStreamConfig(config.Config):
    stream = config.Field("Name of stream to clear", str, "main")

class combineSlicesConfig(config.Config):
    from_stream = config.Field("Name of stream to take extensions from", str,
                               None, optional=False)
    ids = config.Field("List of (1-indexed IDs of extensions to combine)", str,
                       None)

class copyInputsConfig(config.Config):
    pass

class flushPixelsConfig(config.Config):
    force = config.Field("Force write-to-disk?", bool, False)

class getListConfig(config.Config):
    purpose = config.Field("Purpose of list", str, None, optional=True)
    max_frames = config.RangeField("Maximum number of frames", int, None, min=1, optional=True)

class mergeInputsConfig(config.Config):
    pass

class rejectInputsConfig(config.Config):
    at_start = config.RangeField("Number of files to remove from start of list", int, 0, min=0)
    at_end = config.RangeField("Number of files to remove from end of list", int, 0, min=0)

class removeFromInputsConfig(config.Config):
    tags = config.Field("List of tags for selection", str, None, optional=True)

class selectFromInputsConfig(config.Config):
    tags = config.Field("List of tags for selection", str, None, optional=True)

class showInputsConfig(config.Config):
    purpose = config.Field("Purpose of displaying list", str, None, optional=True)

class showListConfig(config.Config):
    purpose = config.Field("Purpose of displaying list", str, 'all')

class sliceIntoStreamsConfig(config.Config):
    root_stream_name = config.Field("Root name for streams", str, "ext")
    copy = config.Field("Populate streams with deepcopies of the slices?", bool, False)

class sortInputsConfig(config.Config):
    descriptor = config.Field("Name of descriptor for sorting", str, 'filename')
    reverse = config.Field("Reverse order of sort?", bool, False)

class transferAttributeConfig(config.Config):
    suffix = config.Field("Filename suffix", str, "_attributeTransferred", optional=True)
    source = config.Field("Stream to transfer from", str, None)
    attribute = config.Field("Name of attribute to transfer", str, None)

class writeOutputsConfig(config.Config):
    overwrite = config.Field("Overwrite exsting files?", bool, True)
    outfilename = config.Field("Output filename", str, None, optional=True)
    prefix = config.Field("Prefix for output files", str, '', optional=True)
    suffix = config.Field("Suffix for output files", str, '', optional=True)
    strip = config.Field("Strip prefix/suffix from filenames?", bool, False)
