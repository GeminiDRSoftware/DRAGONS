import os

import FitsStorageConfig as fsc
import fscpatch

fscpatch.local_fsc_patch(fsc)


print "lfs9: %s" %fsc.fsc_localmode
