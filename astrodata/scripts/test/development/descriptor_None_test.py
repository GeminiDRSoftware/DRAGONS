from astrodata.Descriptors import DescriptorValue
from pprint import pformat
import sys


dval = DescriptorValue(10)

db = DescriptorValue(None)

print db.collapse_value()
print str(db)

ndict = {}
vdict = {}
complexdict = {}

for i in range(1,4):
    for nam in ["SCI", "VAR", "DQ"]:
        ndict[(nam,i)] = None
        vdict[(nam,i)] = 13
        complexdict[(nam,i)] = i*10
        
print "ndict = \n"
print pformat(ndict)
print "[]" * 10
print "vdict = \n"
print pformat(vdict)
print "[]"*10
print "complexdict = \n"
print pformat(complexdict)
print "[]"*10


dnone = DescriptorValue(None)
dn = DescriptorValue(ndict)
dv = DescriptorValue(vdict)
dcomplex = DescriptorValue(complexdict)


dnone.id = "DescriptorValue(None)"
dn.id = "DescriptorValue(--collapsable dict of Nones--)"
dv.id = "DescriptorValue(--collapsable dict of ints--)"
dcomplex.id = "DescriptorValue(--collapsable on extver--)"
dvals = [dnone, dn,dv,dcomplex]


for descval in dvals:
    print "-"*20
    print descval.id,":",str(descval)
    print "-"*20

print "+"*20
print "extver collaps on complex value"
for extver in dcomplex.ext_vers():
    print "dcomplex.collapse_value(%d)=%s" % (extver, dcomplex.collapse_value(extver))
print "+"*20
    
print "dcomplex.collapse_by_extver(self) ==> ", pformat(dcomplex.collapse_by_extver())