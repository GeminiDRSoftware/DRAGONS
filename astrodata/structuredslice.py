from Structures import centralStructureIndex, retrieve_structure_obj

csi = centralStructureIndex

for sname in csi:
    struct = retrieve_structure_obj(sname)
    exec("%s = struct" % (sname))
