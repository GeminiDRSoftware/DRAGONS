class Requirement(object):
    def __or__(self, orwith):
        return OrReq([self, orwith])
    def __and__(self, andwith):
        return AndReq([self, andwith])

    def satisfied_by(self, hdulist):
        return False

class OrReq(Requirement):
    orList = None
    def __init__(self, *args):
        """Arguments are put in list of requirements to OR, 
            @param args: Variable length argument list, all arguments are
            put into the list of requirments to or. Iterable
            arguments are added to the list with list.extend()
            and non-iterable arguments are added with list.append().
            @type args: Instances of classes descending from
            AstroDataType.Requirement.
        """
        self.orList = []
        for arg in args:
            if hasattr(arg,'__iter__'):
                self.orList.extend(arg)
            else:
                self.orList.append(arg)
        
    def satisfied_by(self, hdulist):
        for req in self.orList:
            if req.satisfied_by(hdulist) == True:
                return True
        return False    
OR=OrReq

class NotReq(Requirement):
    req = None
    def __init__(self, req):
        self.req = req
    def satisfied_by(self, hdulist):
        return not self.req.satisfied_by(hdulist)
NOT=NotReq
        
class AndReq(Requirement):
    """Arguments are put in list of requirements to AND, 
        @param args: Variable length argument list, all arguments are
        put into the list of requirments to and. Iterable
        arguments are added to the list with list.extend()
        and non-iterable arguments are added with list.append().
        @type args: Instances of classes descending from
        AstroDataType.Requirement.
    """
    and_list = None
    def __init__(self, *args):
        self.and_list = []
        for arg in args:
            if hasattr(arg,'__iter__'):
                self.and_list.extend(arg)
            else:
                self.and_list.append(arg)
    def satisfied_by(self, hdulist):
        for req in self.and_list:
            if not req.satisfied_by(hdulist):
                return False
        
        return True
AND=AndReq            