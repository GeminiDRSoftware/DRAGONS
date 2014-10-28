_throw_descriptor_exception = False

def set_descriptor_throw(flag = True):
    """ This function stores a flag the descriptor system uses to throw exceptions
    when descriptors do, as opposed to the default behavior of catching the exception
    and returning None. """
    global _throw_descriptor_exception
    _throw_descriptor_exception = flag
     
def get_descriptor_throw():
    global _throw_descriptor_exception
    return _throw_descriptor_exception