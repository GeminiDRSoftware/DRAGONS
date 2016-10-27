from ..utils import logutils
log = logutils.get_logger(__name__)

class ExternalTaskInterface(object):
    """
    The External Task Interface base class. This is a way for the Recipe 
    System to interact with ouside software. It prepares, executes, recovers,
    and cleans all files and parameters pertaining to any external task
    that interfaces with the recipe system.
    """
    param_objs = None
    file_objs = None
    rc = None
    def __init__(self, rc=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("ExternalTaskInterface __init__")
        self.rc = rc
        self.param_objs = [] 
        self.file_objs = []

    def run(self):
        log.debug("ExternalTaskInterface.run()")
        self.prepare()
        self.execute()
        self.recover()
        self.clean()

    def add_param(self, param):
        log.debug("ExternalTaskInterface.add_param()")
        self.param_objs.append(param)

    def add_file(self, file):
        log.debug("ExternalTaskInterface.add_file()")
        self.file_objs.append(file)
        
    def prepare(self):
        log.debug("ExternalTaskInterface.prepare()")
        for par in self.param_objs:
            par.prepare()
        for fil in self.file_objs:
            fil.prepare()

    def execute(self):
        log.debug("ExternalTaskInterface.execute()")
        pass 
    
    def recover(self):
        log.debug("ExternalTaskInterface.recover()")
        for par in self.param_objs:
            par.recover()
        for fil in self.file_objs:
            fil.recover()
    
    def clean(self):
        log.debug("ExternalTaskInterface.clean()")
        for par in self.param_objs:
            par.clean()
        for fil in self.file_objs:
            fil.clean()
        
        

