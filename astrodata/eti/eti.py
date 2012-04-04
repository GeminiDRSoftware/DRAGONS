
class ExternalTaskInterface(object):

    param_dict = None
    file_dict = None
    rc = None
    def __init__(self, rc=None):
        print ("ExternalTaskInterface __init__")
        self.rc = rc
        self.param_dict = {}
        self.file_dict = {}

    def run(self):
        print("ExternalTaskInterface.run()")
        self.prepare()
        self.execute()
        self.recover()

    def add_param(self, param):
        print("ExternalTaskInterface.add_param()")
        self.param_dict.update({repr(param.__class__):param})

    def add_file(self, fil):
        print("ExternalTaskInterface.add_file()")
        self.file_dict.update({repr(fil.__class__):fil})
        
    def prepare(self):
        print("ExternalTaskInterface.prepare()")

        for key in self.param_dict:
            self.param_dict[key].prepare()
        for key in self.file_dict:
            self.file_dict[key].prepare()

    def recover(self):
        print("ExternalTaskInterface.recover()")
        for key in self.param_dict:
            self.param_dict[key].recover()
        for key in self.file_dict:
            self.file_dict[key].recover()
    
    def execute(self):
        print("ExternalTaskInterface.execute()")
        pass 
        
        

    
