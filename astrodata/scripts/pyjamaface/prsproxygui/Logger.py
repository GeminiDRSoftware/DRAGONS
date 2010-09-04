from pyjamas.ui.Grid import Grid

class Logger(Grid):
    instances = []
    
    def __init__(self):
        Logger.instances.append(self)

        Grid.__init__(self)

        self.targets=[]
        self.targets.append("app")
        #self.targets.append("ui")
        self.resize(len(self.targets)+1, 2)
        self.setBorderWidth("1")
        self.counter=0
        
        self.setHTML(0, 0, "<b>Log</b>")
        self.setText(1, 0, "app")
        for i in range(len(self.targets)):
            target=self.targets[i]
            self.setText(i+1, 0, target)
    
    @classmethod
    def getSingleton(self):
        return Logger.singleton
    
    def setSingleton(self):
        Logger.singleton = self
    
    def addTarget(self, target):
        self.targets.append(target)
        self.resize(len(self.targets)+1, 2)
        self.setText(len(self.targets), 0, target)
        return self.targets.index(target)
    
    @classmethod
    def write(cls, target, message):    
        for logger in cls.instances:
            logger.onMessage(target, message)
    
    def onMessage(self, target, message):
        self.counter+=1
        
        if target=='':
            target='app'
        target_idx=self.targets.index(target)
        
        # add new target
        if target_idx<0:
            target_idx=self.addTarget(target)
        
        target_row=target_idx+1     
        old_text=self.getHTML(target_row, 1)
        log_line=self.counter + ": " + message

        if old_text=='&nbsp;':
            new_text=log_line            
        else:
            new_text=old_text + "<br>" + log_line
        self.setHTML(target_row, 1, new_text) 

