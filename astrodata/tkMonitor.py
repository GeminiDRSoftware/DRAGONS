#tkmonitor
# This program creates a text area which contains room for multiple lines of text 
from Tkinter import *
import threading
from datetime import datetime
from datetime import timedelta
from copy import copy

import re, string, time
from time import sleep
 
class GUIExcept:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised in Recipe System"):
        """This constructor takes a message to print to the user."""
        self.message = msg
    def __str__(self):
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message
        
class StatusBar:
    dt = None
    total = None
    co = None
    def __init__(self, master, co = None):
        self.label = Label(master, text="", bd=1, relief=SUNKEN, anchor=W)
        self.label.pack(side=BOTTOM, fill=X)
        self.dt = datetime.now()
        self.label.after(100, self.timer)
        self.total = timedelta(seconds=60.)
        self.co = co
        
    def timer(self, *args):
        if self.co != None:
            if not self.co.is_paused():
                rightnow = datetime.now()
                timeused = rightnow - self.dt
                self.total = self.total - timeused
                self.dt = rightnow
                self.set(str(self.total))
        self.label.after(100, self.timer)
       
    def set(self, format, *args):
        self.label.config(text=format % args)
        self.label.update_idletasks()
    
    def clear(self):
        self.label.config(text="")
        self.label.update_idletasks()
         
class TkRecipeControl( threading.Thread):
    pcqid = 0
    control = None
    # main = None
    monitor = None
    bReady = False
    xloc = 0
    cmdQueue = []
    controlWindows = []
    recipes = None
    initX = 0
    initY = 0
    statusBar = None
    mainWindow = None
    killed = False
    
    def __init__(self, recipe=None, recipes=None, b_thread=True):
        if recipes != None: # then it has a list interface please
            self.recipes = copy(recipes)
        elif recipe !=None:
            self.recipes = [recipe]
        else:
            raise GUIExcept()
        
        if (b_thread):
            threading.Thread.__init__(self)
        else:
            self.run()
            
    def new_control_window(self, recipe, context):
        self.cmdQueue.append( 
            { "cmd": "newRecipe", 
              "recipe" : recipe,
              "context": context
            }
        )
        
    def running(self, recipe):
        self.cmdQueue.append(
            { "cmd": "running",
              "recipe":recipe,
            }
        )
    def done(self):
        self.cmdQueue.append(
            { "cmd": "done",
            }
        )
        
    def quit(self):
        self.cmdQueue.append(
            { "cmd": "quit",
            }
        )
    
    def iq_log(self, name, val, timestr):
        self.cmdQueue.append(
            {   "cmd":"iqlog",
                "name":name,
                "val":val,
                "timestr" : timestr
            }
        )
            
    def set_context(self, co):
        self.co = co
        if (self.control != None):
            self.control.co = co
        if (self.mainWindow != None):
            self.mainWindow.co = co
        if (self.monitor != None):
            self.monitor.co = co
        if (self.statusBar != None):
            self.statusBar.co = co
            
            
    def run(self):
        self.mainWindow = Tk()
        self.mainWindow.title("Recipe Manager Control")
        self.recBtns = {}
        self.mainWindow.protocol("WM_DELETE_WINDOW", self.finish) 

        #self.statusBar = StatusBar(self.mainWindow)
        self.recButFrame = Frame(self.mainWindow)
        self.recButFrame.pack(side=TOP)
        bw = self.recButFrame
        for rec in self.recipes:
            recBtn = Button(bw, text=rec)
            recBtn.pack(side=LEFT, fill = BOTH, expand=1)
            self.recBtns.update({rec: recBtn})
        #add iq log
        self.iqLogFrame = Frame(self.mainWindow)
        iqw = self.iqLogFrame
        iqw.pack(side=BOTTOM, expand = 1, fill=BOTH)
        self.iqscroll = Scrollbar(iqw)
        self.iqscroll.pack(side=RIGHT, fill=Y)
        self.iqLogText = Text(iqw, height=5, width=60, background="black")
        self.iqLogText.pack(side=BOTTOM, expand=1, fill=BOTH)
        self.iqLogText.config(yscrollcommand=self.iqscroll.set)
        self.iqLogText.tag_config("iqtime", 
                                foreground="green") 
        self.iqLogText.tag_config("iqname", 
                                foreground="yellow")
        self.iqLogText.tag_config("iqval", 
                                foreground="white")
                                

        self.iqscroll.config(command=self.iqLogText.yview)
        
        #start command queue timer
        self.mainWindow.after(100, self.process_cmd_queue)
                
        self.mainWindow.geometry("+0+0")
        self.mainWindow.update()
        geom = self.mainWindow.geometry()
        # 95: ", geom
        
        mong = re.match(r'(?P<w>.*?)x(?P<h>.*?)\+(?P<x>.*?)\+(?P<y>.*?)$', geom)
        self.initY = string.atoi(mong.group("h")) + 35
        # print "TK98: " , self.initY
        self.bReady = True
        self.mainWindow.mainloop()
        #print "done with mainloop"
        
    def finish(self):
        for cw in self.controlWindows:
            cw.context.finish()
            cw.destroy()
        self.mainWindow.destroy()
        self.mainWindow.quit()
        self.killed=True
        
    def process_cmd_queue(self):
        # to reschedule this function on a tk timer
        reschedule = True
        
        #schedule myself again
        myqueue = copy(self.cmdQueue)
        self.cmdQueue = []
        for cmdevent in myqueue:
            cmd = cmdevent["cmd"]
            # self.iqLogText.insert(END, "tk148:"+cmd+"\n")
            #print "TK85: ",cmd
            if  cmd == "running":
                recipe = cmdevent["recipe"]
                for rec in self.recBtns.keys():
                    if rec == recipe:
                        self.recBtns[rec].configure(foreground = "#00c000")
                    else:
                        self.recBtns[rec].configure(foreground = "black")
            elif cmd == "iqlog":
                # this is funky, to get a reversed order log, with the best at top.

                self.iqLogText.insert(END, cmdevent["timestr"], "iqtime")
                self.iqLogText.insert(END, string.center(cmdevent["name"],20,"-"), "iqname")
                self.iqLogText.insert(END, string.ljust(cmdevent["val"], 20)+"\n", "iqval")
                self.iqLogText.see(END + "- 2 chars")
            elif cmd == "quit":
                self.mainWindow.destroy()
                self.mainWindow.quit()
                reschedule = False
            elif cmd == "done":
                for rec in self.recBtns.keys():
                    self.recBtns[rec].configure(foreground = "black")
                
            elif cmd == "newRecipe":
                context =  cmdevent["context"]
                recipe = cmdevent["recipe"]
                # print "TK92: ", recipe, context

                controlWindow = Toplevel(self.mainWindow)
                self.controlWindows.append(controlWindow)
                controlWindow.protocol("WM_DELETE_WINDOW", context.finish)
                
                cw = controlWindow # alias
                cw.context = context
                cw.recipe  = recipe
                                
                cw.title("Recipe: %s" % recipe)
                controlFrame = Frame(cw)
                rc = RecipeControl(controlFrame, controlWindow)
                
                rc.co = context
                rc.co.add_callback("pause", rc.sg_pause)
                monitorFrame = Frame(cw)
                MonitorRecipe(monitorFrame).co = context
                
                controlFrame.pack(side=TOP, expand=1, fill=X)
                monitorFrame.pack(side=BOTTOM, expand=1, fill=BOTH)
                controlWindow.update()
                
                geom = controlWindow.geometry()
                # print "TK144: ",geom 
                mong = re.match(r'(?P<w>.*?)x(?P<h>.*?)\+(?P<x>.*?)\+(?P<y>.*?)$', geom)
                geomstr = "%dx%d+%d+%d" % (string.atoi(mong.group("w")),
                                           string.atoi(mong.group("h")),
                                           self.initX,
                                           self.initY)
                # print "TK149:", geomstr
                controlWindow.geometry(geomstr)
                self.initX += 25 #string.atoi(mong.group("w"))+5
                self.initY += 20
                self.set_context(context)
                
                #self.monitor = MonitorWindow(root)
        
        if reschedule:
            self.pcqid = self.mainWindow.after(10, self.process_cmd_queue)
            
class RecipeControl:
    co = None
    bPaused = False # toggles pause function purpose
    
    def __init__(self, window, closer = None):
        self.window = window
        if closer == None:
            closer = window

        self.pauseBtn = Button(self.window, text="pause", command=self.pause)
        self.pauseBtn.pack(side=LEFT, fill = X, expand=1)
        
        self.closeBtn = Button(self.window, text="close", command=closer.destroy)
        self.closeBtn.pack(side=RIGHT, fill= X, expand =1)
    
        self.cancelBtn = Button(self.window, text="cancel", command=self.cancel)
        self.cancelBtn.pack(side=RIGHT, fill=X, expand =1)

        #self.poll()
        
    def pause(self):
        if self.co:
            if self.bPaused:
                self.bPaused = False
                self.pauseBtn.configure(text="pause")
                self.co.unpause()
            else:
                self.bPaused = True
                self.pauseBtn.configure(text="resume")
                self.co.request_pause()
            

    def cancel(self):
        if self.co:
            self.co.is_finished(True)
        
        
    def poll(self):
        self.window.after(100,self.poll)
        # if self.co != None:
        #    print "co",self.co
        
    def sg_pause(self):
        """callback for system generated pause"""
        if not self.bPaused:
            self.bPaused = True
            self.pauseBtn.configure(text="resume")

    def cancel(self):
        if self.co:
            self.co.is_finished(True)
        
        
    def poll(self):
        self.window.after(100,self.poll)
        # if self.co != None:
        #    print "co",self.co
        
        
class MonitorRecipe:
    co = None
    lastTS = None
    durlablist = None # duration labels list for "begin" marks
    already_paused = False
    already_waiting_for_pause = False
    def __init__(self, window):
        self.window = window
    
        self.durlablist = {}
        self.scrollbar = Scrollbar(self.window)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.textWdg = Text(self.window, height=50, width=60, background="white")
        self.textWdg.pack(side=TOP, expand=1, fill=BOTH)
        self.textWdg.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.textWdg.yview)
        
        self.textWdg.tag_config("text", justify=CENTER)
        self.textWdg.tag_config("bold", font = ("times", 13,"bold"),justify=CENTER)
        self.textWdg.tag_config("data", foreground="blue", font=("courier", 12, "bold"), justify=CENTER)
        self.textWdg.tag_config("arrow", font = ("OpenSymbol", 24),justify=CENTER)
        self.textWdg.tag_config("urgent", 
                                foreground="red", 
                                font=("arial", 18, "bold"), 
                                justify=CENTER)
        self.textWdg.tag_config("warn", 
                                foreground="#c0c000", 
                                font=("arial", 16, "bold"), 
                                justify=CENTER)
        self.textWdg.tag_config("info", 
                                foreground="green", 
                                font=("arial", 16, "bold"), 
                                justify=CENTER)
        
        self.poll()
    
    
    def poll(self):        
        if self.co != None:
            sh = self.co.stephistory
            
            if self.co.pause_requested():
                if not self.already_waiting_for_pause:
                    self.textWdg.insert(END, "...waiting for pause...\n", "info")
                    self.already_waiting_for_pause = True
            else:
                self.already_waiting_for_pause = False
            if self.co.paused:
                
                if not self.already_paused:
                    self.already_paused = True
                    self.textWdg.insert(END, "PAUSED\n", "warn")
                    label = Label(self.textWdg, foreground="#c0c000",text="duration")
                    self.textWdg.insert(END, " ", "text")
                    self.textWdg.window_create(END, window = label, align=CENTER)
                    self.textWdg.insert(END, " \n", "text")
                    self.durlablist.update({("pause",0):(label,datetime.now())})
            else:
                self.already_paused = False
                if ("pause",0) in self.durlablist:
                    del (self.durlablist[("pause",0)])
                    
            
            
            marks = sh.keys()
            marks.sort()
            prevmark = None
            for mark in marks:
                marktype = sh[mark]["mark"]
                stepname = sh[mark]["stepname"]
                
                if self.lastTS == None or self.lastTS < mark:
                    
                    if marktype == "begin":
                        # put inputs
                        self.textWdg.insert(END, self.co.inputs_as_str() + "\n", "data")
                        self.textWdg.insert(END, unichr(8595)+"\n","arrow")
                    
                        self.textWdg.insert(END, stepname+"\n", "bold")
                        label = Label(self.textWdg, text="duration")
                        self.textWdg.insert(END, " ", "text")
                        self.textWdg.window_create(END, window = label, align=CENTER)
                        self.textWdg.insert(END, " \n", "text")
                        indent = sh[mark]["indent"]
                        self.durlablist.update({(sh[mark]["stepname"],indent):(label,mark)})
                        
                        self.textWdg.see(END)
                    
                    if marktype == "end":
                        indent = sh[mark]["indent"]
                        # print "TK292: ",indent
                        beginmark = self.co.get_begin_mark(stepname, indent = indent)
                        # print "TK293: ", beginmark, "\n", sh[mark]
                        if beginmark == None:
                            raise "Bad PROBLEM\nCorrupted stephistory\n END with no BEGIN"
                        else:
                            dtime = mark - beginmark[0]
                            elabel = self.durlablist[(stepname, indent)][0]
                            elabel.config(text = str(dtime))
                            del self.durlablist[(stepname,indent)]
                        if (prevmark != None):
                            if stepname != sh[prevmark]["stepname"]:
                                self.textWdg.insert(END, stepname +" (end)\n", "bold")
                        self.textWdg.insert(END, unichr(8595)+"\n","arrow")
                        self.textWdg.see(END)
                    
                    
                    
                    #selef.textWdg.insert(END, str(mark)+ "\n", "text")
                    self.lastTS = mark
                prevmark = mark
                
        # update durations, because of nesting this is any arbitrary labels
        # print "TK322:-----"
        for durstep in self.durlablist:
            # print "TK322: ", durstep
            if not self.co.paused or (durstep == "pause"):
                label = self.durlablist[durstep][0]
                tstamp = self.durlablist[durstep][1]
                dtime = datetime.now() - tstamp
                label.config(text = str(dtime))
        # print "TK322:-----"
                
        if self.co == None or (self.co.finished != True):
            self.window.after(100, self.poll)
        else:
            self.textWdg.insert(END, self.co.status, "urgent")

        
        
