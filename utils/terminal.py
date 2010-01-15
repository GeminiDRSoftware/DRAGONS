import os
import sys, re
import textwrap

import unicodedata, re
# os.environ["TERM"] = "vt50"

forceWidth = None
forceHeight = None

class TerminalController:
    """
    A class that can be used to portably generate formatted output to
    a terminal.  
    
    `TerminalController` defines a set of instance variables whose
    values are initialized to the control sequence necessary to
    perform a given action.  These can be simply included in normal
    output to the terminal:

        >>> term = TerminalController()
        >>> print 'This is '+term.GREEN+'green'+term.NORMAL

    Alternatively, the `render()` method can used, which replaces
    '${action}' with the string required to perform 'action':

        >>> term = TerminalController()
        >>> print term.render('This is ${GREEN}green${NORMAL}')

    If the terminal doesn't support a given action, then the value of
    the corresponding instance variable will be set to ''.  As a
    result, the above code will still work on terminals that do not
    support color, except that their output will not be colored.
    Also, this means that you can test whether the terminal supports a
    given action by simply testing the truth value of the
    corresponding instance variable:

        >>> term = TerminalController()
        >>> if term.CLEAR_SCREEN:
        ...     print 'This terminal supports clearning the screen.'

    Finally, if the width and height of the terminal are known, then
    they will be stored in the `COLS` and `LINES` attributes.
    """
    # Cursor movement:
    BOL = ''             #: Move the cursor to the beginning of the line
    UP = ''              #: Move the cursor up one line
    DOWN = ''            #: Move the cursor down one line
    LEFT = ''            #: Move the cursor left one char
    RIGHT = ''           #: Move the cursor right one char

    # Deletion:
    CLEAR_SCREEN = ''    #: Clear the screen and move to home position
    CLEAR_EOL = ''       #: Clear to the end of the line.
    CLEAR_BOL = ''       #: Clear to the beginning of the line.
    CLEAR_EOS = ''       #: Clear to the end of the screen

    # Output modes:
    BOLD = ''            #: Turn on bold mode
    BLINK = ''           #: Turn on blink mode
    DIM = ''             #: Turn on half-bright mode
    REVERSE = ''         #: Turn on reverse-video mode
    NORMAL = ''          #: Turn off all modes

    # Cursor display:
    HIDE_CURSOR = ''     #: Make the cursor invisible
    SHOW_CURSOR = ''     #: Make the cursor visible

    # Terminal size:
    COLS = None          #: Width of the terminal (None for unknown)
    LINES = None         #: Height of the terminal (None for unknown)

    # Foreground colors:
    BLACK = BLUE = GREEN = CYAN = RED = MAGENTA = YELLOW = WHITE = ''
    
    # Background colors:
    BG_BLACK = BG_BLUE = BG_GREEN = BG_CYAN = ''
    BG_RED = BG_MAGENTA = BG_YELLOW = BG_WHITE = ''
    
    _STRING_CAPABILITIES = """
    BOL=cr UP=cuu1 DOWN=cud1 LEFT=cub1 RIGHT=cuf1
    CLEAR_SCREEN=clear CLEAR_EOL=el CLEAR_BOL=el1 CLEAR_EOS=ed BOLD=bold
    BLINK=blink DIM=dim REVERSE=rev UNDERLINE=smul NORMAL=sgr0
    HIDE_CURSOR=cinvis SHOW_CURSOR=cnorm""".split()
    _COLORS = """BLACK BLUE GREEN CYAN RED MAGENTA YELLOW WHITE""".split()
    _ANSICOLORS = "BLACK RED GREEN YELLOW BLUE MAGENTA CYAN WHITE".split()

    def __init__(self, term_stream=sys.stdout):
        """
        Create a `TerminalController` and initialize its attributes
        with appropriate values for the current terminal.
        `term_stream` is the stream that will be used for terminal
        output; if this stream is not a tty, then the terminal is
        assumed to be a dumb terminal (i.e., have no capabilities).
        """
        self.setupStripping()
        # Curses isn't available on all platforms
        try: import curses
        except: return

        # If the stream isn't a tty, then assume it has no capabilities.
        if not term_stream.isatty(): return

        # Check the terminal type.  If we fail, then assume that the
        # terminal has no capabilities.
        try: curses.setupterm()
        except: return

        # Look up numeric capabilities.
        self.COLS = curses.tigetnum('cols')
        self.LINES = curses.tigetnum('lines')
        
        # Look up string capabilities.
        for capability in self._STRING_CAPABILITIES:
            (attrib, cap_name) = capability.split('=')
            setattr(self, attrib, self._tigetstr(cap_name) or '')

        # Colors
        set_fg = self._tigetstr('setf')
        if set_fg:
            for i,color in zip(range(len(self._COLORS)), self._COLORS):
                setattr(self, color, curses.tparm(set_fg, i) or '')
        set_fg_ansi = self._tigetstr('setaf')
        if set_fg_ansi:
            for i,color in zip(range(len(self._ANSICOLORS)), self._ANSICOLORS):
                setattr(self, color, curses.tparm(set_fg_ansi, i) or '')
        set_bg = self._tigetstr('setb')
        if set_bg:
            for i,color in zip(range(len(self._COLORS)), self._COLORS):
                setattr(self, 'BG_'+color, curses.tparm(set_bg, i) or '')
        set_bg_ansi = self._tigetstr('setab')
        if set_bg_ansi:
            for i,color in zip(range(len(self._ANSICOLORS)), self._ANSICOLORS):
                setattr(self, 'BG_'+color, curses.tparm(set_bg_ansi, i) or '')

    def setupStripping(self):
        all_chars = (unichr(i) for i in xrange(0x10000))
        control_chars = ''.join(c for c in all_chars if unicodedata.category(c) == 'Cc')
        # or equivalently and much more efficiently
        control_chars = ''.join(map(unichr, range(0,32) + range(127,160)))

        self.control_char_re = re.compile('[%s]' % re.escape(control_chars))

    def _tigetstr(self, cap_name):
        # String capabilities can include "delays" of the form "$<2>".
        # For any modern terminal, we should be able to just ignore
        # these, so strip them out.
        import curses
        cap = curses.tigetstr(cap_name) or ''
        return re.sub(r'\$<\d+>[/*]?', '', cap)

    def renderLen(self, template):
        a = re.sub(r'\$\$|\${\w+}', "", template)
        return len (a)
        

    def printablestr(self, s):
        return self.control_char_re.sub('', s)

        
    def render(self, template):
        """
        Replace each $-substitutions in the given template string with
        the corresponding terminal control string (if it's defined) or
        '' (if it's not).
        """
        if template == None:
            raise "it's none"
        return re.sub(r'\$\$|\${\w+}', self._render_sub, template)
        
    def lenstr(self, coloredstr):
        clean = self.cleanstr(coloredstr)
        l = len (clean)
        return l
        
    def cleanstr(self, coloredstr):
        clean = re.sub(r'\$\$|\${\w+}', "", coloredstr)
        return clean

    def _render_sub(self, match):
        s = match.group()
        if s == '$$': return s
        else: 
            try:
                return getattr(self, s[2:-1])
            except:
                return ""

#######################################################################
# Example use case: progress bar
#######################################################################

class ProgressBar:
    """
    A 3-line progress bar, which looks like::
    
                         
       Header
        20% [===========----------------------------------]
                           progress message

    The progress bar is colored, if the terminal supports color
    output; and adjusts to the width of the terminal.
    """
    BAR = '%3d%% ${GREEN}[${BOLD}%s%s${NORMAL}${GREEN}]${NORMAL}\n'
    HEADER = '${BOLD}${CYAN}%s${NORMAL}\n\n'
        
    def __init__(self, term, header):
        self.term = term
        if not (self.term.CLEAR_EOL and self.term.UP and self.term.BOL):
            raise ValueError("Terminal isn't capable enough -- you "
                             "should use a simpler progress dispaly.")
        self.width = self.term.COLS or 75
        self.bar = term.render(self.BAR)
        self.header = self.term.render(self.HEADER % header.center(self.width))
        self.cleared = 1 #: true if we haven't drawn the bar yet.
        self.update(0, '')

    def update(self, percent, message):
        if self.cleared:
            sys.stdout.write(self.header)
            self.cleared = 0
        n = int((self.width-10)*percent)
        sys.stdout.write(
            self.term.BOL + self.term.UP + self.term.CLEAR_EOL +
            (self.bar % (100*percent, '='*n, '-'*(self.width-10-n))) +
            self.term.CLEAR_EOL + message.center(self.width))

    def clear(self):
        if not self.cleared:
            sys.stdout.write(self.term.BOL + self.term.CLEAR_EOL +
                             self.term.UP + self.term.CLEAR_EOL +
                             self.term.UP + self.term.CLEAR_EOL)
            self.cleared = 1
            
class FilteredStdout(object):
    term = None
    realstdout = None
    filters = None
    lastFendline = os.linesep
    writingForIraf = False
    lastWriteForIraf = False
    _linestart = True # generally true we start on a new line
    _needprefix = True
    curline = 0
    lastPrefix = ""
    lastPrefixLine = None
    
    def __init__(self, rso = None):
        # grab standard out
        if rso:
            self.realstdout = rso
        else:
            self.realstdout = sys.stdout
        self.filters = []
        self.term = TerminalController()
        
    def getLinestart(self):
        return self._linestart
        
    def setLinestart(self, to):
        if to == True:
            if self._linestart == False:
                self.curline += 1
        
        self._linestart = to
        
    linestart = property(getLinestart, setLinestart)

    def getNeedprefix(self):
        return self._needprefix
        
    def setNeedprefix(self, to):
        self._needprefix = to
        
    needprefix = property(getNeedprefix, setNeedprefix)
        
    def addFilter(self, nf):
        nf.term = self.term
        self.filters.insert(0,nf)
        self.realstdout.flush()
        nf.fout = self
        
    def removeFilter(self, nf):
        self.filters.remove(nf)
        self.realstdout.flush()
        
    def directWrite(self, out):
        self.realstdout.write(out)
        
    def write(self, out, forceprefix = None):
        # make termlog none to turn off debugging log
        
        termlog = None # open("termlog", "a")
        if termlog:
            termlog.write("\n"+"*"*40 + "\n")
            st = tb.extract_stack()
            termlog.write("STACK\n")
            for fr in st:
                if fr != st[-1]:
                    termlog.write("\t"+repr(fr)+"\n")
            
        #IRAF handling
        if self.lastWriteForIraf == True and self.writingForIraf == False:
            # nasty recursive call if you don't change the lastWriteForIraf flag
            self.lastWriteForIraf = False
            #self.write("\n")
        elif self.lastWriteForIraf == False and self.writingForIraf == True:
            # starting to write
            self.lastWriteForIraf = True
            self.needprefix = True
            self.write("\n")
        
        # create data structures
        out0 = out
        cleanout = self.term.cleanstr(out)
        lines = out.split(os.linesep)
        cleanlines = cleanout.split(os.linesep)
        lines0 = out0.split(os.linesep)
        
        lineslen = len(lines)
        # filter lines
        for i in range(0,lineslen):
            line = lines[i]
            for f in self.filters:
                if termlog:
                    termlog.write("\n"+repr(f))
                line = f.morph(line)
            lines[i] = line
        
        
            
            
        # ()()PREFIX()()()()()()()()()
        # Create Correct Prefix For these Line
        #
        # to debug write vs filter behavior override with uniform prefix --> 
        #          prefix = self.term.render("${NORMAL}${BOLD}prefix: ${NORMAL}")
        preprefix = None
        postprefix = None
        if forceprefix:
            prefix = forceprefix [0] + self.term.render(forceprefix[1]) + forceprefix[2]
            prefixclen = self.term.lenstr(prefix)
            prefix = self.term.render(prefix)
        else:
            # get prefix from top filter
            f0 = self.filters[0]
            if termlog:
                termlog.write("getting prefix from %s" % repr(f0))
            if f0.prefix:
                prefix = f0.preprefix + f0.prefix + f0.postprefix
                prefixclen = self.term.lenstr(prefix)
                prefix = self.term.render(prefix)

            else:
                prefixclen = 0
                prefix = self.term.render("")
        
        if self.lastPrefix != prefix:
            self.needprefix = True
            self.linestart = True
            self.lastPrefix = prefix
            self.realstdout.write("\n")
            self.needprefix = True
            self.linestart = True
        
        self.lastPrefix = prefix
        #
        # end MAKING PREFIX block
        # ()()()()()()()()()()()()()()
        
        #log all the elements
        
        if termlog:
            ps = """\n---------------------------------
             out0: %(out0)s
      out == out0: %(outcompare)s
           lines0: %(lines0)s
         cleanout: %(cleanout)s
       cleanlines: %(cleanlines)s
         lineslen: %(lineslen)s
           prefix: %(prefix)s
        linestart: %(linestart)s
       needprefix: %(needprefix)s
 lastWriteForIraf: %(lwfi)s
   writingForIraf: %(w4i)s
---------------------------------
""" % {     "out0": repr(out0),
            "outcompare": repr(out == out0),
            "lines0": repr(lines0),
            "cleanout": repr(cleanout),
            "cleanlines": repr(cleanlines),
            "lineslen": repr(lineslen),
            "prefix": repr(prefix),
            "linestart": repr(self.linestart),
            "needprefix": repr(self.needprefix),
            "lwfi":repr(self.lastWriteForIraf),
            "w4i":repr(self.writingForIraf)
            }
            termlog.write(ps)
            
        lastline = lineslen - 1
        for i in range(0,lineslen):
            propagateLinestart = False
            line = lines[i]
            if self.linestart:
                if self.needprefix:
                    self.realstdout.write(prefix)
                    self.needprefix = False
                    if termlog:
                        termlog.write("\nwrote prefix: %s\nset needprefix to False" % repr(prefix))
            # check for special " " the print statement sends... don't print
            #  if start of line.
            if (out == " "):
                if self.linestart:
                    # line = self.term.render("${RED}${REVERSE}X${NORMAL}")
                    line = ""
                    propagateLinestart = True
                else:
                    # line = self.term.render("${REVERSE}X${NORMAL}")
                    line = " "
                    
                    
            # ()()()()()()()()()()()()()()
            # WRITING MAIN LINE OUT
            (x,y) = getTerminalSize()
            if termlog:
                termlog.write("(wid,height)+prefixclen+linelen: "+str((x,y))+"+"+str(prefixclen)+"+"+str(len(cleanlines[i])))
            # note, this should us the length prefix clean!
            wid = x-prefixclen
            if len(cleanlines[i]) < wid:
                self.realstdout.write(line)
            else:        
                # loop and write continuations
                sublines = textwrap.wrap(line, wid)
                if sublines:
                    self.realstdout.write(sublines[0])
                    for subline in sublines[1:]:
                        self.realstdout.write("\n")
                        self.realstdout.write("."*(prefixclen-1)+" ")
                        self.realstdout.write(subline)
                else:
                    self.realstdout.write(line)
                    
                
            # ()()()()()()()()()()()()()()
            
            if termlog:
                termlog.write("\nwrote line: %s" % repr(line))
                
            if termlog:
                termlog.write("\nlen(cleanlines[%d] = %d" % (i, len(cleanlines[i])))
            if len(cleanlines[i]) > 0 and not propagateLinestart:
                self.linestart = False
                if termlog:
                    termlog.write("\nlinestart = False")
            else:
                if termlog:
                    termlog.write("\nlinestart unchanged (== %s)" % repr(self.linestart))
                    
            if i < lastline:
                self.realstdout.write(os.linesep)
                self.linestart = True
                self.needprefix = True
                if termlog:
                    termlog.write("\nwrote os.linesep, set linestart and needprefix to True")

        # IRAF HANDLING
        self.lastWriteForIraf = self.writingForIraf
            
        if termlog:
            termlog.write("\n"+"*"*40 + "\n")
                
        
    def flush(self):
        self.realstdout.flush()

class Filter(object):
    preprefix  = "${BOLD}"
    postprefix = "${NORMAL}"
    term    = None
    on      = True
    prefix  = None
    fout    = None
    def __init__(self, prefix=None, preprefix = "${BOLD}", postprefix= "${NORMAL}"):
        self.prefix = prefix
        self.preprefix = preprefix
        self.postprefix = postprefix
        
    def morph(self, arg):
        return arg 
    def clear(self):
        self.prefix = None
        
    def addPrefix(self, out):
        if self.prefix:
            return self.pretag+self.prefix+self.posttag + out
        else:
            return out    

class ColorFilter(Filter):
    
    def morph(self, arg):
        if self.on:
            out = self.term.render(arg)
            return out
        else:
            return ""
            
import traceback as tb
class PrimitiveFilter(Filter):
    _prefix = "primitive: "
    
    def getPrefix(self):
        h = ""
        st = tb.extract_stack()
        started = True # ignore deep part of stack
        for f in st:
            if started:
                if f[2]:
                    nam = f[2]
                else:
                    nam = "{}"
                if ("rimitives" in f[0]) or ("string" in f[0]):
                    h += (nam+": ")
        return h
    def setPrefix(self, val):
        self._prefix = val
        return
    prefix = property(getPrefix, setPrefix)
    
    def morph(self, arg):
        return (arg)
            
            
class IrafStdout():
    fout = None
    ifilter = None
    def __init__(self, fout = sys.stdout):
        self.fout = fout
        self.ifilter = IrafFilter()
        
    def write(self, out):
        out = self.ifilter.morph(out) 
        pout = out
        #pout = re.sub("\n", "\n${REVERSE}IRAF${NORMAL}: ", out)
        if hasattr(self.fout, "writingForIraf"):
            self.fout.writingForIraf = True
        self.fout.write(pout)
        if hasattr(self.fout, "writingForIraf"):
            self.fout.writingForIraf = False
    
    def flush(self):
        self.fout.flush()
        
class IrafFilter(Filter):
    prefix = "IRAF: "
    def morph(self, arg, first = True):
        if "PANIC" in arg or "ERROR" in arg:
            arg = "${RED}" + arg + "${NORMAL}"
        else:
            arg = "${BLUE}"+arg+"${NORMAL}"
        
        return (arg)

def getTerminalSize():
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct, os
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
        '1234'))
        except:
            return None
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        try:
            cr = (env['LINES'], env['COLUMNS'])
        except:
            cr = (25, 80)
    if forceWidth:
        cr = ( cr[0], forceWidth)
    if forceHeight:
        cr = ( forceHeight, cr[1])
    return int(cr[1]), int(cr[0])
