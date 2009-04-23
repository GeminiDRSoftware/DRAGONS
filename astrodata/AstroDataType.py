import pyfits
import os
import re
import AstroData
import sys

from ConfigSpace import configWalk

verbose = False
verboseLoadTypes = True
verbt = False
ldebug = False

class LibraryNotLoaded:
    """Class for raising a particular exceptions"""
    pass
    
class BadArgument:
    """Class for raising a particular exception"""
    pass
    
class DataClassification(object):
    """
    The DataClassification Class encapsulates a single classification type, and
    knows how to recognize that type when given a pyfits.HDUList instance.
    Matching is currently done against PHU header keys, though the object
    is designed to be able to look elsewhere for classifying information.
    Classification configurations are classes subclassed from DataClassification
    with the class variables set appropriately to indicate the PHU requirements.

    The DataClassification class also allows specifying one type as dependant on
    another type, in which case the other type will try to match the PHU headers
    defined for it. When used through GeminiData applicable classification names are
    cached so the PHU is not checked repeatedly.
    
    This object is not intended for general us, and is a worker class for
    the L{ClassificationLibrary}, from the users point of view data classifications
    are handled as strings, i.e. classification names.  L{ClassificationLibrary}
    is therefore the proper interface to use
    for retrieving type information. However, most users will use the L{AstroData}
    classification interface which in turn rely on L{ClassificationLibrary}.
    
    NOTE: The configuration system and public interface makes a distinction between
    "typology" classifications and "processing status" classifications. Technically
    there is no real difference between these two types of classification, the 
    difference occurs in the purpose of the two, and the interfaces allow getting
    one or the other type, or both. In principle however, typology classifications
    relate to instrument-modes or other classifications that more or less still apply
    to the data after it has been transformed by processing (e.g. GMOS_IMAGE data
    is still GMOS_IMAGE data after flat fielding), and processing status 
    classifications will fail to apply after processing (e.g. GMOS_UNPREPARED data
    is no longer GMOS_UNPREPARED after running prepare, but changes instead 
    to GMOS_PREPARED).
    """
    # MEMBER VARIABLES
    # to protect from editing -via the web editing interface
    #   - this should be True, by default, it is false
    editprotect = False
    
    # Parameter Requirement Dictionary
    phuReqs = {}
    
    # Classifications have names
    #type name
    name = "Unclassified"
    
    # So the type knows its source file... if we don't store them on disk
    # then this would become some other locator.  Valuable if there is a
    # reason to update type definitions dynamically (i.e. to support a 
    # type definition editor)
    fullpath = ""
    
    # list of types this type depends on (e.g. to be GMOS-IFU you also must be GMOS type)
    typeReqs = []

    # classification library
    library = None
    
    usage = ""    
    
    # RAW TYPE: this is the raw type of the data, None, means search parent types and use their setting
    # if need be. This support calling descriptors in raw mode.
    rawType = None
    
    phuReqDocs = {
        "FILTER3" : "Specifies Instrument Mode for NIRI, Imaging vs. Spectroscopy",
        "INSTRUME": "Gemini-wide header specifying observing intrument",
        "MASKNAME": "Name of the Mask for GMOS MOS Spectroscopy",
        "OBSMODE" : "GMOS header specifying which mode of IMAGE, IFU, MOS, or LONGSLIT.",
        "OBSERVAT": "Name of the Observatory, Gemini-South or Gemini-North",
        "TELESCOP": "Name of the Observatory, Gemini-South or Gemini-North"
        }
   
    def assertType(self, hdulist):
        """
        This function will check to see if the given HDUList instance is
        of its classification. Currently this function checks PHU keys
        in the C{hdulist} argument as well as check to see if any classifications
        upon which this classification is dependent apply.  To extend
        what is checked to other details, such as headers in data extensions,
        this function must change or be overridden by a child class.
        @param hdulist: an HDUList as returned by pyfits.open()
        @type hdulist: pyfits.HDUList
        @return: C{True} if this class applies to C{hdulist}, C{False} otherwise.
        @rtype: bool
        """
        # print dir(fitsFile)
        if (ldebug):
            print "asserting %s on <||%s||>" % (self.name, hdulist)
            
        if (self.library == None):
            raise LibraryNotLoaded()

        try:
            phuCards = hdulist[0].header.ascard
            phuCardsKeys = phuCards.keys()
        except KeyError:
            return False
            
        numreqs = len(self.phuReqs) + len (self.typeReqs)
        numsatisfied = 0
        numviolated  = 0
        
        # CHECK THE typeReqs
        for typ in self.typeReqs:
            #print "type(%s)" % typ
            if (verbt) : print self.typeReqs
            if (self.library.checkType(typ, hdulist)):   #'hdulist' should be called 'hdulist'
                if(verbt) : print "satisfied"
                numsatisfied = numsatisfied + 1
            else:
                if (verbt) : print "unsatisfied"
                return False

        # CHECK THE phuReqs
        for reqkey in self.phuReqs.keys():
            # Note: the key can have special modifiers, so far, to indicate the
            # key is a regular expression (otherwise it's a string literal)
            # and also to indicate when a flag is not required but PROHIBITED.
            # other modifiers can be supported, they should appear in a comma 
            # separated list within "{}" before the KEY's text, e.g "{re}.*?PREPARE"
            
               
            # assume no key mods
            mods_re = False
            mods_prohibit = False
            
            m = re.match(r"(\{(?P<modifiers>.*?)\}){0,1}(?P<key>.*?)$", reqkey)
            #print "reqkey = %s" % reqkey
            #print self.library.typesDict
            
            # cleanreqkey... POSSIBLY A REGULAR EXPRESSION!!!
            cleanreqkeyPRE = reqkey
            if (m):
                modsstr = m.group("modifiers")
                if modsstr != None:
                    mods = modsstr.split(",")
                    if ("re" in mods):
                        mods_re = True
                    if ("prohibit" in mods):
                        mods_prohibit = True
                        #print "prohibiting %s (%s)" % (reqkey, m.group("key"))
                    
                    cleanreqkeyPRE = m.group("key")
            
            
            #print "cleanreqkey %s" % cleanreqkey
#            print "in? %s" % str(cleanreqkey in phuCards.keys())
#            print  str(phuCards.keys())
#                    
            #before checking the value, check if it exists
            # get list of keys
            if (mods_re):
                cleanreqkeylist = AstroData.reHeaderKeys(cleanreqkeyPRE,hdulist[0].header)
                if (cleanreqkeylist == None or len(cleanreqkeylist) == 0):
                    # no keys match, if this is a prohibited flag, return true
                    if (mods_prohibit == True):
                        numsatisfied += 1
                        continue
                    else: # otherwise false
                        return False
            else:
                cleanreqkeylist = [ cleanreqkeyPRE ]
            
            # @note: Here is a little trickery supporting "prohibited" tags and re Keys
            # ... to make this work for regular expression
            # lists of keys that matched, and treat the original non re-case as a special
            # case involves changing the 'return FALSE' logic... to return false ALL the
            # keys must fail to match, so we will start with a count equal to the number
            # of keys in the list, and only return false if all of them result in False
            numviablekeys = len(cleanreqkeylist)
            
            for cleanreqkey in cleanreqkeylist:

                if cleanreqkey in phuCardsKeys:
                    # else no mods, so they remain false            
                    try:
                        #print "reqkey %s" % reqkey
    #                    print "re %s" % self.phuReqs[reqkey]
    #                    print "phuReqs %s" % self.phuReqs.keys()
                        if (re.match(str(self.phuReqs[reqkey]), str(phuCards[cleanreqkey].value) )):
                            if (verbose) : 
                                print "%s header matched %s with value of %s" % (fname, reqkey, phuCards[reqkey])
                            if (mods_prohibit == False): # in which case the requirement is that it NOT exist
                                numsatisfied = numsatisfied + 1
                            else:
                                numviablekeys -= 1
                                if numviablekeys == 0:
                                    return False #only takes one failed requirement to fail the instance
                        else :
                            if (verbose) : 
                                print "%s header DID NOT match %s with value of %s" % (fname, reqkey, phuCards[reqkey])
                            if (mods_prohibit == True): # then the requirement is that it NOt exist, and this satisfies
                                numsatisfied = numsatisfied + 1
                            else:
                                numviablekeys -= 1
                                if numviablekeys == 0:                     
                                    return False
                    except KeyError:
                        if (mods_prohibit == True):
                            numsatisfied = numsatisfied + 1
                        else:
                            if (verbose) : 
                                print "%s header DID NOT match %s with value of %s" % (fname, reqkey, phuCards[reqkey])
                            numviablekeys -= 1
                            if numviablekeys == 0:
                                return False
                else:
                    # it's not in there... fail unless prohibited
                    if (mods_prohibit == True):
                        numsatisfied += 1
                    else:
                        numviablekeys -= 1
                        if numviablekeys == 0:
                            # note: this shouldn't happen for rekeys because the keys are already only those in the
                            # PHU checked above
                            return False

        
        if (verbt):
            print ("numreqs=%d\nnumsatisfied=%d\nnumviolated=%d" % (numreqs, numsatisfied, numviolated))
            
        if (verbose) : print "match"
        
        # if you made it here
        return True
    
            
    def isSubtypeOf(self,supertype):
        """This function is used to check type relationships. For this type to be
            a "subtype" of the given type both must occur in a linked tree.
            A
            node which is a relative leaf is a considered a subtype of its relative root.
            This is used to resolve conflicts that can occur when objects, features
            or subsystems are associated with a given classification. Since AstroData
            instances generally have more than one classification which applies to them
            and some associated objects, features, or subsystems require that
            they are the only one ultimately associated with a particular AstroData
            instance, such as is the case with Descriptors, this function is used
            to correct the most common case of this, in which one of the types is
            a subtype of the other, and therefore can be taken to override the
            parent level association.
            @param supertype: string name for "supertype"
            @type supertype: string
            @rtype: Bool
            @returns: True if the classification detected by this DataClassification 
            is subtype of the named C{supertype}.
        """
        # the task is to seek supertype in my parent (typeReqs).
        if supertype in self.typeReqs:
            #easy to check by string if in typeReqs (aka parent types)
            return True
            
        # otherwise check each supertype (recursively)
        for typ in self.typeReqs:
            to = self.library.getTypeObj(typ)
            if to != None:
                if to.isSubtypeOf(supertype):
                    return True
        
        # didn't make it... no one is the super
        return False
        
        
    def pythonClass(self):
        ''' This function generates a DataClassification Class based on self.
        The purpose of this is to support classification editors.
        @returns: a string containing python source code for this instance. Note, of 
        course, if you add functions or members to a child class derived from
        DataClassification, they will not be recognized by this function and
        will not be represented in the outputed code.
        @rtype: string
        '''
            
        class_templ='''
class %(typename)s(DataClassification):
    name="%(typename)s"
    usage = "%(usage)s"
    typeReqs= %(typeReqs)s
    phuReqs= %(phuReqs)s

newtypes.append(%(typename)s())
'''
        code = class_templ % { "typename":self.name,
                                "phuReqs": str(self.phuReqs),
                                "typeReqs": str(self.typeReqs),
                                "usage": self.usage
                               }
        
        return code 
              
    def writeClass(self):
        '''Generates python code and writes it to fullpath if not protected.  Note, to
        "protect" a file one must access the .py definition directly and set 
        'editprotect = True.

        NOTE: this is a fairly dangerous function to call, it will overwrite the file
        from which this classification was originally loaded. This is intended only
        to be called from DataClassification definition editors, presumably using
        a dedicated server in which it is OK to perhaps stomp the original file,
        considering one loses the previous file. That is, this is used in a 
        browser based interface in which the code is under version control, and 
        not used for image processing, but only to support the editor.  The revision
        control gives us safety against stomping a classification by mistake,
        and the fact that it is an isolated set of the code not used for data 
        processing ensures that any mistakes do not affect our running system since
        we do not check these changes in automatically, but do so manually after they 
        can be checked.
        '''
        if (self.editprotect == False):
            code = self.pythonClass()
            outf = open(self.fullpath, "w")
            outf.write(code)
            outf.close()
        
        
        
    def htmlEditForm(self):
        '''This function returns a full HTML block of text ready to present as a web form
        to allow for online editing of types. This supports the browser/web based
        editing interface for DataClassification types.
        @returns: a string containing an HTML form filled with default values taken
        from this instance's class members.
        @rtype: string
        '''
        
        edit_templ = '''
            <?xml version="1.0" encoding="iso-8859-1"?>
            <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
                  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
            <html xmlns="http://www.w3.org/1999/xhtml">
            <head>
              <meta http-equiv="content-type" content="text/html; charset=iso-8859-1" />
              <title>edit type</title>
              <meta name="generator" content="Amaya 9.54, see http://www.w3.org/Amaya/" />
            </head>

            <body style="background-color:#d0e0ff">
            <p><b>Editing AstroDataType: %(typename)s</b></p>

            <form action="edittype.py" method="POST">
              <p>Name of Type: <input type="text" name="typename" value="%(typename)s"/>
               <input type="submit" value="Submit" />
              </p>

              <p><strong>Usage appears in online documentation:</strong></p>

              <p>Usage: <input type="text" name="usage" value="%(usage)s" size="80"/> </p>

              <p><strong>List of types which must also apply to the data for this type to
              apply:</strong></p>

              <p>Special Type Of: <input type="text" name="typeReqs" value="%(typeReqs)s" size = "80"/></p>

              <p>PHU Characteristics (required value can be a regular expression):</p>

              <p>Keyword: <input type="text" name="key1" value="%(key1)s" /> Required Value: <input
              type="text" name="val1" value="%(val1)s"/></p>
              <p>Keyword: <input type="text" name="key2" value="%(key2)s"/> Required Value: <input
              type="text" name="val2" value="%(val2)s"/></p>
              <p>Keyword: <input type="text" name="key3" value="%(key3)s"/> Required Value: <input
              type="text" name="val3" value="%(val3)s"/></p>
              <p>Keyword: <input type="text" name="key4" value="%(key4)s"/> Required Value: <input
              type="text" name="val4" value="%(val4)s"/></p>
              <p>Keyword: <input type="text" name="key5" value="%(key5)s"/> Required Value: <input
              type="text" name="val5" value="%(val5)s"/></p>
              <p>Keyword: <input type="text" name="key6" value="%(key6)s"/> Required Value: <input
              type="text" name="val6" value="%(val6)s"/></p>
              <p>Keyword: <input type="text" name="key7" value="%(key7)s"/> Required Value: <input
              type="text" name="val7" value="%(val7)s"/></p>
              <p>Keyword: <input type="text" name="key8" value="%(key8)s"/> Required Value: <input
              type="text" name="val8" value="%(val8)s"/></p>
              <p>Keyword: <input type="text" name="key9" value="%(key9)s"/> Required Value: <input
              type="text" name="val9" value="%(val9)s"/></p>
              <p>Keyword: <input type="text" name="key10" value="%(key10)s"/> Required Value: <input
              type="text" name="val10" value="%(val10)s"/></p>

            </form>
            </body>
            </html>
        '''
        
        formDict = {}
        formDict["typename"] = self.name
        formDict["usage"]    = self.usage
        #typeReqs
        typs = ""
        for typn in self.typeReqs:
            typs = typs + typn
        formDict["typeReqs"] = typs
        ind = 1 #using this to find correct form element
        for prkey in self.phuReqs:
            kstr = "key%d" % ind
            vstr = "val%d" % ind
            formDict[kstr] = prkey
            formDict[vstr] = self.phuReqs[prkey]
            ind = ind+1
            
        #NOTE: hardcoded, will change when I put in javascript to have
        #      arbitrary number of PHUReqs... debugging with hard coded 10
        #      for time being
        
        for emptyI in range(ind,11):
            kstr = "key%d" % emptyI
            vstr = "val%d" % emptyI 
            formDict[kstr] = ""
            formDict[vstr] = ""
        
        
        #editFormPage = edit_templ  + str(formDict)
        editFormPage = edit_templ % formDict
        
        
        return editFormPage
        
    def htmlDoc(self):
        '''
        This function returns a string representation for an HTML table which documents this
        particular classification using the definition itself. Note that the C{usage} member
        of DataClassification is used to describe the use of this classification and will
        be used to provide context and purpose of the classification
        when generating the documentation.
        @return: HTML table
        @rtype: string
        '''

        div_templ = '''
            <a name="%(typename)s"/>
            <div class="dataclassspec" style="padding:3px;border:1px solid #404040">
            
            <b style="font-size:1.15em">%(typename)s</b>&nbsp;&nbsp;(<i>%(fullpath)s</i>)</a>
                %(dictcommands)s
                 <div style="margin:2px;padding:1px;font-size:.8em">
                    %(typeapplic)s
                </div>
                <div style="margin:2px;padding:1px;">
                <i>Must Also Classify As:</i>:
                <b>%(typereqs)s</b>
                </div>
                <table width="100%%" border=1>
                    <colgroup span="3">
                        <col width="15em"/>
                        <col />
                        <col />
                    </colgroup>
                    <thead>
                        <tr style="background-color:#a0a0a0;">
                            <td >PHU Header Keyword</td>
                            <td>Regular Expression Requirement</td>
                            <td>Requirement Comment</td>
                        </tr>
                    </thead>
                    %(clrows)s
                </table>
            </div>
            '''

        # create dict commands
        if (self.editprotect == False):
            dictcommands ="""<br><small>
            (<a href="typedict.py?edit=%(typename)s">modify</a>)
            (<a href="typedelete.py?delete=%(typename)s">delete</a>)
            </small><br>
            """ % {"typename":self.name}
        else:
            dictcommands ="(cannot modify)"
        dictcommands
        # make the rows
        therows = ""
        for reqkey, reqre in self.phuReqs.items():
            try:
                comment = self.phuReqDocs[reqkey]
            except KeyError:
                comment = "&nbsp;"

            
            thisrow = '''
                <tr>
                    <td>%(parmkey)s</td>
                    <td>%(re)s</td>
                    <td>%(comment)s</td>
                </tr>
                ''' % { "parmkey":reqkey, "re":reqre , "comment":comment }
            therows = therows + thisrow
        # make the typereqs string
        if (self.typeReqs == None) or (len(self.typeReqs) == 0):
            trstr = "none"
        else:
            trstr = ""
            for trs in self.typeReqs:
                trstr = trstr + " <a href=\"#%(typ)s\">%(typ)s</a>" % {"typ":trs} 

        retstr = div_templ % {  "dictcommands":dictcommands,
                                "clrows":therows, 
                                "typename":self.name, 
                                "fullpath":self.fullpath,
                                "typeapplic":self.usage,
                                "typereqs":trstr
                                }

        return retstr   

    # graphviz section, uses DOT language to make directed graphs
    # of the type dictionary
    def gvizLinks(self):
        """
        This function supports the automatic generation of a class graph
        driven by the Classification Library. This system builds a script
        for "dot".  The ClassifcationLibrary class handles the script template
        and calls DataClassification functions to get the component strings.
        Links are directed from parent type to sub type.
        @return: a string containing node links for dot script graph visualization
        language
        @rtype: string
        """
        linkstr = ""
        for req in self.typeReqs:
            linkstr = linkstr + "\t %(from)s -> %(to)s; \n" \
                % { "from":self.name, 
                    "to":req, 
                    "url": ("typedict.py#%s" % self.name )
                }
        return linkstr
        
    def gvizNodes(self):
        """This function supports the automatic generation of a class graph
        which is driven by the Classification Library. This function returns
        a "dot language" representation of the node, which can contain things
        such as an URL to click (we use SVG output), or any other information
        about the node.
        @return: A representation of the node  in "dot" language for graphing purposes
        @rtype: String
        """
        nodestr = "%(name)s [URL=\"typedict.py#%(name)s\",tooltip=\"%(tip)s\"];\n" \
                  % {"name":self.name,"tip":self.usage.replace("\n", "")}
        return nodestr
        
class ORClassification(DataClassification): 
    """This class is a special type of DataClassification. When a classification
    refers to other types it is dependent on, the data in question must be all
    the specified types. In contrast to that, this class allows
    specification of set of DataClassification
    names on which this DataClassification depends, but instead of all of them
    having to apply, it is sufficient if any one of them apply. 
    
    B{NOTE}: This class does not check the PHU or standard dependencies and is 
    used ONLY to check the "OR" requirement."""

    typeORs = []

    def assertType(self, hdulist):
        """This function works just as regular assertType except
        that it calls all the members of a list of possible satisfiers...
        e.g. GMOS type can use this to check if the image is either
        GMOS_N or GMOS_S. NOTE: this function in this class ONLY checks
        the typeORs list, and will not at this time do header checking.
        
        @param hdulist: an HDUList as returned by pyfits.open()
        @type hdulist: HDUList
        @return: C{True} if this class applies to C{hdulist}, C{False} otherwise.
        """
        satisfies = False
        
        for typ in self.typeORs:
            if (self.library.checkType(typ, hdulist ) == True):
                satisfies = True
                break
            
        # NOTE: 
        return satisfies
        
    # graphviz section, uses DOT language to make directed graphs
    # of the type dictionary
    def gvizLinks(self):
        """
        This function supports the automatic generation of a class graph
        driven by the Classification Library. This system builds a script
        for "dot".  The ClassifcationLibrary class handles the script template
        and calls DataClassification functions to get the component strings.
        Links are directed from parent type to sub type.
        @return: String containing node links for dot script
        @rtype: String
        """
        linkstr = ""
        for req in self.typeORs:
            linkstr = linkstr + "\t %(from)s -> %(to)s [style=dotted]; \n" \
                % { "from":self.name, 
                    "to":req, 
                    "url": ("typedict.py#%s" % self.name )
                }
        return linkstr
        
    def gvizNodes(self):
        """This function support the automatic generation of a class graph
        which is driven by the Classification Library. This function returns
        a "dot language" representation of the node, which can contain things
        such as an URL to click (we use SVG output), or any other information
        about the node.
        @return: A representation of the node  in "dot" language for graphing purposes
        @rtype: String
        """
        nodestr = "%(name)s [shape=ellipse,URL=\"typedict.py#%(name)s\",tooltip=\"%(tip)s\"];\n" \
                  % {"name":self.name,"tip":self.usage.replace("\n", "")}
        return nodestr  
	            
    def htmlDoc(self):
        '''
        This function returns a string representation for an HTML table which documents this
        particular classification using the definition itself. Note that the C{usage} member
        of DataClassification is used to describe the use of this classification and will
        be used to provide context and purpose of the classification
        when generating the documentation.
        @return: HTML table
        @rtype: string
        '''

        div_templ = '''
            <div class="dataclassspec" style="padding:3px;border:1px solid #404040">
            <a name="%(typename)s">
            <b style="font-size:1.15em">%(typename)s</b>&nbsp;&nbsp;(<i>%(fullpath)s</i>)</a>
                 <div style="margin:2px;padding:1px;font-size:.8em">
                    %(typeapplic)s
                </div>
                <div style="margin:2px;padding:1px;">
                <i>Must Be One Or More Of:</i>:
                <b>%(typereqs)s</b>
                </div>
        %(phutable)s
        </div>
            '''
            
        phutable_templ = '''
                <table width="100%%" border=1>
                    <colgroup span="3">
                        <col width="15em"/>
                        <col />
                        <col />
                    </colgroup>
                    <thead>
                        <tr style="background-color:#a0a0a0;">
                            <td >PHU Header Keyword</td>
                            <td>Regular Expression Requirement</td>
                            <td>Requirement Comment</td>
                        </tr>
                    </thead>
                    %(clrows)s
                </table>'''

        # make the rows
        therows = ""
        if (self.phuReqs == None) or  (len(self.phuReqs) == 0):
            phutable = ""
        else:
            for reqkey, reqre in self.phuReqs.items():
                try:
                    comment = self.phuReqDocs[reqkey]
                except KeyError:
                    comment = "&nbsp;"


                thisrow = '''
                    <tr>
                        <td>%(parmkey)s</td>
                        <td>%(re)s</td>
                        <td>%(comment)s</td>
                    </tr>
                    ''' % { "parmkey":reqkey, "re":reqre , "comment":comment }
                therows = therows + thisrow
            phutable = phutable_templ % {"clrows":therows}

            # make the typereqs string
            if (self.typeORs == None) or (len(self.typeORs) == 0):
                trstr = "none"
            else:
                trstr = ""
            first = True
            sep = ""
            if (first == True):
                first = False
                sep = ""
            else:
                sep = ", "
                trstr = trstr + sep + " <a href=\"#%(typ)s\">%(typ)s</a>" % {"typ":trs} 

            retstr = div_templ % {  "phutable":phutable,
                    "typename":self.name, 
                                    "fullpath":self.fullpath,
                                    "typeapplic":self.usage,
                                    "typereqs":trstr
                                    }

        return retstr   

              
class CLAlreadyExists:
    '''
    This class exists to return a singleton of the ClassificationLibrary instance.
    See L{ClassificationLibrary} for more information.
    
    B{NOTE}: This should be refactored into a non-exception calling version that uses
    the __new__ operator instead of __init__. This method is required because __init__
    cannot return a value, so instead throws an exception if the ClassficationLibrary
    has already been created. Please use the AstroData interfaces for the 
    classification to access data classification information so that your code does not
    break when this is refactored. That interface is far more convienent and in most
    cases you only need to use the ClassificationLibrary if you are working with
    data and therefore have access to an AstroData instance.
    '''
    clInstance = None           
              
class ClassificationLibrary (object):
    '''
    This class exists as the proper full interface to the classification features,
    though most users should merely use the classification interface provided
    through AstroData. Single DataClassification 
    class instances can report if their own classifications apply, but only a complete 
    library encapsulates
    the whole classification system.
    To find if a single classification applies, the coder 
    asks the library by providing the classification by name. DataClassification 
    objects
    are not passed around, but used only to detect the classfication.
    Script authors therefore should not 
    converse directly with the data classification classes, and instead
    allow them to be managed by the Library. 
    
    This Library also knows how to produce HTML documentation of itself, so 
    that the classification
    definitions can be the sole source of such information and thus keep the 
    documentation as up to date as possible.
    
    @note: Classification Names, aka Gemini Type Names, are strings, the 
    python objects are not used to 
    report types, only detect them.  When passed in and out of functions
    classifications are always represented by their string names.
    
    @note: This class is a singleton, which means on the second attempt to create 
    it, instead of a new instance one will recieve
    a pointer to the instance already created. This ensures that only one 
    library will be loaded in
    a given program, which desireable for efficiency and coherence to a single
    type family for a given processing
    session.  
    
    The way this is accomplished is that the constructor for ClassificationLibrary keeps a class static variable
    with it's own instance pointer, if this pointer is already set, the constructor (aka __init__()) throws an
    exception, an instance of CLAlreadyExists which will contain the reference to the ClassificationLibrary instance.
    
    To make this work it is not advised to instantiate the ClassificationLibrary in a single regular call like
    C{cl = ClassificationLibrary()}, instead, use code such as the following::
    
      if (self.classificationLibrary == None):
            try:
                self.classificationLibrary = ClassificationLibrary()
            except CLAlreadyExists, s:
                self.classificationLibrary = s.clInstance
                
        return self.classificationLibrary
    
    The L{AstroData.getClassificationLibrary} function retrieves the instance handle 
    this way.
    
    This method is slated to be replaced, to avoid being affected by the change
    use the AstroData class' interface to classification features.
    '''
    
    definitionsPath = None # set in __init__(..)
    definitionsStorageREMask = None # set in __init(..)
    
    #THIS CLASS IS BUILT TO ACT AS A SINGLETON... ONE INSTANCE IS REFERED TO BY ALL
    # AstroData instances, for example.
    
    __single = None

    # type dictionaries
    typeDict = None
    statusDict = None
    typologyDict = None
    
    # This is the directory to look for parameter requirements (which serve as type definitions)
    def __init__(self, context="default"):
        """
        Default context is currently 'default'.
        
        @keyword context: A name for the context from which to retrieve the library.
        @type context: string
        """
        
        if (ClassificationLibrary.__single):
            # then we exist already
            #print "CLASS:", ClassificationLibrary.__single
            cli = CLAlreadyExists()
            cli.clInstance = ClassificationLibrary.__single
            raise cli
        else:
            #print "NEW CLASS"
            # organize when additional contexts are available
            
            # NOTE: Use this file's name to get path to types
            rootpath = os.path.dirname(os.path.abspath(__file__))
            self.definitionsPath = rootpath+"/types" # see above  os.path.join(self.phuReqDir, "gdtypedef."+tname)
            self.definitionsStorageREMask = r"gemdtype\.(?P<modname>.*?)\.py$"
            # self.definitionsStatusPath = self.definitionsPath+"/status"
            # self.definitionsTypologyPath = self.definitionsPath+"/types"
            self.typesDict = {}  #dict of DataClassifications
            self.typologyDict = {}
            self.statusDict = {}
            self.load()
            ClassificationLibrary.__single = self

    def load(self):
        """
        This function loads the classification library, in general 
        by obtaining python modules containing classes 
        which descend from DataClassification (or share its interface)
        and evaluating them.
        @returns: Nothing
        """
        
        if (verbose):
            print __file__
            print "definitionsPath=",  self.definitionsPath
            print "definitionsStorageREMask=", self.definitionsStorageREMask
 
        self.load_types("types", self.typesDict, self.typologyDict)
        self.load_types("status", self.typesDict, self.statusDict)
        
    def load_types(self, spacename,globaldict, tdict):
        """
        This function loads all the modules matching a given naming convention
        (regular expression "B{gemdtype\.(?P<modname>.*?)\.py$}")
        recursively within the given path. Loaded files are assumed to be
        either DataClassifications or share DataClassification's interface.
        Note, multiple DataClassifications can be put in the same file, 
        and it is important to add the new class to the C{newtypes} variable
        in the definition file, or else the load function will not see it.
        """
        print "AD858:", spacename
        for root, dirn, files in configWalk(spacename):
            #print "_+_+",root, dirn,files
            for dfile in files:
                if (re.match(self.definitionsStorageREMask, dfile)):
                    # this is a definition file then, according to the filenaming convention, that is.
                    
                    fullpath = os.path.join(root, dfile)
                    defsFile = open(fullpath)
                    newtypes=[]
                    exec defsFile
                    defsFile.close()
                    # newtype is declared here and used in the definition file to 
                    # pack in new types and return them to this scope.
                    for newtype in newtypes:
                        newtype.fullpath = fullpath
                        newtype.library = self
                        globaldict[newtype.name] = newtype
                        tdict[newtype.name] = newtype
                else :
                    if (verbose) : print "ignoring %s" % dfile

    def checkType(self, typename, dataset):
        """
        This function will check to see if a given type applies to a given dataset.
        
        @param typename: then name of the type to check the C{dataset} against.
        @type typename: string
        
        @param dataset: the data set in question
        @type dataset: either L{AstroData} instance or B{string}
        @returns: True if the type applies to the dataset, False otherwise
        @rtype: Bool
        """
        
        if  isinstance(dataset, AstroData.AstroData):
            hdulist = dataset.getHDUList()
            freeHDUList = True
        else:
            if  (not isinstance(dataset, pyfits.HDUList)):
                raise BadArgument()
            freeHDUList = False
            hdulist = dataset
        
        retval = self.typesDict[typename].assertType(hdulist)
        
        if freeHDUList == True:
            hdulist = dataset.releaseHDUList()
        
        return retval
    
    def getTypeObj(self,typename):
        """Generally users do not need DataClassification instances, however
        if you really do need that object, say to write an editor... this function
        will retrieve it.
        @param typename: the name of the classification for which you want the
        associated DataClassification instance
        @type typename: string
        @returns: the correct DataClassification instance
        @rtype: DataClassification
        """
        
        try:
            rettype = self.typesDict[typename]
            return rettype
        except KeyError:
            return None

    def discoverTypes(self, dataset, all = False):
        """This function returns a list of string names for the classifications
        which apply to this dataset.
        @param dataset: the data set in question
        @type dataset: either L{AstroData} instance or L{HDUList}
        @param all: flag to drive if a dictionary of three different lists
        is returned, if C{all} is True.  If False, a list of all status and processing
        types is returned as a list of strong names.  If True, a dictionary 
        is returned.  Element with the key "typology" will contain the typology
        related classifications, key "status" will contain the processing status related
        classifications, and key "all" will contain the union of both sets and is the 
        list returned when "all" is False.
        @type all: Bool
        @returns: the data type classifications which apply to the given dataset
        @rtype: list or dict of lists
                """
        
        retarya = self.discoverClassifications(dataset, self.typologyDict)
        retaryb = self.discoverClassifications(dataset, self.statusDict)
        retary = []
        retary.extend(retarya)
        retary.extend(retaryb)
        
        
        if (all == True):
            retdict = {}
            retdict["all"] = retary
            retdict["typology"] = retarya
            retdict["status"]   = retaryb
        
            return retdict
        else:
            return retary
        # return self.discoverClassifications(dataset, self.typesDict)
    
    def discoverStatus(self, dataset):
        """This function returns a list of string names for the processing
        status related classifications
        which apply to this dataset.
        @param dataset: the data set in question
        @type dataset: either L{AstroData} instance or L{HDUList}
        @returns: the data type classifications which apply to the given dataset
        @rtype: list of strings
        """
        
        return self.discoverClassifications(dataset, self.statusDict)
        
    def discoverTypology(self, dataset):
        """This function returns a list of string names for the typological
        classifications
        which apply to this dataset.
        @param dataset: the data set in question
        @type dataset: either L{AstroData} instance or L{HDUList}
        @returns: the data type classifications which apply to the given dataset
        @rtype: list of strings
        """
        return self.discoverClassifications(dataset, self.typologyDict)
        
    def discoverClassifications(self, dataset, classificationDict):
        """
        discoverClassificatons will return a list of classifications ("data types").
        'dataset' should be an HDUList or AstroData instance.
        
        @param dataset: the data set in question
        @type dataset: either L{AstroData} instance or B{string}
        @return: Returns list of DataClassification names
        @rtype: list of strings
        """
        typeList = []
        closeHdulist = False

        if isinstance(dataset, basestring):
            try:
                hdulist = pyfits.open(dataset)
                closeHdulist = True
            except:
                print ("we have a problem with ", dataset)
                raise()
                
        elif isinstance(dataset, AstroData.AstroData):
            hdulist = dataset.getHDUList()
            closeHdulist = False
            
        for tkey,tobj in classificationDict.items():
            if (tobj.assertType(hdulist)):
                typeList.append(tobj.name)
                
        if (closeHdulist):
                hdulist.close()
                
        return typeList
        
    def htmlDoc(self):
        '''
        Produces a full HTML page of documentation on the current classification 
        dictionary, aka Type Dictionary.
        @returns: HTML for the classification library
        @rtype: string
        '''
        
        page_templ = '''
        <html>
        <head><title>Data Classification</title></head>
        <body>
            <a href="gemdtype.viz.svg">Visualization Graph (requires SVG support)</a><br/>
	    <a href="gemdtype.viz.png">Visualization Graph (PNG image, should work)</a><br/>
        (<a href="typedict.py?edit=NEW">new data type</a>)
        <br/>
	    %(cldiv)s
            <div style="height:100em"><!-- blank space so that targets at bottom scroll target to top of window --></div>
        </body>
        '''
        
        divs = ""
        
        skeys = self.typesDict.keys()
        skeys.sort()
        
        for tkey in skeys:
            tobj = self.typesDict[tkey]
            thisdiv = tobj.htmlDoc()
            divs = ''.join([divs, thisdiv, "<br/><br/>"])
        
        retstr = page_templ % { "cldiv": divs } 
        return retstr

    def gvizDoc(self, writeout = False):
        """This function generates output in the "dot" language, which
            is used as input for the graphviz "dot" program which creates
            directed graphs in many different outputs.  We are interested
            in the SVG output, though it should be sufficient for any
            of the suppreqorted output formats.
            @param writeout: controls if the buffer created is written out
            to a file. Note that at this time the buffer will be written out to 
            a hardcoded directory.
            @type writeout: Bool
            @returns: buffer containing DOT language commands as can be interpreted
            by the graphviz product.
            @rtype: string
        """
        # @@REFACTOR@@ get the string to write out to from a lookup
        
        skeys = self.typesDict.keys()
        skeys.sort()
        gviztempl = """
            digraph classes {
            size = "10,20"
            pagedir="LT"
            labelloc=top
            label="AstroDataType: Data Classifications Graph"
            edge [
                color="#306040"
                arrowhead=odot
                arrowsize=.8
            ];
            node [
                color="#304060"
                fontsize = 8
                shape = house
            ];
            %(nodes)s
            %(links)s
            }
        """
        gvizlinks = ""
        gviznodes = ""
        for tkey in skeys:
            tobj = self.typesDict[tkey]
            gvizlinks = gvizlinks + tobj.gvizLinks()
            gviznodes = gviznodes + tobj.gvizNodes()
        
        retstr = gviztempl % { "links": gvizlinks, "nodes":gviznodes } 
        
        if (writeout):
            fl = open ("/var/www/html/gdwf/gemdtype.viz.dot", mode="w+")
            fl.write(retstr)
            fl.close()
            
            os.system("dot -Tpng -o/var/www/html/gdwf/gemdtype.viz.png -Tsvg -o/var/www/html/gdwf/gemdtype.viz.svg /var/www/html/gdwf/gemdtype.viz.dot")
            
        
        return retstr
    
# @@DOCPROJECT@@: done pass 1
        
