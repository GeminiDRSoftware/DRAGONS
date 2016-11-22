import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt

from geminidr import PrimitivesBASE
from .parameters_bookkeeping import ParametersBookkeeping

from recipe_system.utils.decorators import parameter_override
from recipe_system.cal_service import caches
# ------------------------------------------------------------------------------
@parameter_override
class Bookkeeping(PrimitivesBASE):
    """
    This is the class containing all of the preprocessing primitives
    for the GEMINI level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = None

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(Bookkeeping, self).__init__(adinputs, context, ucals=ucals,
                                         uparms=uparms)
        self.parameters = ParametersBookkeeping
    
    def addToList(self, adinputs=None, stream='main', **params):
        """
        This primitive will update the lists of files to be stacked
        that have the same observationID with the current inputs.
        This file is cached between calls to reduce, thus allowing
        for one-file-at-a-time processing.
        
        Parameters
        ----------
        purpose: str
            purpose/name of this list, used as suffix for files
        """
        log = self.log
        purpose = self.parameters.addToList['purpose']
        suffix = '_{}'.format(purpose) if purpose else '_list'
        
        # Update file names and write the files to disk to ensure the right
        # version is stored before adding it to the list.
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=suffix,
                                              strip=True)
            log.stdinfo("Writing {} to disk".format(ad.filename))
            ad.write(clobber=True)
            try:
                self.stacks[_stackid(purpose, ad)].add(ad.filename)
            except KeyError:
                self.stacks[_stackid(purpose, ad)] = set([ad.filename])

        caches.save_cache(self.stacks, caches.stkindfile)
        return

    def getList(self, adinputs=None, stream='main', **params):
        """
        This primitive will check the files in the stack lists are on disk,
        and then update the inputs list to include all members of the stack 
        for stacking.
        
        Parameters
        ----------
        purpose: str
            purpose/name of list to access
        max_frames: int
            maximum number of frames to return
        to_stream: str
            name of stream to which this list will be reported
        """
        log = self.log
        pars = self.parameters.getList
        purpose = pars.get('purpose', '')
        max_frames = pars['max_frames']
        stream = pars.get('to_stream', 'main')
        stream_list = self.streams[stream] if stream!='main' else self.adinputs

        # Get ID for all inputs; use a set to avoid duplication
        sidset = set()
        [sidset.add(_stackid(purpose, ad)) for ad in self.adinputs]
        
        # Import inputs from all lists
        for sid in sidset:
            stacklist = self.stacks[sid]
            log.stdinfo("List for stack id {}(...):".format(sid[:35]))
            # Limit length of stacklist
            if len(stacklist)>max_frames and max_frames is not None:
                stacklist = sorted(stacklist)[-max_frames:]
            # Add each file to the input list if it's not already there
            for f in stacklist:
                if f not in [ad.filename for ad in stream_list]:
                    stream_list.append(astrodata.open(f))
                log.stdinfo("   {}".format(f))
        
        return

    def showInputs(self, adinputs=None, stream='main', **params):
        """
        A simple primitive to show the filenames for the current inputs to 
        this primitive.
        """
        log = self.log
        log.stdinfo("Inputs to stream {}".format(stream))

        inputs = self.adinputs if stream=='main' else self.streams[stream]
        for ad in inputs:
            log.stdinfo("  {}".format(ad.filename))
        return

    showFiles = showInputs
    
    def showList(self, adinputs=None, stream='main', **params):
        """
        This primitive will log the list of files in the stacking list matching
        the current inputs and 'purpose' value.

        Parameters
        ----------
        purpose: str
            purpose/name of list
        """
        log = self.log
        sidset = set()
        purpose = self.parameters.showList.get('purpose', '')
        if purpose == 'all':
            [sidset.add(sid) for sid in self.stacks.keys()]
        else:
            [sidset.add(_stackid(purpose, ad)) for ad in self.adinputs]
        for sid in sidset:
            stacklist = self.stacks(sid)
            log.status("List for stack id={}".format(sid))
            if len(stacklist) > 0:
                for f in stacklist:
                    log.status("   {}".format(f))
            else:
                log.status("No datasets in list")
        return

    def writeOutputs(self, adinputs=None, stream='main', **params):
        """
        A primitive that may be called by a recipe at any stage to
        write the outputs to disk.
        If suffix is set during the call to writeOutputs, any previous 
        suffixes will be striped and replaced by the one provided.
        examples: 
        writeOutputs(suffix= '_string'), writeOutputs(prefix= '_string') 
        or if you have a full file name in mind for a SINGLE file being 
        ran through Reduce you may use writeOutputs(outfilename='name.fits').

        Parameters
        ----------
        strip: bool
            strip the previous suffix off file name?
        clobber: bool
            overwrite existing files?
        suffix: str
            new suffix to append to output files
        prefix: str
            new prefix to prepend to output files
        outfilename: str
            new filename (applicable only if there's one file to be written)
        """
        log = self.log
        pars = self.parameters.writeOutputs
        sfx = pars['suffix']
        pfx = pars['prefix']
        outfilename = pars['outfilename']
        log.fullinfo("suffix = {}".format(sfx))
        log.fullinfo("prefix = {}".format(pfx))
        
        for ad in self.adinputs:
            if sfx or pfx:
                ad.filename = gt.filename_updater(adinput=ad,
                                prefix=pfx, suffix=sfx, strip=pars["strip"])
                log.fullinfo("File name updated to {}".format(ad.filename))
                outfilename = ad.filename
            elif outfilename:
                # Check that there is not more than one file to be written
                # to this file name, if so throw exception
                if len(self.adinputs) > 1:
                    message = "More than one file was requested to be " \
                              "written to the same name {}".format(outfilename)
                    log.critical(message)
                    raise IOError(message)
            else:
                # If no changes to file names are requested then write inputs
                # to their current file names
                outfilename = ad.filename
                log.fullinfo("not changing the file name to be written "
                             "from its current name")
            
            # Finally, write the file to the name that was decided upon
            log.stdinfo("Writing to file {}".format(outfilename))
            ad.write(filename=outfilename, clobber=pars["clobber"])

        return

# Helper function to make a stackid, without the IDFactory nonsense
def _stackid(purpose, ad):
    return (purpose + ad.group_id()).replace(' ', '_')