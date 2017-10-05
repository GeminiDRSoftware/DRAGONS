#
#                                                                  gemini_python
#
#                                                      primitives_bookkeeping.py
# ------------------------------------------------------------------------------
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
    for the Bookkeeping level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = None

    def __init__(self, adinputs, **kwargs):
        super(Bookkeeping, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersBookkeeping
    
    def addToList(self, adinputs=None, purpose=None, **params):
        """
        This primitive will update the lists of files to be stacked
        that have the same observationID with the current inputs.
        This file is cached between calls to reduce, thus allowing
        for one-file-at-a-time processing.
        
        Parameters
        ----------
        purpose: str (None => "list")
            purpose/name of this list, used as suffix for files
        """
        log = self.log
        suffix = '_{}'.format(purpose) if purpose else '_list'
        
        # Update file names and write the files to disk to ensure the right
        # version is stored before adding it to the list.
        for ad in adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=suffix,
                                              strip=True)
            log.stdinfo("Writing {} to disk".format(ad.filename))
            # Need to specify 'ad.filename' here so writes to current dir
            ad.write(ad.filename, clobber=True)
            try:
                if ad.filename not in self.stacks[_stackid(purpose, ad)]:
                    self.stacks[_stackid(purpose, ad)].append(ad.filename)
            except KeyError:
                # Stack doesn't exist yet, so start it off...
                self.stacks[_stackid(purpose, ad)] = [ad.filename]

        caches.save_cache(self.stacks, caches.stkindfile)
        return adinputs

    def clearAllStreams(self, adinputs=None, **params):
        """
        This primitive clears all streams (except "main") by setting them
        to empty lists.
        """
        log = self.log
        for stream in self.streams.keys():
            if stream != 'main':
                log.fullinfo('Clearing stream {}'.format(stream))
                self.streams[stream] = []
        return adinputs

    def clearStream(self, adinputs=None, **params):
        """
        This primitive clears a stream by returning an empty list, which the
        decorator then pushes into the stream.
        """
        log = self.log
        log.fullinfo('Clearing stream {}'.format(params.get('stream', 'main')))
        return []

    def getList(self, adinputs=None, **params):
        """
        This primitive will check the files in the stack lists are on disk,
        and then update the inputs list to include all members that belong
        to the same stack(s) as the input(s).
        
        Parameters
        ----------
        purpose: str
            purpose/name of list to access
        max_frames: int
            maximum number of frames to return
        """
        log = self.log
        purpose = params.get('purpose', '')
        max_frames = params['max_frames']

        # Get ID for all inputs; use a set to avoid duplication
        sidset = set()
        [sidset.add(_stackid(purpose, ad)) for ad in adinputs]

        adinputs = []
        # Import inputs from all lists
        for sid in sidset:
            stacklist = self.stacks[sid]
            log.stdinfo("List for stack id {}(...):".format(sid[:35]))
            # Limit length of stacklist
            if len(stacklist)>max_frames and max_frames is not None:
                stacklist = stacklist[-max_frames:]
            # Add each file to the input list if it's not already there
            for f in stacklist:
                if f not in [ad.filename for ad in adinputs]:
                    try:
                        adinputs.append(astrodata.open(f))
                    except IOError:
                        log.stdinfo("   {} NOT FOUND".format(f))
                    else:
                        log.stdinfo("   {}".format(f))
        return adinputs

    def selectInputs(self, adinputs=None, **params):
        """
        Selects frames whose tags match any one of a list of supplied tags.
        The user is likely to want to redirect the output list.
        
        Parameters
        ----------
        tags: str/list
            Tags which frames must match to be selected
        """
        required_tags = params.get("tags") or []
        if isinstance(required_tags, str):
            required_tags = required_tags.split(',')

        # This selects AD that match *all* the tags. While possibly the most
        # natural, one can achieve this by a series of matches to each tag
        # individually. There is, however, no way to combine lists produced
        # this way to create one as if produced by matching *any* of the tags.
        # Hence a match to *any* tag makes more sense as the implementation.
        #adoutputs = [ad for ad in adinputs
        #             if set(required_tags).issubset(ad.tags)]
        adoutputs = [ad for ad in adinputs if (set(required_tags) & ad.tags)]
        return adoutputs

    def showInputs(self, adinputs=None, stream='main', **params):
        """
        A simple primitive to show the filenames for the current inputs to 
        this primitive.
        """
        log = self.log
        log.stdinfo("Inputs to stream {}".format(stream))

        for ad in adinputs:
            log.stdinfo("  {}".format(ad.filename))
        return adinputs

    showFiles = showInputs
    
    def showList(self, adinputs=None, purpose=None, **params):
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
        if purpose == 'all':
            [sidset.add(sid) for sid in self.stacks]
        else:
            if purpose is None:
                purpose = ''
            [sidset.add(_stackid(purpose, ad)) for ad in adinputs]
        for sid in sidset:
            stacklist = self.stacks.get(sid, [])
            log.status("List for stack id={}".format(sid))
            if len(stacklist) > 0:
                for f in stacklist:
                    log.status("   {}".format(f))
            else:
                log.status("No datasets in list")
        return adinputs

    def writeOutputs(self, adinputs=None, **params):
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
        sfx = params['suffix']
        pfx = params['prefix']
        log.fullinfo("suffix = {}".format(sfx))
        log.fullinfo("prefix = {}".format(pfx))
        
        for ad in adinputs:
            if sfx or pfx:
                ad.filename = gt.filename_updater(adinput=ad,
                                prefix=pfx, suffix=sfx, strip=params["strip"])
                log.fullinfo("File name updated to {}".format(ad.filename))
                outfilename = ad.filename
            elif params['outfilename']:
                # Check that there is not more than one file to be written
                # to this file name, if so throw exception
                if len(adinputs) > 1:
                    message = "More than one file was requested to be " \
                              "written to the same name {}".format(
                        params['outfilename'])
                    log.critical(message)
                    raise IOError(message)
                else:
                    outfilename = params['outfilename']
            else:
                # If no changes to file names are requested then write inputs
                # to their current file names
                outfilename = ad.filename
                log.fullinfo("not changing the file name to be written "
                             "from its current name")
            
            # Finally, write the file to the name that was decided upon
            log.stdinfo("Writing to file {}".format(outfilename))
            ad.write(outfilename, clobber=params["clobber"])
        return adinputs

# Helper function to make a stackid, without the IDFactory nonsense
def _stackid(purpose, ad):
    return (purpose + ad.group_id()).replace(' ', '_')
