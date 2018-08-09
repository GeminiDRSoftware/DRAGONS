#
#                                                                  gemini_python
#
#                                                      primitives_bookkeeping.py
# ------------------------------------------------------------------------------
import astrodata
import gemini_instruments
import numpy as np
from copy import deepcopy

from gempy.gemini import gemini_tools as gt

from geminidr import PrimitivesBASE
from geminidr import save_cache, stkindfile

from . import parameters_bookkeeping

from recipe_system.utils.decorators import parameter_override

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
        self._param_update(parameters_bookkeeping)

    def addToList(self, adinputs=None, purpose=None):
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
            ad.update_filename(suffix=suffix, strip=True)
            log.stdinfo("Writing {} to disk".format(ad.filename))
            # Need to specify 'ad.filename' here so writes to current dir
            ad.write(ad.filename, overwrite=True)
            try:
                if ad.filename not in self.stacks[_stackid(purpose, ad)]:
                    self.stacks[_stackid(purpose, ad)].append(ad.filename)
            except KeyError:
                # Stack doesn't exist yet, so start it off...
                self.stacks[_stackid(purpose, ad)] = [ad.filename]

        save_cache(self.stacks, stkindfile)
        return adinputs

    def clearAllStreams(self, adinputs=None):
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

    def flushPixels(self, adinputs=None, force=False):
        """
        This primitive saves the inputs to disk and then reopens them so
        the pixel data are out of memory
        """
        def is_lazy(ad):
            """Determine whether an AD object is lazily-loaded"""
            for ndd in ad.nddata:
                for attr in ('_data', '_mask', '_uncertainty'):
                    item = getattr(ndd, attr)
                    if item is not None and not (hasattr(item, 'lazy') and item.lazy):
                        return False
            return True

        log = self.log

        for i, ad in enumerate(adinputs):
            if not force and is_lazy(ad):
                log.fullinfo("{} is lazily-loaded; not writing to "
                          "disk".format(ad.filename))
            else:
                # Write in current directory (hence ad.filename specified)
                log.fullinfo("Writing {} to disk and reopening".format(ad.filename))
                ad.write(ad.filename, overwrite=True)
                # We directly edit elements in the list to ensure the versions
                # in the primitivesClass stream are affected too. We also want
                # the files to retain their orig_filename attributes, which
                # would otherwise change upon loading.
                orig_filename = ad.orig_filename
                adinputs[i] = astrodata.open(ad.filename)
                adinputs[i].orig_filename = orig_filename
        return adinputs

    def getList(self, adinputs=None, **params):
        """
        This primitive will check the files in the stack lists are on disk,
        and then update the inputs list to include all members that belong
        to the same stack(s) as the input(s). All images are cleared from
        memory. If the input(s) come from different stacks, images will be
        collected from the stacks in the order of the inputs, until the
        maximum number of frames is reached.

        Parameters
        ----------
        purpose: str
            purpose/name of list to access
        max_frames: int
            maximum number of frames to return
        """
        log = self.log
        purpose = params["purpose"] or '_list'
        # Make comparison checks easier if there's no limit
        max_frames = params['max_frames'] or 1000000

        # Get stack IDs for all inputs, preserve order
        sid_list = []
        for ad in adinputs:
            sid = _stackid(purpose, ad)
            if sid not in sid_list:
                sid_list.append(sid)
        if len(sid_list) > 1:
            log.warning("The input includes frames from {} different stack"
                        " ids".format(len(sid_list)))

        # LIFO stacklists, so reverse and combine
        all_files = []
        for sid in sid_list:
            stacklist = self.stacks[sid]
            log.debug("List for stack id {} ({}):".format(sid[:35],
                                                          len(stacklist)))
            for f in stacklist:
                log.debug("   {}".format(f))
            all_files.extend(reversed(stacklist))

        adinputs = []
        for f in all_files:
            try:
                adinputs.insert(0, astrodata.open(f))
            except astrodata.AstroDataError:
                log.stdinfo("   Cannot open {}".format(f))
            if len(adinputs) >= max_frames:
                break

        log.stdinfo("Using the following files:")
        adinputs = self.showInputs(adinputs, purpose=None)
        return adinputs

    def rejectInputs(self, adinputs=None, at_start=0, at_end=0):
        """
        This primitive removes a set number of frames from the start and end of the
        input list.

        Parameters
        ----------
        at_start: int
            Number of frames to cull from start of input list
        at_end: int
            Number of frames to cull from end of input list
        """
        log = self.log
        if at_start == 0 and at_end == 0:
            log.stdinfo("No files being removed. Both at_start and at_end are zero.")
            return adinputs

        start_text = ("{} file(s) from start of list".format(at_start)
                      if at_start > 0 else "")
        end_text = ("{} file(s) from end of list".format(at_end)
                    if at_end > 0 else "")
        conjunction = " and " if start_text and end_text else ""
        log.stdinfo("Removing " + start_text + conjunction + end_text + ".")
        return adinputs[at_start:len(adinputs) - at_end]

    def selectFromInputs(self, adinputs=None, tags=None):
        """
        Selects frames whose tags match any one of a list of supplied tags.
        The user is likely to want to redirect the output list.

        Parameters
        ----------
        tags: str/None
            Tags which frames must match to be selected
        """
        if tags is None:
            return adinputs
        required_tags = tags.split(',')

        # Commented lines select AD that match *all* the tags. While possibly
        # more natural, one can achieve this by a series of matches to each tag
        # individually. There is, however, no way to combine lists produced
        # this way to create one as if produced by matching *any* of the tags.
        # Hence a match to *any* tag makes more sense as the implementation.
        #adoutputs = [ad for ad in adinputs
        #             if set(required_tags).issubset(ad.tags)]
        adoutputs = [ad for ad in adinputs if set(required_tags) & ad.tags]
        return adoutputs

    def showInputs(self, adinputs=None, purpose=None):
        """
        A simple primitive to show the filenames for the current inputs to
        this primitive.
        
        Parameters
        ----------
        purpose: str
            Brief description for output
        """
        log = self.log
        if purpose:
            log.stdinfo("Inputs for {}".format(purpose))
        for ad in adinputs:
            log.stdinfo("  {}".format(ad.filename))
        return adinputs

    def showList(self, adinputs=None, purpose='all'):
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
            purpose = purpose or '_list'
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

    def sortInputs(self, adinputs=None, descriptor='filename', reverse=False):
        """
        This sorts the input list according to the values returned by the
        descriptor parameter.
        
        Parameters
        ----------
        descriptor: str
            name of descriptor on which to sort (can also be "filename")
        reverse: bool
            return list sorted in reverse order?
        """
        log = self.log

        # Check the attribute/descriptor exists
        try:
            attr_list = [getattr(ad, descriptor) for ad in adinputs]
        except AttributeError:
            log.warning("Invalid sorting descriptor/attribute. Cannot sort "
                        "input list.")
            return adinputs

        # Might be callable (a descriptor) or not ("filename")
        try:
            list_to_sort = [attr() for attr in attr_list]
        except TypeError:
            list_to_sort = attr_list

        log.stdinfo("Sorting input list according to {}".format(descriptor))
        # Sort (equivalent of np.argsort)
        index_order = sorted(range(len(adinputs)), key=list_to_sort.__getitem__)
        if reverse:
            index_order = list(reversed(index_order))
        return [adinputs[i] for i in index_order]


    def transferAttribute(self, adinputs=None, source=None, attribute=None):
        """
        This primitive takes an attribute (e.g., "mask", or "OBJCAT") from
        the AD(s) in another ("source") stream and applies it to the ADs in
        this stream. There must be either the same number of ADs in each
        stream, or only 1 in the source stream.
        
        Parameters
        ----------
        source: str
            name of stream containing ADs whose attributes you want
        attribute: str
            attribute to transfer from ADs in other stream
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        if source not in self.streams.keys():
            log.info("Stream {} does not exist so nothing to transfer".format(source))
            return adinputs

        source_length = len(self.streams[source])
        if not (source_length == 1 or source_length == len(adinputs)):
            log.warning("Incompatible stream lengths: {} and {}".
                        format(len(adinputs), source_length))
            return adinputs

        log.stdinfo("Transferring attribute {} from stream {}".format(attribute, source))

        # Keep track of whether we find anything to transfer, as failing to
        # do so might indicate a problem and we should warn the user
        found = False
        for ad1, ad2 in zip(*gt.make_lists(adinputs, self.streams[source])):
            # Attribute could be top-level or extension-level
            # Use deepcopy so references to original object don't remain
            if hasattr(ad2, attribute):
                try:
                    setattr(ad1, attribute, deepcopy(getattr(ad2, attribute)))
                except ValueError:  # data, mask, are gettable not settable
                    pass
                else:
                    found = True
                    continue
            for ext1, ext2 in zip(ad1, ad2):
                if hasattr(ext2, attribute):
                    setattr(ext1, attribute, deepcopy(getattr(ext2, attribute)))
                    found = True

        if not found:
            log.warning("Did not find any {} attributes to transfer".format(attribute))
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
        overwrite: bool
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
                ad.update_filename(prefix=pfx, suffix=sfx, strip=params["strip"])
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
            ad.write(outfilename, overwrite=params["overwrite"])
        return adinputs

# Helper function to make a stackid, without the IDFactory nonsense
def _stackid(purpose, ad):
    return (purpose + ad.group_id()).replace(' ', '_')
