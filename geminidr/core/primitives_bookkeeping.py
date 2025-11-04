#
#                                                                  gemini_python
#
#                                                      primitives_bookkeeping.py
# ------------------------------------------------------------------------------
import os
from copy import deepcopy
from itertools import zip_longest

import astrodata, gemini_instruments

from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import parameter_override, capture_provenance

from geminidr import PrimitivesBASE, save_cache, stkindfile
from geminidr.core import parameters_bookkeeping


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class Bookkeeping(PrimitivesBASE):
    """
    This is the class containing all of the preprocessing primitives
    for the Bookkeeping level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = None

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
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

    def appendStream(self, adinputs=None, from_stream=None, copy=None):
        """
        This primitive takes the AstroData objects in a stream and appends them
        (order unchanged) to the end of the current stream. If requested, the
        stream whose ADs are being appended is deleted.

        Parameters
        ----------
        new_stream: str
            name of stream whose ADs are going to be appended to the working
            stream
        copy: bool
            append full deepcopies of the AD objects?
        """
        log = self.log
        try:
            stream = self.streams[from_stream]
        except KeyError:
            log.warning(f"There is no stream called '{from_stream}'. "
                        f"Continuing without appending any images.")
            return adinputs

        log.info(f"Appending {len(stream)} frames from stream '{from_stream}'.")
        if copy:
            adinputs.extend([deepcopy(ad) for ad in stream])
        else:
            adinputs.extend(stream)
        return adinputs

    def clearAllStreams(self, adinputs=None):
        """
        This primitive clears all streams (except "main") by setting them
        to empty lists.
        """
        log = self.log
        for stream in self.streams:
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

    def combineSlices(self, adinputs=None, from_stream=None, **params):
        """
        This primitive combines extensions from AD objects in an input stream
        with those in a reference stream.

        Parameters
        ----------
        from_stream : str
            name of stream containing ADs with extensions to combine from
        ids : str
            A 1-indexed, comma-separated string of id numbers of the extensions
            to take from `from_stream` and combine with those in `adinputs`.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        ids = params["ids"]

        if from_stream not in self.streams.keys():
            log.warning(f"Stream {from_stream} does not exist so nothing "
                        "to transfer")
            return adinputs

        source_length = len(self.streams[from_stream])
        if not (source_length == 1 or source_length == len(adinputs)):
            log.warning("Incompatible stream lengths: "
                        f"{len(adinputs)} and {source_length}")
            return adinputs

        # Get the list of extension IDs to combine from the secondary stream.
        # Note that extension IDs are 1-indexed, though it doesn't require
        # any specific handling in the code here.
        ids = [] if ids is None else sorted([int(n) for n in ids.split(',')])

        log.stdinfo(f"Combining slices from stream {from_stream}")
        log.debug(f"Extension IDs to combine: {ids}")

        adoutputs = []
        for ad1, ad2 in zip(*gt.make_lists(adinputs, self.streams[from_stream])):

            adout = astrodata.create(ad1.phu)
            adout.filename = ad1.filename
            adout.orig_filename = ad1.orig_filename

            for ext1, ext2 in zip_longest(ad1, ad2):

                if (ext1 is not None and ext2 is not None) and ext1.id in ids:
                    adout.append(ext2)
                elif (ext1 is None and ext2 is not None):
                    # If ad2 has more extensions than ad1, just append them.
                    adout.append(ext2)
                else:
                    # If ID not in the list of IDs to combine, or ad1 has more
                    # extensions than ad2.
                    adout.append(ext1)

            adoutputs.append(adout)

        return adoutputs

    def copyInputs(self, adinputs=None, **params):
        """
        This primitive results in deepcopies of the AD objects in one stream
        being placed in another stream. All the work is handled by the
        decorator, using the instream/outstream keywords, which is why this
        primitive appears to do nothing.
        """
        return adinputs

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

    def mergeInputs(self, adinputs=None):
        """
        This primitive takes all the inputs in a stream and makes a single
        AstroData object containing all the extensions from all the inputs.
        """
        log = self.log

        new_ad = astrodata.create(adinputs[0].phu)
        new_ad.filename = adinputs[0].filename
        new_ad.orig_filename = adinputs[0].orig_filename
        for ad in adinputs:
            log.stdinfo(f"Appending {len(ad)} extensions from {ad.filename}")
            for ext in ad:
                new_ad.append(ext)

        return [new_ad]

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

    def removeFromInputs(self, adinputs=None, tags=None):
        """
        Removes frames whose tags match any one of a list of supplied tags.
        The user is likely to want to redirect the output list.

        Parameters
        ----------
        tags: str/None
            Tags which frames must match to be selected
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        log.debug("Removing inputs with tags: {}".format(tags))

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
        adoutputs = [ad for ad in adinputs if not (set(required_tags) & ad.tags)]
        log.debug("Remaining files:")
        if adoutputs:
            for ad in adoutputs:
                log.debug("    {}".format(ad.filename))
        else:
            log.debug("    No files remaining")

        return adoutputs

    def selectFromInputs(self, adinputs=None, tags=None):
        """
        Selects frames whose tags match any one of a list of supplied tags.
        The user is likely to want to redirect the output list.

        Parameters
        ----------
        tags: str/None
            Tags which frames must match to be selected
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        log.debug("Selecting inputs with tags: {}".format(tags))

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
        if adoutputs:
            log.debug("Selected files:")
            for ad in adoutputs:
                log.debug("    {}".format(ad.filename))
        else:
            log.debug("    No files selected")

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

    def sliceIntoStreams(self, adinputs=None, root_stream_name=None, copy=True):
        """
        This primitive slices each input AstroData object into separate AD
        objects with one slice each, and puts them into separate streams. The
        stream "index0" will contain all the first slices from all the AD
        objects, "index1" all the second slices, and so on. These streams will
        not be the same length if the input AD objects have different lengths.

        Parameters
        ----------
        root_stream_name: str
            base name for the streams (to be succeeded by 1, 2, 3, ...)
        copy: bool
            make full deepcopies of the slices?
        """
        log = self.log
        streams = set()
        for ad in adinputs:
            for i in range(len(ad)):
                stream_name = f'{root_stream_name}{ad[i].id}'
                if copy:
                    new_ad = deepcopy(ad[i])
                else:
                    new_ad = astrodata.create(ad.phu)
                    new_ad.append(ad[i])
                    new_ad.filename = ad.filename
                    new_ad.orig_filename = ad.orig_filename

                filename, filetype = os.path.splitext(ad.filename)
                fields = filename.rsplit('_', 1)
                if len(fields) == 1:
                    new_ad.update_filename(suffix=f'_{stream_name}')
                else:
                    new_ad.update_filename(suffix=f'_{stream_name}_{fields[1]}', strip=True)

                try:
                    self.streams[stream_name].append(new_ad)
                except KeyError:
                    self.streams[stream_name] = [new_ad]
                streams.add(stream_name)

        log.stdinfo(f'Created {len(streams)} streams by slicing input files.')
        for stream in streams:
            log.debug(f'Files in stream {stream}:')
            for ad in self.streams[stream]:
                log.debug(f'    {ad.filename}')
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

    def transferAttribute(self, adinputs=None, suffix=None, source=None, attribute=None):
        """
        This primitive takes an attribute (e.g., "mask", or "OBJCAT") from
        the AD(s) in another ("source") stream and applies it to the ADs in
        this stream. There must be either the same number of ADs in each
        stream, or only 1 in the source stream.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        source: str
            name of stream containing ADs whose attributes you want
        attribute: str
            attribute to transfer from ADs in other stream
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        if source not in self.streams.keys():
            log.info(f"Stream {source} does not exist so nothing to transfer")
            return adinputs

        source_length = len(self.streams[source])
        if not (source_length == 1 or source_length == len(adinputs)):
            log.warning("Incompatible stream lengths: "
                        f"{len(adinputs)} and {source_length}")
            return adinputs

        log.stdinfo(f"Transferring attribute '{attribute}' from stream {source}")

        # Keep track of whether we find anything to transfer, as failing to
        # do so might indicate a problem and we should warn the user
        found = False

        for ad1, ad2 in zip(*gt.make_lists(adinputs, self.streams[source])):
            # Attribute could be top-level or extension-level
            # Use deepcopy so references to original object don't remain
            try:
                setattr(ad1, attribute,
                        deepcopy(getattr(ad2, attribute)))
            except (AttributeError, ValueError):  # data, mask, are gettable not settable
                pass
            else:
                found = True
                ad1.update_filename(suffix=suffix, strip=True)
                continue

            for ext1, ext2 in zip(ad1, ad2):
                if hasattr(ext2, attribute):
                    setattr(ext1, attribute,
                            deepcopy(getattr(ext2, attribute)))
                    found = True
            if found:
                ad1.update_filename(suffix=suffix, strip=True)

        # Do not report this if the above loop never ran because of no adinputs!
        if not found and len(adinputs):
            log.warning(f"Did not find any {attribute} attributes to transfer")

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
                    raise RuntimeError(message)
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
