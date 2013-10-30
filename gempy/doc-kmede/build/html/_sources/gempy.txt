dtest
======


        This function uses the CL script gireduce to subtract the overscan
        from the input images.

        WARNING: 
            The gireduce script used here replaces the previously
            calculated DQ frames with its own versions.  This may be corrected
            in the future by replacing the use of the gireduce
            with a Python routine to do the overscan subtraction.

        note 
            The inputs to this function MUST be prepared.

        String representing the name of the log file to write all log messages to
        can be defined, or a default of 'gemini.log' will be used.  If the file
        all ready exists in the directory you are working in, then this file will
        have the log messages during this function added to the end of it.

        ## FOR FUTURE

        This function has many GMOS dependencies that would be great to work out
        so that this could be made a more general function (say at the Gemini level).
        In the future the parameters can be looked into and the CL script can be
        upgraded to handle things like row based overscan calculations/fitting/modeling...
        vs the column based used right now, add the model, nbiascontam, ... params to the
        functions inputs so the user can choose them for themselves.
        While for now, GMOS is the only instrument that overscanSubtract is needed
        for, it would be great to offer this type of function for future instruments
        and or telescopes :-D Dream the Dream!

        ###

        @param adIns: Astrodata input flat(s) to be combined and normalized
        @type adIns: Astrodata objects, either a single or a list of objects

