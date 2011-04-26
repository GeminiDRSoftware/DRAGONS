
Random notes to be organized in FUTURE:
=======================================

	standard structure of User Level Functions:
	------------------------------------------
	
	run the science function manager
	set the ulf_key keyword for time stamping the ulf
	
	try:
	    for ad in adInputs:
	        if ulf_key exists:
	            print message
	            adOut = deepcopy(ad)
	        else:
	            check other inputs (like flats)
	            adOut = deepcopy(ad)
	            do the actual work on output object (adOut)
	            rename the output object (adOut.filename)
	            timestamp stuff
	
	        append output object to output list
	
	except:
		log exception info using sys.exec
	    raise
	
	return output list 
