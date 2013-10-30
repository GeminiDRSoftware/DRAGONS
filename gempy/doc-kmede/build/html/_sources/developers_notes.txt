
Random notes to be organized in FUTURE:
=======================================

	deepcopy issues:
	
	While performing modifications on an astrodata object during processing,
	if a copy of an object is made without the use of copy.deepcopy the original
	object will take on the same changes as the copied version.  Thus, using
	copy.deepcopy is the best solution to this dilemma, but also comes with 
	issues to take care of for a perfect copy that maintains the originals
	purity.  Currently there is only one thing to worry about, 
	geminiTools.fileNameUpdater.  This function has a 'strip'=True/False
	argument that uses the PHU key 'ORIGNAME' to start the renaming of the 
	file name from this key's value, which should be the file name prior to 
	any processing and renaming; BUT if this key does not exist in the PHU
	geminiTools.fileNameUpdater will attempt to added it using the astrodata
	function storeOriginalName which requires the astrodata objects private 
	member variable __origFilename that copy.deepcopy does not copy the new 
	version of the object.  Thus, if any future renaming of the object is 
	to occur, the storeOriginalName must be performed on the original object
	prior to copying so that the 'ORIGNAME' PHU key has the objects original
	name stored for copy.deepcopy transfer and thus available for future 
	strip=True renaming of the copied object.
