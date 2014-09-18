# Document strings for astrodata modules.
#
# AstroData 
#
# ------------------------------------------------------------------------------
SUBDATA_INFO_STRING = """


Sub-data Information:

An AstroData instance (AD) is always associated with a second AstroData
instance, or sub-data(AD[]).  This allows users the convenience of accessing
header and image data directly (ex ad[0].data, ad('SCI', 1).data).  Both the
AD and sub-data share objects in memory, which cause many aliases (see below).
Also note that the sub-data mode='update' property cannot be changed and 
AD.filename is assigned to AD[].filename but cannot be changed by the sub-data
instance.
                
      Name Mapping for AD and sub-data(AD[]) for a 3 ext. MEF

AD.phu == AD.hdulist[0] == AD[0].hdulist[0] == AD('SCI', 1).hdulist[0] 
                        == AD[1].hdulist[0] == AD('SCI', 2).hdulist[0]
                        == AD[2].hdulist[0] == AD('SCI', 3).hdulist[0]
                        == AD[0].phu == AD('SCI', 1).phu 
                        == AD[1].phu == AD('SCI', 2).phu
                        == AD[2].phu == AD('SCI', 3).phu

AD.phu.header == all of the above with .header appended

AD[0].data == AD.hdulist[1].data == AD.('SCI', 1).data          
AD[1].data == AD.hdulist[2].data == AD.('SCI', 2).data          
AD[2].data == AD.hdulist[3].data == AD.('SCI', 3).data          
    
AD[0].header == AD.hdulist[1].header == AD.('SCI', 1).header          
AD[1].header == AD.hdulist[2].header == AD.('SCI', 2).header        
AD[2].header == AD.hdulist[3].header == AD.('SCI', 3).header

                     Relationship to pyfits

The AD creates a pyfits HDUList (if not supplied by one) and attaches it 
to itself as AD.hdulist.  The sub-data also creates its own unique HDUList as 
AD[?].hdulist or AD('?', ?).hdulist, but shares in memory the phu (including
the phu header) with the primary AD HDUList. 

The AD.hdulist may have more than one extension, however, the sub-data is 
limited to one extension. This sub-data hdulist extension shares memory with
its corresponding AD.hdulist extension (ex. AD[0].hdulist[1] == AD.hdulist[1])

One important difference to note is that astrodata begins its first element 
'0' with data (ImageHDU), where pyfits HDUList begins its first element '0'
with meta-data (PrimaryHDU). This causes a 'one off' discrepancy.

Flags    default    Description
-----    -------    -----------
as_html   False     return html

oid       False     include object ids

table     False     show BinTableHDU contents

help      False     show help information    
"""
# ------------------------------------------------------------------------------
