gnirsConfigDict = {
        # Dictionary keys are in the following order:
        # prism, decker, grating, camera
        # Used every combination of prism, grating and camera available in
        #   gnirs$data/nsappwave.fits r1.43, EH, February 1, 2013
        # Dictionary values are in the following order:
        # mdf, offsetsection, pixscale, mode
        
        # ShortBlue_G5513 [GS, decommissioned June 2005], 32/mm
        ( "MIR_G5511" , "SC_Long" , "32/mm_G5506" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-ls-short-32-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        ( "MIR_G5511" , "SC_XD/IFU" , "32/mm_G5506" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "SC_XD" , "32/mm_G5506" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "IFU" , "32/mm_G5506" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "** ENG 49450 **" , "32/mm_G5506" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        
        ( "SXD_G5509" , "SC_XD/IFU" , "32/mm_G5506" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-xd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "SC_XD" , "32/mm_G5506" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-xd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "IFU" , "32/mm_G5506" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-xd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        
        # ShortBlue_G5513 [GS, decommissioned June 2005], 111/mm
        ( "MIR_G5511" , "SC_Long" , "111/mm_G5505" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-ls-short-111-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        ( "MIR_G5511" , "SC_XD/IFU" , "111/mm_G5505" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "SC_XD" , "111/mm_G5505" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "IFU" , "111/mm_G5505" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        
        ( "SXD_G5509" , "SC_XD/IFU" , "111/mm_G5505" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-xd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "SC_XD" , "111/mm_G5505" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-xd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "IFU" , "111/mm_G5505" , "ShortBlue_G5513" ) : ( "gnirs$data/gnirs-xd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        
        # ShortBlue_G5521 [GS, installed June 2005], 32/mm
        ( "MIR_G5511" , "SC_Long" , "32/mm_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ls-short-32-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        ( "MIR_G5511" , "SC_XD/IFU" , "32/mm_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "SC_XD" , "32/mm_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "IFU" , "32/mm_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "** ENG 49450 **" , "32/mm_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15", "IFU" ),
        
        ( "SXD_G5509" , "SC_XD/IFU" , "32/mm_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "SC_XD" , "32/mm_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "IFU" , "32/mm_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        
        # ShortBlue_G5521 [GS, installed June 2005], 32/mmSB
        ( "MIR_G5511" , "SC_Long" , "32/mmSB_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ls-short-32-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        ( "MIR_G5511" , "SC_XD/IFU" , "32/mmSB_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "SC_XD" , "32/mmSB_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "IFU" , "32/mmSB_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        
        ( "SXD_G5509" , "SC_XD/IFU" , "32/mmSB_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "SC_XD" , "32/mmSB_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "IFU" , "32/mmSB_G5506" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        
        # ShortBlue_G5521 [GS, installed June 2005], 111/mm
        ( "MIR_G5511" , "SC_Long" , "111/mm_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ls-short-111-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        ( "MIR_G5511" , "SC_XD/IFU" , "111/mm_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "SC_XD" , "111/mm_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "IFU" , "111/mm_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        
        ( "SXD_G5509" , "SC_XD/IFU" , "111/mm_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "SC_XD" , "111/mm_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "IFU" , "111/mm_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        
        # ShortBlue_G5521 [GS, installed June 2005], 111/mmSB
        ( "MIR_G5511" , "SC_Long" , "111/mmSB_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ls-short-111-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        ( "MIR_G5511" , "SC_XD/IFU" , "111/mmSB_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "SC_XD" , "111/mmSB_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "IFU" , "111/mmSB_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        
        ( "SXD_G5509" , "SC_XD/IFU" , "111/mmSB_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "SC_XD" , "111/mmSB_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        ( "SXD_G5509" , "IFU" , "111/mmSB_G5505" , "ShortBlue_G5521" ) : ( "gnirs$data/gnirs-xd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        
        # ShortBlue_G5538 [GN, currently in storage, replaced with G5540], 32/mmSB
        ( "MIR_G5537" , "SC_Long" , "32/mmSB_G5533" , "ShortBlue_G5538" ) : ( "gnirs$data/gnirsn-ls-short-32-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        
        ( "SB+SXD_G5536" , "SCXD_G5531" , "32/mmSB_G5533" , "ShortBlue_G5538" ) : ( "gnirs$data/gnirsn-sxd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        
        # ShortBlue_G5538 [GN, currently in storage, replaced with G5540], 111/mmSB
        ( "MIR_G5537" , "SC_Long" , "111/mmSB_G5534" , "ShortBlue_G5538" ) : ( "gnirs$data/gnirsn-ls-short-111-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        
        ( "SB+SXD_G5536" , "SCXD_G5531" , "111/mmSB_G5534" , "ShortBlue_G5538" ) : ( "gnirs$data/gnirsn-sxd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        
        # ShortBlue_G5540 [GN, installed October 2012], 32/mmSB
        ( "MIR_G5537" , "SC_Long" , "32/mmSB_G5533" , "ShortBlue_G5540" ) : ( "gnirs$data/gnirsn-ls-short-32-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        
        ( "SB+SXD_G5536" , "SCXD_G5531" , "32/mmSB_G5533" , "ShortBlue_G5540" ) : ( "gnirs$data/gnirsn-sxd-short-32-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        
        # ShortBlue_G5540 [GN, installed October 2012], 111/mmSB
        ( "MIR_G5537" , "SC_Long" , "111/mmSB_G5534" , "ShortBlue_G5540" ) : ( "gnirs$data/gnirsn-ls-short-111-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        
        ( "SB+SXD_G5536" , "SCXD_G5531" , "111/mmSB_G5534" , "ShortBlue_G5540" ) : ( "gnirs$data/gnirsn-sxd-short-111-mdf.fits" , "[1:190,*]" , "0.15" , "XD" ),
        
        # LongBlue_G5515, 10/mmLB
        ( "MIR_G5511" , "LC_Long" , "10/mmLB_G5507" , "LongBlue_G5515" ) : ( "gnirs$data/gnirsn-ls-long-10-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        ( "LXD_G5508" , "LC_XD" , "10/mmLB_G5507" , "LongBlue_G5515" ) : ( "gnirs$data/gnirsn-lxd-long-10-mdf.fits" , "[1:190,*]" , "0.05" , "XD" ),
        
        ( "SXD_G5509" , "LC_XD" , "10/mmLB_G5507" , "LongBlue_G5515" ) : ( "gnirs$data/gnirsn-sxd-long-10-mdf.fits" , "[1:190,*]" , "0.05" , "XD" ),
        
        # LongBlue_G5515, 32/mmLB
        ( "MIR_G5511" , "LC_Long" , "32/mmLB_G5506" , "LongBlue_G5515" ) : ( "gnirs$data/gnirsn-ls-long-32-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        ( "LXD_G5508" , "LC_XD" , "32/mmLB_G5506" , "LongBlue_G5515" ) : ( "gnirs$data/gnirsn-lxd-long-32-mdf.fits" , "[1:190,*]" , "0.05" , "XD" ),
        
        ( "SXD_G5509" , "LC_XD" , "32/mmLB_G5506" , "LongBlue_G5515" ) : ( "gnirs$data/gnirsn-sxd-long-32-mdf.fits" , "[1:190,*]" , "0.05" , "XD" ),
        
        # LongBlue_G5515, 111/mmLB
        ( "MIR_G5511" , "LC_Long" , "111/mmLB_G5505" , "LongBlue_G5515" ) : ( "gnirs$data/gnirsn-ls-long-111-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        # LongBlue_G5542 [GN, installed October 2009], 10/mmLB
        ( "MIR_G5537" , "LC_Long" , "10/mmLB_G5532" , "LongBlue_G5542" ) : ( "gnirs$data/gnirsn-ls-long-10-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        ( "LB+LXD_G5535" , "LCXD_G5531" , "10/mmLBLX_G5532" , "LongBlue_G5542" ) : ( "gnirs$data/gnirsn-lxd-long-10-mdf.fits" , "[1:190,*]" , "0.05" , "XD" ),
        
        ( "LB+SXD_G5536" , "LCXD_G5531" , "10/mmLBSX_G5532" , "LongBlue_G5542" ) : ( "gnirs$data/gnirsn-sxd-long-10-mdf.fits" , "[1:190,*]" , "0.05" , "XD" ),
        
        # LongBlue_G5542 [GN, installed October 2009], 32/mmLB
        ( "MIR_G5537" , "LC_Long" , "32/mmLB_G5533" , "LongBlue_G5542" ) : ( "gnirs$data/gnirsn-ls-long-32-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        ( "LB+LXD_G5535" , "LCXD_G5531" , "32/mmLB_G5533" , "LongBlue_G5542" ) : ( "gnirs$data/gnirsn-lxd-long-32-mdf.fits" , "[1:190,*]" , "0.05" , "XD" ),
        
        ( "LB+SXD_G5536" , "LCXD_G5531" , "32/mmLB_G5533" , "LongBlue_G5542" ) : ( "gnirs$data/gnirsn-sxd-long-32-mdf.fits" , "[1:190,*]" , "0.05" , "XD" ),
        
        # LongBlue_G5542 [GN, installed October 2009], 111/mmLB
        ( "MIR_G5537" , "LC_Long" , "111/mmLB_G5534" , "LongBlue_G5542" ) : ( "gnirs$data/gnirsn-ls-long-111-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        # ShortRed_G5514 [GS, decommissioned June 2005], 111/mm
        ( "MIR_G5511" , "SC_Long" , "111/mm_G5505" , "ShortRed_G5514" ) : ( "gnirs$data/gnirs-ls-short-111-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        ( "MIR_G5511" , "SC_XD/IFU" , "111/mm_G5505" , "ShortRed_G5514" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "SC_XD" , "111/mm_G5505" , "ShortRed_G5514" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "IFU" , "111/mm_G5505" , "ShortRed_G5514" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        
        # ShortRed_G5522 [GS, installed June 2005], 32/mmSR
        ( "MIR_G5511" , "SC_Long" , "32/mmSR_G5506" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ls-short-32-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        ( "MIR_G5511" , "SC_XD/IFU" , "32/mmSR_G5506" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "SC_XD" , "32/mmSR_G5506" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "IFU" , "32/mmSR_G5506" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ifu-short-32-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        
        # ShortRed_G5522 [GS, installed June 2005]. 111/mm
        ( "MIR_G5511" , "SC_Long" , "111/mm_G5505" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ls-short-111-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        ( "MIR_G5511" , "SC_XD/IFU" , "111/mm_G5505" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "SC_XD" , "111/mm_G5505" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "IFU" , "111/mm_G5505" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        
        # ShortRed_G5522 [GS, installed June 2005]. 111/mmSR
        ( "MIR_G5511" , "SC_Long" , "111/mmSR_G5505" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ls-short-111-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        ( "MIR_G5511" , "SC_XD/IFU" , "111/mmSR_G5505" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "SC_XD" , "111/mmSR_G5505" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        ( "MIR_G5511" , "IFU" , "111/mmSR_G5505" , "ShortRed_G5522" ) : ( "gnirs$data/gnirs-ifu-short-111-mdf2.fits" , "[900:1024,*]" , "0.15" , "IFU" ),
        
        # ShortRed_G5539 [GN, currently in storage], 32/mmSR
        ( "MIR_G5537" , "SC_Long", "32/mmSR_G5533", "ShortRed_G5539") : ( "gnirs$data/gnirsn-ls-short-32-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        
        # ShortRed_G5539 [GN, currently in storage], 111/mmSR
        ( "MIR_G5537" , "SC_Long", "111/mmSR_G5534", "ShortRed_G5539") : ( "gnirs$data/gnirsn-ls-short-111-mdf.fits" , "[850:1024,*]" , "0.15" , "LS" ),
        
        # LongRed_G5516, 10/mmLR
        ( "MIR_G5511" , "LC_Long" , "10/mmLR_G5507" , "LongRed_G5516") : ( "gnirs$data/gnirsn-ls-long-10-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        # LongRed_G5516, 32/mmLR
        ( "MIR_G5511" , "LC_Long" , "32/mmLR_G5506" , "LongRed_G5516" ) : ( "gnirs$data/gnirs-ls-long-32-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        # LongRed_G5516. 111/mmLR
        ( "MIR_G5511" , "LC_Long" , "111/mmLR_G5505" , "LongRed_G5516" ) : ( "gnirs$data/gnirs-ls-long-111-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        # LongRed_G5543 [GN, installed October 2009], 10/mmLR
        ( "MIR_G5537" , "LC_Long" , "10/mmLR_G5532" , "LongRed_G5543") : ( "gnirs$data/gnirsn-ls-long-10-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        # LongRed_G5543 [GN, installed October 2009], 32/mmLR
        ( "MIR_G5537" , "LC_Long" , "32/mmLR_G5533" , "LongRed_G5543" ) : ( "gnirs$data/gnirsn-ls-long-32-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
        
        # LongRed_G5543 [GN, installed October 2009], 111/mmLR
        ( "MIR_G5537" , "LC_Long" , "111/mmLR_G5534" , "LongRed_G5543" ) : ( "gnirs$data/gnirsn-ls-long-111-mdf.fits" , "[1:30,*]" , "0.05" , "LS" ),
    }
