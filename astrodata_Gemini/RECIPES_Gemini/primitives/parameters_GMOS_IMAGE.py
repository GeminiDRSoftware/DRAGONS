{   "showParams": {"test": 
                   {
                    "default": "GMOS_SPECT_SETTING",
                    "recipeOverride": True,
                    "uiLevel": "debug",
                    "userOverride":True
                    
                    },
                 
                 "otherTest":
                    {"default": "default",
                     "userOverride":True}
                },
    "pause": {"message":
              { "default":"Pausing Reduction by Control System Request",
               "type":"string",
               "a":"default comes first, the rest alphabetically",
               "note1":"these are just test parameters...",
               "note2":"pause doesn't need a 'message' parameter"
               }
              },
    "overscanSubtract":{"fl_over":
                        {
                          'default': True ,
                          'recipeOverride': True,
                          'type': 'bool',
                          'userOverride':True                                      
                         },
                         "fl_trim":
                         {
                          'default': False ,
                          'recipeOverride': True,
                          'type': 'bool',
                          'userOverride':True    
                          },
                          "fl_vardq":
                          {
                           'default': False ,
                          'recipeOverride': True,
                          'type': 'bool',
                          'userOverride':True    
                           },
                          "outpref":
                          {
                           'default': 'oversub' ,
                          'recipeOverride': True,
                          'type': 'str',
                          'userOverride':True    
                           }
                        }
            
}
