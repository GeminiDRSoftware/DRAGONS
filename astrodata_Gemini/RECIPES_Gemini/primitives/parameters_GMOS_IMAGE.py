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
                          'default': True , #maybe hardcode this as this is the purpose of this prim, so setting it to False is pointless
                          'recipeOverride': True,
                          'type': 'bool',
                          'userOverride':True                                      
                         },
                         "fl_trim":
                         {
                          'default': False , #trim the overscan region after it has been subracted? maybe hardcode this as there is a separate prim to do this
                          'recipeOverride': False,
                          'type': 'bool',
                          'userOverride':False    
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
                           'default': '_oversubed' , #value to be post pended to this primitives outputs
                          'recipeOverride': True,
                          'type': 'str',
                          'userOverride':False    
                           }
                        },
        "overscanTrim":{"outsuffix":
                          {
                           'default': '_overtrimd' , #value to be post pended to this primitives outputs
                          'recipeOverride': True,
                          'type': 'str',
                          'userOverride':False    
                           }
                        } 
}
