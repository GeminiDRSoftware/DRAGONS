{ "showParams": {"test": 
                 {
                    "default": "GMOS_IMAGE_SETTING",
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
    "attachMDF":{"outsuffix":
                          {
                           'default': '_mdf' , #value to be post pended to this primitives outputs
                          'recipeOverride': True,
                          'type': 'str',
                          'userOverride':False    
                           }
                        } 
            
}
