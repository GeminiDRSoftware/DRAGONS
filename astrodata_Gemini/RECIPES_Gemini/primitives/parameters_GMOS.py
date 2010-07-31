{    "overscanSubtract":{"fl_over":
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
                        },
        'storeProcessedBias':{'clob':
                              {
                                'default': False,
                                'recipeOverride': True,
                                'type': 'bool',
                                'userOverride':True  
                               }
                              
                              } ,
            'storeProcessedFlat':{'clob':
                              {
                                'default': False,
                                'recipeOverride': True,
                                'type': 'bool',
                                'userOverride':True  
                               }
                              
                              } ,
             "biasCorrect":{"fl_over":
                        {
                          'default': False , 
                          'recipeOverride': True,
                          'type': 'bool',
                          'userOverride':True                                      
                         },
                         "fl_trim":
                         {
                          'default': False , 
                          'recipeOverride': False,
                          'type': 'bool',
                          'userOverride':False    
                          },
                          "fl_vardq":
                          {
                           'default': True ,
                          'recipeOverride': True,
                          'type': 'bool',
                          'userOverride':True    
                           },
                          "outpref":
                          {
                           'default': '_biassub' , #value to be post pended to this primitives outputs
                          'recipeOverride': True,
                          'type': 'str',
                          'userOverride':False    
                           }
                          },
            "makeNormalizedFlat":{"fl_over":
                        {
                          'default': False , 
                          'recipeOverride': False,
                          'type': 'bool',
                          'userOverride':False                                      
                         },
                         "fl_trim":
                         {
                          'default': False , 
                          'recipeOverride': False,
                          'type': 'bool',
                          'userOverride':False    
                          },
                          "fl_vardq":
                          {
                           'default': True ,
                          'recipeOverride': True,
                          'type': 'bool',
                          'userOverride':True    
                           },
                          "outpref":
                          {
                           'default': '_normalized' , #value to be post pended to this primitives outputs
                          'recipeOverride': True,
                          'type': 'str',
                          'userOverride':False    
                           },
                          'fl_bias':
                          {
                           'default': False , 
                          'recipeOverride': False,
                          'type': 'bool',
                          'userOverride':False 
                           }
                          }
}
