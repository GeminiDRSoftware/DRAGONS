{    "overscanSubtract":{"fl_trim":
                         {
                          'default': False , 
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
                           'default': '_normalized' , #value to be post pended to this primitives outputs (used by CLManager not CL directly for this prim)
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
