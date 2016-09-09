class docstrings:

    def instrument(self):
        """
        Return the instrument value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the instrument used for the observation
        """
        pass
    
    def object(self, format=None, **args):
        """
        Return the object value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the name of the target object observed
        """
        pass
    
    def telescope(self, format=None, **args):
        """
        Return the telescope value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: string as default (i.e., format=None)
        :return: the telescope used for the observation
        """
        pass
    
    def ut_date(self, format=None, **args):
        """
        Return the ut_date value
        
        :param dataset: the data set
        :type dataset: AstroData
        :param format: the return format
        :type format: string
        :rtype: datetime as default (i.e., format=None)
        :return: the UT date at the start of the observation
        """
        pass
