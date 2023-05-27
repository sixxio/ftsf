import numpy as np

class Scaler:
    '''
    Instance of default minmax scaler adapted for price scaling.


    '''
    
    __minimum = 0
    __maximum = 0
    
    def __init__(self, params = None):
        '''
        Initializes instance of scaler.

        Args:

            params: List of min and max values.

        Example:
        >>> Scaler([0.2, 2.3])
        '''
        if params != None:
            self.__minimum, self.__maximum = params[0], params[1]

    def fit(self, array_to_fit):
        '''
        Fits scaler on input array.
        '''
        self.__minimum = np.min(array_to_fit)
        self.__maximum = np.max(array_to_fit)
        return self
    
    def scale(self, array) -> np.array:
        '''
        Returns array scaled on data fed during fit.
        '''
        return (array-self.__minimum)/(self.__maximum - self.__minimum)
    
    def fit_scale(self, fit_array, scale_array) -> list:
        '''
        Returns two arrays, scaled on first array range.
        '''
        self.__minimum = np.min(fit_array)
        self.__maximum = np.max(fit_array)
        return (fit_array-self.__minimum)/(self.__maximum - self.__minimum), (scale_array-self.__minimum)/(self.__maximum - self.__minimum)
    
    def unscale(self, array) -> np.array:
        '''
        Returns array, unscaled to source range.
        '''
        return array*(self.__maximum - self.__minimum) + self.__minimum
    
    def params(self) -> list:
        '''
        Returns min and max values, used to scale.
        '''
        return self.__minimum, self.__maximum