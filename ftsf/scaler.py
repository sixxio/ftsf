import numpy as np

class Scaler:
    '''
    Instance of default minmax scaler adapted for price scaling.

    Attributes:

        minimum: Minimum value seen during fit.

        maximum: Maximum value seen during fit.

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

    def fit(self, array):
        '''
        Fits scaler on array.

        Args:

            array: Array to fit scaler on.

        Returns:

            Scaler fitted on array.

        Example:
        >>> scaler.fit([[0.2, 2.3], [0.3,0.2], ..])
        '''
        self.__minimum = np.min(array)
        self.__maximum = np.max(array)
        return self
    
    def scale(self, array) -> np.array:
        '''
        Scales array using min and max found on fit.

        Args:

            array: Array to scale.

        Returns:

            Scaled array.

        Example:
        >>> scaler.scale([[2, 3], [0.3,0.2], ...])
        [[0.6666, 1], [0.11, 0.066], ..]
        '''
        return (array-self.__minimum)/(self.__maximum - self.__minimum)
        
    def unscale(self, array) -> np.array:
        '''
        Scales back array using min and max found on fit.

        Args:

            array: Array to scale back.

        Returns:

            Scaled back array.

        Example:
        >>> scaler.scale([[0.6666, 1], [0.11, 0.066], ..])
        [[2, 3], [0.3,0.2], ..]
        '''
        return array*(self.__maximum - self.__minimum) + self.__minimum
    
    def params(self) -> list:
        '''
        Shows min and max values, used to scale.

        Returns:

            min and max parameters of scaler.

        Example:
        >>> scaler.params()
        (2, 29)
        '''
        return self.__minimum, self.__maximum