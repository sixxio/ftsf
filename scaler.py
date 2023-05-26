import numpy as np

class Scaler:
    '''
    Represents default minmax scaler adapted for price scaling.
    
    Methods:
    
    fit(array) - defines minimum and maximum values for scaling;
    
    scale(array) - scales array by defined minimum and maximum values;
    
    fit_scale(fit_array, scale_array) - defines minimum and maximum from fit_array, then scales fit_array and scale_array using defined parameters;
    
    unscale(array) - scales array back to source range;
    
    params() - returns minimum and maximum parameters of scaling.
    '''
    
    minimum, maximum = 0, 0
    
    def __init__(self, params = None):
        if params != None:
            self.minimum, self.maximum = params[0], params[1]

    def fit(self, array_to_fit) -> any:
        '''
        Fits scaler on input array.
        '''
        self.minimum = np.min(array_to_fit)
        self.maximum = np.max(array_to_fit)
        return self
    
    def scale(self, array) -> np.array:
        '''
        Returns array scaled on data fed during fit.
        '''
        return (array-self.minimum)/(self.maximum - self.minimum)
    
    def fit_scale(self, fit_array, scale_array) -> list:
        '''
        Returns two arrays, scaled on first array range.
        '''
        self.minimum = np.min(fit_array)
        self.maximum = np.max(fit_array)
        return (fit_array-self.minimum)/(self.maximum - self.minimum), (scale_array-self.minimum)/(self.maximum - self.minimum)
    
    def unscale(self, array) -> np.array:
        '''
        Returns array, unscaled to source range.
        '''
        return array*(self.maximum - self.minimum) + self.minimum
    
    def params(self) -> list:
        '''
        Returns min and max values, used to scale.
        '''
        return self.minimum, self.maximum