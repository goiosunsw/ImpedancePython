import numpy as np
import matplotlib.pyplot as pl

class DuctSection:
    '''Base class for a 1D resonator section'''
    def __init__(self):
        pass
    
    
    def __str__(self):
        '''Prints the geometrical dimensions of this section'''
        pass
        
       
class MiddleSection(self):
    '''Any section connecting to another section in either end'''
    def __init__(self, **kwargs):
        self.__SetTravellingWaveTransferMatrix(**kwargs)
        
    def ApplyTransferMatrix(self, inMattrix):
        '''Calculates the transfer matrix to apply 
        to the downstream sections
        
        in Mattrix is the down stream transfer mattrix'''
        try:
            
    
    def __SetTravellingWaveTransferMatrix(self, **kwargs):
        '''Set the transfer matrix in terms of travelling wave coefficients
        
        MiddleSection.__SetTravellingWaveTransferMatrix(self, tmOO=1, tmOI=0, tmIO=0, tmII=1)
            _     _ _            _ _     _
            | pO1 | | tmOO  tmOI | | pO2 |
        M = |     | |            | |     |
            | pI1 | | tmIO  tmII | | pI2 |
            -     - -            - -     -
        
        where PO1, PI1 are the outgoing incoming pressure waves at input
        and PO2, PI2 at the output
        '''
        
        mattrix_coeffs = ('tmOO','tmOI','tmIO','tmII',)
        
        for key, value in kwargs.items():
            if key in mattrix_coeffs:
            try:
                value(0.0)
                setattr(self, key, value)
            except TypeError:
                setattr(self, key, np.vectorize(lambda x: value))
        
class TerminationImpedance(DuctSection):
    '''Base class for a termination impedance'''
    def __init__(self):
        self.zl = np.vectorize(lambda x: 0.0)
        
    def __call__(self, freq):
        '''Return the complex value of the impedance at 
        frequency'''
        
        return self.zl(freq)
        
    def plot(self, fig=None, fmin=0.0, fmax=4000.0, npoints=200):
        if not fig:
            fig=pl.figure
        fvec = np.linspace(fmin,fmax,npoints)
        pl.plot(fvec,self(fvec))
        return fig

class PerfectOpenEnd(TerminationImpedance):
    '''Ideal open end impedance
    Load impedance Zl(f) = 0
    Reflection function R(f) = -1 '''
    def __init__(self):
        self.zl = np.vectorize(lambda x: 0.0)


class PerfectClosedEnd(TerminationImpedance):
    '''Ideal open end impedance
    Load impedance Zl(f) = 0
    Reflection function R(f) = 1 '''
    def __init__(self):
        self.zl = np.vectorize(lambda x: np.Inf)


class StraightSection(MiddleSection):
    '''A straight tube element, either conical or cylindrical'''
    def __init__(self, r_in, r_out, length):
        '''Initialise duct section, with parameters:
        * r_in, r_out: input and output radiuses
        * length: (of the duct section)'''
        self.r_in   = r_in
        self.r_out  = r_out
        self.length = length
        
        self.x1 = self.set_apex_distance()
        self.x2 = self.set_base_distance()
        
        
class OneDResonator:
    '''A one dimensional resonator,
    incorporating several elements'''
    def __init__(self):
        self.middle_elements = []
        self.termination = None
        self.set_perfect_open_pipe()
        
    def set_perfect_open_pipe(self):
        '''Set the termination impedance to a static pressure
        (Zl = 0)'''
        self.termination = PerfectOpenEnd()
        
    def set_perfect_closed_pipe(self):
        '''Set the termination impedance to a rigid wall
        (Zl = Inf)'''
        self.termination = PerfectClosedEnd()
        
    def get_input_impedance(self):
        pass
        
    def draw(self):
        '''Draw resonator in a window'''
        fig = pl.figure()
        fig.canvas.set_window_title('Test')
        return fig
        
    def plot_impedance(self, fig=None, fmin=0.0, fmax=4000.0, npoints=200):
        '''Plot the input impedance of the complete resonator'''
        if not fig:
            fig=pl.figure
        fvec = np.linspace(fmin,fmax,npoints)
        pl.plot(fvec,self.get_input_impedance(fvec))
        return fig
        