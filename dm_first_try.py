import perceval as pcvl
import numpy as np
import scipy as scp
from perceval import SVDistribution, StateVector




class DensityMatrix():

    """

    """

    def __init__(self, svd : Union[SVDistribution, StateVector, BasicState], max_n: int = None, min_n: int = 0):

        #self.min_n = min_n

        if max_n == None:
            max_n = 0
            for key, prob in svd.items()
                if key.n > max_n:
                    max = key.n
            self.max_n = max_n
        else:
            self.max_n = max_n


        self.matrix = scp.Sparse()
