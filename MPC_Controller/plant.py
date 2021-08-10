from systems import NoisyDtLTISystem


class Plant( NoisyDtLTISystem ):
    def __init__ ( self, Ts, Sw, Sv, x0 ):
        
        # state space matrices,
        # A = [[0, -0.999],
        #         [1, 2]]
        # B = [[5.903e-7],
        #        [5.314e-7]]
        # C = [[1, 0]]
        # [position, velocity, current]
        R = 100
        L = 1.3e-3
        A = [[0, 1, 0],
                [0.0640, 0, 0],
                [0, 0, -R/L]]
        B = [[0],
               [0],
               [1/L]]
        C = [[1, 0, 0]]
        D = [[0]]

        NoisyDtLTISystem.__init__( self, A, B, C, D, Ts, Sw, Sv, x0 )
