import numpy as np


class early_stopping(object):

    @staticmethod

    def stopping(n:int,
                 epsilon: float,
                 accuracy: list):
        
        if len(accuracy) <= n: 
            print(f'The first {n} epochs have not finished yet')
        elif len(accuracy) > n:
            n_accuracy = np.array(accuracy[-n:])
            differences = np.diff(n_accuracy)
            print(f'Boolean list of checking {np.abs(differences) <= epsilon}')
            return(sum(np.abs(differences)<= epsilon) == (n-1))

