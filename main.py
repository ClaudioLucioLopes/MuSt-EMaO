import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from pymoo.factory import get_problem,get_reference_directions
from pymoo.visualization.scatter import Scatter
from MuSt_EMaO import MuSt_EMaO

if __name__ == '__main__':
    seed_p = 7
    n_points = 100
    nb_eval = 20000
    n_obj = 3

    problem = get_problem("C2DTLZ2", n_obj=n_obj)

    gamma=1/2

    data_unit_simplex = get_reference_directions("energy", n_obj, int(n_points), seed=seed_p)

    ##########################################################################################################
    # MuSt_EMaO:
    # Input: a problem test set, number of solutions, number of objectives, a seed, a total number of solution
    #       evaluations, a gamma to apply in the T configuration set(tested values are 2/3, 1/3 and 1/2),
    #       a log print-- verbose,
    #       an EMaO algorithm (3 algorithms are available in this implementation: NSGA-III, MOEA/D, C-TAEA) ,
    # Output: ND solutions in objective space with number of solutions, number of generations, number of
    #        solutions evaluations executed, the final reference vector in Stage 3 with i=2
    ##########################################################################################################
    final_PF, total_ngen, total_neval, MSEA_resolution = MuSt_EMaO(problem, n_points, n_obj, seed_p, nb_eval, gamma,
                                                                   True, alg='NSGA-III')

    # Visualization
    plot = Scatter()
    plot.add(final_PF)
    plot.show(block=True)
