import numpy as np
from pymoo.factory import get_problem,get_reference_directions
from pymoo.visualization.scatter import Scatter
from MuSt_EMaO import MuSt_EMaO

if __name__ == '__main__':
    seed_p = 1
    n_points = 100
    nb_eval = 20000
    n_obj = 3

    problem = get_problem("C2DTLZ2", n_obj=n_obj)
    config_test_T = np.array([ [ 0.25, 0.25, 0.5 ] ])
    confT = config_test_T[ 0 ]

    data_unit_simplex = get_reference_directions("energy", n_obj, int(n_points), seed=seed_p)

    final_PF, total_ngen, total_neval, MSEA_resolution = MuSt_EMaO(problem, n_points, n_obj, seed_p, nb_eval, confT,
                                                                   True, alg='NSGA3')

    plot = Scatter()
    plot.add(final_PF)
    plot.show(block=True)
    plot.interactive(False)
