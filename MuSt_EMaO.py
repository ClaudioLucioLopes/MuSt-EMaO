from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga3 import NSGA3,associate_to_niches
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.factory import get_reference_directions,get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np


##########################################################################################################
#Input: a problem test set, number of solutions, number of objectives, a reference vector,
#       a sampling population,number of generations, number of solutions evaluations,
#       a termination criteria, and a seed
#Output: ND extended solutions in decision variables, ND extended solutions in objective,
#        ND final solutions in decision variables,ND final solutions in objective,
#        ideal_point,nadir_point,number of generations executed, number of solutions evaluations executed
##########################################################################################################
def call_NSGA3(problem,n_points,n_obj,data_unit_simplex,sampling_p,n_gen,n_eval,termination=None,seed_p=1):
    if len(sampling_p)==0:
        algorithm = NSGA3(pop_size=n_points,n_obj=n_obj,
                      ref_dirs=data_unit_simplex,seed=seed_p)
    else:
        if sampling_p.shape[0] <=data_unit_simplex.shape[0]:
            increase = data_unit_simplex.shape[0] -sampling_p.shape[0]
            random_points_var = random_float(increase, problem.n_var, problem.xl, problem.xu)
            sampling_p = np.concatenate([sampling_p,random_points_var])

        algorithm = NSGA3(pop_size=n_points,n_obj=n_obj,
                          ref_dirs=data_unit_simplex,
                          sampling=sampling_p,seed=seed_p)
    # execute the optimization
    if termination == None:
        if n_eval!=0:
            termination = get_termination("n_eval", n_eval)
        else:
            termination = get_termination('n_gen', n_gen)

    res = minimize(problem,algorithm,termination)

    n_var = problem.n_var
    if (np.sum(res.pop.get('F') == None)) == 1:  # all points are infeasible:
        n_evals = res.algorithm.evaluator.n_eval
        n_gen = res.algorithm.n_gen
        return np.full([ 1, n_var ], np.nan), np.full([ 1, n_obj ], np.nan), \
               np.full([ 1, n_var ], np.nan), np.full([ 1, n_obj ], np.nan), np.full([ 1, n_obj ], np.nan), np.full(
            [ 1, n_obj ], np.nan), n_evals, n_gen

    # Verify the if the solution is factible
    if (np.sum(res.pop.get('G')==None) > 0) :
        x_var = res.pop.get('X')
        f_var = res.pop.get('F')
    else:
        x_var = res.pop.get('X')[ np.sum(res.pop.get('G') > 0, axis=1) == 0, : ]
        f_var = res.pop.get('F')[ np.sum(res.pop.get('G') > 0, axis=1) == 0, : ]

    non_dominated_index = NonDominatedSorting(method="fast_non_dominated_sort").do(f_var,
                                                                                   only_non_dominated_front=True)
    x_var = (x_var[ non_dominated_index, : ])
    f_var = (f_var[ non_dominated_index, : ])

    ideal_point = np.min(f_var,axis=0)
    nadir_point = np.max(f_var,axis=0)
    n_evals = res.algorithm.evaluator.n_eval
    n_gen = res.algorithm.n_gen
    return x_var,f_var,res.X,res.F,ideal_point,nadir_point,n_evals,n_gen

##########################################################################################################
#Input: a problem test set, number of solutions, number of objectives, a reference vector,
#       a sampling population,number of generations, number of solutions evaluations,
#       a termination criteria, and a seed
#Output: ND extended solutions in decision variables, ND extended solutions in objective,
#        ND final solutions in decision variables,ND final solutions in objective,
#        ideal_point,nadir_point,number of generations executed, number of solutions evaluations executed
##########################################################################################################
def call_CTAEA(problem,n_points,n_obj,data_unit_simplex,sampling_p,n_gen,n_eval,termination=None,seed_p=1):
    if len(sampling_p)==0:
        algorithm = CTAEA(ref_dirs=data_unit_simplex,seed=seed_p)
    else:
        if sampling_p.shape[0] <=data_unit_simplex.shape[0]:
            increase = data_unit_simplex.shape[0] -sampling_p.shape[0]
            random_points_var = random_float(increase, problem.n_var, problem.xl, problem.xu)
            sampling_p = np.concatenate([sampling_p,random_points_var])
        algorithm = CTAEA(ref_dirs=data_unit_simplex,sampling=sampling_p,seed=seed_p)
    # execute the optimization
    if termination == None:
        if n_eval!=0:
            termination = get_termination("n_eval", n_eval)
        else:
            termination = get_termination('n_gen', n_gen)

    res = minimize(problem,algorithm,termination)
    n_var = problem.n_var
    if (np.sum(res.pop.get('F') == None)) == 1:
        n_evals = res.algorithm.evaluator.n_eval
        n_gen = res.algorithm.n_gen
        return np.full([ 1, n_var ], np.nan), np.full([ 1, n_obj ], np.nan), \
               np.full([ 1, n_var ], np.nan), np.full([ 1, n_obj ], np.nan), np.full([ 1, n_obj ], np.nan), np.full(
            [ 1, n_obj ], np.nan), n_evals, n_gen


    # Verify the if the solution is factible
    if (np.sum(res.pop.get('G')==None) > 0) : #there is no constraints
        x_var = res.pop.get('X')
        f_var = res.pop.get('F')
    else:
        x_var = res.pop.get('X')[ np.sum(res.pop.get('G') > 0, axis=1) == 0, : ]
        f_var = res.pop.get('F')[ np.sum(res.pop.get('G') > 0, axis=1) == 0, : ]

    non_dominated_index = NonDominatedSorting(method="fast_non_dominated_sort").do(f_var,
                                                                                   only_non_dominated_front=True)
    x_var = (x_var[ non_dominated_index, : ])
    f_var = (f_var[ non_dominated_index, : ])

    ideal_point = np.min(f_var,axis=0)
    nadir_point = np.max(f_var,axis=0)


    n_evals = res.algorithm.evaluator.n_eval
    n_gen = res.algorithm.n_gen
    return x_var,f_var,res.X,res.F,ideal_point,nadir_point,n_evals,n_gen

##########################################################################################################
#Input: a problem test set, number of solutions, number of objectives, a reference vector,
#       a sampling population,number of generations, number of solutions evaluations,
#       a termination criteria, and a seed
#Output: ND extended solutions in decision variables, ND extended solutions in objective,
#        ND final solutions in decision variables,ND final solutions in objective,
#        ideal_point,nadir_point,number of generations executed, number of solutions evaluations executed
##########################################################################################################
def call_MOEAD(problem,n_points,n_obj,data_unit_simplex,sampling_p,n_gen,n_eval,termination=None,seed_p=1):
    if len(sampling_p)==0:
        algo_m = MOEAD(data_unit_simplex, prob_neighbor_mating=0.7,n_neighbors=int(0.3*n_points),seed=seed_p)
    else:
        if sampling_p.shape[0] < data_unit_simplex.shape[0]:
            n_sampling = data_unit_simplex.shape[0] -sampling_p.shape[0]
            sampling_p_add = np.random.rand(n_sampling, problem.n_var)
            sampling_p = np.concatenate([sampling_p,sampling_p_add])
        else:
            n_sampling = data_unit_simplex.shape[0]
            sample_dus = np.random.choice([i for i in range(sampling_p.shape[0])],n_sampling,replace=False)
            sampling_p = sampling_p[sample_dus,:]
        algo_m = MOEAD(data_unit_simplex, prob_neighbor_mating=0.7, sampling=sampling_p, n_neighbors=int(0.3 * n_points),seed=seed_p)
    if termination == None:
        if n_eval!=0:
            termination = get_termination("n_eval", n_eval)
        else:
            termination = get_termination('n_gen', n_gen)

    res = minimize(problem,algo_m,termination)

    # Verify the if the solution is factible
    if (np.sum(res.pop.get('G')==None) > 0) : #There is no constraints:
        x_var = res.pop.get('X')
        f_var = res.pop.get('F')
    else:
        x_var = res.pop.get('X')[ np.sum(res.pop.get('G') > 0, axis=1) == 0, : ]
        f_var = res.pop.get('F')[ np.sum(res.pop.get('G') > 0, axis=1) == 0, : ]

    n_var = problem.n_var
    if len(f_var)==0:  # all points are infeasible:
        n_evals = res.algorithm.evaluator.n_eval
        n_gen = res.algorithm.n_gen
        return np.full([ 1, n_var ], np.nan), np.full([ 1, n_obj ], np.nan), \
               np.full([ 1, n_var ], np.nan), np.full([ 1, n_obj ], np.nan), np.full([ 1, n_obj ], np.nan), np.full(
            [ 1, n_obj ], np.nan), n_evals, n_gen

    non_dominated_index = NonDominatedSorting(method="fast_non_dominated_sort").do(f_var,
                                                                                   only_non_dominated_front=True)
    x_var = (x_var[ non_dominated_index, : ])
    f_var = (f_var[ non_dominated_index, : ])
    f_var.shape

    _, index = np.unique(f_var, axis=0, return_index=True)
    x_var = (x_var[ index, : ])
    f_var = (f_var[ index, : ])

    ideal_point = np.min(f_var,axis=0)
    nadir_point = np.max(f_var,axis=0)
    n_evals = res.algorithm.evaluator.n_eval
    n_gen = res.algorithm.n_gen
    return x_var,f_var,res.X,res.F,ideal_point,nadir_point,n_evals,n_gen

##########################################################################################################
#Input: a pareto front, a reference vector,ideal_point,nadir_point,
#Output: active reference vector, inactive reference vector, distance matrix from solutions to references,
#        closest solutions to each active reference, index of the closest solutions to each active reference
##########################################################################################################
def Classify(PF, reference, ideal_point, nadir_point):
    niche_of_individuals, dist_to_niche, dist_matrix = \
        associate_to_niches(PF, reference, ideal_point, nadir_point)
    dist_matrix.shape

    closest_PF = np.unique(dist_matrix[ :, np.unique(niche_of_individuals) ].argmin(axis=0))

    not_in_niche_of_individuals = [ i for i in range(reference.shape[ 0 ]) if
                             i not in np.unique(niche_of_individuals) ]
    not_in_niche_of_individuals = np.array(not_in_niche_of_individuals)
    len(not_in_niche_of_individuals)
    len(np.unique(niche_of_individuals) )

    if len(not_in_niche_of_individuals) == 0:
        ref_not_in_closest_reference=np.array([])
    else:
        ref_not_in_closest_reference = np.array(reference[ not_in_niche_of_individuals, : ])

    final_closest = PF[ closest_PF, : ]
    closest_non_dominated_index = NonDominatedSorting(method="fast_non_dominated_sort").do(final_closest,
                                                                                   only_non_dominated_front=True)
    final_closest = final_closest[ closest_non_dominated_index, : ]
    return reference[ np.unique(niche_of_individuals), : ],ref_not_in_closest_reference,dist_matrix,final_closest,closest_PF[closest_non_dominated_index]

def squared_dist(A, B):
    return ((A[ :, None ] - B[ None, : ]) ** 2).sum(axis=2)

def calc_potential_energy(A, d):
    i, j = np.triu_indices(len(A), 1)
    D = np.sqrt(squared_dist(A, A)[ i, j ])
    energy = np.log((1 / D ** d).mean())
    return energy

def random_float(n_row, n_col, low, high):
    return np.random.sample((n_row, n_col)) * (high - low) + low

##########################################################################################################
#Input: a pareto front, a number of desired points
#Output: final pareto front with the desired number of points
# ##########################################################################################################
def Reduce(pareto_front, n_points):
    if pareto_front.shape[0 ] > n_points:
        number_of_points_extract = pareto_front.shape[0 ] - n_points
        d = pareto_front.shape[ 1 ] ** 2
        for i_delete in range(number_of_points_extract):
            energy_temp = [ ]
            for i in pareto_front.tolist():
                aux_B = np.array([ i_a for i_a in pareto_front.tolist() if i != i_a ])
                energy_temp.append(calc_potential_energy(aux_B, d))
            exclude = np.argmin(energy_temp)
            pareto_front = np.delete(pareto_front, exclude, axis=0)
        present_graph = pareto_front
    else:
        present_graph = pareto_front

    return present_graph

##########################################################################################################
#Input: a problem test set, number of solutions, number of objectives, a seed, a total number of solution
#       evaluations, a test T configuration(percentage to be applied in each stage), a log print-- verbose,
#       an EMaO algorithm ,
#Output: ND solutions in objective space with number of solutions, number of generations, number of
#        solutions evaluations executed, the final reference vector in Stage 3 with i=2
##########################################################################################################
def MuSt_EMaO(problem, n_points, n_obj, seed_p, total_neval_par, confT, logprint,  alg='NSGA3'):
    total_ngen = 0
    total_neval = 0

    #Step 1
    E1=int(total_neval_par*confT[0])

    R_1 = get_reference_directions("energy", n_obj,int(n_points), seed=seed_p)
    if alg=='NSGA3':
        S1_all_x_var,S1_all_f_var,S1_ND_x_var,S1_ND_f_var,ideal_point_S1,nadir_point_S1,n_eval_1,n_gen1 = call_NSGA3(problem,n_points,n_obj,R_1,[],0,E1,None,seed_p)
    elif alg=='MOEAD':
        S1_all_x_var, S1_all_f_var, S1_ND_x_var, S1_ND_f_var, ideal_point_S1, nadir_point_S1, n_eval_1, n_gen1 = call_MOEAD(
            problem, n_points, n_obj, R_1, [ ], 0, E1, None, seed_p)
    elif alg=='CTAEA':
        S1_all_x_var, S1_all_f_var, S1_ND_x_var, S1_ND_f_var, ideal_point_S1, nadir_point_S1, n_eval_1, n_gen1 = call_CTAEA(
            problem, n_points, n_obj, R_1, [ ], 0, E1, None, seed_p)
    total_neval+=n_eval_1
    total_ngen += n_gen1



    print('Stage 1 #FO: ', n_eval_1)
    R_1_c,R_1_cbar,_,closest_S1,closest_S1_index=Classify(S1_all_f_var, R_1, ideal_point_S1, nadir_point_S1)

    S1_T1_1c_x_var =S1_all_x_var[closest_S1_index,:]
    S1_T1_1c_f_var = S1_all_f_var[closest_S1_index,:]
    S1_T1_1c_f_var = S1_T1_1c_f_var[ ~np.isnan(S1_T1_1c_f_var).any(axis=1) ]
    S1_T1_1c_x_var = S1_T1_1c_x_var[ ~np.isnan(S1_T1_1c_x_var).any(axis=1) ]


    N1 = S1_T1_1c_f_var[np.sum(np.isnan(S1_T1_1c_f_var),axis=1)==0,].shape[0]

    if logprint:
        print('N1: ',N1)
        print('N1 reference lines: ', R_1_c.shape[0])
        print('N1 reference lines not used: ', R_1_cbar.shape[0])

    E2 = int(total_neval_par * confT[ 1 ] )
    E3 = int(total_neval_par * confT[ 2 ] )

    if (N1 == n_points):
        if alg == 'NSGA3':
            S23_all_x_var, S23_all_f_var, S23_ND_x_var, S23_ND_f_var, ideal_point_S23, nadir_point_S23, n_eval_23, n_gen23 = call_NSGA3(
                problem, n_points, n_obj, R_1, S1_T1_1c_x_var, 0, E2+E3, None, seed_p)
        elif alg == 'MOEAD':
            S23_all_x_var, S23_all_f_var, S23_ND_x_var, S23_ND_f_var, ideal_point_S23, nadir_point_S23, n_eval_23, n_gen23 = call_MOEAD(
                problem, n_points, n_obj, R_1, S1_T1_1c_x_var, 0, E2+E3, None, seed_p)
        elif alg == 'CTAEA':
            S23_all_x_var, S23_all_f_var, S23_ND_x_var, S23_ND_f_var, ideal_point_S23, nadir_point_S23, n_eval_23, n_gen23 = call_CTAEA(
                problem, n_points, n_obj, R_1, S1_T1_1c_x_var, 0, E2+E3, None, seed_p)
        total_neval += n_eval_23
        total_ngen += n_gen23
        # Finish the process
        if S23_all_f_var.shape[0] < n_points:
            S_final_f_var = np.concatenate([ S1_T1_1c_f_var, S23_all_f_var])
            non_dominated_index = NonDominatedSorting(method="fast_non_dominated_sort").do(S_final_f_var,
                                                                                       only_non_dominated_front=True)

            S_final_f_var = (S_final_f_var[ non_dominated_index, : ])
            S_final_f_var = S_final_f_var[~np.isnan(S_final_f_var).any(axis=1)]

            ideal_point_S = np.nanmin(S_final_f_var, axis=0, )
            nadir_point_S = np.nanmax(S_final_f_var, axis=0)

            R_3i_c_aux, R_3i_cbar_aux, _, closest_S3i_aux, closest_S3i_index_aux = Classify(S_final_f_var, R_1,
                                                                                            ideal_point_S,
                                                                                            nadir_point_S)
            S_final_f_var_aux = S_final_f_var[ closest_S3i_index_aux, : ]
            if S_final_f_var_aux.shape[ 0 ] >= n_points:
                S_final_f_var = S_final_f_var_aux
        else:
            S_final_f_var = S23_all_f_var
        S_final = Reduce(S_final_f_var, n_points)
        R_3=R_1
    else:
        if alg=='NSGA3':
            S2_all_x_var,S2_all_f_var,S2_ND_x_var,S2_ND_f_var,ideal_point_S2,nadir_point_S2,n_eval_2,n_gen2 = call_NSGA3(problem,R_1_cbar.shape[0],n_obj,R_1_cbar,S1_all_x_var,0,E2,None,seed_p)
        elif alg=='MOEAD':
            S2_all_x_var,S2_all_f_var,S2_ND_x_var,S2_ND_f_var,ideal_point_S2,nadir_point_S2, n_eval_2, n_gen2 = call_MOEAD(problem, R_1_cbar.shape[0], n_obj, R_1_cbar, S1_all_x_var,0,E2, None,seed_p)
        elif alg=='CTAEA':
            S2_all_x_var,S2_all_f_var,S2_ND_x_var,S2_ND_f_var,ideal_point_S2,nadir_point_S2, n_eval_2, n_gen2 = call_CTAEA(problem, R_1_cbar.shape[0],
                                                                                                    n_obj, R_1_cbar, S1_all_x_var, 0,E2, None,seed_p)

        total_neval += n_eval_2
        total_ngen += n_gen2

        S12_x_var = np.concatenate([ S1_T1_1c_x_var, S2_all_x_var ])
        S12_f_var = np.concatenate([ S1_T1_1c_f_var, S2_all_f_var ])

        non_dominated_index = NonDominatedSorting(method="fast_non_dominated_sort").do(S12_f_var,
                                                                                       only_non_dominated_front=True)

        S12_x_var = (S12_x_var[ non_dominated_index, : ])
        S12_f_var = (S12_f_var[ non_dominated_index, : ])
        S12_f_var = S12_f_var[~np.isnan(S12_f_var).any(axis=1)]
        S12_x_var = S12_x_var[ ~np.isnan(S12_x_var).any(axis=1) ]

        print('Stage 2 #FO: ', n_eval_2)



        ideal_point_S12 = np.nanmin(np.vstack([ideal_point_S1,ideal_point_S2]),axis=0,)
        nadir_point_S12 = np.nanmax(np.vstack([nadir_point_S1,nadir_point_S2]),axis=0)


        R_12_c, R_12_cbar, _, closest_S12, closest_S12_index = Classify(S12_f_var, R_1, ideal_point_S12,
                                                                        nadir_point_S12)


        S_0_3c_x_var = S12_x_var[ closest_S12_index, : ]
        S_0_3c_f_var = S12_f_var[ closest_S12_index, : ]


        len(R_12_c)
        len(R_12_cbar)
        len(closest_S12_index)

        N2 = S_0_3c_f_var.shape[ 0 ]

        if logprint:
            print('N2: ', N2)

        # Step3
        if S_0_3c_f_var.shape[ 0 ] <=n_points:
            S_iminus1_3_f_var = np.concatenate([S1_all_f_var,S2_all_f_var])
            S_iminus1_3_x_var = np.concatenate([ S1_all_x_var, S2_all_x_var ])

            S_i_3c_all_x_var =[]
            S_i_3c_all_f_var = [ ]
            size_S_iminus1_3c = S_0_3c_f_var.shape[ 0 ]
            N3=n_points

            for i_stage3 in range(2):
                N3 = int(n_points * (N3 / size_S_iminus1_3c))
                R_3 = get_reference_directions("energy", n_obj, N3, seed=seed_p)
                if alg == 'NSGA3':
                    S3i_all_x_var, S3i_all_f_var, S3i_ND_x_var, S3i_ND_f_var, \
                    ideal_point_S3i, nadir_point_S3i, n_eval_3i, n_gen3i = call_NSGA3(problem, N3, n_obj,
                                                                                  R_3,
                                                                                  S_iminus1_3_x_var, 0, int(E3/2), None, seed_p)
                elif alg == 'MOEAD':
                    S3i_all_x_var, S3i_all_f_var, S3i_ND_x_var, S3i_ND_f_var, \
                    ideal_point_S3i, nadir_point_S3i, n_eval_3i, n_gen3i = call_MOEAD(problem, N3, n_obj,
                                                                                  R_3,
                                                                                  S_iminus1_3_x_var, 0, int(E3/2), None, seed_p)
                elif alg == 'CTAEA':
                    S3i_all_x_var, S3i_all_f_var, S3i_ND_x_var, S3i_ND_f_var, \
                    ideal_point_S3i, nadir_point_S3i, n_eval_3i, n_gen3i= call_CTAEA(problem, N3, n_obj,
                                                                                  R_3,
                                                                                  S_iminus1_3_x_var, 0, int(E3/2), None, seed_p)

                total_ngen+=n_gen3i
                total_neval+= n_eval_3i

                S3i_all_x_var = np.concatenate([ S3i_all_x_var, S_iminus1_3_x_var ])
                S3i_all_f_var = np.concatenate([ S3i_all_f_var, S_iminus1_3_f_var ])

                non_dominated_index = NonDominatedSorting(method="fast_non_dominated_sort").do(S3i_all_f_var,
                                                                                               only_non_dominated_front=True)
                S3i_all_x_var = (S3i_all_x_var[ non_dominated_index, : ])
                S3i_all_f_var = (S3i_all_f_var[ non_dominated_index, : ])

                ideal_point_S3i = np.nanmin(S3i_all_f_var, axis=0, )
                nadir_point_S3i = np.nanmax(S3i_all_f_var, axis=0)

                R_3i_c, R_3i_cbar, _, closest_S3i, closest_S3i_index = Classify(S3i_all_f_var, R_3, ideal_point_S3i,
                                                                                nadir_point_S3i)

                S_i_3c_x_var = S3i_all_x_var[ closest_S3i_index, : ]
                S_i_3c_f_var = S3i_all_f_var[ closest_S3i_index, : ]
                S_i_3c_f_var.shape

                S_i_3c_all_x_var.append(S_i_3c_x_var)
                S_i_3c_all_f_var.append(S_i_3c_f_var)

                S_iminus1_3_f_var = S3i_all_f_var
                S_iminus1_3_x_var = S3i_all_x_var

                size_S_iminus1_3c = S_i_3c_f_var.shape[0]

                if logprint:
                    print('Stage 3 iteration :  ',i_stage3+1)
                    print('S_c: ', S_i_3c_f_var.shape[ 0 ])
                    print('Reisz-energy: ', R_3.shape[ 0 ])

            if S_i_3c_all_f_var[1].shape[0] < n_points:
                S_final_3c_f_var = np.concatenate([S_0_3c_f_var,*S_i_3c_all_f_var])
                S_final_3c_x_var = np.concatenate([S_0_3c_x_var,*S_i_3c_all_x_var])
            else:
                S_final_3c_f_var = S_i_3c_all_f_var[1]
                S_final_3c_x_var = S_i_3c_all_x_var[1]
        else:
            raise Exception("Sorry, we will not use T3 at all!!!! This is not permitted.")

        non_dominated_index = NonDominatedSorting(method="fast_non_dominated_sort").do(S_final_3c_f_var,
                                                                                       only_non_dominated_front=True)

        S_final_3c_f_var = (S_final_3c_f_var[ non_dominated_index, : ])
        S_final_3c_f_var = S_final_3c_f_var[~np.isnan(S_final_3c_f_var).any(axis=1)]
        S_final = Reduce(S_final_3c_f_var, n_points)
    if logprint:
        print('Final:  ', S_final.shape[ 0 ])
        print('Reisz-energy: ', R_3.shape[ 0 ])
    return S_final,total_ngen,total_neval,R_3