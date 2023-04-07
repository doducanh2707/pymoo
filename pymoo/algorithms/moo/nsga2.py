import numpy as np
import warnings

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import *
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible
from pymoo.util.randomized_argsort import randomized_argsort

# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape

    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = algorithm.tournament_type
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):

        a, b = P[i, 0], P[i, 1]
        a_cv, a_f, b_cv, b_f = pop[a].CV[0], pop[a].F, pop[b].CV[0], pop[b].F
        rank_a, cd_a = pop[a].get("rank", "crowding")
        rank_b, cd_b = pop[b].get("rank", "crowding")
        # if at least one solution is infeasible
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)
        # both solutions are feasible
        else:

            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(a_f, b_f)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b

            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')

            else:
                raise Exception("Unknown tournament type.")

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(S[i]):
                S[i] = compare(a, cd_a, b, cd_b, method='larger_is_better', return_random_if_equal=True)

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(RankAndCrowding):
    
    def __init__(self, nds=None, crowding_func="cd"):
        warnings.warn(
                "RankAndCrowdingSurvival is deprecated and will be removed in version 0.8.*; use RankAndCrowding operator instead, which supports several and custom crowding diversity metrics.",
                DeprecationWarning, 2
            )
        super().__init__(nds, crowding_func)

# =========================================================================================================
# Implementation
# =========================================================================================================
class KneePointSelection(RankAndCrowding):
        def __init__(self, nds=None, crowding_func="cd"):
            super().__init__(nds, crowding_func)
            self.t = np.zeros(100)
            self.r = np.ones(100)
        def _do(self, problem,pop,*args,n_survive=None,prob=0.9, **kwargs):
            # get the objective space values and objects
            F = pop.get("F").astype(float, copy=False)

            # the final indices of surviving individuals
            survivors = []

            # do the non-dominated sorting until splitting front
            fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
            for k, front in enumerate(fronts):
                # current front sorted by crowding distance if splitting
                if len(survivors) + len(front) > n_survive:
                    I = []
                    n_selected = n_survive - len(survivors)
                    F_i = F[front]
                    E = np.array(sorted(F_i,key=lambda x: x[0]))[[[0,-1]]]
                    approx_ideal = F_i.min(axis=0)
                    approx_nadir = F_i.max(axis =0)
                    # Tinh vung lan can cua 1 diem
                    R  = np.zeros(approx_ideal.shape)
                    self.r[k] = self.r[k] * (np.e ** (-(1-(self.t[k])/0.5)/2))
                    for j in range(2):
                        R[j] = (approx_nadir[j]-approx_ideal[j])*self.r[k]
                    # Xay dung duong noi 2 diem cuc bien va tinh khoang cach 
                    a = E[1][1] - E[0][1]
                    b = E[0][0] - E[1][0]
                    c = a*(E[1][0]) + b*(E[1][1])
                    dist = [-(a*x[0]+b*x[1]-c)/np.sqrt(a **2 + b ** 2) for x in F_i]
                    index = np.argsort(dist)
                    remove = []
                    for idx in index:
                        if len(I)  == n_selected:
                            break
                        if idx in remove: 
                            continue
                        I.append(idx)
                        for i in range(len(front)):
                            if i in remove or i in I:
                                continue
                            if np.abs(F_i[i][0] - F_i[idx][0]) <=  R[0] or np.abs(F_i[i][1] - F_i[idx][1]) <=  R[1]: 
                                remove.append(i)
                    # Crowding distance
                    crowding_of_front = \
                        self.crowding_func.do(
                            F[front, :],
                            n_remove=0
                        )
                        
                    cd = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                    if len(I) < n_selected:
                        for idx in cd:
                            if len(I)  == n_selected:
                                break
                            if idx in I:
                                continue
                            I.append(idx)
                            
                # otherwise take the whole front unsorted
                else:
                    # calculate the crowding distance of the front
                    crowding_of_front = \
                        self.crowding_func.do(
                            F[front, :],
                            n_remove=0
                        )
                    I = np.arange(len(front))
                # save rank and crowding in the individual class
                for j, i in enumerate(front):
                    pop[i].set("rank", k)
                    pop[i].set("crowding", crowding_of_front[j])

                # extend the survivors by all or selected individuals
                survivors.extend(front[I])
            return pop[survivors]
        
        
class KneePointSelection_v2(RankAndCrowding):
        def __init__(self, nds=None, crowding_func="cd"):
            super().__init__(nds, crowding_func)
            self.t = np.zeros(100)
            self.r = np.ones(100)
        def _do(self, problem,pop,*args,n_survive=None, **kwargs):
            # get the objective space values and objects
            F = pop.get("F").astype(float, copy=False)
            # the final indices of surviving individuals
            survivors = []

            # do the non-dominated sorting until splitting front
            fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
            for k, front in enumerate(fronts):
                I = []
                F_i = F[front]
                E = np.array(sorted(F_i,key=lambda x: x[0]))[[[0,-1]]]
                approx_ideal = F_i.min(axis=0)
                approx_nadir = F_i.max(axis =0)
                # Tinh vung lan can cua 1 diem
                R  = np.zeros(approx_ideal.shape)
                self.r[k] = self.r[k] * (np.e ** (-(1-(self.t[k])/0.5)/2))
                for j in range(2):
                    R[j] = (approx_nadir[j]-approx_ideal[j])*self.r[k]
                # Xay dung duong noi 2 diem cuc bien va tinh khoang cach 
                a = E[1][1] - E[0][1]
                b = E[0][0] - E[1][0]
                c = a*(E[1][0]) + b*(E[1][1])
                dist = [-(a*x[0]+b*x[1]-c)/np.sqrt(a **2 + b ** 2) for x in F_i]
                index = np.argsort(dist)
                remove = []
                for idx in index:
                    if idx in remove: 
                        continue
                    I.append(idx)
                    for i in range(len(front)):
                        if i in remove or i in I:
                            continue
                        if np.abs(F_i[i][0] - F_i[idx][0]) <=  R[0] or np.abs(F_i[i][1] - F_i[idx][1]) <=  R[1]: 
                            remove.append(i)
                crowding_of_front = \
                        self.crowding_func.do(
                            F[front, :],
                            n_remove=0
                        )
                # save rank and crowding in the individual class
                for j, i in enumerate(front):
                    pop[i].set("rank", k)
                    pop[i].set("crowding", crowding_of_front[j])
                # extend the survivors by all or selected individuals
                survivors.extend(front[I])
            for k, front in enumerate(fronts):
                I = []
                for i in np.arange(len(front)):
                    if front[i] not in survivors:
                        I.append(i)
                # current front sorted by crowding distance if splitting
                if len(survivors) + len(I) > n_survive:
                    # Define how many will be removed
                    n_remove = len(survivors) + len(front) - n_survive
                    # re-calculate the crowding distance of the front
                    crowding_of_front = \
                        self.crowding_func.do(
                            F[front, :],
                            n_remove=0
                        )
                    tmp = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                    I = []
                    for ii in tmp: 
                        if front[ii] not in survivors:
                            I.append(ii)
                    I = I[:-n_remove]
                # extend the survivors by all or selected individuals
                survivors.extend(front[I])
            return pop[survivors]
               
class NSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=PermutationRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                #  crossover=SBX(eta=15, prob=0.9),                
                #  mutation=PM(eta=20),
                crossover=OrderCrossover(),
                mutation=InversionMutation(),
                 survival=KneePointSelection_v2(),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


parse_doc_string(NSGA2.__init__)
