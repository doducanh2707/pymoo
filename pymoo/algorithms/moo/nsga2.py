import numpy as np
import warnings

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible


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

class RankAndCrowdingCustom(RankAndCrowding):
        def __init__(self, nds=None, crowding_func="cd"):
            warnings.warn(
                    "RankAndCrowdingSurvival is deprecated and will be removed in version 0.8.*; use RankAndCrowding operator instead, which supports several and custom crowding diversity metrics.",
                    DeprecationWarning, 2
                )
            super().__init__(nds, crowding_func)
        def _do(self, problem,pop,*args,n_survive=None,prob=0.9, **kwargs):
            # get the objective space values and objects
            F = pop.get("F").astype(float, copy=False)

            # the final indices of surviving individuals
            survivors = []

            # do the non-dominated sorting until splitting front
            fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
            for k, front in enumerate(fronts):
                
                I = np.arange(len(front))

                # current front sorted by crowding distance if splitting
                if len(survivors) + len(I) > n_survive*prob:
                    break

                # otherwise take the whole front unsorted
                else:
                    # calculate the crowding distance of the front
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
            if len(survivors < n_survive):
                last = np.array([])
                for ii,front in enumerate(fronts[k:]):
                    last = np.concatenate(last,front)
                    crowding_of_front = \
                        self.crowding_func.do(
                            F[front, :],
                            n_remove=0
                        )
                    for j, i in enumerate(front):
                        pop[i].set("rank", k+ii)
                        pop[i].set("crowding", crowding_of_front[j])
                survivors.extend(np.random.choice(last,n_survive-len(survivors)))
            return pop[survivors]

class NSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowdingCustom(),
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
