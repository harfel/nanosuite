"""Assorted utilities

NOTE: This module is considered experimental. Anything in this
namespace may change, or be removed between versions without prior
notice.
"""
import numpy as np

def burst(species: int, stoichiometries: np.ndarray, impurity: float, state: np.ndarray):
    """Model instantaneous reactions among impure species

    For a given species with given impurity, calculate the maximal
    possible reaction flow of a side reaction (given by its
    stoichiometry) where one of the reaction educts is depleted.
    Return a new state with this reaction applied.

    Parameters
    ----------
    species: int
        index of impure species in state vector
    stoichiometries: np.array
        stoichiometric coeffcients of the impure reaction
    impurity: float
        impurity of impure species (between 0 and 1)
    state: np.ndarray
        State vector to which impure reaction is applied
    """
    flux = np.min([
        -(impurity if idx==species else 1.)/stoich*conc
        for idx, (conc, stoich) in enumerate(zip(state.T, stoichiometries))
        if stoich < 0
    ], axis=0)
    return state + flux.reshape(-1,1)*stoichiometries
