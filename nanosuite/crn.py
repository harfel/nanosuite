from warnings import warn
import csv
import re
import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
from lmfit import Parameter, Parameters


class CRN:
    def __init__(self, species=None):
        self.species = species or []
        self.complexes = []
        self.reactions = {}
        self.parameters = Parameters()

    def _repr_html_(self):
        return (
            '<table>'
            + '\n'.join(
                f'''<tr>
                    <td style="text-align: right">{' + '.join(reaction[0])}</td>
                    <td style="text-align: center">&LongRightArrow;</td>
                    <td style="text-align: left">{' + '.join(reaction[1])}</td>
                    <td style="text-align: left">{rate.name} = {rate.value:.2g}</td>
                </tr>'''
                for reaction, rate in self.reactions.items()
            )
            + '</table>'
        )

    @property
    def complex_graph(self):
        return np.array([
            [
                compl.count(name)
                for compl in self.complexes
            ]
            for name in self.species
        ])

    @property
    def complex_adjacency(self):
        return np.array([
            [
                self.reactions.get((lhs, rhs), 0.)
                for lhs in self.complexes
            ]
            for rhs in self.complexes
        ])

    def scale_concentration_unit(self, factor):
        """Scale reaction rate constants to account for a change in concentration unit
        
        For example, if current rate constants are given in M^-1s^-1, the call
        crn.scale_concentration_unit(1e-9) will rescale those to nM^-1s^-1.
        """
        for reaction, rate in self.reactions.items():
            rate.value *= factor**(len(reaction[0])-1)

    def add_reaction(self, lhs, rhs, rate):
        if rate.name in self.parameters:
            raise ValueError(f"Parameter '{rate.name}' is already used.")

        # collect species and complexes
        for name in lhs+rhs:
            if name not in self.species:
                self.species.append(name)

        for compl in [lhs, rhs]:
            if compl not in self.complexes:
                self.complexes.append(compl)

        self.reactions[lhs, rhs] = rate
        self.parameters[rate.name] = rate

    def rate_law(self, repeats=1):
        """Mass action kinetics derived from complex graph

        This method generates mass action kinetic equations for a
        chemical reaction network using the method of van der Schaft et al.
        (2011) SIAM J Appl Math 73(2):953-973.

        Params
        ------
        Z: the complex map, n x m numpy array
        A: the augmented complex graph adjacency, m x x numpy array

        Returns
        -------
        A function with signature func(t, x) where
        t is the time (not used) and x is a numpy array of length n
        indicating the system state.
        """
        # calculate graph Laplacian
        sum_diag = np.diag(np.sum(self.complex_adjacency, axis=0))
        laplacian = sum_diag - self.complex_adjacency

        Z = block_diag(*repeats*[self.complex_graph])
        L = block_diag(*repeats*[laplacian])

        def kinetics(_, state):
            # complex_graph.T @ log(state) with convention 0*inf = 0
            with np.errstate(invalid='ignore'):
                tmp = np.log(state, out=-np.inf*np.ones_like(state), where=(state != 0))
                tmp = np.nansum(Z*tmp, axis=0)
            return -Z @ L @ np.exp(tmp)

        return kinetics

    def integrate(self, initial_condition, t0=0., t_eval=None):
        r = 1 if len(initial_condition.shape) == 1 else initial_condition.shape[0]
        kinetics = self.rate_law(r)
        t_eval = t_eval if t_eval is not None else np.linspace(0, 100, 101)
        res = solve_ivp(kinetics, (t0, t_eval[-1]), initial_condition.flatten(),
                        t_eval=t_eval, vectorized=True)
        return res.y.reshape(initial_condition.shape+t_eval.shape)

    @staticmethod
    def _parse_reaction(string):
        def parse_complex(string):
            return tuple(sorted(name.strip() for name in string.split('+')))

        lhs, _, rhs = string.partition('->')
        lhs = parse_complex(lhs)
        rhs = parse_complex(rhs)
        return lhs, rhs

    @classmethod
    def from_string(cls, string, species=None):
        crn = cls(species=species)

        # parse reaction
        reactions = [] # reaction list: (lhs, rhs, name, val)
        for raw_line in string.split('\n'):
            line, _, __ = raw_line.partition('#') # remove comments
            line = line.strip()
            if not line:
                continue

            reaction, sep, rate_constant = line.partition(';')

            if '->' in reaction:
                lhs, rhs = cls._parse_reaction(reaction)
                # format of rate_constant string: [identifier][=][float-literal]
                if not sep:
                    reactions.append((lhs, rhs, None, 1))
                else:
                    param, equals, val = rate_constant.partition('=')
                    if equals:
                        reactions.append((lhs, rhs, param.strip(), float(val)))
                    else:
                        try:
                            val = float(param)
                            reactions.append((lhs, rhs, None, val))
                        except ValueError:
                            reactions.append((lhs, rhs, param.strip(), 1))

            else:
                raise ValueError(f"Invalid input: {raw_line.strip()}")

        # name unnamed constants
        bound_names = [name for lhs, rhs, name, val in reactions]
        free_names = [name for idx, _ in enumerate(reactions)
                      if (name := f'k{idx}') not in bound_names]

        # add reactions
        for lhs, rhs, name, val in reactions:
            if not name:
                name = free_names.pop(0)
            constant = Parameter(name, value=val, vary=True, min=0)
            crn.add_reaction(lhs, rhs, constant)

        return crn

    @classmethod
    def from_kinDA(cls, path, species=None):
        crn = cls(species=species)
        with open(path) as csvfile:
            reader = csv.reader(csvfile)

            # skip to reaction rate data table
            while (row := next(reader)) != ['# REACTION RATE DATA']:
                pass
            # and table header
            assert next(reader)[0] == 'reaction'

            # parse each reaction in the table
            idx = 0 # reaction index
            while len(row := next(reader)) == 5:
                reaction, k_forward, _, k_backward = row[:4]
                lhs, rhs = cls._parse_reaction(reaction)

                # skip reactions that do not convert species
                if lhs == rhs:
                    continue

                pattern = re.compile(r"\[Complex\(([^\)]*)\)\]")
                lhs = tuple(pattern.match(name).group(1) for name in lhs)
                rhs = tuple(pattern.match(name).group(1) for name in rhs)

                kf = float(k_forward)
                kb = float(k_backward)
                k_effective = kf*kb/(kf+kb)

                constant = Parameter(f"k{idx}", value=k_effective, vary=True, min=0)
                idx += 1

                crn.add_reaction(lhs, rhs, constant)

            return crn
