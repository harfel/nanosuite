0.3.1 (Unreleased)
------------------

Added:
- CRN.trajectory returns a Trajectory object for integration and fitting
- CRN.equilibrium returns an Equilibrium object that can be fit to experimental data
- ns.jupyter.ion and ioff allow to turn interactive output on or off

Changed:
- CRN.equilibrate now accepts 2D initial conditions with specialized CRN.params
- CRN.param t0 is now fixed by default

Deprecated:
- CRN.integrate is deprecated in favor of CRN.trajectory().eval
- CRN.fit is deprecated in favor of CRN.trajectory().fit
- CRN.equilibrate is deprecated in favor of CRN.equilibrium().eval

Fixed:
- ParameterMap.specification for behaves better with absent species
- CRN.parametrize_for is now more careful when substituting specialized parameters in expressions
