0.3.1 (Unreleased)
------------------

Added:
- CRN.trajectory returns a Trajectory object for integration and fitting
- CRN.equilibrium returns an Equilibrium object that can be fit to experimental data
- ns.jupyter.ion and ioff allow to turn interactive output on or off

Changed:
- CRN.equilibrate now accepts 2D initial conditions with specialized CRN.params
- Trajectory.fit accepts argument observe, which can be a species name of custom conversion function

Deprecated:
- CRN.integrate is deprecated in favor of CRN.trajectory().eval
- CRN.fit is deprecated in favor of CRN.trajectory().fit
- CRN.equilibrate is deprecated in favor of CRN.equilibrium().eval
- Argument conversion to Trajectory.fit is deprecated in favor of observe

Fixed:
- ParameterMap.specification for behaves better with absent species
- CRN.parametrize_for is now more careful when substituting specialized parameters in expressions
