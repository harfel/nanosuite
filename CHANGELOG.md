0.3.2 (Unreleased)
------------------

Changed:
- Demote Assay.rfu.group and Assay.setup.group from multiindex level to coordinate
- Assay.setup now uses 'sample' as dimension name rather than 'content'


0.3.1
-----

Added:
- CRN.trajectory returns a Trajectory object for integration and fitting
- CRN.equilibrium returns an Equilibrium object that can be fit to experimental data
- Assay.annotate allows to add arbitrary annotations to assay samples
- ns.jupyter.ion and ioff allow to turn interactive output on or off

Changed:
- CRN.equilibrate now accepts 2D initial conditions with specialized CRN.params
- CRN.param t0 is now fixed by default
- Perform weighted fit if error argument is provided to Trajectory.fit, Equilibrium.fit
- Start time is no longer regarded as fit parameter by default. Set `CRN.params['t0'].vary = False`
- Improve sample map visualization
- Use hvplot for interactive assay visualization

Deprecated:
- CRN.integrate is deprecated in favor of CRN.trajectory().eval
- CRN.fit is deprecated in favor of CRN.trajectory().fit
- CRN.equilibrate is deprecated in favor of CRN.equilibrium().eval

Fixed:
- ParameterMap.specification for behaves better with absent species
- CRN.parametrize_for is now more careful when substituting specialized parameters in expressions
- More care is taken with edge cases when reading assay excel setup files
