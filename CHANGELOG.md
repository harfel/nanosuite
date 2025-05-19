0.3.1rc1
--------

Added:
- CRN.equilibrium returns a model that can be fit to experimental data

Changed:
- CRN.equilibrate now accepts 2D initial conditions with specialized CRN.params

Fixed:
- ParameterMap.specification for behaves better with absent species
- CRN.parametrize for is now more careful when substituting specialized parameters in expressions
