nanosuite
=========

nanosuite is a growing suite of utilities to power computational workflows in the design and analysis of complex reaction systems, targeting dynamic DNA nanotechnology.


Highlights
----------
* plate reader assay visualization and analysis
* easy definition of reaction models
* simultaneous parameter fit to multiple samples


Installation
------------

```sh
$ pip install git+https://github.com/harfel/nanosuite.git
```

See available github branches for access to developer versions.


Getting started
---------------

The following code demonstrates how to define a reaction model and simulate it for a range of initial conditions:

```python
import nanosuite as ns
from matplotlib import pyplot as plt


# Define a chemical reaction network via a simple language
model = ns.crn.from_string("""
        Sensor + Target -> Intermediate           ; k1=10
    Intermediate + Fuel -> Signal + Sensor + Waste; k2=3
""")

# Set up initial states with varying concentrations of target
initial = model.state(Sensor=10,
                      Fuel=10,
                      Target=[0.001, 0.01, 0.1, 1])

# And simulate the system for 100 time units
traj = model.simulate(initial, 100)

# Plot signal concentration over time
plt.xlabel("Time")
lpt.ylabel("Concentration [mM]")
for conc, sample in zip(initial.sel(species='Target'), traj.sel(species='Signal')):
    plt.plot(sample.time, sample, label=f"{str(conc.data)} mM")
plt.legend()
plt.show()
```

And this is how to fit models to experimental data

```python
# Load assay setup and results
assay = ns.Assay(setup_file = './nanosuite/examples/edc_setup.xlsx',
                 rfu_file = './nanosuite/examples/edc_RFU.xlsx')

# Convert fluoresence to concentrations
experiment = assay.to_concentrations(pos_conc=10)

# Fit signal concentration to all experimental data
fit = model.fit(experiment, initial, conversion=lambda state: state.sel(species='Signal'))

print(fit.params)
```

More examples are found in the `nanosuite/examples` folder.


Getting involved
----------------
Use the issue tracker to submit feedback, bug reports and feature requests.
Please get in touch if you would like to contribute to the project.

License
-------
FIXME: license
