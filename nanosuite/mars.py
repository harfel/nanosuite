from openpyxl import load_workbook
import numpy as np


class Assay:
	def __init__(self, path, resolution=1, contents=None):
		"""Clariostar plate reader data as saved by Mars.

		If contents is given, it must be a mapping from strings to well indices.
		"""
		DEACTIVATE_INFO_CELL = 11, 1
		TIME_ROW = 14
		CONTENT_COL = 1
		SAMPLE_FIRST_ROW = 15
		SAMPLE_FIRST_COL = 3

		self.path = path

		wb = load_workbook(self.path)
		ws = wb["Table All Cycles"]

		SAMPLE_LAST_ROW = ws.max_row

		# read deactivated wells from header info
		self.deactivated = [
			well.strip()
			for well in ws.cell(*DEACTIVATE_INFO_CELL).value.split(':')[-1].split(';')
		]

		# time and raw read information (incl. deactivated wells)
		self.times = np.array([ cell.value for cell in np.array(ws[TIME_ROW][2::resolution])])
		self.wells = np.array([
			[cell.value for cell in ws[y][2::resolution]]
			for y in range(SAMPLE_FIRST_ROW, SAMPLE_LAST_ROW+1)
		])

		# mapping of content to well indices (excl. deactivated wells)
		if not contents:
			contents = {}
			for idx, row in enumerate(ws[SAMPLE_FIRST_ROW: SAMPLE_LAST_ROW]):
				if row[0].value in self.deactivated: continue
				content = row[CONTENT_COL].value
				contents[content] = contents.get(content, []) + [idx]
		self.contents = contents

	def __repr__(self):
		return f'<Assay "{self.path}">'


if __name__ == '__main__':
	from scipy.linalg import block_diag
	from scipy.integrate import solve_ivp
	from matplotlib import pyplot as plt
	from lmfit import create_params, Parameter, minimize, fit_report

	assay = Assay("./Clariostar - Raw Data Output (4 repeats with excluded wells).xlsx", resolution=1)
	
	initial = np.array([
	    [0.,    2, 0, 3, 0], # Sample X1
	    [0.001, 2, 0, 3, 0], # Sample X2
	    [0.002, 2, 0, 3, 0], # Sample X3
	    [0.003, 2, 0, 3, 0], # Sample X4
	    [0.004, 2, 0, 3, 0], # Sample X5
	    [0.005, 2, 0, 3, 0], # Sample X6
	    [0.01,  2, 0, 3, 0], # Sample X7
	    [0.02,  2, 0, 3, 0], # Sample X8
	    [0.03,  2, 0, 3, 0], # Sample X9
	    [0.04,  2, 0, 3, 0], # Sample X10
	    [0.05,  2, 0, 3, 0], # Sample X11
	    [0.1,   2, 0, 3, 0], # Sample X12
	])
	
	mean = np.array([
	    np.mean(assay.wells[idx], axis=0)
	    for idx in assay.contents.values()
	])
	
	std = np.array([
	    np.std(assay.wells[idx], axis=0)
	    for idx in assay.contents.values()
	])
	
	def mass_action(Z, A):
	    # calculate graph Laplacian
	    D = np.diag(np.sum(A, axis=0))
	    L = D - A
	
	    def kinetics(t, x):
	        # Z.T @ log(x) with convention 0*inf = 0
	        lg = np.log(x, out=-np.inf*np.ones_like(x), where=(x!=0))
	        lg = np.nansum(Z*lg, axis=0)
	        lg = -Z @ L @ np.exp(lg)
	        return lg
	
	    return kinetics
	
	params = create_params(
	    t0 = {'value': 0, 'max': 0, 'vary': False},
	    k0 = {'value': 0.0, 'min': 0, 'vary': False},
	    k1 = {'value': 0.01, 'min': 0},
	    k2 = {'value': 1, 'min': 0},
	)

	def edc(params, x0, t_eval=None):
	    """
	    input + probe       -> intermediate,   k1
	    intermediate + fuel -> input + signal, k2
	    probe + fuel        -> input + signal, k0
	    """
	    t0 = params['t0'].value
	    k0 = params['k0'].value
	    k1 = params['k1'].value
	    k2 = params['k2'].value
	
	    # complex graph
	    Z = np.array([
	        [ 1, 0, 0, 0, 1, ], # input
	        [ 1, 0, 1, 0, 0, ], # probe
	        [ 0, 1, 0, 1, 0, ], # intermediate
	        [ 0, 1, 1, 0, 0, ], # fuel
	        [ 0, 0, 0, 0, 1, ], # signal
	    ])
	
	    # reaction rate constant matrix  (complex adjacency matrix)
	    A = np.array([
	        [ 0,  0,  0,  0,  0],  # input + probe
	        [ 0,  0,  0,  0,  0],  # intermedia + fuel
	        [ 0,  0,  0,  0,  0],  # probe + fuel
	        [ k1, 0,  0,  0,  0],  # intermediate
	        [ 0,  k2, k0, 0,  0],  # input + signal
	    ])

	    # copy reaction system to fit number of initial conditions
	    r = x0.shape[0] if len(x0.shape)==2 else 1
	    Z = block_diag(*r*[Z])
	    A = block_diag(*r*[A])
	    
	    ode = mass_action(Z, A)
	    return solve_ivp(ode, (t0, 1440), x0.flatten(),
	                     t_eval=t_eval, vectorized=True).y  

	def toRFU(conc, signal_idx=4, probe_idx=1, signal_rfu=18_000, probe_rfu=400):
	    return signal_rfu*conc[signal_idx::5] + probe_rfu*conc[probe_idx::5]
	
	def residuals(params, data, error=1):
	    model = toRFU(edc(params, initial, t_eval=assay.times))
	    return (model-data)**2/error**2
	
	fit = minimize(residuals, 
	               params,
	               args=(mean, std),
	               )#method='nelder')

	plt.ion()

	fig, ax = plt.subplots()
	
	plt.title("Mean and standard deviations")
	plt.xlabel("time [min]")
	plt.ylabel("RFU")
	
	labels = [f'{1000*init:.0f}pM' for init in initial[:,0]]
	
	from matplotlib import cm
	from matplotlib import colors
	cols = [cm.get_cmap('gist_ncar')(i/len(labels)) for i,_ in enumerate(labels)]
	cols = [colors.to_hex(color) for color in cols]
	
	for idx, label in enumerate(labels):
	    #plt.fill_between(assay.times, mean[idx]-std[idx], mean[idx]+std[idx],
	    #                 color=cols[idx], alpha=0.2)
	    plt.plot(assay.times, toRFU(edc(fit.params, initial[idx], t_eval=assay.times)),
	             color=cols[idx], label=label)
	
	plt.legend()
	plt.grid()
	plt.show()
	
	plt.pause(-1)
