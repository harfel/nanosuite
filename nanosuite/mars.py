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
		], dtype=float)

		# mapping of content to well indices (excl. deactivated wells)
		if not contents:
			contents = {}
			for idx, row in enumerate(ws[SAMPLE_FIRST_ROW: SAMPLE_LAST_ROW]):
				if row[0].value in self.deactivated: continue
				content = row[CONTENT_COL].value
				contents[content] = contents.get(content, []) + [idx]
		self.contents = contents

		self._mean = None
		self._std = None

	def __repr__(self):
		return f'<Assay "{self.path}">'

	@property
	def mean(self):
		if self._mean is None:
			self._mean = np.array([
				np.mean(self.wells[idx], axis=0)
				for idx in self.contents.values()
			])
		return self._mean

	@property
	def std(self):
		if self._std is None:
			self._std = np.array([
				np.std(self.wells[idx], axis=0)
				for idx in self.contents.values()
			])
			# replace 0 std values by smallest positive value
			self._std[self._std==0] = np.min(self._std, where=self._std!=0, initial=np.inf)
		return self._std
