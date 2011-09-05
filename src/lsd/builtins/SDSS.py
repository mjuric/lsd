import os
from .misc import FileTable

class RunToMJD(object):
	def __init__(self, runlist=None):
		if runlist is None:
			runlist = os.getenv('SDSS_RUNLIST', None)

		if runlist is None:
			from .. import config
			runlist = os.path.join(config.data_dir, 'sdss', 'opRunlist.par')

		self.run2mjd = FileTable(runlist).map('run', 'mjd_ref', 'field_ref')

	def __call__(self, run, field=0, rowc=0):
		data = self.run2mjd(run)

		offs_obj = rowc * 0.396 / (360.985647 * 3600.)
		offs_fld = 36*(field - data['field_ref']) / (24.*3600.)
		halfexp = (53.907456 / 2.) / (24.*3600.)

		mjd = data['mjd_ref'] + offs_fld + offs_obj - halfexp

		return mjd

from ..utils import LazyCreate

mjd_obs = LazyCreate(RunToMJD)
