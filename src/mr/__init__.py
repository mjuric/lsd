#

import core
from core import Pool

core.mr_init()
core.logger.debug("fn=%s, cwd=%s, argv=%s, len(env)=%s" % (core.fn, core.cwd, core.argv, len(core.env)))
