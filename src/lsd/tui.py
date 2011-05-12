# Text UI helpers. Just import this from every LSD command
# Import it as 'from lsd.tui import *'

__all__ = ['TUIException', 'tui_getopt']

##################################

import sys, exceptions, getopt, os, logging

logger = logging.getLogger('lsd.tui')

class TUIException(Exception):
	pass;

def suppress_keyboard_interrupt_message():
	old_excepthook = sys.excepthook

	def new_hook(type, value, traceback):
		if isinstance(value, TUIException):
			old_excepthook(type, value, None)
		elif type == exceptions.KeyboardInterrupt:
			pass
		else:
			old_excepthook(type, value, traceback)

	sys.excepthook = new_hook

def tui_getopt(short, long, argn, usage, argn_max=None, stdopts=True):
	""" A streamlined getopt adapted to LSD tool needs.
	    Also adds standard option existing on all LSD
	    commands (e.g., --db)

	    short -- short options
	    long  -- long options
	    argn  -- number of cmdline args
	    usage -- a callable to call to print usage info
	    argn_max -- max. number of cmdline args. If
	                None, will be equal to argn. If
	                -1, equals infinity
	"""
	if stdopts:
        	short += 'd:'
        	long.append('db=')

	try:
		optlist, args = getopt.getopt(sys.argv[1:], short, long)
	except getopt.GetoptError, err:
		print str(err)
		usage()
		exit(-1)

	if argn_max == None:
		argn_max = argn
	elif argn_max == -1:
		argn_max = 1e10
	if not (argn <= len(args) <= argn_max):
		print "Error: Incorrect number of command line arguments (%d, expected %d)." % (len(args), argn)
		usage()
		exit(-1)

        if stdopts:
        	stdopts = tui_getstdopts(optlist)

        	return optlist, stdopts, args
        else:
                return optlist, args

def tui_getstdopts(optlist):
	""" Parses standard options out of command
	    line arguments and environment variables.
	"""
	dbdir = os.getenv('LSD_DB', None)

	for o, a in optlist:
		if o in ('-d', '--db'):
			dbdir = a

	if dbdir is None:
		raise TUIException('No database specified. Please specify a '
			'database using the --db option or LSD_DB environment '
			'variable.')

	return (dbdir,)

_default_logging_config = """
version: 1

formatters:
  simple:
    format: '%(levelname)s: %(message)s'
  detailed:
    format: '%(asctime)s.%(msecs)03d %(processName)s[%(process)d] %(levelname)-8s {%(module)s:%(funcName)s:%(lineno)d}: %(message)s'
    datefmt: '%a, %d %b %Y %H:%M:%S'

handlers:
  console:
    class: logging.StreamHandler
    level: ERROR
    formatter: simple
    stream: ext://sys.stderr
  file:
    class: logging.FileHandler
    level: INFO
    formatter: detailed
    filename: /dev/null

loggers:
  lsd:
    level: DEBUG
    handlers: [console]
    propagate: 0
  surveys:
    level: DEBUG
    handlers: [console]
    propagate: 0

root:
  level: DEBUG
  handlers: [console]
"""

def startup():
	## Setup various useful text UI handlers ##################
	suppress_keyboard_interrupt_message()

	## Setup logging ##
	import logging.config, yaml

	lsdlogrc = "%s/.lsdlogrc" % os.environ["HOME"]
	defaultcfg = False
	if os.getenv("LOG_CONFIG") is not None:
	        # External YAML file with log configuration, explicitly set
		cfg = yaml.load(open(os.getenv("LOG_CONFIG")))
	elif os.path.exists(lsdlogrc):
	        # External YAML file with log configuration
		cfg = yaml.load(open(lsdlogrc))
	else:
	        # Default configuration, possibly partially overridden
		cfg = yaml.load(_default_logging_config)
		defaultcfg = True

        ## Increase the logging level everywhere if debugging is turned on
        if int(os.getenv("DEBUG", 0)):
                for h in cfg['handlers'].itervalues():
        		h['level'] = 'DEBUG'

        ## Redirect output of all loggers to a log file, if requested
        if "LOG" in os.environ:
		cfg['handlers']['file']['filename'] = os.environ["LOG"]
		cfg['root']['handlers'] += ['file']
		for lcfg in cfg['loggers'].itervalues():
			lcfg['handlers'] += ['file']
        elif defaultcfg:
                ## Remove the unused file handler
		del cfg['handlers']['file']

	## Set the name of the current process to the filename of the executable
	from multiprocessing import current_process
	current_process().name = sys.argv[0].split('/')[-1]

        logging.config.dictConfig(cfg)
        logging.getLogger().name = current_process().name

	logger.info("Started %s", ' '.join(sys.argv))
	logger.debug("Debug messages turned ON")

startup()
