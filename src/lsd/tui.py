# Text UI helpers. Just import this from every LSD command
# Import it as 'from lsd.tui import *'

__all__ = ['TUIException', 'tui_getopt']

##################################

import sys, exceptions, getopt, os, logging

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

def startup():
	## Setup various useful text UI handlers ##################
	suppress_keyboard_interrupt_message()

	## Setup logging ##
	name = sys.argv[0].split('/')[-1]
	format = '%(asctime)s.%(msecs)03d %(name)s[%(process)d] %(levelname)-8s {%(module)s:%(funcName)s}: %(message)s'
	datefmt = '%a, %d %b %Y %H:%M:%S'
	level = logging.DEBUG if (os.getenv("DEBUG", 0) or os.getenv("LOGLEVEL", "info") == "debug") else logging.INFO
	filename = ('%s.log' % name) if os.getenv("LOG", None) is None else os.getenv("LOG")
	logging.basicConfig(filename=filename, format=format, datefmt=datefmt, level=level)

	logger = logging.getLogger()
	logger.name = name
	logging.info("Started %s", ' '.join(sys.argv))
	logging.debug("Debug messages turned ON")

	# Log WARNING and above to console
	ch = logging.StreamHandler()
	ch.setLevel(logging.WARNING)
	ch.setFormatter(logging.Formatter('%(levelname)s: %(name)s[%(process)d]: %(message)s'))
	logger.addHandler(ch)

startup()
