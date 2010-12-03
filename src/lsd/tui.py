# Text UI helpers. Just import this from every LSD command
# Import it as 'from lsd.tui import *'

__all__ = ['TUIException', 'tui_getopt']

##################################

import sys, exceptions, getopt, os

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

def tui_getopt(short, long, argn, usage, argn_max=None):
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

	stdopts = tui_getstdopts(optlist)

	return optlist, stdopts, args

def tui_getstdopts(optlist):
	""" Parses standard options out of command
	    line arguments and environment variables.
	"""
	dbdir = os.getenv('LSDDB', None)

	for o, a in optlist:
		if o in ('-d', '--db'):
			dbdir = a

	if dbdir is None:
		raise TUIException('No database specified. Please specify a '
			'database using the --db option or LSDDB environment '
			'variable.')

	return (dbdir,)

## Setup various useful text UI handlers ##################
suppress_keyboard_interrupt_message()
