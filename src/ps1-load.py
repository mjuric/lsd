#!/usr/bin/env python

import argparse
import lsd.utils as utils
import lsd
import lsd.pool2 as pool2
import numpy as np
import subprocess
import textwrap
import os, errno, sys
import shutil
import re
import time
from collections import defaultdict

class Updater(object):
	auto_logdir = False

	def __init__(self, survey, dbpath, catpath, environ_script, lsd_bin, logdir):
		self.dbpath = dbpath
		self.catpath = catpath
		self.survey = survey
		self.njobs = 80

		if len(environ_script):
			self.environ_source = "source %s" % (environ_script)
		else:
			self.environ_source = ''

		if lsd_bin is not None:
			self.lsd_prefix = os.path.normpath(lsd_bin) + '/'
		else:
			self.lsd_prefix = ''

		if logdir is not None:
			self.auto_logdir = False
			self.logdir = os.path.normpath(logdir)
		else:
			self.auto_logdir = True
			t = time.time()
			self.logdir =  "ps1_load_log.%s.%03d" % (time.strftime("%Y%m%d-%H%M%S", time.gmtime(t)), int(1e3*(t-int(t))))
		utils.mkdir_p(self.logdir)

		self.cat_filelist = os.path.join(self.logdir, 'cat-filelist.txt')
		self.db_filelist = os.path.join(self.logdir, 'db-filelist.txt')
		self.new_filelist = os.path.join(self.logdir, 'new-filelist.txt')
	
		self.ingest_sh = os.path.join(self.logdir, 'ingest.sh')
		self.build_objects_sh = os.path.join(self.logdir, 'build_objects.sh')

		self.import_bsub = os.path.join(self.logdir, 'import.bsub')
		self.jobid_fn = os.path.join(self.logdir, 'jobid')
		self.postprocess_sh = os.path.join(self.logdir, 'postprocess.sh')

	def __del__(self):
		if self.auto_logdir:
			try:
				os.rmdir(self.logdir)
			except:
				pass

	def list_smf(self):
		print "Getting the list of SMF files"
		try:
			utils.shell("find {catpath} -name '*.smf' > {cat_filelist} 2> {logdir}/cat-filelist.log".format(catpath=self.catpath, cat_filelist=self.cat_filelist, logdir=self.logdir))
			catsmf = [ s.strip() for s in open(self.cat_filelist).xreadlines() ]
			print "\tSMF files in catalog: %d" % len(catsmf)
		except subprocess.CalledProcessError:
			print >>sys.stderr, "Enumeration of catlog files failed. See cat-filelist.log for details."
			print >>sys.stderr, "Snippet:"
			for k, l in enumerate(file('cat-filelist.log').xreadlines()):
				print >>sys.stderr, "   ", l.strip()
				if k == 10: break
			exit()

	def list_db(self):
		print "Getting the list of already loaded SMF files"
		db = lsd.DB(self.dbpath)
		if db.table_exists('ps1_exp'):
			# Try loading from summary file that lsd-import-smf makes for us, if it exists
			try:
				uri = 'lsd:ps1_exp:metadata:all_exposures.txt'
				with db.table('ps1_exp').open_uri(uri) as f:
					smf_fn = [ s.strip() for s in f.xreadlines() ]
			except IOError:
				smf_fn = db.query("select smf_fn from ps1_exp").fetch(progress_callback=pool2.progress_pass).smf_fn
		else:
			assert not db.table_exists('ps1_det')
			smf_fn = []
		print "\tExposures in database: %d" % len(smf_fn)
		np.savetxt(self.db_filelist, smf_fn, fmt='%s')

	def new_smfs(self):
		print "Computing the list of new exposures..."
		dbsmf = [ s.strip() for s in open(self.db_filelist).xreadlines() ]
		catsmf = [ s.strip() for s in open(self.cat_filelist).xreadlines() ]

		def parse_paths(filelist):
			for path in filelist:
				# Matching and parsing the following endings:
				#    /o5280g0308o.151120.cm.62546.smf
				#    /o5255g0108.142838.cm.53295.smf
				#    o5280g0308o.151120.cm.62546
				#    o5255g0108.142838.cm.53295
				fn = path.split('/')[-1]
				(id, r1, _, r2) = fn.split('.')[:4]

				# Remove leading/trailing o
				if id[0]  == 'o': id = id[1:]
				if id[-1] == 'o': id = id[:-1]

				ver = (int(r1), int(r2))

				#print '[%s]' % fn, id, ver
				yield (path, id, ver)

		indb = dict((id, fn) for (fn, id, _) in parse_paths(dbsmf))
		assert len(indb) == len(dbsmf)

		# Keep only the files not in the database
		newcat = list((id, ver, fn) for (fn, id, ver) in parse_paths(catsmf) if id not in indb)
		print "\tNew SMF files (incl. duplicates): %d" % len(newcat)

		# Keep only the newest version of each file (if there are duplicates
		newcat = dict((id, (ver, fn)) for (id, ver, fn) in sorted(newcat))
		print "\tNew SMF files (w/o duplicates): %d" % len(newcat)

		with open(self.new_filelist, 'w') as fp:
			for _, fn in newcat.itervalues():
				fp.write(fn + '\n')

	def ingest(self):
		print "Loading new data"

		newsmf = [ s.strip() for s in open(self.new_filelist).xreadlines() ]
		if len(newsmf) != 0:
			pp = r"""
			#!/bin/bash

			{environ_source}

			SMFLIST={smffiles}
			SURVEY={survey}

			export LSD_DB={db}
			export PIXLEVEL=6

			{lsd_prefix}lsd-import-smf -c -f $SURVEY ps1_det ps1_exp $SMFLIST > {logdir}/ingest.log 2>&1
			""".format(db=self.dbpath, survey=self.survey, logdir=self.logdir, smffiles=self.new_filelist, environ_source=self.environ_source, lsd_prefix=self.lsd_prefix)

			pp = textwrap.dedent(pp).lstrip()
			with open(self.ingest_sh, 'w') as fp: fp.write(pp)

			try:
				out = utils.shell("bash %s" % self.ingest_sh)
			except subprocess.CalledProcessError:
				print >>sys.stderr, "Error while loading new data. See ingest.log for details."
		else:
			print "\tNo new files to import."


	def mk_bsub(self):
		print "Creating LSF job submission script"
		bsub = r"""
		#!/bin/bash

		#BSUB -u mjuric@cfa.harvard.edu
		#BSUB -J "lsdimport[1-{njobs}]"
		#BSUB -e {logdir}/worker_outs/task-%I.err
		#BSUB -o {logdir}/worker_outs/task-%I.out
		#BSUB -q itc
		#--BSUB -n 2
		#--BSUB -R "span[ptile=2]"

		{environ_source}

		SMFLIST={smffiles}
		SURVEY={survey}
		OUT={logdir}/worker_outs/$LSB_JOBINDEX.out

		mkdir -p {logdir}/worker_logs {logdir}/worker_outs

		export LSD_DB={db}
		export PIXLEVEL=6
		export LOG={logdir}/worker_logs/$LSB_JOBINDEX.log
		export NWORKERS=1

		{lsd_prefix}lsd-import-smf -c -m import -o $((LSB_JOBINDEX-1)) -s {njobs} -f $SURVEY ps1_det ps1_exp $SMFLIST > $OUT 2>&1
		""".format(db=self.dbpath, survey=self.survey, njobs=self.njobs, logdir=self.logdir, smffiles=self.new_filelist, environ_source=self.environ_source, lsd_prefix=self.lsd_prefix)
		bsub = textwrap.dedent(bsub).lstrip()

		open(self.import_bsub, 'w').write(bsub)

	def bsub(self):
		print "Submitting LSF job"
		newsmf = [ s.strip() for s in open(self.new_filelist).xreadlines() ]
		if len(newsmf) != 0:
			# Start a transaction. While lsd-import-smf would also start
			# a transaction, it would also join one (perhaps even a 
			# failed one), if it exists. So we start it here to ensure
			# no other transaction is running and we're starting with
			# a clean slate.
			db = lsd.DB(self.dbpath)
			db.begin_transaction()

			out = utils.shell("bsub < %s" % self.import_bsub).strip()
			#out = "Job <27448054> is submitted to queue <itc>"
			print "\t%s" % out

			# Try to parse out the job number
			m = re.match(r"Job <(\d+)> is submitted to queue <(\w+)>", out)
			(jobid, queue) = map(m.group, [1, 2])
			with open(self.jobid_fn, 'w') as fp:
				fp.write("%d %s\n" % (int(jobid), queue))
		else:
			print "\tNo new files, not starting import."
			try:
				os.unlink(self.jobid_fn)
			except:
				pass

	def wait(self):
		print "Waiting for completion"

		try:
			jobid, queue = file(self.jobid_fn).read().strip().split()
		except IOError as e:
			if e.errno != errno.ENOENT: raise
			print "\tNot running."
			return

		nretry = 30
		report = ""
		while True:
			## "27043806   jsteine RUN   itc        iliadaccess hero4207    *_a_aas4_2 Feb 23 18:12"
			try:
				out = utils.shell("bjobs -q {queue} {jobid} 2> /dev/null | grep {jobid}".format(queue=queue, jobid=jobid)).strip().split('\n')
			except subprocess.CalledProcessError:
				# It takes some time for a job to become visible
				nretry -= 1
				if nretry == 0: raise
				time.sleep(2)
				continue

			stats = defaultdict(int)
			for line in out:
				state = line.split()[2]
				stats[state] += 1
			newreport = ', '.join("%s: %d/%d" % (state, njobs, self.njobs) for state, njobs in stats.iteritems())
			if newreport == report:
				continue

			report = newreport
			print "\t%s" % report

			if stats['DONE'] == self.njobs:
				os.unlink(self.jobid_fn)
				break
			else:
				time.sleep(5)

	def postprocess(self):
		print "Post-processing"

		newsmf = [ s.strip() for s in open(self.new_filelist).xreadlines() ]
		if len(newsmf) != 0:
			pp = r"""
			#!/bin/bash

			{environ_source}

			SMFLIST={smffiles}
			SURVEY={survey}

			export LSD_DB={db}

			{lsd_prefix}lsd-import-smf -m postprocess $SURVEY ps1_det ps1_exp $SMFLIST > {logdir}/postprocess.log 2>&1
			""".format(db=self.dbpath, survey=self.survey, logdir=self.logdir, smffiles=self.new_filelist, environ_source=self.environ_source, lsd_prefix=self.lsd_prefix)
			pp = textwrap.dedent(pp).lstrip()

			with open(self.postprocess_sh, 'w') as fp: fp.write(pp)

			try:
				out = utils.shell("bash %s" % self.postprocess_sh)
			except subprocess.CalledProcessError:
				print >>sys.stderr, "Error while postprocessing. See postprocess.log for details."
		else:
			print "\tNo new files imported, no need to postprocess."

	def build_objects(self):
		print "Building object catalog"

		newsmf = [ s.strip() for s in open(self.new_filelist).xreadlines() ]
		if len(newsmf) != 0:
			pp = r"""
			#!/bin/bash

			{environ_source}

			export LSD_DB={db}

			{lsd_prefix}lsd-make-object-catalog --auto --fov-radius=2 ps1_obj ps1_det ps1_exp  > {logdir}/build_objects.log 2>&1
			""".format(db=self.dbpath, logdir=self.logdir, environ_source=self.environ_source, lsd_prefix=self.lsd_prefix)
			pp = textwrap.dedent(pp).lstrip()

			with open(self.build_objects_sh, 'w') as fp: fp.write(pp)

			try:
				out = utils.shell("bash %s" % self.build_objects_sh)
			except subprocess.CalledProcessError:
				print >>sys.stderr, "Error while postprocessing. See build-objects.log for details."
		else:
			print "\tNo new files imported, no need to postprocess."

all_stages1 = [ 'list_db', 'list_smf', 'new_smfs', 'ingest', 'mk_bsub', 'bsub', 'wait', 'postprocess', 'build_objects' ]	# All available stages
all_stages  = [ 'list_db', 'list_smf', 'new_smfs', 'ingest', 'build_objects' ]	# Default stages to be executed

def csv_arg(value):
	return [ s.strip() for s in value.split(',') ]

import argparse
parser = argparse.ArgumentParser(description='Update PS1 LSD database with SMF files',
	formatter_class=argparse.RawDescriptionHelpFormatter,
	epilog="""Examples:

	ps1-load.py mdf /n/panlfs/datastore/ps1-md-cat/gpc1/MD??.nt
	ps1-load.py 3pi /n/panlfs/datastore/ps1-3pi-cat/gpc1/ThreePi.nt
.
""")
parser.add_argument('survey', type=str, help='Three letter survey ID (usually "3pi" or "mdf")')
parser.add_argument('smf_path', type=str, help='Path to SMF files (may include wildcards)', nargs='+')
parser.add_argument('--db', default=os.getenv('LSD_DB', None), type=str, help='Path to LSD database')
parser.add_argument('--stages', default=all_stages, type=csv_arg,
	help='Execute only the specified import stages (comma separated list). Available stages are: '
	+ ', '.join(all_stages1))
parser.add_argument('--environ-script', default="~mjuric/projects/lsd/scripts/lsd-setup-odyssey.sh", type=str, help='Script that will set up the LSD environment')
parser.add_argument('--lsd-bin', default=None, type=str, help='Path to LSD binaries, if they\'re not in $PATH')
parser.add_argument('--logdir', default=None, type=str, help='Directory where intermediate scripts and logs \
will be generated and stored. If left unspecified, a directory with a timestamp-based name will be autocreated.')

args = parser.parse_args()

stages = args.stages
u = Updater(survey=args.survey, dbpath=args.db, catpath=' '.join(args.smf_path), environ_script=args.environ_script, lsd_bin=args.lsd_bin, logdir=args.logdir)

# Execute all requested stages
for k, s in enumerate(stages):
	fun = getattr(u, s)
	print "[%d/%d]  " % (k+1, len(stages)),
	fun()
