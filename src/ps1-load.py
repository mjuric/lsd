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

# Synopsis:
#  update-ps1.py <survey> <dbpath> <catpath>

class Updater(object):

	def __init__(self, survey, dbpath, catpath):
		self.dbpath = dbpath
		self.catpath = catpath
		self.survey = survey
		self.njobs = 80

	def list_smf(self):
		print "Getting the list of SMF files"
		try:
			utils.shell("find %s -name '*.smf' > cat-filelist.txt 2> cat-filelist.log" % (self.catpath))
			catsmf = [ s.strip() for s in open('cat-filelist.txt').xreadlines() ]
			print "\tSMF files in catalog: %d" % len(catsmf)
		except subprocess.CalledProcessError:
			print >>sys.stderr, "Enumeration of catlog files failed. See cat-filelist.log for details."
			print >>sys.stderr, "Snippet:"
			for k, l in enumerate(file('cat-filelist.log').xreadlines()):
				print >>sys.stderr, "   ", l.strip()
				if k == 10: break
			exit()

	def list_db(self):
		print "Getting the list of exposures present in the database"
		db = lsd.DB(self.dbpath)
		if db.table_exists('ps1_exp'):
			smf_fn = db.query("select smf_fn from ps1_exp").fetch(progress_callback=pool2.progress_pass).smf_fn
		else:
			assert not db.table_exists('ps1_det')
			smf_fn = []
		print "\tExposures in database: %d" % len(smf_fn)
		np.savetxt('db-filelist.txt', smf_fn, fmt='%s')

	def new_smfs(self):
		print "Computing the list of new exposures..."
		dbsmf = [ s.strip() for s in open('db-filelist.txt').xreadlines() ]
		catsmf = [ s.strip() for s in open('cat-filelist.txt').xreadlines() ]

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

		with open('new-filelist.txt', 'w') as fp:
			for _, fn in newcat.itervalues():
				fp.write(fn + '\n')

	def drop(self):
		print "Dropping existing catalog tables"
		try:
			shutil.rmtree(self.dbpath + '/ps1_det')
		except OSError as e:
			if e.errno != errno.ENOENT: raise

		try:
			shutil.rmtree(self.dbpath + '/ps1_exp')
		except OSError as e:
			if e.errno != errno.ENOENT: raise

		try:
			os.unlink(self.dbpath + '/.ps1_det:ps1_exp.join')
		except OSError as e:
			if e.errno != errno.ENOENT: raise


	def begin_transaction(self):
		print "Beginning transaction"

		transdir = self.dbpath + '/.transaction'
		if os.path.exists(transdir + '/open'):
			raise Exception("Open transaction already exists. Rollback or commit before starting another one.")

		utils.mkdir_p(transdir)

		# If the tables are not there, no transaction is needed
		if    os.path.exists(self.dbpath + '/ps1_det') \
		   or os.path.exists(self.dbpath + '/ps1_exp') \
		   or os.path.exists(self.dbpath + '/ps1_obj') \
		   or os.path.exists(self.dbpath + '/_ps1_obj_to_ps1_det') \
		   or os.path.exists(self.dbpath + '/.ps1_det:ps1_exp.join'):
			try:
				utils.shell("rsync -a --delete {db}/ps1_det {db}/ps1_exp {db}/.ps1_det:ps1_exp.join {db}/ps1_obj {db}/_ps1_obj_to_ps1_det {trans} 2> trans-begin.log".format(db=self.dbpath, trans=transdir))
			except subprocess.CalledProcessError:
				print >>sys.stderr, "Transaction start failed. See trans-begin.log for details."
				print >>sys.stderr, "Snippet:"
				for k, l in enumerate(file('trans-begin.log').xreadlines()):
					print >>sys.stderr, "   ", l.strip()
					if k == 10: break
				exit()

		utils.shell("touch {trans}/open".format(trans=transdir))

	def commit_transaction(self):
		print "Committing transaction",

		#... enumerate tables in .transaction, and swap them with those in main directory ...

		transdir = self.dbpath + '/.transaction'
		utils.shell("rm -f {trans}/open".format(trans=transdir))

	def mk_bsub(self):
		print "Creating LSF job submission script"
		bsub = r"""
		#!/bin/bash

		#BSUB -u mjuric@cfa.harvard.edu
		#BSUB -J "lsdimport[1-{njobs}]"
		#BSUB -e {logdir}/task-%I.err
		#BSUB -o {logdir}/task-%I.out
		#BSUB -q itc
		#--BSUB -n 2
		#--BSUB -R "span[ptile=2]"

		source ~mjuric/projects/lsd/scripts/lsd-setup-odyssey.sh dev

		SMFLIST={smffiles}
		SURVEY={survey}

		export LSD_DB={db}
		export PIXLEVEL=6
		export LOG={logdir}/import.$LSB_JOBINDEX.log
		export NWORKERS=1

		mkdir -p {logdir}

		lsd-import-smf -c -m import -o $((LSB_JOBINDEX-1)) -s {njobs} -f $SURVEY ps1_det ps1_exp $SMFLIST \
			> {logdir}/output.$LSB_JOBINDEX.log 2>&1
		""".format(db=self.dbpath, survey=self.survey, njobs=self.njobs, logdir='import_log', smffiles='new-filelist.txt')
		bsub = textwrap.dedent(bsub).lstrip()

		open('import.bsub', 'w').write(bsub)

	def submit(self):
		print "Submitting LSF job"
		newsmf = [ s.strip() for s in open('new-filelist.txt').xreadlines() ]
		if len(newsmf) != 0:
			out = utils.shell("bsub < import.bsub").strip()
			#out = "Job <27448054> is submitted to queue <itc>"
			print "\t%s" % out

			# Try to parse out the job number
			m = re.match(r"Job <(\d+)> is submitted to queue <(\w+)>", out)
			(jobid, queue) = map(m.group, [1, 2])
			with open('import_log/jobid', 'w') as fp:
				fp.write("%d %s\n" % (int(jobid), queue))
		else:
			print "\tNo new files, not starting import."
			try:
				os.unlink('import_log/jobid')
			except:
				pass

	def wait(self):
		print "Waiting for completion"

		try:
			jobid, queue = file('import_log/jobid').read().strip().split()
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
				os.unlink('import_log/jobid')
				break
			else:
				time.sleep(5)

	def postprocess(self):
		print "Post-processing"

		newsmf = [ s.strip() for s in open('new-filelist.txt').xreadlines() ]
		if len(newsmf) != 0:
			pp = r"""
			#!/bin/bash

			source ~mjuric/projects/lsd/scripts/lsd-setup-odyssey.sh dev

			SMFLIST={smffiles}
			SURVEY={survey}

			export LSD_DB={db}

			mkdir -p {logdir}

			rm -f $LSD_DB/ps1_det/tablet_tree.pkl
			rm -f $LSD_DB/ps1_exp/tablet_tree.pkl

			lsd-import-smf -m postprocess $SURVEY ps1_det ps1_exp $SMFLIST > {logdir}/postprocess.log 2>&1
			""".format(db=self.dbpath, survey=self.survey, logdir='import_log', smffiles='new-filelist.txt')
			pp = textwrap.dedent(pp).lstrip()

			with open('postprocess.sh', 'w') as fp: fp.write(pp)

			try:
				out = utils.shell("bash postprocess.sh")
			except subprocess.CalledProcessError:
				print >>sys.stderr, "Error while postprocessing. See postprocess.log for details."
		else:
			print "\tNo new files imported, no need to postprocess."

if True:
	all_stages = [ 'list_db', 'list_smf', 'new_smfs', 'mk_bsub', 'begin_transaction', 'submit', 'wait', 'postprocess', 'commit_transaction' ]

	def csv_arg(value):
		return [ s.strip() for s in value.split(',') ]

	import argparse
	parser = argparse.ArgumentParser(description='Update PS1 LSD database with SMF files')
	parser.add_argument('survey', type=str, help='Three letter survey ID (3pi or mdf)')
	parser.add_argument('smf_path', type=str, help='Path to SMF files (may include wildcards)')
	parser.add_argument('--db', default=os.getenv('LSD_DB', None), type=str, help='Path to LSD database')
	parser.add_argument('--stages', default=all_stages, type=csv_arg,
		help='Execute only the specified import stages (comma separated list). Available stages are: '
		+ ', '.join(all_stages))

	args = parser.parse_args()
	
	stages = args.stages
	u = Updater(survey=args.survey, dbpath=args.db, catpath=args.smf_path)
else:
	# For debugging/development
	#

	#u = Updater(survey='3pi', dbpath="/n/pan/mjuric/db_test", catpath='/n/panlfs/datastore/ps1-3pi-cat/gpc1/ThreePi.nt/2011/02')
	#u = Updater(survey='3pi', dbpath="/n/pan/mjuric/db", catpath='/n/panlfs/datastore/ps1-3pi-cat/gpc1/ThreePi.nt')
	#u = Updater(survey='mdf', dbpath="/n/pan/mjuric/db", catpath='/n/panlfs/datastore/ps1-md/gpc1/MD??.nt')
	u = Updater(survey='3pi', dbpath="/n/pan/mjuric/ps1_lap", catpath='/n/panlfs/mjuric/catalogs/smf')
	#stages = [ 'list_db', 'list_smf', 'new_smfs', 'mk_bsub', 'begin_transaction' ]
	#stages = [ 'submit', 'wait', 'postprocess', 'commit_transaction' ]
	stages = [ 'list_db', 'list_smf', 'new_smfs', 'mk_bsub', 'begin_transaction', 'submit', 'wait', 'postprocess', 'commit_transaction' ]
	#stages = [ 'list_db', 'new_smfs', 'mk_bsub', 'submit', 'wait', 'postprocess' ]
	#stages = [ 'list_smf', 'list_db', 'new_smfs', 'mk_bsub' ]
	#stages = ['mk_bsub', 'submit', 'wait', 'postprocess' ]
	#stages = ['wait']
	#stages = ['postprocess']

# Execute all requested stages
for k, s in enumerate(stages):
	fun = getattr(u, s)
	print "[%d/%d]  " % (k+1, len(stages)),
	fun()
