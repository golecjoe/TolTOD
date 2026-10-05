import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm

def make_tod_plots(tod,plotdir=None):
	for count, i in enumerate(tod['apt_uid']):
		plt.figure()

		plt.plot(tod['signal'][count,:])

		plt.xlabel('Sample')
		plt.ylabel('Signal (mJy/beam)')

		plt.title(f'Detector UID {i}')
		if plotdir is not None:
			plotfile = plotdir / f"signal_uid_{i}.png"
			plt.savefig(plotfile,bbox_inches='tight')
		plt.close()
	plt.figure()

	plt.plot(tod['cm'][:])

	plt.xlabel('Sample')
	plt.ylabel('Signal (mJy/beam)')
	plt.title('Common Mode')
	if plotdir is not None:
		plotfile = plotdir / f"commonmode.png"
		plt.savefig(plotfile,bbox_inches='tight')
	plt.close()

def make_cmresid_plots(tod,plotdir=None):
	for count, i in enumerate(tod['apt_uid']):
		plt.figure()

		plt.plot(tod['signal'][count,:]-tod['cm'])

		plt.xlabel('Sample')
		plt.ylabel('Signal (mJy/beam)')

		plt.title(f'Detector UID {i}')
		if plotdir is not None:
			plotfile = plotdir / f"cmresid_uid_{i}.png"
			plt.savefig(plotfile,bbox_inches='tight')
		plt.close()


def make_spikefiltered_plots(datfilt,tod,plotdir=None,thresh=8):
	deviation = np.median(np.abs(datfilt), axis=1)
	for count, i in enumerate(tod['apt_uid']):
		plt.figure()

		plt.plot(datfilt[count,:])

		plt.axhline(thresh*deviation[count],linestyle='--',c='k')
		plt.axhline(-thresh*deviation[count],linestyle='--',c='k')

		plt.xlabel('Sample')
		plt.ylabel('Filter Response')

		plt.title(f'Detector UID {i}')
		if plotdir is not None:
			plotfile = plotdir / f"datfilt_uid_{i}.png"
			plt.savefig(plotfile,bbox_inches='tight')
		plt.close()

# def make_jumpfiltered_plots(datfilt,tod,plotdir=None,thresh=8):
# 	deviation = np.median(np.abs(datfilt), axis=1)
# 	dat_med = np.median(datfilt, axis=1, keepdims=True)

# 	det_thresh = thresh * np.median(
# 	    np.abs(datfilt - dat_med),
# 	    axis=1
# 	)
# 	for count, i in enumerate(tod['apt_uid']):
# 		plt.figure()

# 		plt.plot(datfilt[count,:])

# 		plt.axhline(dat_med[count] + det_thresh[count],linestyle='--',c='k')
# 		plt.axhline(dat_med[count] - det_thresh[count],linestyle='--',c='k')

# 		plt.xlabel('Sample')
# 		plt.ylabel('Filter Response')

# 		plt.title(f'Detector UID {i}')
# 		if plotdir is not None:
# 			plotfile = plotdir / f"datfilt_uid_{i}.png"
# 			plt.savefig(plotfile,bbox_inches='tight')
# 		plt.close()

def make_jumpfiltered_plots(
	datfilt,
	tod,
	plotdir=None,
	thresh=8,
	width=10,
	pad=2,
):
	ndet, n = datfilt.shape

	edge = pad * width

	# Only use the valid portion of the filtered timestream
	# when calculating median and threshold
	dat_valid = datfilt[:, edge:n-edge]

	dat_med = np.median(
		dat_valid,
		axis=1,
		keepdims=True
	)

	det_thresh = thresh * np.median(
		np.abs(dat_valid - dat_med),
		axis=1
	)

	for count, uid in enumerate(tod['apt_uid']):

		plt.figure()

		plt.plot(datfilt[count, :])

		med_i = dat_med[count, 0]

		plt.axhline(
			med_i + det_thresh[count],
			linestyle='--',
			c='k'
		)

		plt.axhline(
			med_i - det_thresh[count],
			linestyle='--',
			c='k'
		)

		plt.xlabel('Sample')
		plt.ylabel('Filter Response')

		plt.title(f'Detector UID {uid}')

		if plotdir is not None:
			plotfile = plotdir / f"datfilt_uid_{uid}.png"
			plt.savefig(plotfile, bbox_inches='tight')
			plt.close()
	
def plot_spikes(tod,spikelist,window=10,plotdir=None):
	for count, i in enumerate(tod['apt_uid']):
		if len(spikelist[count])==0:
			continue
		else:
			todspike=1
			if i==0.0:
				print(spikelist[count])
			for j in spikelist[count]:

				sampwindow = window*tod['samprate']
				firstindex = int(j-(0.5*sampwindow))
				lastindex = int(j+(0.5*sampwindow))

				if firstindex<0:
					samples = np.arange(0,lastindex)
				elif lastindex>len(tod['signal'][count,:]):
					samples = np.arange(firstindex,len(tod['signal'][count,:]))
				else:
					samples = np.arange(int(j-(0.5*sampwindow)),int(j+(0.5*sampwindow)))



				#samples = np.arange(int(j-(0.5*sampwindow)),int(j+(0.5*sampwindow)))

				time = np.arange(samples.size)/tod['samprate']

				time = time-(np.mean(time))

				plt.figure()

				if firstindex<0:
					plt.plot(time,tod['signal'][count,:lastindex])
				elif lastindex>len(tod['signal'][count,:]):
					plt.plot(time,tod['signal'][count,firstindex:])
				else:
					plt.plot(time,tod['signal'][count,firstindex:lastindex])

				# plt.plot(tmpsignal[count,int(j-(20.*window)):int(j+(20.*window))],c='r')
				
				plt.xlabel('Time relative to spike (sec)')
				plt.ylabel('Signal (mJy/beam)')

				plt.title(f'Detector UID {i} Spike {todspike}/samp {j}')
				if plotdir is not None:
					tmpdir = Path(plotdir /f'uid_{i}/')
					tmpdir.mkdir(parents=True, exist_ok=True)
					plotfile = tmpdir / f"uid_{i}_spike{todspike}.png"
					plt.savefig(plotfile,bbox_inches='tight')
				plt.close()
				todspike+=1

def make_psd_qc_plots(tod,Awn,nu0,alpha,plotdir,
					Awnthresh = 0.2, fkneethresh = 2.5, alphathresh = 2.5):
	plt.figure()

	plt.semilogy(tod['apt_uid'],Awn,'.')

	plt.axhline(np.median(Awn),c='k')
	plt.axhline((Awnthresh*np.std(Awn))+np.median(Awn),linestyle='--',c='k')
	plt.axhline((-Awnthresh*np.std(Awn))+np.median(Awn),linestyle='--',c='k')

	plt.xlabel('UID')
	plt.ylabel('WN Amplitude (mJy/beam)^2/Hz')
	plotfile = plotdir / f"Awn_summary.png"
	plt.savefig(plotfile,bbox_inches='tight')
	plt.close()

	plt.figure()

	plt.plot(tod['apt_uid'],nu0,'.')
	plt.axhline(np.median(nu0),c='k')
	plt.axhline((fkneethresh*np.std(nu0))+np.median(nu0),linestyle='--',c='k')
	plt.axhline((-fkneethresh*np.std(nu0))+np.median(nu0),linestyle='--',c='k')

	plt.xlabel('UID')
	plt.ylabel('f_knee (Hz)')
	plotfile = plotdir / f"fknee_summary.png"
	plt.savefig(plotfile,bbox_inches='tight')
	plt.close()

	plt.figure()

	plt.plot(tod['apt_uid'],alpha)
	plt.axhline(np.median(alpha),c='k')
	plt.axhline((alphathresh*np.std(alpha))+np.median(alpha),linestyle='--',c='k')
	plt.axhline((-alphathresh*np.std(alpha))+np.median(alpha),linestyle='--',c='k')

	plt.xlabel('UID')
	plt.ylabel('1/f index')
	plotfile = plotdir / f"alpha_summary.png"
	plt.savefig(plotfile,bbox_inches='tight')
	plt.close()

	plt.figure()

	plt.semilogy(nu0,Awn,'.')

	plt.axhline(np.median(Awn),c='k')
	plt.axhline((Awnthresh*np.std(Awn))+np.median(Awn),linestyle='--',c='k')
	plt.axhline((-Awnthresh*np.std(Awn))+np.median(Awn),linestyle='--',c='k')
	
	plt.axvline(np.median(nu0),c='k')
	plt.axvline((fkneethresh*np.std(nu0))+np.median(nu0),linestyle='--',c='k')
	plt.axvline((-fkneethresh*np.std(nu0))+np.median(nu0),linestyle='--',c='k')

	plt.xlabel('f_knee')
	plt.ylabel('A_wn')
	plotfile = plotdir / f"fkneevsWN_summary.png"
	plt.savefig(plotfile,bbox_inches='tight')
	plt.close()

	plt.figure()

	plt.plot(nu0,alpha,'.')

	plt.axvline(np.median(nu0),c='k')
	plt.axvline((fkneethresh*np.std(nu0))+np.median(nu0),linestyle='--',c='k')
	plt.axvline((-fkneethresh*np.std(nu0))+np.median(nu0),linestyle='--',c='k')

	plt.axhline(np.median(alpha),c='k')
	plt.axhline((alphathresh*np.std(alpha))+np.median(alpha),linestyle='--',c='k')
	plt.axhline((-alphathresh*np.std(alpha))+np.median(alpha),linestyle='--',c='k')

	plt.xlabel('f_knee')
	plt.ylabel('alpha')
	plotfile = plotdir / f"fkneevsalpha_summary.png"
	plt.savefig(plotfile,bbox_inches='tight')
	plt.close()

	plt.semilogx(Awn,alpha,'.')

	plt.axhline(np.median(alpha),c='k')
	plt.axhline((alphathresh*np.std(alpha))+np.median(alpha),linestyle='--',c='k')
	plt.axhline((-alphathresh*np.std(alpha))+np.median(alpha),linestyle='--',c='k')

	plt.axvline(np.median(Awn),c='k')
	plt.axvline((Awnthresh*np.std(Awn))+np.median(Awn),linestyle='--',c='k')
	plt.axvline((-Awnthresh*np.std(Awn))+np.median(Awn),linestyle='--',c='k')

	plt.xlabel('A_wn')
	plt.ylabel('alpha')
	plotfile = plotdir / f"Awnvsalpha_summary.png"
	plt.savefig(plotfile,bbox_inches='tight')
	plt.close()

def make_spikefiltered_plots_psd(datfilt,freq,tod,plotdir=None,thresh=8):
	deviation = np.median(np.abs(datfilt), axis=1)
	for count, i in enumerate(tod['apt_uid']):
		plt.figure()

		plt.plot(freq[1:],datfilt[count,1:])

		plt.axhline(thresh*deviation[count],linestyle='--',c='k')
		plt.axhline(-thresh*deviation[count],linestyle='--',c='k')

		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Filter Response')

		plt.ylim(-1.5*thresh*deviation[count],1.5*thresh*deviation[count])

		plt.title(f'Detector UID {i}')
		if plotdir is not None:
			tmpdir = Path(plotdir /f'psdspikefilts/')
			tmpdir.mkdir(parents=True, exist_ok=True)
			plotfile = tmpdir / f"psdspikefilt_uid_{i}.png"
			plt.savefig(plotfile,bbox_inches='tight')
		plt.close()


def make_cmfit_plots(tod,gainfact,offsetfact,rmslist,plotdir,
						gainthresh = 3., offsetthresh = 3., rmsthresh=3.):

	med_gainfact = np.median(gainfact)
	mad_gainfact = np.median(np.abs(gainfact - med_gainfact))
	plt.figure()
	plt.plot(tod['apt_uid'],gainfact)
	plt.axhline(np.median(gainfact),c='k')
	plt.axhline((gainthresh*mad_gainfact)+np.median(gainfact),linestyle='--',c='k')
	plt.axhline((-gainthresh*mad_gainfact)+np.median(gainfact),linestyle='--',c='k')
	plt.xlabel('UID')
	plt.ylabel('Gain Factor to CM')
	plotfile = plotdir / f"gainfactor.png"
	plt.savefig(plotfile,bbox_inches='tight')
	plt.close()

	med_offfact = np.median(offsetfact)
	mad_offfact = np.median(np.abs(offsetfact - med_offfact))

	plt.figure()
	plt.plot(tod['apt_uid'],offsetfact)
	plt.axhline(np.median(offsetfact),c='k')
	plt.axhline((offsetthresh*mad_offfact)+np.median(offsetfact),linestyle='--',c='k')
	plt.axhline((-offsetthresh*mad_offfact)+np.median(offsetfact),linestyle='--',c='k')
	plt.xlabel('UID')
	plt.ylabel('Offset Factor to CM')
	plotfile = plotdir / f"offsetfactor.png"
	plt.savefig(plotfile,bbox_inches='tight')
	plt.close()

	med_rmsfact = np.median(rmslist)
	mad_rmsfact = np.median(np.abs(rmslist - med_rmsfact))

	plt.figure()
	plt.plot(tod['apt_uid'],rmslist)
	plt.axhline(np.median(rmslist),c='k')
	plt.axhline((rmsthresh*mad_rmsfact)+np.median(rmslist),linestyle='--',c='k')
	plt.axhline((-rmsthresh*mad_rmsfact)+np.median(rmslist),linestyle='--',c='k')
	plt.xlabel('UID')
	plt.ylabel('Min Chi Square ')
	plotfile = plotdir / f"minrms.png"
	plt.savefig(plotfile,bbox_inches='tight')
	plt.close()

	plt.figure()

	sc = plt.scatter(offsetfact,gainfact,c=rmslist,s=30,cmap='viridis')

	plt.axhline(1.0,linestyle='--',c='k')
	plt.axvline(0.0,linestyle='--',c='k')

	plt.xlabel('Offset Factor to CM')
	plt.ylabel('Gain Factor to CM')

	cbar = plt.colorbar(sc)
	cbar.set_label('Min Chi Squared',rotation=-90)
	plotfile = plotdir / f"cm_offset_summary.png"
	plt.savefig(plotfile,bbox_inches='tight')
	plt.close()


def make_final_tod_plots(tod, cutdict,plot_all_TODs=False, plotdir=None):

	# Cuts you want listed in the QC box
	cut_keys = list(cutdict.keys())

	if plot_all_TODs:

		for count, i in enumerate(tod['apt_uid']):

			fig, ax = plt.subplots(figsize=(10, 5))

			if cutdict['master_cuts'][count]:
				ax.plot(tod['signal'][count, :], c='k')
			else:
				ax.plot(tod['signal'][count, :], c='r')

			ax.set_xlabel('Sample')
			ax.set_ylabel('Signal (mJy/beam)')
			ax.set_title(f'Detector UID {i}')

			# Leave room on the right for the cut summary
			fig.subplots_adjust(right=0.78)

			# Header
			ax.text(
				1.03, 0.95,
				'Detector Cuts',
				transform=ax.transAxes,
				fontsize=11,
				fontweight='bold',
				va='top'
			)

			# List each cut
			y = 0.88

			for key in cut_keys:

				passed = bool(cutdict[key][count])

				color = 'green' if passed else 'red'

				ax.text(
					1.03, y,
					key,
					transform=ax.transAxes,
					fontsize=10,
					color=color,
					va='top'
				)

				y -= 0.06

			if plotdir is not None:
				plotfile = plotdir / f"signal_uid_{i}.png"
				plt.savefig(plotfile, bbox_inches='tight')
			plt.close()
	# if good_signal.size == 0 or good_signal.shape[0] == 0:
	# 	plt.figure()

	# 	plt.text(
	#         0.5, 0.5,
	#         "NO GOOD DETECTORS",
	#         color="red",
	#         fontsize=24,
	#         ha="center",
	#         va="center",
	#         transform=plt.gca().transAxes
	#     )
	# 	plt.xlabel('Sample')
	# 	plt.ylabel('Signal (mJy/beam)')
	# 	if plotdir is not None:
	# 		plotfile = plotdir / f"goodTODSummary.png"
	# 		plt.savefig(plotfile, bbox_inches='tight')
	# 	plt.close()
	# else:
	# 	plt.figure()

	# 	for count, i in enumerate(tod['apt_uid']):
	# 		if cutdict['master_cuts'][count]:
	# 			plt.plot(tod['signal'][count, :],c='grey',alpha=0.2)
	# 	plt.plot(np.median(tod['signal'][cutdict['master_cuts'], :],axis=0),c='k')
	# 	plt.xlabel('Sample')
	# 	plt.ylabel('Signal (mJy/beam)')
	# 	if plotdir is not None:
	# 		plotfile = plotdir / f"goodTODSummary.png"
	# 		plt.savefig(plotfile, bbox_inches='tight')
	# 	plt.close()


	# plt.figure()

	# for count, i in enumerate(tod['apt_uid']):
	# 	if not cutdict['master_cuts'][count]:
	# 		plt.plot(tod['signal'][count, :],c='grey',alpha=0.2)
	# plt.plot(np.median(tod['signal'][~cutdict['master_cuts'], :],axis=0),c='k')

	# plt.xlabel('Sample')
	# plt.ylabel('Signal (mJy/beam)')
	# if plotdir is not None:
	# 	plotfile = plotdir / f"badTODSummary.png"
	# 	plt.savefig(plotfile, bbox_inches='tight')
	# plt.close()


	good_signal = tod['signal'][cutdict['master_cuts'], :]

	if good_signal.size == 0 or good_signal.shape[0] == 0:
		plt.figure(figsize=(10, 5))

		plt.text(
			0.5, 0.5,
			"NO GOOD DETECTORS",
			color="red",
			fontsize=24,
			ha="center",
			va="center",
			transform=plt.gca().transAxes
		)

		plt.xlabel("Sample")
		plt.ylabel("Signal (mJy/beam)")

		if plotdir is not None:
			plotfile = plotdir / "goodTODSummary_density.png"
			plt.savefig(plotfile, bbox_inches="tight")

		plt.close()

		plt.figure()

		plt.text(
			0.5, 0.5,
			"NO GOOD DETECTORS",
			color="red",
			fontsize=24,
			ha="center",
			va="center",
			transform=plt.gca().transAxes
		)
		plt.xlabel('Sample')
		plt.ylabel('Signal (mJy/beam)')
		if plotdir is not None:
			plotfile = plotdir / f"goodTODSummary.png"
			plt.savefig(plotfile, bbox_inches='tight')
		plt.close()

	else:

		plt.figure()

		for count, i in enumerate(tod['apt_uid']):
			if cutdict['master_cuts'][count]:
				plt.plot(tod['signal'][count, :],c='grey',alpha=0.2)
		plt.plot(np.median(tod['signal'][cutdict['master_cuts'], :],axis=0),c='k')
		plt.xlabel('Sample')
		plt.ylabel('Signal (mJy/beam)')
		if plotdir is not None:
			plotfile = plotdir / f"goodTODSummary.png"
			plt.savefig(plotfile, bbox_inches='tight')
		plt.close()

		nsamp = good_signal.shape[1]

		# Make an x coordinate for every detector/sample value
		x = np.tile(np.arange(nsamp), good_signal.shape[0])
		y = good_signal.ravel()

		plt.figure(figsize=(10, 5))

		# Density of timestreams
		# plt.hist2d(
		#     x,
		#     y,
		#     bins=[500, 200],
		#     cmap='Greys',
		#     cmin=1
		# )

		plt.hist2d(
			x,
			y,
			bins=[500, 200],
			cmap='Greys',
			norm=LogNorm()
		)

		# Median across detectors at each sample
		median_tod = np.median(good_signal, axis=0)

		plt.plot(
			np.arange(nsamp),
			median_tod,
			c='r',
			lw=1.5,
			label='Median'
		)

		plt.xlabel('Sample')
		plt.ylabel('Signal (mJy/beam)')
		plt.colorbar(label='Number of detectors')
		plt.legend()

		if plotdir is not None:
			plotfile = plotdir / "goodTODSummary_density.png"
			plt.savefig(plotfile, bbox_inches='tight')

		plt.close()

	bad_signal = tod['signal'][~cutdict['master_cuts'], :]

	if bad_signal.size == 0 or bad_signal.shape[0] == 0:

		plt.figure()

		plt.text(
			0.5, 0.5,
			"NO BAD DETECTORS!?!",
			color="red",
			fontsize=24,
			ha="center",
			va="center",
			transform=plt.gca().transAxes
		)
		
		plt.xlabel('Sample')
		plt.ylabel('Signal (mJy/beam)')
		if plotdir is not None:
			plotfile = plotdir / f"badTODSummary.png"
			plt.savefig(plotfile, bbox_inches='tight')
		plt.close()


		plt.figure(figsize=(10, 5))

		plt.text(
			0.5, 0.5,
			"NO BAD DETECTORS!?!",
			color="red",
			fontsize=24,
			ha="center",
			va="center",
			transform=plt.gca().transAxes
		)

		plt.xlabel("Sample")
		plt.ylabel("Signal (mJy/beam)")

		if plotdir is not None:
			plotfile = plotdir / "badTODSummary_density.png"
			plt.savefig(plotfile, bbox_inches="tight")

		plt.close()

	else:

		plt.figure()

		for count, i in enumerate(tod['apt_uid']):
			if not cutdict['master_cuts'][count]:
				plt.plot(tod['signal'][count, :],c='grey',alpha=0.2)
		plt.plot(np.median(tod['signal'][~cutdict['master_cuts'], :],axis=0),c='k')

		plt.xlabel('Sample')
		plt.ylabel('Signal (mJy/beam)')
		if plotdir is not None:
			plotfile = plotdir / f"badTODSummary.png"
			plt.savefig(plotfile, bbox_inches='tight')
		plt.close()

		nsamp = bad_signal.shape[1]

		# Make an x coordinate for every detector/sample value
		x = np.tile(np.arange(nsamp), bad_signal.shape[0])
		y = bad_signal.ravel()

		plt.figure(figsize=(10, 5))

		# Density of timestreams
		# plt.hist2d(
		#     x,
		#     y,
		#     bins=[500, 200],
		#     cmap='Greys',
		#     cmin=1
		# )

		plt.hist2d(
			x,
			y,
			bins=[500, 200],
			cmap='Greys',
			norm=LogNorm()
		)

		# Median across detectors at each sample
		median_tod = np.median(bad_signal, axis=0)

		plt.plot(
			np.arange(nsamp),
			median_tod,
			c='r',
			lw=1.5,
			label='Median'
		)

		plt.xlabel('Sample')
		plt.ylabel('Signal (mJy/beam)')
		plt.colorbar(label='Number of detectors')
		plt.legend()

		if plotdir is not None:
			plotfile = plotdir / "badTODSummary_density.png"
			plt.savefig(plotfile, bbox_inches='tight')

		plt.close()
	
	   


def make_array_cut_summary(qcdir):
	"""
	Plot good and bad detector positions for each TolTEC array.

	Only networks with a ``cut_summary.csv`` file are included. Missing
	networks are skipped so the figure can be made from partial analyses.

	Parameters
	----------
	qcdir : str or pathlib.Path
		Observation QC directory containing ``nw_*`` subdirectories.

	Returns
	-------
	pathlib.Path
		Path to the generated summary PNG.
	"""
	import csv
	from matplotlib.markers import MarkerStyle

	qcdir = Path(qcdir)
	array_networks = {
		"a1100": (0, 1, 2, 3, 4, 5, 6),
		"a1400": (7, 8, 9, 10),
		"a2000": (11, 12),
	}

	fig, axes = plt.subplots(
		1,
		3,
		figsize=(18, 6),
		constrained_layout=True,
	)

	for ax, (array_name, network_ids) in zip(axes, array_networks.items()):
		x_positions = []
		y_positions = []
		flags = []
		included_networks = []

		for network_id in network_ids:
			csv_path = qcdir / f"nw_{network_id}" / "cut_summary.csv"

			if not csv_path.exists():
				continue

			included_networks.append(network_id)

			with csv_path.open(newline="") as csv_file:
				reader = csv.DictReader(csv_file)
				required_columns = {
					"apt_x_t",
					"apt_y_t",
					"new_apt_flags",
				}
				missing_columns = required_columns.difference(
					reader.fieldnames or ()
				)

				if missing_columns:
					raise ValueError(
						f"{csv_path} is missing columns: "
						+ ", ".join(sorted(missing_columns))
					)

				for row in reader:
					x_positions.append(float(row["apt_x_t"]))
					y_positions.append(float(row["apt_y_t"]))
					flags.append(int(float(row["new_apt_flags"])))

		if not x_positions:
			ax.text(
				0.5,
				0.5,
				"No analyzed networks",
				transform=ax.transAxes,
				ha="center",
				va="center",
				fontsize=14,
			)
			ax.set_title(array_name)
			ax.set_xlabel("Detector X Position")
			ax.set_ylabel("Detector Y Position")
			ax.set_aspect("equal", adjustable="box")
			continue

		x_positions = np.asarray(x_positions)
		y_positions = np.asarray(y_positions)
		flags = np.asarray(flags)
		good_detectors = flags == 0
		bad_detectors = ~good_detectors
		detector_colors = np.where(good_detectors, "green", "red")

		# Group detectors by physical location. Rounding avoids treating tiny
		# floating-point differences as separate positions.
		location_groups = {}
		for detector_index, (x_position, y_position) in enumerate(
			zip(x_positions, y_positions)
		):
			location_key = (
				round(float(x_position), 6),
				round(float(y_position), 6),
			)
			location_groups.setdefault(location_key, []).append(detector_index)

		single_indices = []
		left_half_indices = []
		right_half_indices = []

		for detector_indices in location_groups.values():
			if len(detector_indices) == 1:
				single_indices.extend(detector_indices)
			else:
				left_half_indices.append(detector_indices[0])
				right_half_indices.append(detector_indices[1])
				single_indices.extend(detector_indices[2:])

		marker_size = 24
		marker_options = {
			"s": marker_size,
			"edgecolors": "black",
			"linewidths": 0.25,
		}

		if single_indices:
			ax.scatter(
				x_positions[single_indices],
				y_positions[single_indices],
				c=detector_colors[single_indices],
				marker="o",
				**marker_options,
			)

		if left_half_indices:
			ax.scatter(
				x_positions[left_half_indices],
				y_positions[left_half_indices],
				c=detector_colors[left_half_indices],
				marker=MarkerStyle("o", fillstyle="left"),
				**marker_options,
			)
			ax.scatter(
				x_positions[right_half_indices],
				y_positions[right_half_indices],
				c=detector_colors[right_half_indices],
				marker=MarkerStyle("o", fillstyle="right"),
				**marker_options,
			)

		# Empty full-circle markers provide a simple good/bad color legend.
		ax.scatter(
			[],
			[],
			c="green",
			s=marker_size,
			label=f"Good ({np.sum(good_detectors)})",
		)
		ax.scatter(
			[],
			[],
			c="red",
			s=marker_size,
			label=f"Bad ({np.sum(bad_detectors)})",
		)

		network_text = ", ".join(str(nw) for nw in included_networks)
		ax.set_title(f"{array_name}\nNetworks: {network_text}")
		ax.set_xlabel("Detector X Position")
		ax.set_ylabel("Detector Y Position")
		ax.set_aspect("equal", adjustable="box")
		ax.legend()

	fig.suptitle(
		f"{qcdir.parent.name} Detector Cut Summary",
		fontsize=18,
		fontweight="bold",
	)

	output_path = qcdir / "array_cut_summary.png"
	fig.savefig(output_path, bbox_inches="tight", facecolor="white")
	plt.close(fig)

	return output_path


def make_qc_summary_pdfs(qcdir):
	"""
	Create per-network and all-network QC summary PDFs.

	Each page contains the bad and good TOD density plots, the PSD summary,
	and the master-cut summary for one network.

	Parameters
	----------
	qcdir : str or pathlib.Path
		Observation QC directory containing ``nw_*`` subdirectories.

	Returns
	-------
	network_pdf_paths : list[pathlib.Path]
		Paths to the per-network summary PDFs.
	combined_pdf_path : pathlib.Path
		Path to the multi-page observation summary PDF.
	"""
	from matplotlib.backends.backend_pdf import PdfPages

	qcdir = Path(qcdir)

	image_specs = (
		(
			"Bad TODs",
			Path("final_tods") / "badTODSummary_density.png",
		),
		(
			"Good TODs",
			Path("final_tods") / "goodTODSummary_density.png",
		),
		(
			"Power Spectral Densities",
			Path("tod_psds") / "psd_summary.png",
		),
		(
			"Master Cuts",
			Path("cuts_figs") / "mastercutssummary.png",
		),
	)

	# Only include directories named nw_<integer>.
	network_dirs = sorted(
		(
			path
			for path in qcdir.glob("nw_*")
			if path.is_dir() and path.name[3:].isdigit()
		),
		key=lambda path: int(path.name[3:]),
	)

	network_pages = []

	for network_dir in network_dirs:
		image_paths = [
			(title, network_dir / relative_path)
			for title, relative_path in image_specs
		]

		existing_paths = [
			image_path
			for _, image_path in image_paths
			if image_path.exists()
		]

		# Networks with no detectors do not produce QC images or a PDF page.
		if not existing_paths:
			continue

		# A partial set indicates that network processing did not finish.
		missing_paths = [
			image_path
			for _, image_path in image_paths
			if not image_path.exists()
		]

		if missing_paths:
			missing_text = "\n".join(str(path) for path in missing_paths)
			raise FileNotFoundError(
				f"Cannot build {network_dir.name} summary. "
				f"Missing QC images:\n{missing_text}"
			)

		network_pages.append((network_dir, image_paths))

	if not network_pages:
		raise FileNotFoundError(
			f"No complete network QC image sets were found in {qcdir}"
		)

	combined_pdf_path = qcdir / f"{qcdir.parent.name}_summary.pdf"
	network_pdf_paths = []

	with PdfPages(combined_pdf_path) as combined_pdf:
		for network_dir, image_paths in network_pages:
			fig, axes = plt.subplots(
				2,
				2,
				figsize=(17, 11),
				constrained_layout=True,
			)

			for ax, (title, image_path) in zip(axes.flat, image_paths):
				ax.imshow(plt.imread(image_path))
				ax.set_title(title, fontsize=14)
				ax.axis("off")

			fig.suptitle(
				f"{qcdir.parent.name} — {network_dir.name}",
				fontsize=18,
				fontweight="bold",
			)

			network_pdf_path = (
				network_dir / f"{network_dir.name}_summary.pdf"
			)

			fig.savefig(
				network_pdf_path,
				format="pdf",
				bbox_inches="tight",
				facecolor="white",
			)

			combined_pdf.savefig(
				fig,
				bbox_inches="tight",
				facecolor="white",
			)

			plt.close(fig)
			network_pdf_paths.append(network_pdf_path)

	return network_pdf_paths, combined_pdf_path




