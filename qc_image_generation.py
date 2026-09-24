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


def make_final_tod_plots(tod, cutdict, plotdir=None):

	# Cuts you want listed in the QC box
	cut_keys = list(cutdict.keys())

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


	good_signal = tod['signal'][cutdict['master_cuts'], :]

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
	
	   


