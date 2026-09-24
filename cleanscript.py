from tod_io import read_in_nw_from_netCDF, make_new_timestream_nc_file
from other_processing import *
from qc_image_generation import *
from detection_functions import find_spikes,find_jumps
from fourier_processing import make_psds,fit_psds
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.signal import find_peaks

all_obsnums = [152390,152392,152419,152431,152433]

input_dir_base = Path('/work/toltec/commissioning2025-test/2025-C1-COM-01/jgolec/NGC4449/reduced_TODs/redu00')
output_dir_base = Path('/work/toltec/commissioning2025-test/2025-C1-COM-01/jgolec/NGC4449/reduced_TODs/step1')

tmpfilepath = 'data/toltec_commissioning_science_152390_rtc_timestream.nc'
newfilepath = 'data/toltec_commissioning_science_152390_rtc_timestream_step1.nc'


#######################
## Define Parameters ##
#######################

# Define CM Cut Parameters

gainmedthresh = 3. # +/- MAD from median of fitted gains to cut
offsetmedthresh = 3. # +/- MAD from median of fitted DC offsets to cut
rmsmedthresh = 2. # +/- MAD from median of resulting residual rms to cut

# Define Spike Cut Parameters

spikethresh = 8 # number of MAD 
spikeendtrim = 50 # number of samples at start/end of TOD to ignore in spike filter (this prevents detecting ringing as spikes)
med_spikenum_thresh = 10. # factors of median above the median number of spikes to threshold on

# Define Jump Cut Parameters

jumpthresh = 10.
jumpsearchwidth = 10

jumpiterations = 2


despiketods = True
run_tod_dejump = False

total_t1 = time.time()


for obsnum in all_obsnums:
	initialncfilepath = input_dir_base / f'{obsnum}/raw/toltec_commissioning_science_{obsnum}_rtc_timestream.nc'
	finalncfilepath = output_dir_base / f'{obsnum}/raw/toltec_commissioning_science_{obsnum}_rtc_timestream.nc'

	outputdir = Path(output_dir_base / f'{obsnum}/raw/')
	outputdir.mkdir(parents=True, exist_ok=True)

	for i in [0,1,2,3,4,5,6,7,8,9,11,12]:

		tmpnw = i
		tmpobsnum = obsnum


		outputdir = Path(output_dir_base / f'outputs/obs_{tmpobsnum}/')
		outputdir.mkdir(parents=True, exist_ok=True)

		tmpqcbasedir = Path(output_dir_base / f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/')
		tmpqcbasedir.mkdir(parents=True, exist_ok=True)


		print('Reading in the file')
		tmptod = read_in_nw_from_netCDF(initialncfilepath,tmpnw)
		print('Finished reading in the file')

		totalpsdcuts = tmptod['apt_uid']>-1
		despikecutmask = tmptod['apt_uid']>-1
		jump_cut_mask = tmptod['apt_uid']>-1
		numberofspikescut = tmptod['apt_uid']>-1
		totalcmcuts = tmptod['apt_uid']>-1

		if tmptod['ndet']==0:
			continue

		master_cut_list = np.zeros(len(tmptod['apt_uid']))
		tmpcutsqcpath = Path(output_dir_base /f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/cuts_figs/')
		tmpcutsqcpath.mkdir(parents=True, exist_ok=True)


		tmpdd, tmppred2, tmpcm = fit_cm_plus_poly(tmptod['signal'],full_out=True)

		tmptod['signal'] = tmpdd#-tmppred2
		tmptod['cm'] = tmpcm

		tmpqcbasedir_init = Path(output_dir_base / f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/initial_tods/')
		tmpqcbasedir_init.mkdir(parents=True, exist_ok=True)

		# make_tod_plots(tmptod,plotdir=tmpqcbasedir_init)
		

		tmpqcbasedir_init = Path(output_dir_base / f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/cm_fitting/')
		tmpqcbasedir_init.mkdir(parents=True, exist_ok=True)

		tmpgainfact, tmpoffsetfact, tmprms = fit_gain_and_offset_to_cm(tmptod,plotdir=None)

		tmptod['cm_gain_factor'] = tmpgainfact
		tmptod['cm_offset_factor'] = tmpoffsetfact

		plt.figure()

		for count,i in enumerate(tmptod['apt_uid']):
			plt.plot(tmptod['signal'][count,:],c='grey',alpha=0.2)
		plt.plot(tmptod['cm'],c='k')

		plt.xlabel('Sample')
		plt.ylabel('Signal (mJy/beam)')
		plt.title('CM and Corrected TODs')
		plotfile = tmpqcbasedir_init / f"cmcorrected_summary.png"
		plt.savefig(plotfile,bbox_inches='tight')
		plt.close()





		make_cmfit_plots(tmptod,tmpgainfact,tmpoffsetfact,tmprms,tmpqcbasedir_init,
						 gainthresh=gainmedthresh, offsetthresh = offsetmedthresh, rmsthresh=rmsmedthresh)

		med_gainfact = np.median(tmpgainfact)
		mad_gainfact = np.median(np.abs(tmpgainfact - med_gainfact))

		med_offfact = np.median(tmpoffsetfact)
		mad_offfact = np.median(np.abs(tmpoffsetfact - med_offfact))

		med_rmsfact = np.median(tmprms)
		mad_rmsfact = np.median(np.abs(tmprms - med_rmsfact))

		gainupper = (gainmedthresh*mad_gainfact)+np.median(tmpgainfact) #(gainmedthresh*np.std(tmpgainfact))+np.median(tmpgainfact)
		gainlower = (-gainmedthresh*mad_gainfact)+np.median(tmpgainfact) #(-gainmedthresh*np.std(tmpgainfact))+np.median(tmpgainfact)

		offsetupper = (offsetmedthresh*mad_offfact)+np.median(tmpoffsetfact) #(offsetmedthresh*np.std(tmpoffsetfact))+np.median(tmpoffsetfact)
		offsetlower = (-offsetmedthresh*mad_offfact)+np.median(tmpoffsetfact) #(-offsetmedthresh*np.std(tmpoffsetfact))+np.median(tmpoffsetfact)


		rmsupper = (rmsmedthresh*mad_rmsfact)+np.median(tmprms) #(rmsmedthresh*np.std(tmprms))+np.median(tmprms)
		rmslower = (-rmsmedthresh*mad_rmsfact)+np.median(tmprms) #(rmsmedthresh*np.std(tmprms))+np.median(tmprms)


		gaincuts = (tmpgainfact < gainupper) & (tmpgainfact > gainlower)
		offsetcuts = (tmpoffsetfact < offsetupper) & (tmpoffsetfact > offsetlower)
		rmscuts = (tmprms < rmsupper) & (tmprms > rmslower)

		totalcmcuts = gaincuts & offsetcuts & rmscuts
		print('##########################################')
		print('Gain Cut Fraction = ',len(tmptod['apt_uid'][~gaincuts])/len(tmptod['apt_uid']))
		print('Offset Cut Fraction = ',len(tmptod['apt_uid'][~offsetcuts])/len(tmptod['apt_uid']))
		print('RMS Cut Fraction = ',len(tmptod['apt_uid'][~rmscuts])/len(tmptod['apt_uid']))

		print('Fraction of Detectors cut by CM cuts = ',len(tmptod['apt_uid'][~totalcmcuts])/len(tmptod['apt_uid']))

		print('##########################################')

		plt.figure()

		plt.plot(tmptod['apt_x_t'][totalcmcuts],tmptod['apt_y_t'][totalcmcuts],'.',c='g')
		plt.plot(tmptod['apt_x_t'][~totalcmcuts],tmptod['apt_y_t'][~totalcmcuts],'.',c='r')
		plt.xlabel('Detector X Pos')
		plt.ylabel('Detector Y Pos')

		frac_cut = np.sum(~totalcmcuts) / len(totalcmcuts)

		plt.text(
			0.98,
			0.98,
			f'Fraction of Detectors Cut = {frac_cut:.3f}',
			transform=plt.gca().transAxes,
			ha='right',
			va='top'
		)
		plt.title('CM Cut Summary')
		plotfile = tmpcutsqcpath / f"cmcutssummary.png"
		plt.savefig(plotfile,bbox_inches='tight')
		plt.close()


		# tmpqcbasedir_init = Path(output_dir_base / f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/cmresid_tods/')
		# tmpqcbasedir_init.mkdir(parents=True, exist_ok=True)
		# make_cmresid_plots(tmptod,plotdir=tmpqcbasedir_init)


		print('Running Spike Finder')

		

		tmpspikes, tmpdatfilt = find_spikes(tmptod['signal'],thresh=spikethresh,end_trim = spikeendtrim)


		tmpqcbasedir_init = Path(output_dir_base / f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/spikefilt_tods/')
		tmpqcbasedir_init.mkdir(parents=True, exist_ok=True)

		# make_spikefiltered_plots(tmpdatfilt,tmptod,plotdir=tmpqcbasedir_init,thresh=tmpthresh)

		tmpwindow = 30 #seconds
		# plot_spikes(tmptod,tmpspikes,window=tmpwindow,plotdir=tmpqcbasedir_init)
		numspikes = []
		for i in range(tmptod['ndet']):
			numspikes.append(len(tmpspikes[i]))

		numspikes = np.array(numspikes)

		
		print(f'Number of TODs with spike counts greater than {med_spikenum_thresh} times the median = ',len(np.where(numspikes>med_spikenum_thresh*np.median(numspikes))[0]))
		print('Fraction of TODs = ',len(np.where(numspikes>med_spikenum_thresh*np.median(numspikes))[0])/tmptod['ndet'])


		numberofspikescut = numspikes <=med_spikenum_thresh*np.median(numspikes)
		plt.figure()
		plt.hist(numspikes,bins=np.linspace(0,max(numspikes)/10,50))
		plt.axvline(np.median(numspikes),c='k')
		plt.axvline(med_spikenum_thresh*np.median(numspikes),linestyle='--',c='k')

		plt.xlabel('Number of Spikes')
		plt.ylabel('Count')
		plotfile = tmpqcbasedir_init / f"spikehist.png"
		plt.savefig(plotfile,bbox_inches='tight')
		plt.close()

		plt.figure()

		plt.plot(tmptod['apt_x_t'][numberofspikescut],tmptod['apt_y_t'][numberofspikescut],'.',c='g')
		plt.plot(tmptod['apt_x_t'][~numberofspikescut],tmptod['apt_y_t'][~numberofspikescut],'.',c='r')

		frac_cut = np.sum(~numberofspikescut) / len(numberofspikescut)

		plt.text(
			0.98,
			0.98,
			f'Fraction of Detectors Cut = {frac_cut:.3f}',
			transform=plt.gca().transAxes,
			ha='right',
			va='top'
		)
		plt.xlabel('Detector X Pos')
		plt.ylabel('Detector Y Pos')
		plt.title('Spike Cut Summary')
		plotfile = tmpcutsqcpath / f"spikecutssummary.png"
		plt.savefig(plotfile,bbox_inches='tight')
		plt.close()

		print('Spike Finder finished')


		

		if despiketods:
			print('Running Despiker')
			tmpqcbasedir_init = Path(output_dir_base / f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/despike_tods/')
			tmpqcbasedir_init.mkdir(parents=True, exist_ok=True)
			t1 = time.time()
			longincidentthresh = 10.
			replacesampfracs, tmplongdespikecount = despike_timestreams(tmptod,tmpspikes,lookaheadwindow=5,noisewindowtime=0.5,cleanwindowsamps=20,plotdir=tmpqcbasedir_init, threshtimeblock = longincidentthresh)
			print('Time to despike = ',time.time()-t1)
			plt.figure()
			plt.hist(replacesampfracs,bins=np.linspace(0,max(replacesampfracs),50))
			med_num_thresh_replace = 3.
			plt.axvline(np.median(replacesampfracs),c='k')
			plt.axvline(med_num_thresh_replace*np.median(replacesampfracs),linestyle='--',c='k')

			plt.xlabel('Fraction of samples replaced')
			plt.ylabel('Count')
			plotfile = tmpqcbasedir_init / f"fracdatareplacedhist.png"
			plt.savefig(plotfile,bbox_inches='tight')
			plt.close()

			plt.figure()
			plt.hist(tmplongdespikecount,bins=np.linspace(0,max(tmplongdespikecount),20))
			plt.axvline(np.median(replacesampfracs),c='k')
			plt.axvline(med_num_thresh_replace*np.median(replacesampfracs),linestyle='--',c='k')

			plt.xlabel(f'Number of Incidences > {longincidentthresh} seconds')
			plt.ylabel('Count')
			plotfile = tmpqcbasedir_init / f"longincidenthist.png"
			plt.savefig(plotfile,bbox_inches='tight')
			plt.close()

			despikecutthresh = max([med_num_thresh_replace*np.median(replacesampfracs),0.05])

			despikecutmask = replacesampfracs<despikecutthresh

			print('Fraction of Detectors cut in despiking = ',len(tmptod['apt_uid'][~despikecutmask])/len(tmptod['apt_uid']))
			plt.figure()
			plt.plot(tmptod['apt_x_t'][despikecutmask],tmptod['apt_y_t'][despikecutmask],'.',c='g')
			plt.plot(tmptod['apt_x_t'][~despikecutmask],tmptod['apt_y_t'][~despikecutmask],'.',c='r')
			plt.xlabel('Detector X Pos')
			plt.ylabel('Detector Y Pos')
			frac_cut = np.sum(~despikecutmask) / len(despikecutmask)

			plt.text(
				0.98,
				0.98,
				f'Fraction of Detectors Cut = {frac_cut:.3f}',
				transform=plt.gca().transAxes,
				ha='right',
				va='top'
			)
			plt.title('Despike Cut Summary')
			plotfile = tmpcutsqcpath / f"despikecutssummary.png"
			plt.savefig(plotfile,bbox_inches='tight')
			plt.close()

		print('Running jump detector')

		for iteration in range(jumpiterations):

			print(f'starting jump iter {iteration+1} of {jumpiterations}')

			tmpjumps, tmpjumpfilt = find_jumps(tmptod['signal'],pad=4,thresh=jumpthresh,width=jumpsearchwidth)

			tmpqcbasedir_init = Path(output_dir_base / f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/jumpfilt_tods_iter{iteration}/')
			tmpqcbasedir_init.mkdir(parents=True, exist_ok=True)

			make_jumpfiltered_plots(tmpjumpfilt,tmptod,plotdir=tmpqcbasedir_init,thresh=jumpthresh,pad=4)

			numjumps = []
			for i in range(tmptod['ndet']):
				numjumps.append(len(tmpjumps[i]))

			med_num_thresh = 3
			print('Median number of jumps = ',np.median(numjumps))
			# print(f'Number of TODs with jump counts greater than {med_num_thresh} times the median = ',len(np.where(numjumps>med_num_thresh*np.median(numjumps))[0]))
			# print('Fraction of TODs = ',len(np.where(numjumps>med_num_thresh*np.median(numjumps))[0])/tmptod['ndet'])
			numjumps = np.array(numjumps)
			jump_cut_mask = ~(numjumps>min([3,np.median(numjumps)]))
			print('Fraction of Detectors Cut From Jumps = ',np.sum(~jump_cut_mask)/len(jump_cut_mask))

			plt.figure()

			plt.plot(tmptod['apt_x_t'][jump_cut_mask],tmptod['apt_y_t'][jump_cut_mask],'.',c='g')
			plt.plot(tmptod['apt_x_t'][~jump_cut_mask],tmptod['apt_y_t'][~jump_cut_mask],'.',c='r')
			plt.xlabel('Detector X Pos')
			plt.ylabel('Detector Y Pos')
			frac_cut = np.sum(~jump_cut_mask) / len(jump_cut_mask)

			plt.text(
				0.98,
				0.98,
				f'Fraction of Detectors Cut = {frac_cut:.3f}',
				transform=plt.gca().transAxes,
				ha='right',
				va='top'
			)
			plt.title('Jump Cut Summary')
			plotfile = tmpcutsqcpath / f"jumpcutssummary.png"
			plt.savefig(plotfile,bbox_inches='tight')
			plt.close()



			plt.figure()
			plt.hist(numjumps,bins=np.linspace(0,max(numjumps),20))
			plt.axvline(np.median(numjumps),c='k')
			plt.axvline(med_num_thresh*np.median(numjumps),linestyle='--',c='k')

			plt.xlabel('Number of Jumps')
			plt.ylabel('Count')
			plotfile = tmpqcbasedir_init / f"jumphist.png"
			plt.savefig(plotfile,bbox_inches='tight')
			plt.close()

			if run_tod_dejump:
				print('running DeJumper')
				tmpjumpamps = dejump_timestream(tmptod,tmpjumps,width=jumpsearchwidth,plotwidth=2.*122.,plotdir=tmpqcbasedir_init)

				plt.figure()
				plt.hist(tmpjumpamps,bins=np.linspace(0,max(tmpjumpamps),50))
				plt.axvline(np.median(tmpjumpamps),c='k')

				plt.xlabel('Jump Amplitudes (mJy/beam)')
				plt.ylabel('Count')
				plotfile = tmpqcbasedir_init / f"jumpamps.png"
				plt.savefig(plotfile,bbox_inches='tight')
				plt.close()

		print('Computing PSDs')

		tmpqcbasedir_init = Path(output_dir_base /f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/tod_psds/')
		tmpqcbasedir_init.mkdir(parents=True, exist_ok=True)

		tmppsd, tmpfreqs = make_psds(tmptod,plotdir=tmpqcbasedir_init)

		plt.figure()


		for i in range(tmppsd.shape[0]):
			plt.semilogy(tmpfreqs[1:],tmppsd[i,1:],c='grey',alpha=0.2)
		plt.semilogy(tmpfreqs[1:],np.median(tmppsd[:,1:], axis=0),c='k')

		plt.xlabel('Frequency (Hz)')
		plt.ylabel('PSD (mJy/beam)^2/Hz')
		plotfile = tmpqcbasedir_init / f"psd_summary.png"
		plt.savefig(plotfile,bbox_inches='tight')
		plt.close()

		print('Fitting PSDs')
		tmpAwn, tmpnu0, tmpalpha, tmpwhitenedpsds = fit_psds(tmptod,tmppsd,tmpfreqs,plotdir=tmpqcbasedir_init)

		wnmedthresh = 3.
		fkneethresh = 2.5
		alphathresh = 2.5

		make_psd_qc_plots(tmptod,tmpAwn,tmpnu0,tmpalpha,tmpqcbasedir_init,
						  Awnthresh = wnmedthresh, fkneethresh = fkneethresh, alphathresh = alphathresh)

		Awnupper = (wnmedthresh*np.std(tmpAwn))+np.median(tmpAwn)
		Awnlower = (-wnmedthresh*np.std(tmpAwn))+np.median(tmpAwn)

		fkneeupper = (fkneethresh*np.std(tmpnu0))+np.median(tmpnu0)
		fkneelower = (-fkneethresh*np.std(tmpnu0))+np.median(tmpnu0)

		alphaupper = (alphathresh*np.std(tmpalpha))+np.median(tmpalpha)
		alphalower = (-alphathresh*np.std(tmpalpha))+np.median(tmpalpha)

		Awncuts = (tmpAwn < Awnupper) & (tmpAwn > Awnlower)
		fkneecuts = (tmpnu0 < fkneeupper) & (tmpnu0 > fkneelower)
		alphacuts = (tmpalpha < alphaupper) & (tmpalpha > alphalower)

		totalpsdcuts = Awncuts & fkneecuts & alphacuts

		print('Fraction of Detectors cut in PSD analysis= ',len(tmptod['apt_uid'][~totalpsdcuts])/len(tmptod['apt_uid']))
		plt.figure()
		plt.plot(tmptod['apt_x_t'][totalpsdcuts],tmptod['apt_y_t'][totalpsdcuts],'.',c='g')
		plt.plot(tmptod['apt_x_t'][~totalpsdcuts],tmptod['apt_y_t'][~totalpsdcuts],'.',c='r')
		plt.xlabel('Detector X Pos')
		plt.ylabel('Detector Y Pos')
		frac_cut = np.sum(~totalpsdcuts) / len(totalpsdcuts)

		plt.text(
			0.98,
			0.98,
			f'Fraction of Detectors Cut = {frac_cut:.3f}',
			transform=plt.gca().transAxes,
			ha='right',
			va='top'
		)
		plt.title('PSD Cut Summary')
		plotfile = tmpcutsqcpath / f"psdcutssummary.png"
		plt.savefig(plotfile,bbox_inches='tight')
		plt.close()



		tmpthresh = 200
		tmpendtrim = 50

		tmppsdspikes, tmppsddatfilt = find_spikes(tmpwhitenedpsds,thresh=tmpthresh,end_trim = tmpendtrim)

		make_spikefiltered_plots_psd(tmppsddatfilt,tmpfreqs,tmptod,plotdir=tmpqcbasedir_init,thresh=tmpthresh)

		allpsdspikeind = []
		allpsdspikefreq = []

		for i in range(tmptod['ndet']):
			for j in range(len(tmppsdspikes[i])):
				allpsdspikeind.append(tmppsdspikes[i][j])
				allpsdspikefreq.append(tmpfreqs[tmppsdspikes[i][j]])

		allpsdspikefreq = np.array(allpsdspikefreq)
		allpsdspikeind = np.array(allpsdspikeind)

		spikefreqbins = np.linspace(0, 61, 150)
		counts, bin_edges = np.histogram(allpsdspikefreq, bins=spikefreqbins)
		bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

		# Find distinct peaks in the histogram
		peak_idx, properties = find_peaks(counts)

		# Sort peaks by histogram count
		peak_idx = peak_idx[np.argsort(counts[peak_idx])[::-1]]

		# Take the two strongest distinct peaks
		top_two_idx = peak_idx[:5]

		top_two_freqs = bin_centers[top_two_idx]
		top_two_counts = counts[top_two_idx]

		for freq, count in zip(top_two_freqs, top_two_counts):
			print(f"{freq:.2f} Hz: {count} occurrences")

		# Plot
		plt.figure()

		plt.hist(allpsdspikefreq, bins=spikefreqbins)

		for freq in top_two_freqs:
			plt.axvline(freq, linestyle="--", label=f"{freq:.2f} Hz")

		plt.xlabel("Spike Frequency (Hz)")
		plt.ylabel("Count")
		plt.legend()

		plotfile = tmpqcbasedir_init / "psd_spike_hist.png"
		plt.savefig(plotfile, bbox_inches="tight")
		plt.close()
		tmpfs = float(np.asarray(tmptod['samprate']).squeeze())
		notchfiltdata = notch_filter(tmptod['signal'], tmpfs, top_two_freqs, width=0.7, axis=-1)
		tmptod['signal'] = notchfiltdata


		mastercuts = totalpsdcuts & despikecutmask & jump_cut_mask & numberofspikescut & totalcmcuts

		mastercuts_dict = {}
		mastercuts_dict['master_cuts'] = mastercuts
		mastercuts_dict['psd_cuts'] = totalpsdcuts
		mastercuts_dict['despike_cuts'] = despikecutmask
		mastercuts_dict['jump_cuts'] = jump_cut_mask
		mastercuts_dict['spike_cuts'] = numberofspikescut
		mastercuts_dict['cm_cuts'] = totalcmcuts

		tmptod['new_apt_flags'] = (~mastercuts).astype(int)

		#print(tmptod['new_apt_flags'])

		tmpqcbasedir_init = Path(output_dir_base / f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/final_tods/')
		tmpqcbasedir_init.mkdir(parents=True, exist_ok=True)

		make_final_tod_plots(tmptod, mastercuts_dict, plotdir=tmpqcbasedir_init)

		print('Total Fraction of Detectors cut = ',len(tmptod['apt_uid'][~mastercuts])/len(tmptod['apt_uid']))
		plt.figure()
		plt.plot(tmptod['apt_x_t'][mastercuts],tmptod['apt_y_t'][mastercuts],'.',c='g')
		plt.plot(tmptod['apt_x_t'][~mastercuts],tmptod['apt_y_t'][~mastercuts],'.',c='r')
		plt.xlabel('Detector X Pos')
		plt.ylabel('Detector Y Pos')
		plt.title('Master Cut Summary')
		frac_cut = np.sum(~mastercuts) / len(mastercuts)

		plt.text(
			0.98,
			0.98,
			f'Fraction of Detectors Cut = {frac_cut:.3f}',
			transform=plt.gca().transAxes,
			ha='right',
			va='top'
		)
		plotfile = tmpcutsqcpath / f"mastercutssummary.png"
		plt.savefig(plotfile,bbox_inches='tight')
		plt.close()

		make_new_timestream_nc_file(initialncfilepath,finalncfilepath,tmptod)

		# tmpqcbasedir_init = Path(f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/notchfiltered_tods/')
		# tmpqcbasedir_init.mkdir(parents=True, exist_ok=True)

		# make_tod_plots(tmptod,plotdir=tmpqcbasedir_init)

		# tmpqcbasedir_init = Path(f'outputs/obs_{tmpobsnum}/qc/nw_{tmpnw}/tod_psds_notchfilt/')
		# tmpqcbasedir_init.mkdir(parents=True, exist_ok=True)

		# tmppsd, tmpfreqs = make_psds(tmptod,plotdir=tmpqcbasedir_init)

		# plt.figure()


		# for i in range(tmppsd.shape[0]):
		# 	plt.semilogy(tmpfreqs[1:],tmppsd[i,1:],c='grey',alpha=0.2)
		# plt.semilogy(tmpfreqs[1:],np.median(tmppsd[:,1:], axis=0),c='k')

		# plt.xlabel('Frequency (Hz)')
		# plt.ylabel('PSD (mJy/beam)^2/Hz')
		# plotfile = tmpqcbasedir_init / f"psd_summary.png"
		# plt.savefig(plotfile,bbox_inches='tight')
		# plt.close()

		# print('Fitting PSDs')
		# tmpAwn, tmpnu0, tmpalpha, tmpwhitenedpsds = fit_psds(tmptod,tmppsd,tmpfreqs,plotdir=tmpqcbasedir_init)

		# make_psd_qc_plots(tmptod,tmpAwn,tmpnu0,tmpalpha,tmpqcbasedir_init)


print('Total Time = ',(time.time()-total_t1)/3600., ' Hours')


	
