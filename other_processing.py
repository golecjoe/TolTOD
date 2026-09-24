from typing_extensions import (
	TYPE_CHECKING,
	List,
	Literal,
	Optional,
	Sequence,
	Tuple,
	Union,
	overload,
)

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import iirnotch, filtfilt
from scipy.optimize import least_squares

def _linfit_2mat(dat, mat1, mat2):
	np1 = mat1.shape[1]
	np2 = mat2.shape[1]
	mm = np.append(mat1, mat2, axis=1)
	lhs = np.dot(mm.transpose(), mm)
	rhs = np.dot(mm.transpose(), dat)
	lhs_inv = np.linalg.inv(lhs)
	fitp = np.dot(lhs_inv, rhs)
	fitp1 = fitp[0:np1].copy()
	fitp2 = fitp[np1:].copy()
	assert len(fitp2) == np2
	return fitp1, fitp2


def fit_cm_plus_poly(
	dat: NDArray[np.floating],
	ord: int = 2,
	cm_ord: int = 1,
	niter: int = 1,
	medsub: bool = False,
	full_out: bool = False,
) -> Union[
	NDArray[np.floating],
	Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]],
]:
	"""
	Fit the common mode with polynomials for drifts across the focal plane.

	Parameters
	----------
	dat : NDArray[np.floating]
		The data to fit common mode out of.
		Should be (ndet, ndata).
	ord : int, default: 2
		Order of the legvander that is used as the non common mode polynomial.
	cm_ord : int, default: 2
		Order of the legvander used for the common mode.
	niter : int, default: 1
	medsub : bool, default: False
		If True, subtract the median before fitting.
		Shouldn'd really make a difference since the CM should include the median.
	full_out : bool, default: False
		If True also return pred2 and cm

	Returns
	-------
	dd : NDArray[np.floating]
		The data with the polynomial drifts subtracted.
	pred2 : NDArray[np.floating]
		The polynomial common mode.
	cm : NDArray[np.floating]
		The median common mode.
	"""
	ndet, n = dat.shape
	if medsub:
		med = np.median(dat, axis=1)
		dat = dat - np.repeat([med], n, axis=0).transpose()

	xx = np.arange(n) + 0.0
	xx = xx - xx.mean()
	xx = xx / xx.max()

	pmat = np.polynomial.legendre.legvander(xx, ord)
	cm_pmat = np.polynomial.legendre.legvander(xx, cm_ord - 1)
	calfacs = np.ones(ndet) * 1.0
	dd = dat.copy()
	pred2 = np.zeros(n)
	cm = np.zeros(n)
	for i in range(niter):
		for j in range(ndet):
			dd[j, :] /= calfacs[j]

		cm = np.median(dd, axis=0)
		cm_mat = np.zeros(cm_pmat.shape)
		for k in range(cm_mat.shape[1]):
			cm_mat[:, k] = cm_pmat[:, k] * cm
		fitp_p, fitp_cm = _linfit_2mat(dat.transpose(), pmat, cm_mat)
		pred1 = np.dot(pmat, fitp_p).transpose()
		pred2 = np.dot(cm_mat, fitp_cm).transpose()
		dd = dat - pred1

	if full_out:
		return dd, pred2, cm  # if requested, return the modelled CM as well
	return dd


# def despike_timestreams(tod,spikelist,window=10,plotdir=None):
#     tmpsignal = tod['signal'][:,:].copy()
#     despikecount = 0
#     for count, i in enumerate(tod['apt_uid']):
#         if len(spikelist[count])==0:
#             continue
#         else:
#             print(f'Despiking TOD {i}')
#             todspike = 1
#             for j in spikelist[count]:
#                 print(f'Despike index {j}')
#                 bracket1_val = tod['signal'][count,int(j-(0.5*window))]
#                 bracket2_val = tod['signal'][count,int(j+(0.5*window))]
#                 noise_est1 = np.std(tod['signal'][count,int(j-window):int(j-(0.5*window))])
#                 noise_est2 = np.std(tod['signal'][count,int(j+(0.5*window)):int(j+window)])
#                 slope = (bracket2_val-bracket1_val)/(int(j+(0.5*window))-int(j-(0.5*window)))
#                 noise = np.random.normal(loc=0.0, scale=0.5*(noise_est1+noise_est2), size=np.arange(int(j-(0.5*window)),int(j+(0.5*window))).size)
#                 xinterp = np.arange(int(j-(0.5*window)),int(j+(0.5*window)))
#                 tod['signal'][count,int(j-(0.5*window)):int(j+(0.5*window))] = bracket1_val+(slope*(xinterp-xinterp[0]))+noise

#                 if despikecount%5000==0:

#                     plt.figure()

#                     plt.plot(tod['signal'][count,int(j-(20.*window)):int(j+(20.*window))])

#                     plt.plot(tmpsignal[count,int(j-(20.*window)):int(j+(20.*window))],c='r')
					
#                     plt.xlabel('Sample')
#                     plt.ylabel('Signal (mJy/beam)')

#                     plt.title(f'Detector UID {i}')
#                     if plotdir is not None:
#                         plotfile = plotdir / f"despiked_uid_{i}_spike{todspike}.png"
#                         plt.savefig(plotfile,bbox_inches='tight')
#                     plt.close()
#                 todspike+=1
#                 despikecount+=1

def despike_timestreams(tod,spikelist,lookaheadwindow=1,noisewindowtime=0.5,cleanwindowsamps=10,
	                    plotdir=None, threshtimeblock = 10):
	window = tod['samprate']*lookaheadwindow
	noisewindow = tod['samprate']*noisewindowtime
	tmpsignal = tod['signal'][:,:].copy()
	despikecount = 0
	replacedsampfraclist = np.zeros(len(tod['apt_uid']))
	longflagsampcount = np.zeros(len(tod['apt_uid']))
	for count, i in enumerate(tod['apt_uid']):
		#print('Despiking UID: ',tod['apt_uid'][count])
		longdespikeflag = 0
		replacedsampscounter = 0
		if len(spikelist[count])==0:
			continue
		else:
			nextspikecount = 0
			tmpcount = 0
			while nextspikecount<len(spikelist[count]):
				#firstindex = int(spikelist[count][nextspikecount]-window)
				firstindex = int(spikelist[count][nextspikecount]-cleanwindowsamps)
				nextspikecount += 1
				if nextspikecount>len(spikelist[count])-1:
					tmpdespikegroup = []
					tmpdespikegroup.append(spikelist[count][nextspikecount-1])
					lastindex = int(spikelist[count][nextspikecount-1]+window)
					tmpcount+=1
				else:
					tmpdespikegroup = []
					tmpdespikegroup.append(spikelist[count][nextspikecount-1])
					lastindex = int(spikelist[count][nextspikecount-1]+window)
					while spikelist[count][nextspikecount]<lastindex:
						tmpdespikegroup.append(spikelist[count][nextspikecount])
						lastindex = int(spikelist[count][nextspikecount]+window)
						nextspikecount+=1
						if nextspikecount>len(spikelist[count])-1:
							break
				#print('Despiking group of indices: ',tmpdespikegroup)
				if len(tmpdespikegroup)==1:
					lastindex = tmpdespikegroup[0]+cleanwindowsamps
				else:
					lastindex = tmpdespikegroup[-1]+cleanwindowsamps

				replacedsampscounter+=(lastindex-firstindex)

				#threshtimeblock = 10.
				if (lastindex-firstindex)/tod['samprate']>threshtimeblock:
					#print(f'Spike incident {tmpcount} is longer than {threshtimeblock} seconds!')
					#print(f'Event length = {(lastindex-firstindex)/tod['samprate']} seconds')
					longflagsampcount[count] += 1

				bracket1_val = tod['signal'][count,firstindex]
				bracket2_val = tod['signal'][count,lastindex]
				noiseestfirstindex = max([int(firstindex-noisewindow),0])
				noise_est1 = np.std(tod['signal'][count,noiseestfirstindex:int(firstindex)])
				noise_est2 = np.std(tod['signal'][count,int(lastindex):int(lastindex+noisewindow)])
				slope = (bracket2_val-bracket1_val)/(lastindex-firstindex)
				totalnoiseest = noise_est1 #0.5*(noise_est1+noise_est2)/np.sqrt(2)
				noise = np.random.normal(loc=0.0, scale=totalnoiseest, size=np.arange(firstindex,lastindex).size)
				xinterp = np.arange(firstindex,lastindex)
				tod['signal'][count,firstindex:lastindex] = bracket1_val+(slope*(xinterp-xinterp[0]))+noise

				#if tmpcount%50==0:
				if False:
					plt.figure()

					plt.plot(tod['signal'][count,noiseestfirstindex:int(lastindex+noisewindow)],c='k')

					plt.plot(tmpsignal[count,noiseestfirstindex:int(lastindex+noisewindow)],c='r')
					
					plt.xlabel('Sample')
					plt.ylabel('Signal (mJy/beam)')

					plt.title(f'Detector UID {i} / spike incident {tmpcount}')
					if plotdir is not None:
						tmpdir = Path(plotdir /f'uid_{i}/')
						tmpdir.mkdir(parents=True, exist_ok=True)
						plotfile = tmpdir / f"despiked_uid_{i}_spike{tmpcount}.png"
						plt.savefig(plotfile,bbox_inches='tight')
						plt.close()
				tmpcount+=1
			#print(f'Replaced samples = {replacedsampscounter} / fraction of data {replacedsampscounter/len(tmpsignal[count,:])}')
			replacedsampfraclist[count] = replacedsampscounter/len(tmpsignal[count,:])
	return np.array(replacedsampfraclist),longflagsampcount


def dejump_timestream(tod,jumplist,width=10,plotwidth=10,plotdir=None):

	jump_amp_list = []

	for count,i in enumerate(tod['apt_uid']):
		for ind in jumplist[count]:
			pre = np.median(tod['signal'][count,int(max(0,ind-width)):ind])
			post = np.median(tod['signal'][count,ind:int(min(tod['nsamp'],ind+width))])
			jump_amp = post-pre

			startind = int(max([0,ind-plotwidth]))
			lastind = int(min([tod['signal'].shape[1],ind+plotwidth]))
			
			todspin_beforedespike = tod['signal'][count,startind:lastind].copy()
			tod['signal'][count,ind:] -= jump_amp
			jump_amp_list.append(jump_amp)

			if False:


				plt.figure()

				plt.plot(range(startind,lastind),tod['signal'][count,startind:lastind],c='k',label='dejumped')

				plt.plot(range(startind,ind),pre*np.ones(len(range(startind,ind))))
				plt.plot(range(ind,lastind),post*np.ones(len(range(ind,lastind))))

				plt.plot(range(startind,lastind),todspin_beforedespike,c='r',label='original')


				# plt.plot(range(startind,lastind),tod['original_signal'][count,startind:lastind],c='r',label='original')


				# plt.plot(range(startind,lastind),tod['cm_gain_factor'][count]*tod['original_signal'][count,startind:lastind]+tod['cm_offset_factor'][count],c='r',label='original')

				plt.axvline(ind,linestyle='--',c='k')


				
				plt.xlabel('Sample')
				plt.ylabel('Signal (mJy/beam)')
				plt.legend(loc='upper right')

				plt.title(f'Detector UID {i} / jump incident {ind}')
				if plotdir is not None:
					tmpdir = Path(plotdir /f'uid_{i}/')
					tmpdir.mkdir(parents=True, exist_ok=True)
					plotfile = tmpdir / f"dejump_uid_{i}_jump{ind}.png"
					plt.savefig(plotfile,bbox_inches='tight')
				plt.close()

	return jump_amp_list



def notch_filter(data, fs, freqs, width=0.5, axis=-1):
    filtered = np.asarray(data).copy()

    for freq in freqs:
        Q = freq / width
        b, a = iirnotch(freq, Q, fs=fs)
        filtered = filtfilt(b, a, filtered, axis=axis)

    return filtered

def fit_gain_and_offset_to_cm(tod,plotdir=None):
	def chisq_func(x,signal,cm):
		return (x[0]*signal+x[1])-cm

	gain_factors = np.empty(len(tod['apt_uid']))
	offset_factors = np.empty(len(tod['apt_uid']))
	minrms = np.empty(len(tod['apt_uid']))

	for count,i in enumerate(tod['apt_uid']):
		x0 = np.array([1.,0.])

		tmpresults = least_squares(chisq_func,x0,args=(tod['signal'][count,:],tod['cm']))

		gain_factors[count] = tmpresults.x[0]
		offset_factors[count] = tmpresults.x[1]

		resid = (tmpresults.x[0]*tod['signal'][count,:]+tmpresults.x[1])-tod['cm']
		minrms[count] = np.sqrt(np.mean(resid**2))

		tod['signal'][count,:] = tmpresults.x[0]*tod['signal'][count,:]+tmpresults.x[1]

		if plotdir is not None:
			plt.figure()

			plt.plot(resid)

			plt.xlabel('Sample')
			plt.ylabel('Residual (mJy/beam)')
			plt.title(f'Detector UID {i}')

			rms = np.sqrt(np.mean(resid**2))

			textstr = (
			    f"Offset = {tmpresults.x[1]:.3f}\n"
			    f"Gain = {tmpresults.x[0]:.3f}\n"
			    f"RMS = {rms:.3f}"
			)

			plt.text(
			    0.95, 0.95,
			    textstr,
			    transform=plt.gca().transAxes,
			    horizontalalignment='right',
			    verticalalignment='top',
			    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
			)

			plotfile = plotdir / f"cmfitresid_uid{i}.png"
			plt.savefig(plotfile,bbox_inches='tight')
			plt.close()


		



	return gain_factors,offset_factors,minrms













