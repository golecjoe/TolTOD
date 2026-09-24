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

def find_spikes(
	dat: NDArray[np.floating], inner: float = 1, outer: float = 10, thresh: float = 8, end_trim: float = 50
) -> Tuple[List[List[int]], NDArray[np.floating]]:
	"""
	Find spikes in a block of timestreams using a difference of gaussians filter.

	Parameters
	----------
	dat : NDArray[np.floating]
		Data to find spikes in along each row.
		Assumed to be 2d.
	inner : float, default: 1
		Sigma of smaller gaussian for filter in samples.
	outer : float, default: 10
		Sigma of larger gaussian for filter in samples.
	thresh : float, default: 8
		Threshold in units of filtered data median absolute deviation to qualify as a spike.

	Returns
	-------
	spikes : list[list[int]]
		List of lists, each list corresponds to a row of dat with the indices of spikes for that row.
	datfilt : NDArray[np.floating]
		The filtered data with spike locations set to 0.
	"""
	ndet, n = dat.shape
	x = np.arange(n)

	# Smaller gaussian
	filt1 = np.exp(-0.5 * x**2 / inner**2)
	filt1 = filt1 + np.exp(-0.5 * (x - n) ** 2 / inner**2)
	filt1 = filt1 / filt1.sum()

	# Larger gaussian
	filt2 = np.exp(-0.5 * x**2 / outer**2)
	filt2 = filt2 + np.exp(-0.5 * (x - n) ** 2 / outer**2)
	filt2 = filt2 / filt2.sum()

	# Apply the difference of gaussian filter
	filt = filt1 - filt2
	filtft = np.fft.rfft(filt)
	datft = np.fft.rfft(dat, axis=1)
	datfilt = np.fft.irfft(filtft * datft, axis=1, n=n)

	datfilt[:,:end_trim] = 0
	datfilt[:,-end_trim:] = 0

	spikes = [[]] * ndet
	deviation = np.median(np.abs(datfilt), axis=1)
	for i in range(ndet):
		ind = np.where(np.abs(datfilt[i, :]) > thresh * deviation[i])[0]
		spikes[i] = list(ind)
		#datfilt[i, ind] = 0
	return spikes, datfilt



def find_jumps(
	dat: NDArray[np.floating],
	width: int = 10,
	pad: int = 2,
	thresh: float = 10,
	rat: float = 0.5,
	dejump: bool = False,
) -> Union[List[List[int]], Tuple[List[List[int]], NDArray[np.floating]]]:
	"""
	Find jumps in a block of timestreams, preferably with the common mode removed.

	Parameters
	----------
	dat : NDArray[np.floating]
		Data to find jumps in along each row.
		Assumed to be 2d.
	width : int, default: 10
		Width in pixels to average over when looking for a jump.
	pad : int, default: 2
		The length in units of width to mask at beginning/end of timestream.
	thresh : float, default: 10
		Threshold in units of filtered data median absolute deviation to qualify as a jump.
	rat : float, default: .5
		The ratio of largest neighboring opposite-sign jump to the found jump.
		If there is an opposite-sign jump nearby, the jump finder has probably just picked up a spike.
	dejump : bool, default: False
		If True return dejumped data.

	Returns
	-------
	jumps: list[list[int]]
		List of lists, each list corresponds to a row of dat with the indices of jumps for that row.
	dat_dejump : NDArray[np.floating]
		The data with jumps removed.
		Only returned if dejump is True.
	"""
	ndet, n = dat.shape

	# make a filter template that is a gaussian with sigma with, sign-flipped in the center
	# so, positive half-gaussian starting from zero, and negative half-gaussian at the end
	x = np.arange(n)
	filt: NDArray[np.floating] = np.exp(-0.5 * x**2 / width**2)
	filt_sub: NDArray[np.floating] = np.exp((-0.5 * (x - n) ** 2 / width**2))
	filt -= filt_sub
	fac = np.abs(filt).sum() / 2.0
	filt /= fac

	dat_filt = np.fft.rfft(dat, axis=1)

	filt_ft = np.fft.rfft(filt)
	dat_filt = dat_filt * np.repeat([filt_ft], ndet, axis=0)
	dat_filt = np.fft.irfft(dat_filt, axis=1, n=n)
	dat_filt_org=dat_filt.copy()
	dat_filt_org[:, 0 : pad * width] = 0
	dat_filt_org[:, -pad * width :] = 0

	# print(dat_filt.shape)
	dat_med = np.median(dat_filt, axis=1, keepdims=True)
	dat_filt[:, 0 : pad * width] = dat_med
	dat_filt[:, -pad * width :] = dat_med
	# det_thresh = thresh * np.median(np.abs(dat_filt), axis=1)
	

	det_thresh = thresh * np.median(
		np.abs(dat_filt - dat_med),
		axis=1
	)
	dat_dejump = dat
	if dejump:
		dat_dejump = dat.copy()
	jumps = [[]] * ndet
	print("have filtered data, now searching for jumps")
	jumps2 = [[] for _ in range(ndet)] #[[]] * ndet
	for i in range(ndet):
		# ind = np.where(np.abs(dat_filt[i, :]) > det_thresh[i])[0]
		med_i = dat_med[i, 0]
		ind = np.where(
			np.abs(dat_filt[i, :] - med_i) > det_thresh[i]
		)[0]

		#jumps2[i] = list(ind)
		while np.max(np.abs(dat_filt[i, :] - med_i)) > det_thresh[i]:
			# ind = (
			# 	np.argmax(np.abs(dat_filt[i, :])) + 1
			# )  # +1 seems to be the right index to use
			ind = (
				np.argmax(np.abs(dat_filt[i, :] - med_i)) + 1
			)
			imin = max(int(ind - width), 0)
			imax = min(int(ind + width), n)
			val = dat_filt[i, ind] - med_i
			if val > 0:
				val2 = np.min(dat_filt[i, imin:imax]- med_i)
			else:
				val2 = np.max(dat_filt[i, imin:imax]- med_i)

			#print("found jump on detector ", i, " at sample ", ind)
			if np.abs(val2 / val) > rat:
				tmpdummy = 1
				#print("I think this is a spike due to ratio ", np.abs(val2 / val))
			else:
				jumps[i].append(ind)
				jumps2[i].append(ind)
			# independent of if we think it is a spike or a jump, zap that stretch of the data
			if dejump:
				dat_dejump[i, ind:] = dat_dejump[i, ind:] + dat_filt[i, ind]
			#dat_filt[i, ind - pad * width : ind + pad * width] = 0
			# Zap region to median, not zero
			lo = max(ind - pad * width, 0)
			hi = min(ind + pad * width, n)

			dat_filt[i, lo:hi] = med_i
		jumps[i].sort()
		jumps2[i].sort()
	if dejump:
		return jumps, dat_dejump
	return jumps2, dat_filt_org