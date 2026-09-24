import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def make_psds(tod,plotdir=None,plotpsds=False):

	data = tod['signal'][:, :]      # shape (n_streams, n_samples)

	fs = tod['samprate']             # Hz
	N = data.shape[1]

	# Optional but usually recommended: remove the mean of each timestream
	data = data - np.mean(data, axis=1, keepdims=True)

	# FFT
	datft = np.fft.rfft(data, axis=1)

	# Frequency axis
	freq = np.fft.rfftfreq(N, d=1/fs)

	# Two-sided PSD normalization, evaluated on rfft bins
	psd = np.abs(datft)**2 / (fs * N)

	# Convert to one-sided PSD:
	# double bins that don't correspond to DC or Nyquist
	if N % 2 == 0:
		psd[:, 1:-1] *= 2
	else:
		psd[:, 1:] *= 2

	if plotpsds:
		for count, i in enumerate(tod['apt_uid']):
			plt.figure()

			plt.semilogy(freq[1:],psd[count,1:])


			plt.xlabel('Frequency (Hz)')
			plt.ylabel('PSD (mJy/beam)^2/Hz')

			plt.title(f'Detector UID {i}')
			if plotdir is not None:
				plotfile = plotdir / f"PSD_uid_{i}.png"
				plt.savefig(plotfile,bbox_inches='tight')
			plt.close()
	return psd, freq

def fit_psds(tod,psd,freq,plotdir=None,plotpsds=False):

	def modelspec(x,nu):
		return x[0] * (1.0 + (nu / x[1])**x[2])

	def chisq_func(x,data,nu):
		return np.log(data) - np.log(modelspec(x, nu))

	Awn = []
	nu0 = []
	alpha = []

	whitened_psds = np.empty(psd.shape)

	for i in range(psd.shape[0]):

		x0 = np.array([np.median(psd[i,np.where(np.logical_and(freq>20,freq<30))]),2,-2])


		tmpresults = least_squares(chisq_func,x0,args=(psd[i,1:],freq[1:]),bounds=((x0[0]/1E4,1E-2,-3),(x0[0]*1E4,35,0)))
		Awn.append(tmpresults.x[0])
		nu0.append(tmpresults.x[1])
		alpha.append(tmpresults.x[2])

		if plotpsds:

			plt.figure()

			plt.loglog(freq[1:],psd[i,1:])
			plt.loglog(freq[1:],modelspec(tmpresults.x,freq[1:]),c='k')


			plt.xlabel('Frequency (Hz)')
			plt.ylabel('PSD (mJy/beam)^2/Hz')

			plt.title(f'Detector UID {tod['apt_uid'][i]}')
			if plotdir is not None:
				plotfile = plotdir / f"fittedPSD_uid_{tod['apt_uid'][i]}.png"
				plt.savefig(plotfile,bbox_inches='tight')
			plt.close()

		whitened_psds[i,1:] = psd[i,1:]/modelspec(tmpresults.x,freq[1:])

	whitened_psds[:,0] = 0


	Awn = np.array(Awn)
	nu0 = np.array(nu0)
	alpha = np.array(alpha)

	return Awn,nu0,alpha,whitened_psds