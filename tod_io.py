import netCDF4 as nc
import numpy as np
from pathlib import Path
import shutil


def read_in_nw_from_netCDF(filepath,nwindex):
	print(f'Reading in network {nwindex} from file: ',filepath)

	tod = {}

	with nc.Dataset(filepath, "r") as data:
		#print(data.variables.keys())

		nw_array = data['apt_nw'][:]
		apt_flag_array = data['apt_flag'][:]

		select_mask = (nw_array == nwindex) & (apt_flag_array == 0)

		signal = data["signal"][:,select_mask]
		print(f'ndet = {signal.shape[1]} ', f'nsamp = {signal.shape[0]}')

		apt_uid = data['apt_uid'][select_mask]
		samprate = data['SAMPRATE'][0]
		print('Sample rate = ',samprate)

		xts = data['apt_x_t'][select_mask]
		yts = data['apt_y_t'][select_mask]
	tod['signal'] = signal.T
	tod['apt_uid'] = apt_uid
	tod['ndet'] = signal.shape[1]
	tod['nsamp'] = signal.shape[0]
	tod['samprate'] = samprate
	tod['apt_x_t'] = xts
	tod['apt_y_t'] = yts
	tod['original_signal'] = signal.T.copy()
	return tod


def make_new_timestream_nc_file(oldpath,newpath,tod):
	# check to see if new file is made yet, if not make it
	source_file = Path(oldpath)
	new_file = Path(newpath)
	if not new_file.exists():
		shutil.copy2(source_file, new_file)


	with nc.Dataset(new_file, "r+") as newfile:
		# Example: modify an existing variable
		select_indices = np.where(np.logical_and(
			np.isin(newfile['apt_uid'][:], tod['apt_uid']),newfile['apt_flag'][:]==0.0)
		)[0]


		#select_mask = np.isin(tod['apt_uid'], newfile['apt_uid'][:])
		# print('len(tod[apt_uid]) = ',len(tod['apt_uid']))
		# print('len(select_mask) = ',len(select_indices))
		print(f"Replacing Data for {len(select_indices)} Detectors")
		# print(f'np.sum(tod[new_apt_flags]) = ',np.sum(tod['new_apt_flags']))
		newfile['apt_flag'][select_indices] = tod['new_apt_flags']
		newfile['signal'][:,select_indices] = tod['signal'].T










