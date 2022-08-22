import h5py
from .base_dataset import BaseVoxelDataset
import pandas as pd
import numpy as np

class GoproEsimH5Dataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the Monash University HDF5 events format
    GOPRO ESIM data, index is different (index + 1)
    """

    def get_frame(self, index):
        """
        discard the first and the last frame in GoproEsim
        """
        return self.h5_file['images']['image{:09d}'.format(index+1)][:]
    
    def get_gt_frame(self, index):
        """
        discard the first and the last frame in GoproEsim
        """
        return self.h5_file['sharp_images']['image{:09d}'.format(index+1)][:]

    def get_events(self, idx0, idx1):
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0  # -1 and 1
        return xs, ys, ts, ps

    def load_data(self, data_path):
        self.data_sources = ('esim', 'ijrr', 'mvsec', 'eccd', 'hqfd', 'unknown')
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]
        print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['flow']) > 0
        self.t0 = self.h5_file['events/ts'][0]
        self.tk = self.h5_file['events/ts'][-1]
        self.num_events = self.h5_file.attrs["num_events"]
        self.num_frames = self.h5_file.attrs["num_imgs"]

        self.frame_ts = []
        for img_name in self.h5_file['images']:
            self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = self.data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx

    def ts(self, index):
        return self.h5_file['events/ts'][index]

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for img_name in self.h5_file['images']:
            end_idx = self.h5_file['images/{}'.format(img_name)].attrs['event_idx']
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class SeemsH5Dataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the Monash University HDF5 events format
    H5 data contains the exposure information
    """

    def get_frame(self, index):
        """
        discard the first and the last frame in GoproEsim
        """
        return self.h5_file['images']['image{:09d}'.format(index)][:]

    def get_gt_frame(self, index):
        """
        discard the first and the last frame in GoproEsim
        """
        return self.h5_file['sharp_images']['image{:09d}'.format(index)][:]

    def get_events(self, idx0, idx1):
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0  # -1 and 1
        return xs, ys, ts, ps

    def load_data(self, data_path):
        self.data_sources = ('esim', 'ijrr', 'mvsec', 'eccd', 'hqfd', 'unknown')
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))

        if self.sensor_resolution is None:
            self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]
        print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['flow']) > 0
        self.t0 = self.h5_file['events/ts'][0]
        self.tk = self.h5_file['events/ts'][-1]
        self.num_events = self.h5_file.attrs["num_events"]
        self.num_frames = self.h5_file.attrs["num_imgs"]

        self.frame_ts = []
        if self.has_exposure_time:
            self.frame_exposure_start=[]
            self.frame_exposure_end = []
            self.frame_exposure_time=[]
        for img_name in self.h5_file['images']:
            self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])
            if self.has_exposure_time:
                self.frame_exposure_start.append(self.h5_file['images/{}'.format(img_name)].attrs['exposure_start'])
                self.frame_exposure_end.append(self.h5_file['images/{}'.format(img_name)].attrs['exposure_end'])
                self.frame_exposure_time.append(self.h5_file['images/{}'.format(img_name)].attrs['exposure_time'])
        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = self.data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx

    def ts(self, index):
        return self.h5_file['events/ts'][index]

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for img_name in self.h5_file['images']:
            end_idx = self.h5_file['images/{}'.format(img_name)].attrs['event_idx']
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


class GoproEsimH52NpzDataset(BaseVoxelDataset):
    """

    """

    def get_frame(self, index):
        """
        discard the first and the last frame in GoproEsim
        """
        return self.h5_file['images']['image{:09d}'.format(index + 1)][:]

    def get_gt_frame(self, index):
        """
        discard the first and the last frame in GoproEsim
        """
        return self.h5_file['sharp_images']['image{:09d}'.format(index + 1)][:]

    def get_events(self, idx0, idx1):
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0  # -1 and 1
        return xs, ys, ts, ps

    def load_data(self, data_path, csv_path):
        self.data_sources = ('esim', 'ijrr', 'mvsec', 'eccd', 'hqfd', 'unknown')
        try:
            self.h5_file = h5py.File(data_path, 'r')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))
        pd_file = pd.read_csv(csv_path, header=None, names=['t', 'path'], dtype={'t': np.float64},
                                  engine='c')

        img_time_stamps = pd_file.t.values.astype(np.float64)
        img_names = pd_file.path.values

        if self.sensor_resolution is None:
            self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]

        print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['flow']) > 0

        self.t0 = self.h5_file['events/ts'][0]
        self.tk = self.h5_file['events/ts'][-1]
        self.num_events = self.h5_file.attrs["num_events"]


        self.frame_ts = list(img_time_stamps)
        self.img_names = list(img_names)
        self.num_frames = len(self.frame_ts)

        self.data_source_idx = -1


    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx

    def ts(self, index):
        return self.h5_file['events/ts'][index]

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for img_name in self.h5_file['images']:
            end_idx = self.h5_file['images/{}'.format(img_name)].attrs['event_idx']
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices


def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    """
    Binary search for a timestamp in an HDF5 event file, without
    loading the entire file into RAM
    @param dset The HDF5 dataset
    @param x The timestamp being searched for
    @param l Starting guess for the left side (0 if None is chosen)
    @param r Starting guess for the right side (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2;
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


