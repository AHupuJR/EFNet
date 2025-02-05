import numpy as np
import torch


def events_to_voxel_grid(events, num_bins, width, height, return_format='CHW'):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param return_format: 'CHW' or 'HWC'
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    # print('DEBUG: voxel.shape:{}'.format(voxel_grid.shape))

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    # print('last stamp:{}'.format(last_stamp))
    # print('max stamp:{}'.format(events[:, 0].max()))
    # print('timestamp:{}'.format(events[:, 0]))
    # print('polarity:{}'.format(events[:, -1]))

    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT # 
    ts = events[:, 0]
    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins # [True True ... True]
    # print('x max:{}'.format(xs[valid_indices].max()))
    # print('y max:{}'.format(ys[valid_indices].max()))
    # print('tix max:{}'.format(tis[valid_indices].max()))

    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width  ## ! ! !
            + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    if return_format == 'CHW':
        return voxel_grid
    elif return_format == 'HWC':
        return voxel_grid.transpose(1,2,0)

def voxel_norm(voxel):
    """
    Norm the voxel

    :param voxel: The unnormed voxel grid
    :return voxel: The normed voxel grid
    """
    nonzero_ev = (voxel != 0)
    num_nonzeros = nonzero_ev.sum()
    # print('DEBUG: num_nonzeros:{}'.format(num_nonzeros))
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = voxel.sum() / num_nonzeros
        stddev = torch.sqrt((voxel ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        voxel = mask * (voxel - mean) / stddev

    return voxel



def filter_event(x,y,p,t, s_e_index=[0,6]):
    '''
    s_e_index: include both left and right index
    '''
    t_1=t.squeeze(1)
    uniqw, inverse = np.unique(t_1, return_inverse=True)
    discretized_ts = np.bincount(inverse)
    index_exposure_start = np.sum(discretized_ts[0:s_e_index[0]])
    index_exposure_end = np.sum(discretized_ts[0:s_e_index[1]+1])
    x_1 = x[index_exposure_start:index_exposure_end]
    y_1 = y[index_exposure_start:index_exposure_end]
    p_1 = p[index_exposure_start:index_exposure_end]
    t_1 = t[index_exposure_start:index_exposure_end]
    
    return x_1, y_1, p_1, t_1

