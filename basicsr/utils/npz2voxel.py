import numpy as np
import os
import time

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
    
def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    # return [os.path.join(looproot, filename)
    #         for looproot, _, filenames in os.walk(rootdir)
    #         for filename in filenames if filename.endswith(suffix)]
    return [filename
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def main():
    
    dataroot = '/work/lei_sun/HighREV/val'
    dataroot = '/work/lei_sun/HighREV/train'

    voxel_root = dataroot + '/voxel'
    if not os.path.exists(voxel_root):
        os.makedirs(voxel_root)
    voxel_bins = 6
    blur_frames = sorted(recursive_glob(rootdir=os.path.join(dataroot, 'blur'), suffix='.png'))
    blur_frames = [os.path.join(dataroot, 'blur', blur_frame) for blur_frame in blur_frames]
    h_lq, w_lq = 1224, 1632
    event_frames = sorted(recursive_glob(rootdir=os.path.join(dataroot, 'event'), suffix='.npz'))
    event_frames = [os.path.join(dataroot, 'event', event_frame) for event_frame in event_frames]

    all_event_lists = []
    for i in range(len(blur_frames)):
        blur_name = os.path.basename(blur_frames[i])  # e.g., SEQNAME_00001.png
        base_name = os.path.splitext(blur_name)[0]   # Remove .png, get SEQNAME_00001
        event_list = sorted([f for f in event_frames if f.startswith(os.path.join(dataroot, 'event', base_name + '_'))])[:-1] # remove the last one because it is out of the exposure time
        all_event_lists.append(event_list)

    total_time = 0
    num_events_processed = 0
    total_events = len(all_event_lists) * len(all_event_lists[0])
    print("loop started")
    start_time = time.time()
    for event_list in all_event_lists:
        loop_start_time = time.time()

        events = [np.load(event_path) for event_path in event_list]
        parts = event_list[0].rsplit('_', 1)
        base_name = parts[0].split('/')[-1]
        print(base_name, 'start', event_list[0])
        all_quad_event_array = np.zeros((0,4)).astype(np.float32)
        for event in events:
            ### IMPORTANT: dataset mistake x and y !!!!!!!!
            ###            Switch x and y here !!!!
            y = event['x'].astype(np.float32)            
            x = event['y'].astype(np.float32)          
            t = event['timestamp'].astype(np.float32)
            p = event['polarity'].astype(np.float32)

            this_quad_event_array = np.concatenate((t,x,y,p),axis=1) # N,4
            all_quad_event_array = np.concatenate((all_quad_event_array, this_quad_event_array), axis=0)
        voxel = events_to_voxel_grid(all_quad_event_array, num_bins=voxel_bins, width=w_lq, height=h_lq, return_format='HWC')
        # print(voxel.dtype, voxel.shape) # float32, (1224, 1632, 6)

        voxel_path = os.path.join(voxel_root, base_name + '.npz')
        # print(f'saving to {voxel_path}')

        # voxel_path = base_name + '.npz'

        np.savez(voxel_path, voxel=voxel)

        loop_end_time = time.time()
        loop_duration = loop_end_time - loop_start_time
        total_time += loop_duration
        num_events_processed += len(events)
        
        print(f"Loop {num_events_processed}/{total_events} took {loop_duration:.2f} seconds.")
        print("")
        

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total time taken: {total_duration:.2f} seconds.")


if __name__ == '__main__':
    main()