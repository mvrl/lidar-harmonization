import numpy as np

def checkerboard(t1, t2, width=8):
    # produce a new tile that checkerboards the fixed tile and the ground truth tile
    
    # create structure
    cb_t1 = t1.copy()
    cb_t2 = t2.copy()
    
    # range for X and Y
    x_min, x_max = cb_t1[:, 0].min(), cb_t1[:, 0].max()
    y_min, y_max = cb_t1[:, 1].min(), cb_t1[:, 1].max()
    
    x_dist = x_max - x_min
    y_dist = y_max - y_min
    
    x_step_size = x_dist // width
    y_step_size = y_dist // width
    
    # create bins
    xr = np.arange(x_min, x_max, x_step_size)
    yr = np.arange(y_min, y_max, y_step_size)
    
    xr = np.concatenate((xr, np.array([x_max])))
    yr = np.concatenate((yr, np.array([y_max])))

    t1_indices = np.full(cb_t1.shape[0], False, dtype=bool)
    t2_indices = np.full(cb_t2.shape[0], False, dtype=bool)

    for i in range(len(xr)-1):
        for j in range(len(yr)-1):
            if (i + j) % 2 == 1:
                # 'white' tile --> t2
                new_indices = (
                    (cb_t2[:, 0] < xr[i+1]) &
                    (cb_t2[:, 0] >= xr[i]) &
                    (cb_t2[:, 1] < yr[j+1]) &
                    (cb_t2[:, 1] >= yr[j])
                )
                
                t2_indices = t2_indices | new_indices
                
            else:
                # 'black' tile --> gt
                new_indices = (
                    (cb_t1[:, 0] < xr[i+1]) &
                    (cb_t1[:, 0] >= xr[i]) &
                    (cb_t1[:, 1] < yr[j+1]) &
                    (cb_t1[:, 1] >= yr[j])
                )
                
                t1_indices = t1_indices | new_indices
                    

    return cb_t1[t1_indices], cb_t2[t2_indices]
