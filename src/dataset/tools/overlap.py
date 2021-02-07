def get_hist_overlap(pc1, pc2, sample_overlap_size=10000, hist_bin_length=100):
    # Params:
    #     pc1: point cloud 1 (np array with shape ([m, k1]))
    #     pc2: point cloud 2 (np array with shape ([n, k2]))
    #
    # k1 and k2 must contain at least x and y coordinates. 
    
    #
    # Returns:
    #     
    
    # define a data range
    pc_combined = np.concatenate((pc1, pc2))
    data_range = np.array(
        [[pc_combined[:, 0].min(), pc_combined[:, 0].max()],
        [pc_combined[:, 1].min(), pc_combined[:, 1].max()],
        [pc_combined[:, 2].min(), pc_combined[:, 2].max()]])
    
    bin_counts = [int((f[1]-f[0])/hist_bin_length) for f in data_range]

    del pc_combined  # save some mem
    
    # define bins based on data_range:
    x_bins = np.linspace(data_range[0][0], data_range[0][1], num=bin_counts[0])
    y_bins = np.linspace(data_range[1][0], data_range[1][1], num=bin_counts[1])
    z_bins = np.linspace(data_range[2][0], data_range[2][1], num=bin_counts[2])
    
    # Collect some number of points as overlap between these point clouds
    # build kd tree so we can search for points in pc2
    kd = kdtree._build(pc2[:, :3])

    # collect a sample of points in pc1 to query in pc2
    sample_overlap = np.random.choice(len(pc1), size=sample_overlap_size)
    pc1_sample = pc1[sample_overlap]

    # query pc1 sample in pc2. note that we want lots of nearby neighbors
    query = kdtree._query(kd, pc1_sample[:, :3], k=150, dmax=1)
    
    # Count the number of neighbors found at each query point
    counts = np.zeros((len(query), 1))
    for i in range(len(query)):
        counts[i][0] = len(query[i])

    # Append this to our sample
    pc1_sample_with_counts = np.concatenate((pc1_sample[:, :3], counts), axis=1)

    # this needs to be transformed such that the points (X, Y) occur in the
    # array `count` times. This will make histogram creation easier.
    rows = []
    for i in range(len(pc1_sample_with_counts)):
        row = pc1_sample_with_counts[i, :3]
        row = np.expand_dims(row, 0)
        if pc1_sample_with_counts[i, 2]:
            duplication = np.repeat(row, pc1_sample_with_counts[i, 3], axis=0)
            rows.append(duplication)
    
    pc1_sample_f = np.concatenate(rows, axis=0)
    
    # build histogram over data
    hist, edges = np.histogramdd(
        pc1_sample_f[:, :3], 
        bins=[x_bins, y_bins, z_bins])

    return (hist, edges), pc1_sample_f


def get_overlap_points(pc, hist_info, c, s=True):
    # Pull points out of `pc` from overlap information to be used in dataset
    # creation.
    #   `hist_info`: tuple (hist, bins) 
    #   `c` : the count of required overlap points to exist in a bin for it to
    #       to count as being "in the overlap." Higher values of c grab points 
    #       more likely to be in the overlap.

    
    indices = np.full(pc.shape[0], False, dtype=bool)
    process_list = []
    hist, (xedges, yedges, zedges) = hist_info
    
    def get_indices(t):
        i, j, k = t
        x1, x2 = xedges[i], xedges[i+1]
        y1, y2 = yedges[j], yedges[j+1]
        z1, z2 = zedges[k], zedges[k+1]
        
        # this is very slow :/
        new_indices = ((x1 <= pc[:, 0]) & (pc[:, 0] < x2) & 
                       (y1 <= pc[:, 1]) & (pc[:, 1] < y2) &
                       (z1 <= pc[:, 2]) & (pc[:, 2] < z2))
        
        return new_indices
        
    h_iter = np.array(np.meshgrid(
        np.arange(hist.shape[0]), 
        np.arange(hist.shape[1]),
        np.arange(hist.shape[2])
    )).T.reshape(-1, 3)
    
    for t in tqdm(h_iter, desc="building process list"):
        i, j, k = t
        if s and hist[i][j][k] > c:
            process_list.append(t)
        if not s and hist[i][j][k] < c:
            process_list.append(t)
            
    process_list = np.array(process_list)    
    for t in tqdm(process_list, desc="compiling indices"):
        indices = indices | get_indices(t)
    
    return indices