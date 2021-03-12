

# Generate harmonization mapping table (hm) to direct the flow of work.
#   This is pretty simple and doesn't need SLURM or anytihng.



# Figure out which scans overlap with the target. Update hm. 
#
#   Probably simple in the initial case. Could create a job array where each
#   job idx is the source scan to check against the target scan. 
#
#   How would this work for intermediate cases? Since we have multiple targets
#   and multiple source scans? Would it be a seperate set of jobs for each 
#   target? 
# 
#   Seperating this step from dataset creation involves repeat work of finding
#   the overlap indices unless you save that information somewhere on the disk.
#   I guess you could do this for outside the overlap region as well. 
#   

# Create the dataset. Since we already have a listing of which scans are
#   overlapping, this is just a matter of loading the scans, getting the overlap
#   indices, and then performing the queries followed by saving all the valid 
#   neighborhoods. 
#
#   With the non-overlapping scans part out of the way, this should be trivial
#   to put on LCC. If overlap regions are on the disk, each task can be given a
#   region to turn into dataset examples.


# Train the model on the newly created dataset.
#
#   I think this has to be just one job. This was fairly slow previously.
#

# Harmonize the current group of sources with the new model. 
#
#   GPU and CPU intensive task. What's the best way to go about doing this?