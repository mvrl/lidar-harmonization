Idea: index the data so some queries are faster

https://www.cs.unc.edu/~isenburg/lastools/download/lasindex_README.txt

Looks like it will take around 2-3 hours to index Kentucky

MAYBE:

 - flush output so we can see job progress w/o having to `ls` the dirs
 - make it only redo work if necessary
 - use a queue system so we can have multiple compute nodes?
 - force the data to be in the correct coordinate system first?

