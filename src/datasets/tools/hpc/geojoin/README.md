# geojoin 

Idea: Provide a list of polygons in some standard format (wky? geojson? or more) and extract individual LAZ files for each one.


Notes:

 - using points from Weilian Song's FARSA paper to get starte
   (/u/vul-d1/scratch/wso226/FARSA_V2/splits/split_backups/20/01_24)

Implementation ideas:

  - hash the file and use that as part of the time so we don't extract the same region multiple times? 

  - first filter based on the tile set?

  - limit it to radii no greater than the buffer (140 meters I think,
    https://kygeonet.ky.gov/kyfromabove/pdfs/Specs_LiDAR_Production.pdf) so we
only need to look at one tile? 

  - output as TF records?

Status: (it is just an idea at this point)

