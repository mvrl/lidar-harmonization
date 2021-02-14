# The tile index was made by navigating to ftp://ftp.kymartian.ky.gov/kyaped/LAZ
# and then (in firefox) right click -> view page source. This produces a text
# file with all lots of information, but primarily there is a column with all 
# the tile names. This can be used to setup a parallel download which can 
# drastically speed up aquisition time. 

# There is, however, tons of garbage data that can be removed, making this 
# more efficient to store. This is a short script that pares this file down.

index_file_name = "list.txt"
new_name = "list_new.txt"
files = []
with open(index_file_name, "r") as f:
    lines = f.readlines()

for l in lines:
    s = l.split(" ")

    if s[1][-4:-1] == "laz":
        filename = s[1].split("\"")[1]
        files.append(filename)

with open(new_name, "w") as f:
    for laz_filename in files:
        f.write(laz_filename)
        f.write("\n")



