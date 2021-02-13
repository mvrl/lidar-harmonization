from src.config.project import Project

p = Project()

dublin_config = {
	"name": "dublin",

	# Creation settings
	"target_scan": '1',
    "igroup_size": 5,
    "igroup_sample_size": 500,
    "max_chunk_size": int(4e6),
    "max_n_size": 150,
    "scans_path": str(p.root / "dataset/dublin/npy/"),
    "save_path": str(p.root / "dataset/dublin/150"),
    "min_overlap_size": 200000,

    # Corruption
    'dorf_path': str(p.root / "dataset/dublin/dorf.json"),
    'mapping_path': str(p.root / "dataset/dublin/mapping.npy"),
    'max_intensity': 512,

    # global shift settings:
    'bounds_path': str(p.root / "dataset/dublin/bounds.npy"),
    'sig_floor': .3,
    'sig_center': .5,
    'sig_l': 100,
    'sig_s': .7
}
