from src.config.project import Project

p = Project()

dublin_config = {
	"name": "dublin",
	"target_scan": '1',
    "igroup_size": 5,
    "igroup_sample_size": 500,
    "max_chunk_size": int(4e6),
    "max_n_size": 150,
    "scans_path": str(p.root / "dataset/dublin/npy/"),
    "save_path": str(p.root / "dataset/dublin/150"),
    "min_overlap_size": 200000
}
