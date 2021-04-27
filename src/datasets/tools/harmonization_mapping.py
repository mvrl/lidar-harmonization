import pandas as pd
import numpy as np
from pathlib import Path
from shutil import copyfile
from src.datasets.tools.transforms import GlobalShift
import code

class HarmonizationMapping:
    def __init__(self, config):

        scans_path = config['dataset']['scans_path']
        target_scan_num = config['dataset']['target_scan']
        harmonization_path = config['dataset']['harmonized_path']

        self.harmonization_path = Path(harmonization_path)
        self.harmonization_path.mkdir(exist_ok=True, parents=True)

        
        # 1. collect all scans
        scans = [str(f) for f in Path(scans_path).glob("*.npy")]

        # 2. select target scan(s)
        target_scan_path = Path(scans_path) / (target_scan_num+".npy")

        # copy to the harmonized path.
        # - one time this just didn't work. Deleting the copy and restarting the
        #     program seemed to work.
        if not (self.harmonization_path / (target_scan_num+".npy")).exists():
            if not config['dataset']['shift']:
                copyfile(
                    str(target_scan_path), 
                    str(self.harmonization_path / (target_scan_num+".npy")))
            else:
                # move this later?
                target = np.load(str(target_scan_path))
                G = GlobalShift(**config["dataset"])
                target = G(target)
                np.save(str(self.harmonization_path / (target_scan_num+".npy")), target)


        if not config['dataset']['create_new']:
            if (self.harmonization_path / "df.csv").exists():
                self.df = pd.read_csv((self.harmonization_path / "df.csv"), index_col=0)
            else:
                exit(f"Couldn't find HM csv file at {self.harmonization_path / 'df.csv'}")
        else:
            if (self.harmonization_path / "df.csv").exists():
                # store a backup just in case
                copyfile(str(self.harmonization_path / "df.csv"),
                         str(self.harmonization_path / "df_old.csv")
                    )
            # initialize the df
            self.df = pd.DataFrame(
                columns=["source_scan", 
                         "harmonization_target", 
                         "source_scan_path", 
                         "harmonization_scan_path", 
                         "processing_stage"])
            
            self.df.source_scan_path = scans
            self.df.harmonization_target = [None]*len(scans)
            self.df.harmonization_scan_path = [None]*len(scans)
            self.df.source_scan = [int(Path(f).stem) for f in scans]
            self.df.processing_stage = [0]*len(scans)

            # setup target scan
            target_scan_num = int(target_scan_num)
            self.df.loc[self.df.source_scan == target_scan_num, "harmonization_target"] = int(target_scan_num)
            self.df.loc[self.df.source_scan == target_scan_num, "harmonization_scan_path"] = str(self.harmonization_path / (str(target_scan_num)+".npy"))
            self.df.loc[self.df.source_scan == target_scan_num, "processing_stage"] = 2

            # need processing stages for each source. Sources start at stage 0.
            #   Stage 0 means that the sources haven't been identified as having
            #   any overlap with a target scan. By extension, they don't have
            #   examples in the dataset, nor do they have the harmonized
            #   version. A source scan enters stage one after overlap in the 
            #   scan has been detected and examples have been added to the
            #   dataset. After a model is trained with the new dataset, this 
            #   source scan can then be harmonized with the target. The source
            #   scan enters stage 2 after it has been harmonized. This source
            #   scan can now be used as a target scan to search for overlap
            #   regions with other soure scans. After all sources have been 
            #   checked for overlap, the stage 2 source scan can then be moved
            #   to stage 3 (done). Stage 3 scans do not have to be used again. 

            # The harmonization is process is finished when all scans are stage
            # 2 or higher OR all scans are stage 3 or stage 0. 

            self.save()

    def __getitem__(self, source_scan_num):
        # return the entire row for a source scan num (float or int or str)
        return self.df.loc[self.df.source_scan == int(source_scan_num)]

    def __len__(self):
        return len(self.df)

    def save(self):
        self.df.to_csv(self.harmonization_path / "df.csv")

    def done(self):
        # there are two conditions for being done. If either are not satisified,
        #  then the whole process is not finished. The first condition is that 
        #  all sources must be harmonized (all scans are stage 2 and above). In
        #  the event that a scan does not contain enough overlap to reach stage
        #  1, all stage 2 and above scans will be harmonized to stage 3 while 
        #  searching for overlap, so there will be no stage 1 or stage 2 sources
        #  remaining. 

        # All scans are harmonized
        cond1 = ((1 not in self.df.processing_stage.values) and 
                 (0 not in self.df.processing_stage.values))
        
        # All scans are harmonized except for stage 0 scans which don't have 
        #   any reasonable overlap
        cond2 = ((2 not in self.df.processing_stage.values) and
                 (1 not in self.df.processing_stage.values))

        return cond1 or cond2

    def add_target(self, source_scan_num, harmonization_target_num):
        self.df.loc[self.df.source_scan == int(source_scan_num), "harmonization_target"] = harmonization_target_num
        self.save()

    def incr_stage(self, source_scan_num):
        self.df.loc[self.df.source_scan == int(source_scan_num), "processing_stage"] += 1
        self.save()

    def get_stage(self, stage_num):
        return self.df.loc[self.df.processing_stage == int(stage_num)].source_scan.values.tolist()

    def add_harmonized_scan_path(self, source_scan_num):
        self.df.loc[self.df.source_scan == int(source_scan_num), "harmonization_scan_path"] = str(self.harmonization_path / (str(source_scan_num)+".npy"))
        self.save()

    def print_mapping(self):
        print("Final Mapping:")
        for idx, row in self.df.iterrows():
            print(f"{row.source_scan}: {row.harmonization_target}")

