from pathlib import Path
from src.config.config import project_info

class Project:
    def __init__(self):
        self.config = project_info

        # setup paths
        self.root = Path(self.config['root'])


if __name__ == "__main__":
    project = Project()

    print(project.config)
    print(project.root)
        
