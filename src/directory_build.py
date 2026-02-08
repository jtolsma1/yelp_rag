import os
import shutil
from pathlib import Path
import src.config as config

class BuildDirectoryStructure:

    def __init__(self,
                sampled_data_path = None,
                processed_data_path = None,
                index_data_path = None,
                index_extensions = None,
                ):
        """
        Defaults to all parameters as set in config.py; overrides parameters when stated in function call.
        @param sampled_data_path_reviews: file path for storing data sampled from the Yelp dataset
        @param processed_data_path: file path for storing data processed through the RAG engine
        @param index_data_path: file path for storing review chunk index and metadata files
        @param index_extensions: file type extensions for index and metadata files
        """

        defaults = {
            "sampled_data_path":config.DATA_DIR_SAMP,
            "processed_data_path":config.DATA_DIR_PROC,
            "index_data_path":config.INDEX_DIR,
            "index_extensions":config.INDEX_EXTENSIONS
        }

        overrides = {
            "sampled_data_path": sampled_data_path,
            "processed_data_path": processed_data_path,
            "index_data_path": index_data_path,
            "index_extensions":index_extensions
        }

        for name, default in defaults.items():
            value = overrides[name] if overrides[name] is not None else default
            setattr(self, name, value)


    def build_data_directories(self):
        """
        Empties the sampled data and processed data directories for a fresh RAG run
        """
        for dir in (self.sampled_data_path,self.processed_data_path):
            dir = Path(dir)
            if dir.exists():
                shutil.rmtree(dir)
            os.mkdir(dir)
    

    def clean_index_directory(self):
        """
        Removes index and metadata files from the index directory while keeping other files (README.md) intact
        """
        for file in Path(config.INDEX_DIR).iterdir():
            if file.is_file():
                if self.index_extensions is None or file.suffix in self.index_extensions:
                    file.unlink()


    def run_build(self):
        """
        Executes all directory building and cleaning operations
        """
        self.build_data_directories()
        self.clean_index_directory()

