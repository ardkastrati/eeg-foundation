## ==== apply PrepPipeline to elements in index ====

import os
import numpy as np
import pandas as pd
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from math import ceil
import json
import time
import mne
import logging

from utils import create_raw, find_bad_by_prep


class Preprocessor:
    def __init__(
        self,
        index_store_dir: str,
        num_threads: int,
    ):
        self.index_store_dir = index_store_dir
        self.num_threads = num_threads

        # Set the logging level to ERROR to avoid excess printing
        logging.basicConfig(level=logging.ERROR)
        pyprep_logger = logging.getLogger("pyprep")
        pyprep_logger.setLevel(logging.ERROR)
        numpy_logger = logging.getLogger("numpy")
        numpy_logger.setLevel(logging.ERROR)
        mne.set_log_level("ERROR")

    def preprocess_chunk(self, index_chunk, idx):

        index_chunk_new = []

        for i, index_element in enumerate(index_chunk):

            print("=" * 100)
            print(index_element["path"])

            ## == load ==
            if index_element["path"].endswith(".edf"):
                print("Don't prep-pipeline .edf data")
                bad_channels = None

            elif index_element["path"].endswith("pkl"):
                with open(index_element["path"], "rb") as file:
                    file_data = pd.read_pickle(file)
                raw = create_raw(
                    data=file_data,
                    ch_names1=index_element["channels"],
                    ch_names2=index_element["channels_mapped"],
                    sr=index_element["sr"],
                )
                ## == pre-process .pkl files ==
                try:
                    bad_raw = find_bad_by_prep(
                        raw.copy(),
                        montage_kind=index_element["montage"],
                        grid_frequency=index_element["GridFrequency"],
                    )
                    bad_channels = list(bad_raw.info["bads"])
                except Exception as e:
                    print("An error occurred:", e, "for file", index_element["path"])
                    bad_channels = None
            else:
                assert False, "Invalid file path."

            ## == Add "PrepPipeline" column (containing bad_channels) ==
            index_element["PrepPipeline"] = bad_channels

            index_chunk_new.append(index_element)

            ## == dump index every 1_000 iterations ==
            if i % 1_000 == 0:
                store_path = os.path.join(
                    self.index_store_dir, f"index_with_bad_channels_{idx}_i={i}.json"
                )
                with open(store_path, "w") as file:
                    json.dump(index_chunk_new, file, indent=4)
                print(f"Dumped file, idx = {idx}, i = {i}")

            ## == iterate ==

        # Final store
        store_path = os.path.join(
            self.index_store_dir, f"index_with_bad_channels_{idx}.json"
        )
        with open(store_path, "w") as file:
            json.dump(index_chunk_new, file, indent=4)
        print(f"Dumped file, idx = {idx}")

        return f"Completed chunk {idx}"


if __name__ == "__main__":
    ## ==== here, we could parallelize (num_threads) ====
    preprocessor = Preprocessor(
        index_store_dir="/itet-stor/maxihuber/net_scratch/stuff/cleaning/final2",
        num_threads=1,
    )
    starting_time = time.time()
    preprocessor.run()
    print("overall, took", time.time() - starting_time, "seconds.")
