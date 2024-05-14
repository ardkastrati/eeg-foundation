import os
import sys
import json
from math import ceil
import sys
from parallel_preprocessor import Preprocessor


def main(index_chunk, idx):
    preparator = Preprocessor(
        index_store_dir="/itet-stor/maxihuber/net_scratch/stuff/cleaning/final2",
        num_threads=1,
    )
    preparator.preprocess_chunk(index_chunk, idx)


if __name__ == "__main__":
    index_path = str(sys.argv[1])
    idx = int(sys.argv[2])
    num_chunks = int(sys.argv[3])

    with open(index_path, "r") as file:
        index = json.load(file)

    # on process 0: collect paths in foundation_prepared that are not contained in the index
    if idx == 0:
        base_dir = "/itet-stor/maxihuber/deepeye_storage/foundation_prepared/"
        paths = os.listdir(base_dir)
        path_dict = {os.path.join(base_dir, path): False for path in paths}

        for index_element in index:
            path_dict[index_element["path"]] = True

        missing_paths = sorted(
            [path for path in path_dict.keys() if not path_dict[path]]
        )
        print("Skipping", len(missing_paths), "in foundation_prepared")

        missing_paths_store_path = f'/itet-stor/maxihuber/net_scratch/stuff/cleaning/missing_paths_{os.getenv("SLURM_ARRAY_JOB_ID")}.txt'
        with open(missing_paths_store_path, "w") as file:
            for path in missing_paths:
                file.write(path + "\n")

    # For testing: process a sublist of the index
    # index = index[:10]

    # split the index file into num_threads many groups
    chunk_size = ceil(len(index) / num_chunks)
    index_chunk = index[idx * chunk_size : (idx + 1) * chunk_size]

    print(f"Processing [{idx * chunk_size}, {(idx + 1) * chunk_size})", file=sys.stderr)

    main(index_chunk, idx)
