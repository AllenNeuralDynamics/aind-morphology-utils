import argparse
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Generator, Tuple, List, Optional

import imageio
import networkx as nx
import numpy as np
import s3fs
import zarr
from pydantic import BaseModel, Field, ValidationError
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from aind_morphology_utils.swc import NeuronGraph


class MovieConfig(BaseModel):
    zarr_path: str = Field(..., description="Path to the zarr dataset.")
    swc_path: str = Field(..., description="Path to the SWC file.")
    frame_dir: str = Field(..., description="Directory to save the frames in.")
    n_frames: int = Field(
        default=-1,
        description="Number of frames to generate. Use -1 to generate all "
        "frames along the longest path.",
    )
    mip_size: int = Field(
        default=0,
        description="Number of frames on either side of swc point to use for "
        "Maximum Intensity Projection.",
    )
    frame_size: Tuple[int, int] = Field(
        default=(512, 512), description="Frame size."
    )
    vmin: int = Field(
        default=0, description="Minimum intensity for rescaling."
    )
    vmax: int = Field(
        default=1000, description="Maximum intensity for rescaling."
    )
    max_workers: int | None = Field(
        default=None,
        description="Maximum number of workers for parallel processing.",
    )


class FrameGenerationStrategy:
    """Abstract class defining the protocol for frame generation strategies."""

    def generate_frames(
        self, coords: List[Tuple[int, int, int]], arr: zarr.core.Array
    ) -> Generator[np.ndarray, None, None]:
        """
        Abstract method to be implemented for generating frames.

        Parameters
        ----------
        coords : List[Tuple[int, int, int]]
            List of coordinates to generate frames.
        arr : zarr.core.Array
            The zarr array containing image data.

        Yields
        ------
        Generator[np.ndarray, None, None]
            A generator yielding frames as numpy arrays.
        """
        raise NotImplementedError


class MaxIntensityProjectionStrategy(FrameGenerationStrategy):
    """
    Strategy for generating frames using Maximum Intensity Projection.
    """

    def __init__(
        self,
        mip_size: int = 10,
        frame_size: Tuple[int, int] = (512, 512),
        vmin: int = 0,
        vmax: int = 1000,
    ):
        self.mip_size = mip_size
        self.frame_size = frame_size
        self.vmin = vmin
        self.vmax = vmax

    def generate_frames(
        self, coords: List[Tuple[int, int, int]], arr: zarr.core.Array
    ) -> Generator[np.ndarray, None, None]:
        """
        Generates frames using the Maximum Intensity Projection strategy.

        Parameters
        ----------
        coords : List[Tuple[int, int, int]]
            List of coordinates to generate frames.
        arr : zarr.core.Array
            The zarr array containing image data.

        Yields
        ------
        Generator[np.ndarray, None, None]
            A generator yielding processed frames as numpy arrays.
        """
        for x_center, y_center, z_center in coords:
            y_start, y_end = max(0, y_center - self.frame_size[0] // 2), min(
                arr.shape[3], y_center + self.frame_size[0] // 2
            )
            x_start, x_end = max(0, x_center - self.frame_size[1] // 2), min(
                arr.shape[4], x_center + self.frame_size[1] // 2
            )
            z_start, z_end = max(0, z_center - self.mip_size), min(
                arr.shape[2], z_center + self.mip_size + 1
            )

            chunk = arr[0, 0, z_start:z_end, y_start:y_end, x_start:x_end]
            mip = np.max(chunk, axis=0)
            rescaled_frame = rescale_intensity(
                mip, in_range=(self.vmin, self.vmax), out_range=np.uint16
            )
            yield rescaled_frame


class SplineInterpolationStrategy(FrameGenerationStrategy):
    # TODO
    pass


class FrameGenerator:
    """
    Generates frames using a specified frame generation strategy.
    """

    def __init__(self, strategy: FrameGenerationStrategy):
        self.strategy = strategy

    def generate(
        self,
        coords: List[Tuple[int, int, int]] | np.ndarray,
        arr: zarr.core.Array,
    ) -> Generator[np.ndarray, None, None]:
        """
        Generates frames based on the provided strategy.

        Parameters
        ----------
        coords : List[Tuple[int, int, int]]
            List of coordinates to generate frames.
        arr : zarr.core.Array
            The zarr array containing image data.

        Yields
        ------
        Generator[np.ndarray, None, None]
            A generator yielding frames as numpy arrays.
        """
        return self.strategy.generate_frames(coords, arr)


class MovieMaker:
    """
    Orchestrates the process of generating and saving frames to create a movie.
    """

    def __init__(self, frame_generator: FrameGenerator, frame_dir: str):
        self.frame_generator = frame_generator
        self.frame_dir = frame_dir

        if os.path.isdir(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)

    def write_frames(
        self, coords: List[Tuple[int, int, int]], arr: zarr.core.Array
    ):
        """
        Creates a movie by generating and saving frames.

        Parameters
        ----------
        coords : List[Tuple[int, int, int]]
            List of coordinates to generate frames.
        arr : zarr.core.Array
            The zarr array containing image data.
        """
        try:
            for i, frame in tqdm(
                enumerate(
                    self.frame_generator.generate(coords, arr),
                )
            ):
                frame_path = os.path.join(self.frame_dir, f"frame_{i:04d}.png")
                imageio.imwrite(frame_path, frame)
                logging.debug(f"Frame {i} saved")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

    def write_frames_parallel(
        self,
        coords: List[Tuple[int, int, int]] | np.ndarray,
        arr: zarr.core.Array,
        max_workers: int = None,
    ):
        """
        Creates a movie by generating and saving frames in parallel.

        Parameters
        ----------
        coords : List[Tuple[int, int, int]]
            List of coordinates to generate frames.
        arr : zarr.core.Array
            The zarr array containing image data.
        max_workers : int, optional
            The maximum number of threads to use for parallel processing.
        """
        chunks = np.array_split(coords, max_workers or os.cpu_count())

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            start_idx = 0
            for chunk in chunks:
                futures.append(
                    executor.submit(self._process_chunk, chunk, arr, start_idx)
                )
                start_idx += len(chunk)

            for future in tqdm(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing chunk: {e}")

    def _process_chunk(
        self,
        chunk: List[Tuple[int, int, int]] | np.ndarray,
        arr: zarr.core.Array,
        start_idx: int,
    ):
        """
        Processes a chunk of coordinates, generating and saving frames.

        Parameters
        ----------
        chunk : List[Tuple[int, int, int] or np.ndarray
            A chunk of coordinates to process.
        arr : zarr.core.Array
            The zarr array containing image data.
        start_idx : int
            The starting index of this chunk in the original list of coordinates.
        """
        for i, frame in enumerate(
            self.frame_generator.generate(chunk, arr), start=start_idx
        ):
            frame_path = os.path.join(self.frame_dir, f"frame_{i:04d}.png")
            imageio.imwrite(frame_path, frame)
            logging.debug(f"Frame {i} saved")


def _swc_to_coords(
    swc_path: str,
) -> Generator[Tuple[int, int, int], None, None]:
    """
    Generate coordinates from an SWC file.

    Parameters
    ----------
    swc_path : str
        The path to the SWC file.

    Yields
    ------
    Generator[Tuple[int, int, int], None, None]
        A generator yielding coordinates as tuples (x, y, z).
    """
    graph = NeuronGraph.from_swc(swc_path)
    longest_path = nx.dag_longest_path(graph)
    for node in longest_path:
        n = graph.nodes[node]
        yield int(n["x"]), int(n["y"]), int(n["z"])


def _open_zarr(
    zarr_path: str,
    dataset: str,
    client_kwargs: Optional[dict] = None,
    max_cache_size: int = 1024**3,
) -> zarr.core.Array:
    """
    Load zarr dataset.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr dataset.
    dataset : str
        Name of the dataset to load.
    client_kwargs : Optional[dict], optional
        Additional keyword arguments for the S3 client.
    max_cache_size : int, optional
        Maximum size of the cache.

    Returns
    -------
    zarr.core.Array
        The loaded zarr array.
    """
    s3 = s3fs.S3FileSystem(anon=False, client_kwargs=client_kwargs)
    store = s3fs.S3Map(root=zarr_path, s3=s3, check=False)
    cache = zarr.LRUStoreCache(store, max_size=max_cache_size)
    return zarr.group(store=cache, overwrite=False)[dataset]


def main():
    parser = argparse.ArgumentParser(description="Movie Maker Script")

    parser.add_argument(
        "--zarr_path",
        help="Path to the zarr dataset.",
        # default="s3://aind-open-data/exaSPIM_674185_2023-10-02_14-06-36_flatfield-correction_2023-11-17_00-33-17_fusion_2023-11-23_12-56-00/fused.zarr"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=-1,
        help="Number of frames to generate. Use -1 to generate all frames "
        "along the longest path.",
    )
    parser.add_argument(
        "--mip_size",
        type=int,
        default=0,
        help="Number of frames on either side of swc point to use for "
        "Maximum Intensity Projection.",
    )
    parser.add_argument(
        "--frame_size",
        nargs=2,
        type=int,
        default=[512, 512],
        help="Frame size as two integers (width height).",
    )
    parser.add_argument(
        "--frame_dir",
        help="Directory to save the frames in.",
        # default="frames-test"
    )
    parser.add_argument(
        "--vmin", type=int, default=0, help="Minimum intensity for rescaling."
    )
    parser.add_argument(
        "--vmax",
        type=int,
        default=1000,
        help="Maximum intensity for rescaling.",
    )
    parser.add_argument(
        "--swc_path",
        help="Path to the SWC file.",
        # default=r"C:\Users\cameron.arshadi\Downloads\N014-674185-dendrite-PG.swc"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of workers for parallel processing.",
    )
    parser.add_argument("--log_level", default="INFO", help="Logging level.")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    try:
        config = MovieConfig(
            zarr_path=args.zarr_path,
            n_frames=args.n_frames,
            mip_size=args.mip_size,
            frame_size=tuple(args.frame_size),
            frame_dir=args.frame_dir,
            vmin=args.vmin,
            vmax=args.vmax,
            swc_path=args.swc_path,
            max_workers=args.max_workers,
        )
    except ValidationError as e:
        logging.error(f"Configuration validation error: {e}")
        return

    logging.basicConfig(level=logging.INFO)

    # Load data and coordinates
    ds = _open_zarr(config.zarr_path, "0")

    if config.n_frames == -1:
        config.n_frames = len(list(_swc_to_coords(config.swc_path)))
    coords = list(_swc_to_coords(config.swc_path))[: config.n_frames]

    # Setup FrameGenerator and MovieMaker
    strategy = MaxIntensityProjectionStrategy(
        config.mip_size, config.frame_size, config.vmin, config.vmax
    )
    frame_generator = FrameGenerator(strategy)
    movie_maker = MovieMaker(frame_generator, config.frame_dir)

    # t0 = time.time()
    # movie_maker.write_frames(coords, ds)
    # t1 = time.time()
    # print(f"Serial processing took {t1 - t0} seconds")

    t0 = time.time()
    movie_maker.write_frames_parallel(coords, ds, config.max_workers)
    t1 = time.time()
    print(f"Parallel processing took {t1 - t0} seconds")


if __name__ == "__main__":
    main()
