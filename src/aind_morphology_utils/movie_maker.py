import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Generator, Tuple, List, Optional

import dask.array
import imageio
import networkx as nx
import numpy as np
import s3fs
import xarray
import zarr
from pydantic import BaseModel, Field, ValidationError
from scipy import interpolate
from dask_image.ndinterp import affine_transform
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
    cache_size: int = Field(
        default=1024**3,
        description="Maximum size of the Zarr cache. Defaults to 1GB.",
    )
    frame_generator: str = Field(
        default="mip",
        description="Frame generation strategy. Must be one of 'mip' or "
        "'spline'.",
    )
    smoothing: int = Field(
        default=1000,
        description="Smoothing for spline interpolation.",
    )


class SplineFitter:
    def __init__(self, k=3, resolution_scale=10, smoothing=200):
        self.resolution_scale = resolution_scale
        self.smoothing = smoothing
        self.k = k
        self._tck = None

    def fit(self, coords: List[Tuple[float, float, float]]) -> None:
        """
        Fit the spline to the given coordinates.

        Parameters
        ----------
        coords : List[Tuple[float, float, float]]
            The coordinates to fit the spline.
        """
        x, y, z = zip(*coords)
        self._tck, _ = interpolate.splprep(
            [x, y, z], k=self.k, s=self.smoothing
        )
        self.spline_length = self._spline_len(
            coords, self._tck, self.resolution_scale
        )

    def query(self, u: np.ndarray) -> np.ndarray:
        """
        Query the spline at given positions.

        Parameters
        ----------
        u : np.ndarray
            The positions along the spline to query.

        Returns
        -------
        np.ndarray
            The resampled coordinates.
        """
        x_fine, y_fine, z_fine = interpolate.splev(u, self._tck)
        resampled_coords = np.array(list(zip(x_fine, y_fine, z_fine)))
        return resampled_coords

    def sample(self, step_size: float = 1.0) -> np.ndarray:
        """
        Sample the spline at a specified step size.

        Parameters
        ----------
        step_size : float
            The step size for sampling.

        Returns
        -------
        np.ndarray
            The sampled coordinates.
        """
        # Determine the number of samples for your desired spacing
        num_samples = int(np.ceil(self.spline_length / step_size))
        return self.query(np.linspace(0, 1, num_samples))

    def _spline_len(
        self,
        coords: List[Tuple[float, float, float]],
        tck: Tuple,
        resolution_scale: int = 10,
    ):
        """
        Calculate the total length of the spline path in a vectorized manner.

        Parameters
        ----------
        coords : List[Tuple[float, float, float]]
            List of coordinates.
        tck : Tuple
            Spline representation obtained from scipy.interpolate.splprep.
        resolution_scale : int
            Scale factor to determine the resolution of the spline
            approximation.
            Increase this value to improve accuracy.

        Returns
        -------
        float
            Total length of the spline.
        """
        u = np.linspace(0, 1, len(coords) * resolution_scale)
        x, y, z = interpolate.splev(u, tck)

        distances = np.sqrt(
            np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2
        )
        spline_length = np.sum(distances)

        return spline_length


class FrameGenerator:
    """Abstract class defining the protocol for frame generation strategies."""

    def generate_frames(
        self,
        coords: List[Tuple[int, int, int]] | np.ndarray,
        arr: zarr.core.Array,
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


class MIPGenerator(FrameGenerator):
    """
    Strategy for generating frames using Maximum Intensity Projections.
    """

    def __init__(
        self,
        mip_size: int = 5,
        frame_size: Tuple[int, int] = (512, 512),
        vmin: int = 0,
        vmax: int = 1000,
    ):
        """
        Initializes the Maximum Intensity Projection strategy.

        Parameters
        ----------
        mip_size : int
            Number of frames on either side of the SWC point to include in the
            MIP.
        frame_size : Tuple[int, int]
            Size of the frame to be generated (width, height).
        vmin : int
            Minimum intensity value for rescaling.
        vmax : int
            Maximum intensity value for rescaling.
        """
        self.mip_size = mip_size
        self.frame_size = frame_size
        self.vmin = vmin
        self.vmax = vmax

    def generate_frames(
        self,
        coords: List[Tuple[int, int, int]] | np.ndarray,
        arr: zarr.core.Array,
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


class InterpolatingGenerator(FrameGenerator):
    def __init__(
        self,
        frame_size: Tuple[int, int] = (512, 512),
        interp_method: str = "linear",
        vmin: int = 0,
        vmax: int = 1000,
    ):
        """
        Initializes the interpolating frame generation strategy. This strategy
        allows for the generation of frames at non-integral coordinates.

        Parameters
        ----------
        frame_size : Tuple[int, int]
            Size of the frame to be generated (width, height).
        interp_method : str
            Interpolation method to use. Must be one of "linear", "nearest",
            "zero", "slinear", "quadratic", "cubic", "previous", "next".
        vmin : int
            Minimum intensity value for rescaling.
        vmax : int
            Maximum intensity value for rescaling.
        """
        self.frame_size = frame_size
        self.interp_method = interp_method
        self.vmin = vmin
        self.vmax = vmax

    def generate_frames(
        self,
        coords: List[Tuple[float, float, float]] | np.ndarray,
        arr: dask.array.Array,
    ) -> Generator[np.ndarray, None, None]:
        """
        Generates frames using spline interpolation strategy.

        Parameters
        ----------
        coords : List[Tuple[float, float, float]]
            List of floating-point coordinates to generate frames.
        arr : dask.array.Array
            The dask Array containing image data.

        Yields
        ------
        Generator[np.ndarray, None, None]
            A generator yielding processed frames as numpy arrays.
        """

        x_fine, y_fine, z_fine = zip(*coords)

        for x, y, z in zip(x_fine, y_fine, z_fine):
            frame = self._get_frame_dask(
                arr,
                (z, y, x),
                self.frame_size,
            )

            rescaled_frame = rescale_intensity(
                frame,
                in_range=(self.vmin, self.vmax),
                out_range=np.uint16,
            )

            yield rescaled_frame

    def _get_frame_xarray(
        self,
        arr: xarray.DataArray,
        center: Tuple[float, float, float],
        frame_size: Tuple[int, int],
    ):
        """
        Extract a 2D plane from a 5D volume using xarray interpolation.

        Parameters
        ----------
        arr : xarray.DataArray
            The 5D image volume as a xarray DataArray (t, c, z, y, x).
        center : tuple
            The (z, y, x) coordinates of the center of the 2D plane.
        frame_size : tuple
            The dimensions (height, width) of the output plane.

        Returns
        -------
        np.ndarray
            Extracted 2D plane as a numpy array.

        """
        # Calculate the frame boundaries considering the frame size
        y_bounds = max(0, center[1] - frame_size[0] / 2), min(
            arr.shape[3], center[1] + frame_size[0] / 2
        )
        x_bounds = max(0, center[2] - frame_size[1] / 2), min(
            arr.shape[4], center[2] + frame_size[1] / 2
        )

        frame = (
            arr.interp(
                y=np.linspace(y_bounds[0], y_bounds[1], frame_size[0]),
                x=np.linspace(x_bounds[0], x_bounds[1], frame_size[1]),
                z=center[0],
                method=self.interp_method,
            )
            .squeeze()
            .compute()
        )
        return frame

    def _get_frame_dask(
        self,
        arr: dask.array.Array,
        center: Tuple[float, float, float],
        frame_size: Tuple[int, int],
    ):
        """
        Extract a 2D plane from a 5D volume using dask.

        Parameters
        ----------
        arr : (dask.array.Array)
            The 5D image volume as a Dask array (t, c, z, y, x).
        center : (tuple)
            The (z, y, x) coordinates of the center of the 2D plane.
        frame_size : (tuple)
            The dimensions (height, width) of the output plane.

        Returns
        -------
        np.ndarray: Extracted 2D plane as a numpy array.
        """
        # Create a 5x5 identity matrix for the affine transformation
        matrix = np.eye(5)

        # Calculate the offset to position the center of the output frame
        # at the specified coordinates
        z_offset = center[0]
        y_offset = center[1] - frame_size[1] / 2
        x_offset = center[2] - frame_size[1] / 2
        translation = [0, 0, z_offset, y_offset, x_offset]

        # Apply the affine transformation
        transformed = affine_transform(
            arr,
            matrix=matrix,
            offset=translation,
            output_shape=(1, 1, 1, *frame_size),
            output_chunks=(1, 1, 1, *frame_size),
            order=1,
        )

        return transformed.squeeze().compute()


class MovieMaker:
    """
    Orchestrates the process of generating and saving frames to create a movie.
    """

    def __init__(self, frame_generator: FrameGenerator, frame_dir: str):
        """
        Initializes the movie maker.

        Parameters
        ----------
        frame_generator : FrameGenerator
           The frame generator to use for creating frames.
        frame_dir : str
           Directory where the frames will be saved.
        """
        self.frame_generator = frame_generator
        self.frame_dir = frame_dir

        os.makedirs(frame_dir, exist_ok=True)

    def write_frames(
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

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
                    logging.exception(f"Error processing chunk: {e}")

    def _process_chunk(
        self,
        chunk: np.ndarray,
        arr: zarr.core.Array,
        start_idx: int,
    ) -> None:
        """
        Processes a given chunk of coordinates, generating and saving frames.

        Parameters
        ----------
        chunk : np.ndarray
            A chunk of coordinates to process.
        arr : zarr.core.Array
            The zarr array containing image data.
        start_idx : int
            The starting index for frame naming in this chunk.
        """
        for i, frame in enumerate(
            self.frame_generator.generate_frames(chunk, arr), start=start_idx
        ):
            frame_path = os.path.join(self.frame_dir, f"frame_{i:05d}.png")
            imageio.imwrite(frame_path, frame)
            logging.debug(f"Frame {i} saved")


def _swc_to_coords(
    swc_path: str,
) -> np.ndarray:
    """
    Generate coordinates from an SWC file.

    Parameters
    ----------
    swc_path : str
        The path to the SWC file.

    Returns
    ------
    np.ndarray
        A numpy array of coordinates.
    """
    graph = NeuronGraph.from_swc(swc_path)
    longest_path = nx.dag_longest_path(graph)
    coords = []
    for node in longest_path:
        n = graph.nodes[node]
        coords.append([int(n["x"]), int(n["y"]), int(n["z"])])
    # remove duplicate nodes
    coords = np.array(coords)
    _, ind = np.unique(coords, axis=0, return_index=True)
    # Maintain input order
    return coords[np.sort(ind)]


def _open_zarr_s3(
    zarr_path: str,
    client_kwargs: Optional[dict] = None,
    max_cache_size: int = 1024**3,
) -> zarr.Group:
    """
    Load zarr dataset.

    Parameters
    ----------
    zarr_path : str
        Path to the zarr dataset.
    client_kwargs : Optional[dict], optional
        Additional keyword arguments for the S3 client.
    max_cache_size : int, optional
        Maximum size of the cache.

    Returns
    -------
    zarr.Group
        The loaded zarr group.
    """
    s3 = s3fs.S3FileSystem(anon=False, client_kwargs=client_kwargs)
    store = s3fs.S3Map(root=zarr_path, s3=s3, check=False)
    cache = zarr.LRUStoreCache(store, max_size=max_cache_size)
    return zarr.group(store=cache, overwrite=False)


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
    parser.add_argument(
        "--smoothing",
        type=int,
        default=1000,
        help="Smoothing for spline interpolation.",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=1024**3,
        help="Maximum size of the Zarr cache. Defaults to 1GB.",
    )
    parser.add_argument(
        "--generator", default="mip", help="Frame generation strategy."
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
            cache_size=args.cache_size,
            frame_generator=args.generator,
            smoothing=args.smoothing,
        )
    except ValidationError as e:
        logging.error(f"Configuration validation error: {e}")
        return

    logging.basicConfig(level=args.log_level)

    ds = _open_zarr_s3(config.zarr_path, max_cache_size=config.cache_size)["0"]

    coords = list(_swc_to_coords(config.swc_path))
    if config.n_frames == -1:
        config.n_frames = len(coords)
    coords = np.array(coords[: config.n_frames])

    if config.frame_generator == "spline":
        # TODO: no hardcode
        fitter = SplineFitter(
            k=3, resolution_scale=10, smoothing=config.smoothing
        )
        fitter.fit(coords)
        frame_step_size = 1.0  # increase for coarser (faster) sampling
        coords = fitter.sample(step_size=frame_step_size)[: config.n_frames]

        frame_generator = InterpolatingGenerator(
            config.frame_size,
            interp_method="linear",
            vmin=config.vmin,
            vmax=config.vmax,
        )

        # I'm convinced wrapping with a dask array is drastically slowing
        # down the interpolation, but our zarr data is not directly readable
        # by xarray.
        ds = dask.array.from_array(ds, chunks=ds.chunks)

    elif config.frame_generator == "mip":
        frame_generator = MIPGenerator(
            config.mip_size, config.frame_size, config.vmin, config.vmax
        )

    else:
        raise ValueError(
            f"Invalid frame generation strategy: {config.frame_generator}"
        )

    movie_maker = MovieMaker(frame_generator, config.frame_dir)

    t0 = time.time()
    movie_maker.write_frames(coords, ds, config.max_workers)
    t1 = time.time()
    logging.info(f"Parallel processing took {t1 - t0} seconds")


if __name__ == "__main__":
    main()
