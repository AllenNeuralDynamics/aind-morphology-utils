import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import zarr

from aind_morphology_utils.movie_maker import (
    MovieConfig,
    FrameGenerationStrategy,
    MaxIntensityProjectionStrategy,
    FrameGenerator,
    MovieMaker,
)


class TestMovieConfig(unittest.TestCase):
    def test_config(self):
        config = MovieConfig(
            zarr_path="test_zarr_path",
            swc_path="test_swc_path",
            frame_dir="test_frames",
            n_frames=100,
            mip_size=10,
            frame_size=(512, 512),
            vmin=0,
            vmax=1000,
        )
        self.assertEqual(config.zarr_path, "test_zarr_path")
        self.assertEqual(config.swc_path, "test_swc_path")
        self.assertEqual(config.frame_dir, "test_frames")
        self.assertEqual(config.n_frames, 100)
        self.assertEqual(config.mip_size, 10)
        self.assertEqual(config.frame_size, (512, 512))
        self.assertEqual(config.vmin, 0)
        self.assertEqual(config.vmax, 1000)


class TestFrameGenerationStrategy(unittest.TestCase):
    def test_generate_frames(self):
        strategy = FrameGenerationStrategy()
        with self.assertRaises(NotImplementedError):
            list(strategy.generate_frames([], None))


class TestMaxIntensityProjectionStrategy(unittest.TestCase):
    def test_generate_frames(self):
        strategy = MaxIntensityProjectionStrategy(frame_size=(128, 128))
        coords = [(100, 100, 10)]
        arr = zarr.array(np.random.rand(1, 1, 20, 512, 512))
        frames = list(strategy.generate_frames(coords, arr))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, (128, 128))


class TestFrameGenerator(unittest.TestCase):
    def test_generate(self):
        strategy = MaxIntensityProjectionStrategy(
            mip_size=1, frame_size=(64, 64), vmin=0, vmax=100
        )
        generator = FrameGenerator(strategy)

        mock_arr = zarr.array(np.random.rand(1, 1, 128, 128, 128))
        mock_coords = [(64, 64, 64)]

        frames = list(generator.generate(mock_coords, mock_arr))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].shape, (64, 64))


class TestMovieMaker(unittest.TestCase):
    def setUp(self):
        self.mock_generator = MagicMock()
        self.mock_generator.generate.return_value = [
            np.zeros((512, 512)) for _ in range(5)
        ]

        self.movie_maker = MovieMaker(self.mock_generator, "test_frames")

    @patch("aind_morphology_utils.movie_maker.os.path.isdir")
    @patch("aind_morphology_utils.movie_maker.os.makedirs")
    @patch("aind_morphology_utils.movie_maker.imageio.imwrite")
    def test_write_frames(self, mock_imwrite, mock_makedirs, mock_isdir):
        mock_isdir.return_value = False

        mock_arr = zarr.array(np.random.rand(1, 1, 5, 5, 5))
        mock_coords = [(2, 2, 2)]

        self.movie_maker.write_frames(mock_coords, mock_arr)
        self.assertEqual(mock_imwrite.call_count, 5)

    @patch("aind_morphology_utils.movie_maker.os.path.isdir")
    @patch("aind_morphology_utils.movie_maker.os.makedirs")
    @patch("aind_morphology_utils.movie_maker.imageio.imwrite")
    def test_write_frames_parallel(
        self, mock_imwrite, mock_makedirs, mock_isdir
    ):
        mock_isdir.return_value = False

        mock_arr = zarr.array(np.random.rand(1, 1, 5, 5, 5), dtype=np.uint16)
        mock_coords = [(2, 2, 2)]

        self.movie_maker.write_frames_parallel(
            mock_coords, mock_arr, max_workers=1
        )
        self.assertEqual(mock_imwrite.call_count, 5)


if __name__ == "__main__":
    unittest.main()
