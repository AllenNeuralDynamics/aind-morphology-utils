import logging
import os
from pathlib import Path

import numpy as np
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from allensdk.core.structure_tree import StructureTree
from allensdk.core.swc import Morphology, read_swc

_LOGGER = logging.getLogger(__name__)


class CCFMorphologyMapper:
    """Class for mapping CCF region information to a Morphology object"""

    def __init__(
        self,
        reference_space_key: str = "annotation/ccf_2017",
        resolution: int = 25,
        cache_dir: str = os.getcwd(),
    ):
        """
        Class constructor.

        Parameters
        ----------
        reference_space_key : str, optional
            The reference space key to use, defaults to 'annotation/ccf_2017'.
        resolution : int, optional
            The resolution of the reference space, defaults to 25 microns.
        cache_dir : str, optional
            The directory to use for caching, defaults to the current working directory.
        """
        self._ref_space_cache = ReferenceSpaceCache(
            resolution,
            reference_space_key,
            manifest=Path(cache_dir) / "manifest.json",
        )
        try:
            (
                self.volume,
                self.meta,
            ) = self._ref_space_cache.get_annotation_volume()
        except MemoryError:
            _LOGGER.error(
                f"Not enough memory available to read the {resolution} um annotation volume."
                f"Please try using a lower resolution."
            )
            raise
        # We will invert this matrix later to map micron coordinates to voxel positions
        self.direction_matrix = self.meta["space directions"]

    def annotate_morphology(
        self, morphology: Morphology, structure_graph_id: int = 1
    ) -> None:
        """
        Annotates a Morphology object with CCF region information.

        Parameters
        ----------
        morphology : Morphology
            The Morphology object to annotate.
        structure_graph_id : int, optional
            The structure graph ID to use, defaults to 1 (adult mouse).
        """
        tree: StructureTree = self._ref_space_cache.get_structure_tree(
            structure_graph_id=structure_graph_id
        )
        inv_dir_mat = np.linalg.inv(self.direction_matrix)
        for c in morphology.compartment_list:
            pos = np.array([c["x"], c["y"], c["z"]])
            pos_px = np.dot(pos, inv_dir_mat).astype(int)
            try:
                ccf_region_id = self.volume[tuple(pos_px)]
            except IndexError:
                # ID of 0 indicates point is outside CCF space
                # This can happen for e.g., poorly registered tracings, or for
                # neurons that project to the spinal cord,
                # or other areas not represented in the CCF space.
                ccf_region_id = 0
            structure = tree.get_structures_by_id([ccf_region_id])[0]
            if structure is not None:
                c["allenInformation"] = structure


if __name__ == "__main__":
    mapper = CCFMorphologyMapper(resolution=25)
    print(mapper.volume.nbytes / 2**20)
    swc = r"C:\Users\cameron.arshadi\Desktop\exaSPIM_609281_2022-11-03_13-49-18\ccf_coords\Neuron_02_2022-11-03.swc_ccf10.swc"
    morph = read_swc(swc)
    mapper.annotate_morphology(morph)
