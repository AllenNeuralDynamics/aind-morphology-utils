import unittest

from allensdk.core.swc import read_swc

from aind_morphology_utils.ccf_annotation import CCFMorphologyMapper
from aind_morphology_utils.utils import read_json
from aind_morphology_utils.writers import MouseLightJsonWriter
from tests.utils import (
    get_test_swc_path,
    get_test_resources_dir,
    get_test_json_path,
    dict_equal,
)


class TestMappingAndJsonCreation(unittest.TestCase):
    def test_mapper_and_dict_builder(self):
        morph = read_swc(str(get_test_swc_path()))
        mapper = CCFMorphologyMapper(
            resolution=10, cache_dir=str(get_test_resources_dir())
        )
        mapper.annotate_morphology(morph)
        id_string = "AA0983"
        sample = {
            "date": "2018-08-01T17:24:59.000Z",
            "strain": "Sim1-Cre"
        }
        label = {
            "virus": "PHP-eB-CAG-FRT-rev-3xGFP+PHP-eB-CAG-flex-rev-Flpo",
            "fluorophore": "Immunolabeled with anti-GFP, Alexa-488"
        }
        comment = (
            "Downloaded 2023/06/20. Please consult Terms-of-Use at https://mouselight.janelia.org when referencing "
            "this reconstruction."
        )
        doi = "10.25378/janelia.7804508"
        d = MouseLightJsonWriter._build_dict(
            morph,
            id_str=id_string,
            sample=sample,
            label=label,
            comment=comment,
            doi=doi
        )
        expected_d = read_json(get_test_json_path())
        self.assertTrue(dict_equal(d, expected_d, rel_tol=1e-4))


if __name__ == "__main__":
    unittest.main()
