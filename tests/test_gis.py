from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cogmap_sim.gis import import_shapefiles, load_graph_json, save_graph_json


class GISTests(unittest.TestCase):
    def test_shp_topology_inference_and_round_trip(self) -> None:
        try:
            import shapefile
        except ImportError:
            self.skipTest("pyshp optional dependency is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roads = str(root / "roads")
            writer = shapefile.Writer(roads, shapeType=shapefile.POLYLINE)
            writer.field("ROAD_ID", "C")
            writer.field("STATUS", "C")
            writer.line([[[0, 0], [1, 0]]]); writer.record("E1", "OPEN")
            writer.line([[[1, 0], [2, 0]]]); writer.record("E2", "CLOSED")
            writer.close()
            aois = str(root / "aois")
            writer = shapefile.Writer(aois, shapeType=shapefile.POINT)
            writer.field("AOI_ID", "C"); writer.field("FUNCTION", "C")
            writer.point(0, 0); writer.record("HOME", "HOME")
            writer.point(2, 0); writer.record("WORK", "WORK")
            writer.close()
            graph = import_shapefiles(roads + ".shp", aois + ".shp", snap_tolerance=0.01)
            self.assertEqual(len(graph.nodes), 3)
            self.assertEqual(len(graph.edges), 2)
            self.assertTrue(all(aoi.feature_kind == "POI" for aoi in graph.aois.values()))
            target = root / "map.json"
            save_graph_json(graph, target)
            loaded = load_graph_json(target)
            self.assertEqual(set(loaded.edges), {"E1", "E2"})
            self.assertTrue(all(aoi.feature_kind == "POI" for aoi in loaded.aois.values()))


if __name__ == "__main__":
    unittest.main()
