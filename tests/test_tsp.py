import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tsp_solver import (
    DEFAULT_DATA,
    distance_matrix,
    held_karp,
    heuristic_tour,
    load_cities,
    plot_route,
    route_names,
    save_results,
    tour_cost,
    validate_tour,
)


class TSPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cities = load_cities(DEFAULT_DATA)
        cls.distances = distance_matrix(cls.cities)

    def test_dataset(self):
        self.assertEqual(len(self.cities), 14)
        self.assertEqual(self.cities["name"].nunique(), 14)
        self.assertTrue(self.cities["latitude"].between(21, 26).all())
        self.assertTrue(self.cities["longitude"].between(119, 123).all())

    def test_distance_matrix(self):
        self.assertEqual(self.distances.shape, (14, 14))
        self.assertTrue(np.allclose(np.diag(self.distances), 0))
        self.assertTrue(np.allclose(self.distances, self.distances.T))
        taipei = self.cities.index[self.cities["name"] == "Taipei"][0]
        kaohsiung = self.cities.index[self.cities["name"] == "Kaohsiung"][0]
        self.assertGreater(self.distances[taipei, kaohsiung], 250)
        self.assertLess(self.distances[taipei, kaohsiung], 400)

    def test_held_karp_on_a_square(self):
        square = np.array(
            [
                [0.0, 1.0, 1.414, 1.0],
                [1.0, 0.0, 1.0, 1.414],
                [1.414, 1.0, 0.0, 1.0],
                [1.0, 1.414, 1.0, 0.0],
            ]
        )
        result = held_karp(square)
        self.assertAlmostEqual(result.cost, 4.0, places=3)
        self.assertTrue(validate_tour(result.tour, 4))
        self.assertAlmostEqual(tour_cost(result.tour, square), 4.0, places=3)

    def test_exact_tour_is_complete(self):
        result = held_karp(self.distances)
        self.assertTrue(validate_tour(result.tour, 14))
        self.assertAlmostEqual(
            result.cost, tour_cost(result.tour, self.distances), places=2
        )
        self.assertGreater(result.cost, 500)

    def test_heuristic_is_valid_and_not_better_than_exact(self):
        exact = held_karp(self.distances)
        heuristic = heuristic_tour(self.distances)
        self.assertTrue(validate_tour(heuristic.tour, 14))
        self.assertGreaterEqual(heuristic.cost + 1e-6, exact.cost)

    def test_route_names(self):
        result = held_karp(self.distances)
        names = route_names(result.tour, self.cities)
        self.assertEqual(len(names), 14)
        self.assertIn("Taipei", names)
        self.assertIn("Kaohsiung", names)

    def test_plot_and_json_exports(self):
        result = held_karp(self.distances)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plot_route(self.cities, result.tour, result.cost, output / "route.png")
            save_results([result], self.cities, output / "summary.json")
            self.assertGreater((output / "route.png").stat().st_size, 10_000)
            self.assertIn("Held-Karp", (output / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
