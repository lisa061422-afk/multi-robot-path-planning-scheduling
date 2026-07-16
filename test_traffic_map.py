import math
import unittest

from traffic_map import TrafficMap


class TrafficMapTests(unittest.TestCase):
    def test_paper_2x2_auto_ports(self):
        tmap = TrafficMap.paper_2x2()

        self.assertEqual(len(tmap.intersection_ids), 4)
        self.assertEqual(len(tmap.port_ids), 8)
        self.assertEqual(tmap.road_ids, (5, 6, 7, 8))
        self.assertEqual(tmap.road_id_between(1, 2), 5)
        self.assertEqual(tmap.road_id_between(2, 3), 6)
        self.assertEqual(tmap.road_id_between(3, 4), 7)
        self.assertEqual(tmap.road_id_between(4, 1), 8)
        self.assertEqual(tmap.find_port(1, "L"), 1)
        self.assertEqual(tmap.find_port(1, "D"), 2)
        self.assertEqual(tmap.find_port(2, "D"), 3)
        self.assertEqual(tmap.find_port(2, "R"), 4)
        self.assertEqual(tmap.find_port(3, "R"), 5)
        self.assertEqual(tmap.find_port(3, "U"), 6)
        self.assertEqual(tmap.find_port(4, "U"), 7)
        self.assertEqual(tmap.find_port(4, "L"), 8)
        self.assertEqual(tmap.port_location(1), (1, "L"))
        self.assertEqual(tmap.port_location(5), (3, "R"))

    def test_two_route_od_has_branch_at_start(self):
        tmap = TrafficMap.paper_2x2()
        entrance = tmap.find_port(1, "L")
        exit_ = tmap.find_port(3, "R")

        vehicle = tmap.vehicle_route_options(1, entrance, exit_)
        paths = {opt.intersections for opt in vehicle.options}

        self.assertEqual(paths, {(1, 2, 3), (1, 4, 3)})
        self.assertEqual(vehicle.branch_intersections, (1,))
        self.assertEqual(vehicle.next_choices_after((1,)), (2, 4))
        self.assertTrue(vehicle.is_branch_after((1,)))
        self.assertEqual(vehicle.next_choices_after((1, 2)), (3,))
        self.assertFalse(vehicle.is_branch_after((1, 2)))

        for opt in vehicle.options:
            self.assertEqual(len(opt.branch_points), 1)
            self.assertEqual(opt.branch_points[0].intersection, 1)
            self.assertEqual(opt.branch_points[0].next_intersections, (2, 4))

        by_path = {opt.intersections: opt for opt in vehicle.options}
        lower_path = by_path[(1, 2, 3)]
        self.assertEqual([t.route_id for t in lower_path.traversals], [11, 10, 3])
        self.assertEqual([t.turn for t in lower_path.traversals], ["straight", "left", "right"])
        self.assertAlmostEqual(lower_path.execution_times[0], 2.0)
        self.assertAlmostEqual(lower_path.execution_times[1], 3.0 * math.pi / 4.0)
        self.assertAlmostEqual(lower_path.execution_times[2], math.pi / 4.0)

    def test_same_intersection_od_has_direct_single_path(self):
        tmap = TrafficMap.paper_2x2()
        entrance = tmap.find_port(1, "D")
        exit_ = tmap.find_port(1, "L")

        options = tmap.route_options(entrance, exit_)

        self.assertEqual(len(options), 1)
        self.assertEqual(options[0].intersections, (1,))
        self.assertEqual(options[0].branch_points, ())

    def test_rectangular_3x3_all_perimeter_ports(self):
        tmap = TrafficMap.rectangular_grid(3, 3)

        self.assertEqual(len(tmap.intersection_ids), 9)
        self.assertEqual(len(tmap.port_ids), 12)

        west_bottom = tmap.find_port(1, "L")
        east_top = tmap.find_port(9, "R")
        options = tmap.route_options(west_bottom, east_top, max_hops=5)

        self.assertTrue(options)
        self.assertTrue(all(opt.intersections[0] == 1 for opt in options))
        self.assertTrue(all(opt.intersections[-1] == 9 for opt in options))

    def test_paper_3x3_has_middle_path_selection(self):
        tmap = TrafficMap.paper_3x3()

        self.assertEqual(len(tmap.intersection_ids), 9)
        self.assertEqual(len(tmap.port_ids), 12)
        self.assertEqual(tmap.road_ids, tuple(range(10, 22)))
        self.assertEqual(tmap.find_port(1, "L"), 1)
        self.assertEqual(tmap.find_port(9, "R"), 7)

        vehicle = tmap.vehicle_route_options(1, 1, 7)
        self.assertGreater(len(vehicle.options), 2)
        self.assertIn(1, vehicle.branch_intersections)
        self.assertTrue(
            any(
                branch.path_index > 0
                for option in vehicle.options
                for branch in option.branch_points
            )
        )

    def test_intersection_time_scale_scales_coarse_and_space_times(self):
        base = TrafficMap.paper_3x3()
        scaled = TrafficMap.paper_3x3(intersection_time_scale=2.0)

        base_traversal = base.traversal_profile(5, "L", "R")
        scaled_traversal = scaled.traversal_profile(5, "L", "R")

        self.assertAlmostEqual(
            scaled_traversal.execution_time,
            2.0 * base_traversal.execution_time,
        )
        self.assertEqual(
            scaled_traversal.space_durations,
            tuple(2.0 * value for value in base_traversal.space_durations),
        )

    def test_paper_3x3_route_options_exclude_u_turn_traversals(self):
        tmap = TrafficMap.paper_3x3()

        for entrance in tmap.port_ids:
            for exit_ in tmap.port_ids:
                if entrance == exit_:
                    continue
                for option in tmap.route_options(entrance, exit_):
                    for traversal in option.traversals:
                        self.assertNotEqual(
                            traversal.exit_dir,
                            traversal.entry_dir,
                            msg=(
                                f"P{entrance}->P{exit_} route {option.intersections} "
                                f"has U-turn at I{traversal.intersection}"
                            ),
                        )

    def test_manual_edges_allow_non_unit_axis_aligned_segments(self):
        tmap = TrafficMap.from_grid(
            coords={1: (0, 0), 2: (3, 0), 3: (3, 1)},
            edges=[(1, 2), (2, 3)],
            name="manual_sparse",
        )

        entrance = tmap.find_port(1, "L")
        exit_ = tmap.find_port(3, "U")
        paths = tmap.enumerate_intersection_paths(entrance, exit_)

        self.assertEqual(paths, ((1, 2, 3),))


if __name__ == "__main__":
    unittest.main()
