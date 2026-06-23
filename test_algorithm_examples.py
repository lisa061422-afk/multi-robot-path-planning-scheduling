import math
import unittest

from coarse_scheduler import (
    apply_entrance_headway,
    apply_relaxed_entrance_headway,
    build_relaxed_vehicle_plan,
    build_vehicle_plan,
    expand_node,
    search_dynamic_codesign_dfs_bb,
    search_relaxed_dfs_bb,
    search_dfs_bb,
)
from main import make_vehicle_plans
from traffic_map import TrafficMap


class AlgorithmExampleTests(unittest.TestCase):
    def assert_valid_relaxed_schedule(self, result, plans, *, lambda_path):
        node = result.best_node
        self.assertTrue(math.isfinite(result.best_g))
        self.assertTrue(all(len(candidates) == 1 for candidates in node.route_candidates))

        delay = sum(seg.delay for seg in result.best_schedule)
        self.assertAlmostEqual(delay, node.g_delay)
        self.assertAlmostEqual(node.g_delay + lambda_path * node.g_path, node.g)
        self.assertAlmostEqual(node.g, result.best_g)

        expected_by_vehicle = {}
        for plan, candidates in zip(plans, node.route_candidates):
            option = plan.route_options[candidates[0]]
            expected_by_vehicle[plan.vehicle_id] = list(
                zip(option.resource_sequence, option.execution_times)
            )

        actual_by_vehicle = {}
        for seg in sorted(result.best_schedule, key=lambda item: (item.vehicle_id, item.task_index)):
            actual_by_vehicle.setdefault(seg.vehicle_id, []).append(seg)
            self.assertGreaterEqual(seg.start_time + 1e-9, seg.requested_time)
            self.assertGreaterEqual(seg.end_time + 1e-9, seg.start_time)

        for vehicle_id, expected in expected_by_vehicle.items():
            actual = actual_by_vehicle.get(vehicle_id, [])
            self.assertEqual(len(actual), len(expected))
            for task_index, (seg, (resource, duration)) in enumerate(
                zip(actual, expected),
                start=1,
            ):
                self.assertEqual(seg.task_index, task_index)
                self.assertEqual(seg.resource, resource)
                self.assertAlmostEqual(seg.end_time - seg.start_time, duration)

        by_resource = {}
        for seg in result.best_schedule:
            by_resource.setdefault(seg.resource, []).append(seg)
        for segs in by_resource.values():
            ordered = sorted(segs, key=lambda item: (item.start_time, item.end_time))
            for left, right in zip(ordered, ordered[1:]):
                self.assertLessEqual(left.end_time, right.start_time + 1e-9)

    def test_single_vehicle_no_contention_has_zero_delay(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=3, exit=4, alpha0=1.5),
        ]

        result = search_dfs_bb(plans, branch_and_bound=False, verbose=False)

        self.assertAlmostEqual(result.best_g, 0.0)
        self.assertEqual(len(result.best_schedule), 1)
        seg = result.best_schedule[0]
        self.assertEqual(seg.vehicle_id, 1)
        self.assertEqual(seg.resource, 2)
        self.assertAlmostEqual(seg.requested_time, 1.5)
        self.assertAlmostEqual(seg.start_time, 1.5)
        self.assertAlmostEqual(seg.delay, 0.0)

    def test_independent_intersections_can_run_in_parallel(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=1, exit=2, alpha0=0.0),
            build_vehicle_plan(tmap, vehicle_id=2, entrance=3, exit=4, alpha0=0.0),
        ]

        result = search_dfs_bb(plans, branch_and_bound=False, verbose=False)

        self.assertAlmostEqual(result.best_g, 0.0)
        self.assertEqual(len(result.best_schedule), 2)
        starts = {
            (seg.vehicle_id, seg.resource): seg.start_time
            for seg in result.best_schedule
        }
        self.assertAlmostEqual(starts[(1, 1)], 0.0)
        self.assertAlmostEqual(starts[(2, 2)], 0.0)

    def test_same_entrance_headway_serializes_initial_alpha(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=1, exit=2, alpha0=0.0),
            build_vehicle_plan(tmap, vehicle_id=2, entrance=1, exit=6, alpha0=0.0),
            build_vehicle_plan(tmap, vehicle_id=3, entrance=1, exit=7, alpha0=1.0),
        ]

        adjusted = apply_entrance_headway(plans, headway=2.0)

        self.assertEqual([plan.alpha0 for plan in adjusted], [0.0, 2.0, 4.0])

    def test_preempted_repeat_task_resets_to_full_duration(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=1, exit=2, alpha0=0.0),
            build_vehicle_plan(tmap, vehicle_id=2, entrance=1, exit=2, alpha0=0.2),
        ]

        result = search_dfs_bb(plans, branch_and_bound=False, verbose=False)
        full_duration = plans[0].durations[0]

        self.assertTrue(
            any(
                node.U_temp == (None, 1)
                and node.ni == (1, 1)
                and abs(node.r[0] - full_duration) <= 1e-9
                for node in result.nodes
            )
        )

    def test_priority_queue_preserves_existing_order_when_new_task_arrives(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=1, exit=2, alpha0=0.0),
            build_vehicle_plan(tmap, vehicle_id=2, entrance=1, exit=2, alpha0=0.2),
            build_vehicle_plan(tmap, vehicle_id=3, entrance=1, exit=2, alpha0=0.4),
        ]

        result = search_dfs_bb(plans, branch_and_bound=False, verbose=False)
        n3_arrival_nodes = [
            node
            for node in result.nodes
            if abs(node.tw - 0.4) <= 1e-9
            and node.U_temp == (None, 1, None)
            and node.d[2] <= 1e-9
        ]
        self.assertTrue(n3_arrival_nodes)

        children, is_leaf = expand_node(result.nodes, n3_arrival_nodes[0].idx, plans)

        self.assertFalse(is_leaf)
        self.assertTrue(children)
        for child in children:
            queue = dict(child.priority_queues)[1]
            self.assertLess(queue.index(1), queue.index(0))

    def test_fixed_route_policy_can_use_manual_path_or_shortest_path(self):
        tmap = TrafficMap.paper_2x2()
        requests = [
            (1, 2, 7, 0.0, [1, 2, 3, 4]),
        ]

        manual = make_vehicle_plans(
            tmap,
            requests,
            Dt=3.0,
            fixed_route_policy="manual_or_shortest",
        )
        shortest = make_vehicle_plans(
            tmap,
            requests,
            Dt=3.0,
            fixed_route_policy="shortest",
        )

        self.assertEqual(manual[0].route.intersections, (1, 2, 3, 4))
        self.assertEqual(shortest[0].route.intersections, (1, 4))
        self.assertLess(shortest[0].free_flow_time, manual[0].free_flow_time)

    def test_relaxed_single_vehicle_prefers_short_path(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_relaxed_vehicle_plan(
                tmap,
                vehicle_id=1,
                entrance=2,
                exit=7,
                alpha0=0.0,
                road_time=3.0,
            )
        ]

        result = search_relaxed_dfs_bb(
            plans,
            lambda_path=1.0,
            branch_and_bound=False,
            verbose=False,
        )

        self.assertAlmostEqual(result.best_g, 0.0)
        self.assertAlmostEqual(result.best_node.g_delay, 0.0)
        self.assertAlmostEqual(result.best_node.g_path, 0.0)
        self.assertEqual(
            tuple(seg.resource for seg in result.best_schedule),
            (1, 4),
        )

    def test_relaxed_contention_returns_finite_joint_solution(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_relaxed_vehicle_plan(tmap, vehicle_id=1, entrance=2, exit=7, alpha0=0.0),
            build_relaxed_vehicle_plan(tmap, vehicle_id=2, entrance=2, exit=8, alpha0=0.0),
        ]

        result = search_relaxed_dfs_bb(
            plans,
            lambda_path=1.0,
            branch_and_bound=True,
            verbose=False,
        )

        self.assertTrue(math.isfinite(result.best_g))
        self.assertGreaterEqual(result.best_node.g_delay, 0.0)
        self.assertGreaterEqual(result.best_node.g_path, 0.0)
        self.assertEqual(
            sorted({seg.vehicle_id for seg in result.best_schedule}),
            [1, 2],
        )

    def test_dynamic_codesign_matches_enumerated_route_choices_on_3x3(self):
        tmap = TrafficMap.paper_3x3()
        plans = [
            build_relaxed_vehicle_plan(
                tmap,
                vehicle_id=1,
                entrance=2,
                exit=6,
                alpha0=0.0,
                road_time=3.0,
            ),
            build_relaxed_vehicle_plan(
                tmap,
                vehicle_id=2,
                entrance=1,
                exit=9,
                alpha0=0.0,
                road_time=3.0,
            ),
        ]
        plans = apply_relaxed_entrance_headway(plans, headway=2.0)

        self.assertEqual([len(plan.route_options) for plan in plans], [10, 10])

        dynamic = search_dynamic_codesign_dfs_bb(
            plans,
            lambda_path=1.0,
            branch_and_bound=True,
            verbose=False,
        )
        enumerated = search_relaxed_dfs_bb(
            plans,
            lambda_path=1.0,
            branch_and_bound=True,
            verbose=False,
        )

        self.assert_valid_relaxed_schedule(dynamic, plans, lambda_path=1.0)
        self.assert_valid_relaxed_schedule(enumerated, plans, lambda_path=1.0)
        self.assertAlmostEqual(dynamic.best_g, enumerated.best_g)
        self.assertAlmostEqual(dynamic.best_node.g_delay, enumerated.best_node.g_delay)
        self.assertAlmostEqual(dynamic.best_node.g_path, enumerated.best_node.g_path)

        selected_extra = 0.0
        for plan, candidates in zip(plans, dynamic.best_node.route_candidates):
            selected = plan.route_options[candidates[0]]
            free_times = [
                sum(option.execution_times) + len(option.edges) * plan.road_time
                for option in plan.route_options
            ]
            selected_time = sum(selected.execution_times) + len(selected.edges) * plan.road_time
            selected_extra += max(0.0, selected_time - min(free_times))
        self.assertAlmostEqual(dynamic.best_node.g_path, selected_extra)


if __name__ == "__main__":
    unittest.main()
