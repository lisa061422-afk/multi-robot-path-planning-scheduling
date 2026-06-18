import math
import unittest

from coarse_scheduler import (
    apply_entrance_headway,
    build_relaxed_vehicle_plan,
    build_vehicle_plan,
    expand_node,
    search_relaxed_dfs_bb,
    search_dfs_bb,
)
from main import make_vehicle_plans
from traffic_map import TrafficMap


class AlgorithmExampleTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
