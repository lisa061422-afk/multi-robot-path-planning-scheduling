import math
import unittest

from coarse_scheduler import (
    apply_entrance_headway,
    build_vehicle_plan,
    count_intersection_demands,
    describe_intersection_demands,
    expand_node,
    search_parallel_dfs_bb,
    search_dfs_bb,
)
from resource_schedulers import CoarseIntersectionScheduler, FiveSpaceScheduler
from traffic_map import TrafficMap


class CoarseSchedulerTests(unittest.TestCase):
    def test_shortest_route_prefers_fewer_intersections(self):
        tmap = TrafficMap.paper_2x2()

        route = tmap.shortest_route_option(3, 4)

        self.assertEqual(route.intersections, (2,))
        self.assertEqual(route.resource_sequence, (2,))

    def test_shortest_route_tie_uses_stable_option_order(self):
        tmap = TrafficMap.paper_2x2()

        route = tmap.shortest_route_option(1, 5, road_time=3.0)

        self.assertEqual(route.intersections, (1, 2, 3))
        self.assertEqual(route.id, 1)

    def test_two_vehicles_branch_on_same_intersection(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=1, exit=5),
            build_vehicle_plan(tmap, vehicle_id=2, entrance=2, exit=6),
        ]

        result = search_dfs_bb(plans, verbose=False)

        self.assertGreater(len(result.nodes), 2)
        self.assertAlmostEqual(result.best_g, math.pi / 4.0 + (3.0 * math.pi / 4.0 - 2.0))

        i1_segments = [seg for seg in result.best_schedule if seg.resource == 1]
        self.assertEqual(len(i1_segments), 2)
        i1_segments = sorted(i1_segments, key=lambda seg: seg.start_time)
        self.assertLessEqual(i1_segments[0].end_time, i1_segments[1].start_time)
        self.assertEqual(i1_segments[0].vehicle_id, 2)

    def test_count_intersection_demands(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=1, exit=5),
            build_vehicle_plan(tmap, vehicle_id=2, entrance=2, exit=6),
        ]

        demand = count_intersection_demands(plans)

        self.assertEqual(demand[1], ((1, 1), (2, 1)))
        self.assertIn(
            "I1: 2 vehicles, 2 visits -> N1(K1), N2(K1)",
            describe_intersection_demands(plans),
        )

    def test_apply_entrance_headway(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=1, exit=5, alpha0=0.0),
            build_vehicle_plan(tmap, vehicle_id=2, entrance=1, exit=6, alpha0=0.0),
            build_vehicle_plan(tmap, vehicle_id=3, entrance=1, exit=7, alpha0=5.0),
        ]

        adjusted = apply_entrance_headway(plans, headway=2.0)

        self.assertEqual([plan.alpha0 for plan in adjusted], [0.0, 2.0, 5.0])

    def test_new_task_can_interrupt_and_reset_running_task(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=1, exit=2, alpha0=0.0),
            build_vehicle_plan(tmap, vehicle_id=2, entrance=1, exit=2, alpha0=0.2),
        ]

        result = search_dfs_bb(plans, branch_and_bound=False, verbose=False)
        full_duration = plans[0].durations[0]

        preempted_children = [
            node
            for node in result.nodes
            if node.U_temp == (None, 1)
            and node.ni == (1, 1)
            and abs(node.r[0] - full_duration) <= 1e-9
        ]

        self.assertTrue(preempted_children)

    def test_waiting_interrupted_task_does_not_reenter_new_interrupt_competition(self):
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

        children, is_leaf = expand_node(
            result.nodes,
            n3_arrival_nodes[0].idx,
            plans,
        )

        self.assertFalse(is_leaf)
        self.assertTrue(children)
        self.assertFalse(any(child.U_temp[0] == 1 for child in children))
        self.assertTrue(any(child.U_temp[2] == 1 for child in children))
        for child in children:
            queue = dict(child.priority_queues)[1]
            self.assertLess(queue.index(1), queue.index(0))

    def test_parallel_dfs_matches_serial_on_small_case(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=1, exit=5),
            build_vehicle_plan(tmap, vehicle_id=2, entrance=2, exit=6),
        ]

        serial = search_dfs_bb(plans, verbose=False)
        parallel = search_parallel_dfs_bb(
            plans,
            frontier_depth=1,
            max_workers=1,
            verbose=False,
        )

        self.assertAlmostEqual(parallel.best_g, serial.best_g)

    def test_scheduler_facade_matches_serial_baseline(self):
        tmap = TrafficMap.paper_2x2()
        plans = [
            build_vehicle_plan(tmap, vehicle_id=1, entrance=1, exit=5),
            build_vehicle_plan(tmap, vehicle_id=2, entrance=2, exit=6),
        ]

        direct = search_dfs_bb(plans, verbose=False)
        via_facade = CoarseIntersectionScheduler().schedule_fixed(plans, verbose=False)

        self.assertEqual(CoarseIntersectionScheduler().name, "coarse_intersection")
        self.assertAlmostEqual(via_facade.best_g, direct.best_g)
        with self.assertRaises(NotImplementedError):
            FiveSpaceScheduler().schedule_fixed(plans, verbose=False)


if __name__ == "__main__":
    unittest.main()
