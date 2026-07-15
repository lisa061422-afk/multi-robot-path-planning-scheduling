import os
import unittest

from trajectory_conflicts import (
    TRAJECTORY_FILTER_ENV,
    route_ids_conflict,
    set_trajectory_conflict_filter,
    simultaneous_prefix,
)


class TrajectoryConflictTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop(TRAJECTORY_FILTER_ENV, None)

    def test_disabled_filter_reproduces_single_runner(self):
        set_trajectory_conflict_filter(False)
        self.assertEqual(simultaneous_prefix((0, 1), (2, 8)), (0,))

    def test_enabled_filter_admits_only_pairwise_compatible_movements(self):
        set_trajectory_conflict_filter(True)
        self.assertFalse(route_ids_conflict(2, 8))
        self.assertTrue(route_ids_conflict(2, 5))
        self.assertEqual(simultaneous_prefix((0, 1, 2), (2, 8, 5)), (0, 1))

    def test_same_entry_or_same_exit_always_conflicts(self):
        # Same D entry: routes 1, 2, 3. Same U exit: routes 2, 6, 10.
        self.assertTrue(route_ids_conflict(1, 2))
        self.assertTrue(route_ids_conflict(2, 6))

    def test_user_defined_non_conflicting_patterns_and_rotations(self):
        pattern_a = ((1, 12), (4, 3), (7, 6), (10, 9))
        pattern_b = ((1, 6), (4, 9), (7, 12), (10, 3))
        for pair in (*pattern_a, *pattern_b):
            with self.subTest(pair=pair):
                self.assertFalse(route_ids_conflict(*pair))


if __name__ == "__main__":
    unittest.main()
