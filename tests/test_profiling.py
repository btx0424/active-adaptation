import io
import unittest
from contextlib import redirect_stdout

from active_adaptation.utils.profiling import ScopedTimer


class ScopedTimerTest(unittest.TestCase):
    def setUp(self):
        ScopedTimer.clear()

    def tearDown(self):
        ScopedTimer.clear()

    def test_clear_allows_a_timer_to_be_reused_under_a_new_parent(self):
        with ScopedTimer("rollout"):
            with ScopedTimer("simulation"):
                pass

        with redirect_stdout(io.StringIO()):
            ScopedTimer.print_summary(clear=True)

        with ScopedTimer("evaluation") as evaluation:
            with ScopedTimer("simulation") as simulation:
                pass

        self.assertIs(simulation.parent, evaluation)


if __name__ == "__main__":
    unittest.main()
