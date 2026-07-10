import time
import torch
from torch.utils._contextlib import _DecoratorContextManager
from typing import List, Dict


class ScopedTimer(_DecoratorContextManager):
    """A context manager for timing code blocks with singleton pattern.

    Each timer name creates a singleton instance that accumulates timing data
    across multiple uses.

    Usage:
    >>> with ScopedTimer("step"):
    ...     time.sleep(1)
    >>> with ScopedTimer("step"):  # Reuses the same timer
    ...     time.sleep(1)
    >>> ScopedTimer.print_summary(clear=True)
    """

    _instances: Dict[str, "ScopedTimer"] = {}
    _stack: List["ScopedTimer"] = []
    _root_nodes: List["ScopedTimer"] = []
    children: List["ScopedTimer"] = []

    def __new__(cls, name: str, sync: bool = False):
        if name not in cls._instances:
            instance = super().__new__(cls)
            instance.name = name
            instance.sync = sync
            instance.time = 0.0
            instance.count = 0
            instance.children = []
            instance.parent = None
            cls._instances[name] = instance
        return cls._instances[name]

    def __init__(self, name: str, sync: bool = False):
        # Update sync flag if it's different
        self.sync = sync

    def clone(self):
        """Required by ``_DecoratorContextManager`` for decorator usage.

        For our singleton timer, cloning just returns ``self``, which is
        sufficient since timing state is accumulated per named timer.
        """
        return self

    def __enter__(self):
        # Update hierarchical parent/roots based on current stack at each use,
        # so timers appear under the scopes where they are most recently used.
        if ScopedTimer._stack:
            parent = ScopedTimer._stack[-1]
        else:
            parent = "root"

        # Re-parent if needed so the summary tree reflects current usage.
        if self.parent is not parent:
            if self.parent is not None:
                raise ValueError(f"Timer {self.name} already has a parent {self.parent.name}")
            # Attach to new parent or root list.
            self.parent = parent
            if parent == "root":
                ScopedTimer._root_nodes.append(self)
            else:
                parent.children.append(self)

        ScopedTimer._stack.append(self)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sync:
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.last_time = self.end - self.start
        self.time += self.last_time
        self.count += 1
        ScopedTimer._stack.pop()

    @staticmethod
    def print_summary(clear: bool = True, depth: int = 3):
        """Print timing summary for all timers."""
        if not ScopedTimer._instances:
            print("No timers recorded.")
            return

        total_time = sum(r.time for r in ScopedTimer._root_nodes)
        print("\n" + "=" * 70)
        print(f"{'Timer Name':<30} {'Count':>8} {'Total (s)':>10} {'Avg (ms)':>10} {'%':>8}")
        print("=" * 70)

        for root in ScopedTimer._root_nodes:
            root.print_recursive(root, 0, clear=clear, max_depth=depth, total_time=total_time)

        print("=" * 70 + "\n")
        if clear:
            ScopedTimer.clear()

    @staticmethod
    def clear():
        """Clear timing values and hierarchy so timer names can be reused elsewhere."""
        for timer in ScopedTimer._instances.values():
            timer.time = 0.0
            timer.count = 0
            timer.children.clear()
            timer.parent = None
        ScopedTimer._root_nodes.clear()
        ScopedTimer._stack.clear()

    def print_recursive(self, node: "ScopedTimer", depth: int = 0, clear: bool = True, max_depth: int = -1, total_time: float = 0.0):
        """Recursively print timer nodes in DFS order."""
        if depth >= max_depth:
            return
        indent = "  " * depth
        avg_ms = (node.time / node.count * 1000) if node.count > 0 else 0
        pct = (node.time / total_time * 100) if total_time > 0 else 0.0
        print(
            f"{indent}{node.name:<30} {node.count:>8} {node.time:>10.4f} {avg_ms:>10.2f} {pct:>7.1f}%"
        )
        if clear:
            node.time = 0
            node.count = 0

        for child in node.children:
            self.print_recursive(child, depth + 1, clear=clear, max_depth=max_depth, total_time=total_time)
