"""Fleet dependency graph with impact analysis."""

from collections import deque

class FleetGraph:
    def __init__(self):
        self._deps: dict[str, list[str]] = {}
        self._rev: dict[str, list[str]] = {}
        self._meta: dict[str, dict] = {}

    def add_crate(self, name: str, meta: dict = None):
        if name not in self._deps:
            self._deps[name] = []
            self._rev[name] = []
            self._meta[name] = meta or {}

    def add_dependency(self, crate: str, depends_on: str):
        self.add_crate(crate)
        self.add_crate(depends_on)
        if depends_on not in self._deps[crate]:
            self._deps[crate].append(depends_on)
        if crate not in self._rev[depends_on]:
            self._rev[depends_on].append(crate)

    def dependents(self, crate: str) -> list[str]:
        return list(self._rev.get(crate, []))

    def dependencies(self, crate: str) -> list[str]:
        return list(self._deps.get(crate, []))

    def impact(self, crate: str) -> dict:
        visited, queue, count = set(), deque(), 0
        for d in self._rev.get(crate, []):
            queue.append(d)
        while queue:
            n = queue.popleft()
            if n not in visited:
                visited.add(n)
                count += 1
                queue.extend(self._rev.get(n, []))
        return {"crate": crate, "direct_dependents": len(self._rev.get(crate, [])),
                "total_impact": count, "affected": sorted(visited)}

    def layer(self, crate: str) -> str:
        if not self._deps.get(crate):
            return "foundation"
        depth = self._max_depth(crate)
        if depth <= 1: return "core"
        if depth <= 3: return "midlayer"
        return "facade"

    def _max_depth(self, crate: str, visited: set = None) -> int:
        if visited is None: visited = set()
        if crate in visited: return 0
        visited.add(crate)
        deps = self._deps.get(crate, [])
        if not deps: return 0
        return 1 + max(self._max_depth(d, visited.copy()) for d in deps)

    @property
    def nodes(self) -> list[str]:
        return list(self._deps.keys())

    @property
    def edges(self) -> list[tuple[str, str]]:
        return [(f, t) for f, ts in self._deps.items() for t in ts]

    @property
    def stats(self) -> dict:
        layers = {}
        for n in self._deps:
            l = self.layer(n)
            layers[l] = layers.get(l, 0) + 1
        return {"crates": len(self._deps), "edges": len(self.edges), "layers": layers}
