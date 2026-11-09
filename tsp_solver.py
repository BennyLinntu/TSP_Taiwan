"""Travelling salesman problem on Taiwan's 14 major cities.

Two Gurobi formulations are included:

1. DFJ (Dantzig-Fulkerson-Johnson) — one binary arc variable and all subset
   subtour-elimination constraints.
2. MTZ (Miller-Tucker-Zemlin) — a compact formulation with continuous order
   variables and big-M constraints.

Gurobi is optional. The exact Held-Karp dynamic program and a 2-opt heuristic
run without a licence, so the project is usable out of the box.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_DATA = WORKSPACE / "data" / "taiwan_cities.csv"
DEFAULT_OUTPUT = WORKSPACE / "results"
EARTH_RADIUS_KM = 6371.0088

LABEL_OFFSETS = {
    "Taipei": (0.06, 0.045),
    "New Taipei": (-0.20, -0.11),
    "Taoyuan": (-0.16, 0.035),
    "Hsinchu": (0.04, 0.015),
    "Miaoli": (0.04, 0.015),
    "Taichung": (0.04, 0.02),
    "Changhua": (0.04, -0.05),
    "Nantou": (0.04, -0.03),
    "Yunlin": (0.04, -0.02),
    "Chiayi": (0.04, 0.015),
    "Tainan": (0.04, 0.015),
    "Kaohsiung": (-0.17, -0.06),
    "Pingtung": (0.05, -0.04),
    "Yilan": (0.04, 0.02),
}


class GurobiUnavailable(RuntimeError):
    """Raised when a Gurobi model is requested without gurobipy installed."""


@dataclass
class TourResult:
    method: str
    cost: float
    tour: list[int]
    runtime: float
    status: str = "optimal"


def load_cities(path: Path | str = DEFAULT_DATA) -> pd.DataFrame:
    cities = pd.read_csv(path)
    required = {"name", "latitude", "longitude"}
    if not required.issubset(cities.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")
    if len(cities) < 3:
        raise ValueError("At least three cities are required.")
    return cities


def haversine_matrix(latitudes, longitudes) -> np.ndarray:
    """Great-circle distances in kilometres."""
    lat = np.radians(np.asarray(latitudes, dtype=float))
    lon = np.radians(np.asarray(longitudes, dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_matrix(cities: pd.DataFrame) -> np.ndarray:
    return haversine_matrix(cities["latitude"], cities["longitude"])


def tour_cost(tour: list[int], distances: np.ndarray) -> float:
    if not tour:
        return math.inf
    return float(
        sum(distances[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    )


def validate_tour(tour: list[int], n: int) -> bool:
    return len(tour) == n and sorted(tour) == list(range(n))


def held_karp(distances: np.ndarray) -> TourResult:
    """Exact TSP with bitmask dynamic programming, O(n² 2ⁿ)."""
    started = time.perf_counter()
    n = len(distances)
    if n > 20:
        raise ValueError("Held-Karp is limited to 20 cities.")

    size = 1 << n
    dp = np.full((size, n), np.inf)
    parent = np.full((size, n), -1, dtype=np.int16)
    dp[1, 0] = 0.0

    for mask in range(1, size):
        if not mask & 1:
            continue
        for last in range(n):
            if not mask & (1 << last) or not np.isfinite(dp[mask, last]):
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                candidate = dp[mask, last] + distances[last, nxt]
                if candidate < dp[new_mask, nxt]:
                    dp[new_mask, nxt] = candidate
                    parent[new_mask, nxt] = last

    full = size - 1
    end = min(range(1, n), key=lambda j: dp[full, j] + distances[j, 0])
    cost = float(dp[full, end] + distances[end, 0])

    tour = [0]
    mask = full
    current = end
    while current != 0:
        tour.append(int(current))
        previous = int(parent[mask, current])
        mask ^= 1 << current
        current = previous
    tour = [0] + list(reversed(tour[1:]))

    return TourResult(
        method="Held-Karp (exact)",
        cost=round(cost, 4),
        tour=tour,
        runtime=round(time.perf_counter() - started, 4),
    )


def nearest_neighbor(distances: np.ndarray, start: int = 0) -> list[int]:
    n = len(distances)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        current = min(unvisited, key=lambda city: distances[current, city])
        tour.append(current)
        unvisited.remove(current)
    return tour


def two_opt(tour: list[int], distances: np.ndarray) -> list[int]:
    """Improve a tour until no 2-opt swap reduces its cost."""
    best = tour[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if tour_cost(candidate, distances) + 1e-9 < tour_cost(best, distances):
                    best = candidate
                    improved = True
        if not improved:
            break
    return best


def heuristic_tour(distances: np.ndarray) -> TourResult:
    started = time.perf_counter()
    route = two_opt(nearest_neighbor(distances), distances)
    return TourResult(
        method="Nearest neighbour + 2-opt",
        cost=round(tour_cost(route, distances), 4),
        tour=route,
        runtime=round(time.perf_counter() - started, 4),
        status="heuristic",
    )


def _require_gurobi():
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise GurobiUnavailable(
            "Install Gurobi and gurobipy to run the MILP formulations: "
            "https://www.gurobi.com/"
        ) from exc
    return gp


def _extract_tour(model, x, n: int) -> list[int]:
    successors = {}
    for i in range(n):
        for j in range(n):
            if i != j and x[i, j].X > 0.5:
                successors[i] = j
    if len(successors) != n:
        raise RuntimeError("The model did not return a complete tour.")
    tour = [0]
    for _ in range(n - 1):
        tour.append(successors[tour[-1]])
    if successors[tour[-1]] != 0:
        raise RuntimeError("The returned route does not close at the depot.")
    return tour


def solve_dfj(
    distances: np.ndarray,
    time_limit: float = 120,
    output_flag: bool = False,
) -> TourResult:
    """Dantzig-Fulkerson-Johnson formulation with all subset cuts."""
    gp = _require_gurobi()
    n = len(distances)
    started = time.perf_counter()

    model = gp.Model("TSP_DFJ")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = time_limit
    x = model.addVars(n, n, vtype=gp.GRB.BINARY, name="x")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1)
        x[i, i].ub = 0

    # Every non-empty subset S of V \ {0} must have at least one outgoing arc.
    others = list(range(1, n))
    for size in range(2, n):
        for subset in itertools.combinations(others, size):
            members = list(subset)
            model.addConstr(
                gp.quicksum(x[i, j] for i in members for j in members if i != j)
                <= size - 1,
                name=f"sec_{'_'.join(map(str, members))}",
            )

    model.setObjective(
        gp.quicksum(
            distances[i, j] * x[i, j]
            for i in range(n)
            for j in range(n)
            if i != j
        ),
        gp.GRB.MINIMIZE,
    )
    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError("Gurobi could not find a feasible tour.")
    tour = _extract_tour(model, x, n)
    return TourResult(
        method="DFJ (Gurobi)",
        cost=round(tour_cost(tour, distances), 4),
        tour=tour,
        runtime=round(time.perf_counter() - started, 4),
        status="optimal" if model.Status == gp.GRB.OPTIMAL else "feasible",
    )


def solve_mtz(
    distances: np.ndarray,
    time_limit: float = 120,
    output_flag: bool = False,
) -> TourResult:
    """Miller-Tucker-Zemlin compact formulation with order variables."""
    gp = _require_gurobi()
    n = len(distances)
    started = time.perf_counter()

    model = gp.Model("TSP_MTZ")
    model.Params.OutputFlag = 1 if output_flag else 0
    model.Params.TimeLimit = time_limit
    x = model.addVars(n, n, vtype=gp.GRB.BINARY, name="x")
    u = model.addVars(n, lb=1, ub=n - 1, vtype=gp.GRB.CONTINUOUS, name="u")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1)
        x[i, i].ub = 0

    model.addConstr(u[0] == 1)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

    model.setObjective(
        gp.quicksum(
            distances[i, j] * x[i, j]
            for i in range(n)
            for j in range(n)
            if i != j
        ),
        gp.GRB.MINIMIZE,
    )
    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError("Gurobi could not find a feasible tour.")
    tour = _extract_tour(model, x, n)
    return TourResult(
        method="MTZ (Gurobi)",
        cost=round(tour_cost(tour, distances), 4),
        tour=tour,
        runtime=round(time.perf_counter() - started, 4),
        status="optimal" if model.Status == gp.GRB.OPTIMAL else "feasible",
    )


def route_names(tour: list[int], cities: pd.DataFrame) -> list[str]:
    return [cities.iloc[index]["name"] for index in tour]


def print_result(result: TourResult, cities: pd.DataFrame) -> None:
    names = route_names(result.tour, cities)
    route = " → ".join(names + [names[0]])
    print(f"\n{result.method}")
    print(f"  cost:    {result.cost:,.2f} km")
    print(f"  runtime: {result.runtime:.3f}s ({result.status})")
    print(f"  route:   {route}")


def plot_route(
    cities: pd.DataFrame,
    tour: list[int],
    cost: float,
    output: Path,
    title: str = "Taiwan TSP — optimal route",
) -> Path:
    ordered = [cities.iloc[i] for i in tour] + [cities.iloc[tour[0]]]
    lons = [city["longitude"] for city in ordered]
    lats = [city["latitude"] for city in ordered]

    fig, ax = plt.subplots(figsize=(8.4, 11))
    ax.set_facecolor("#f5f1e8")
    fig.patch.set_facecolor("#f5f1e8")
    ax.plot(lons, lats, color="#e4572e", linewidth=2.2, zorder=1)
    ax.scatter(
        cities["longitude"],
        cities["latitude"],
        s=70,
        color="#172033",
        edgecolor="#fffdf8",
        linewidth=1.4,
        zorder=2,
    )
    ax.scatter(lons[0], lats[0], s=160, color="#2a9d8f", edgecolor="#fffdf8", linewidth=2, zorder=3)
    for index, city in cities.iterrows():
        dx, dy = LABEL_OFFSETS.get(city["name"], (0.045, 0.02))
        ax.annotate(
            city["name"],
            (city["longitude"], city["latitude"]),
            xytext=(city["longitude"] + dx, city["latitude"] + dy),
            fontsize=7.5,
            color="#34405a",
        )
    ax.set_title(f"{title}\n{cost:.2f} km", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d8d2c6", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def save_results(results: list[TourResult], cities: pd.DataFrame, output: Path) -> Path:
    payload = {
        "cities": len(cities),
        "results": [
            {
                "method": result.method,
                "cost_km": result.cost,
                "runtime_seconds": result.runtime,
                "status": result.status,
                "tour": route_names(result.tour, cities),
            }
            for result in results
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a Taiwan TSP instance.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gurobi",
        action="store_true",
        help="Also solve the DFJ and MTZ models (requires gurobipy + licence).",
    )
    parser.add_argument("--time-limit", type=float, default=120)
    parser.add_argument("--verbose-gurobi", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cities = load_cities(args.data)
    distances = distance_matrix(cities)
    results = [held_karp(distances), heuristic_tour(distances)]

    if args.gurobi:
        try:
            results.append(
                solve_dfj(
                    distances,
                    time_limit=args.time_limit,
                    output_flag=args.verbose_gurobi,
                )
            )
            results.append(
                solve_mtz(
                    distances,
                    time_limit=args.time_limit,
                    output_flag=args.verbose_gurobi,
                )
            )
        except GurobiUnavailable as error:
            print(f"\n[skip] {error}")

    for result in results:
        print_result(result, cities)

    best = min(results, key=lambda result: result.cost)
    plot_path = plot_route(
        cities,
        best.tour,
        best.cost,
        args.output / "route.png",
        title=f"Taiwan TSP — {best.method}",
    )
    json_path = save_results(results, cities, args.output / "summary.json")
    print(f"\nMap: {plot_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
