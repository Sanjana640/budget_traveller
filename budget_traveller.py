"""
Budget Traveller with Coupons
=============================
Find the best flight journey from a start city to a destination city when:
  * you may take at most (max_stops + 1) flights,
  * you have a limited number of coupons (each halves ONE flight price, rounded down),
  * you may rank journeys by CHEAPEST or FASTEST.

Features
  1. Top-K route comparison (default 3) with prices and savings
  2. Smart coupon allocation (coupons go where they save the most money)
  3. Cheapest / Fastest mode (each flight stores price + duration)
  4. Transparency: every decision is explained, and the cheapest answer is
     cross-checked with an independent Dynamic Programming solution.

Run:  python budget_traveller.py
"""

from dataclasses import dataclass
from typing import List, Optional

INF = float("inf")


# --------------------------------------------------------------------------
# 1. DATA MODEL
# --------------------------------------------------------------------------
@dataclass
class Flight:
    src: int
    dst: int
    price: int      # in rupees
    duration: int   # in minutes


@dataclass
class Route:
    flights: List[Flight]          # flights in travel order
    coupon_on: List[bool]          # coupon_on[i] -> coupon used on flights[i]
    paid: List[int]                # price actually paid for each flight

    @property
    def cities(self) -> List[int]:
        return [self.flights[0].src] + [f.dst for f in self.flights]

    @property
    def full_price(self) -> int:
        return sum(f.price for f in self.flights)

    @property
    def final_cost(self) -> int:
        return sum(self.paid)

    @property
    def saved(self) -> int:
        return self.full_price - self.final_cost

    @property
    def total_time(self) -> int:
        return sum(f.duration for f in self.flights)


# --------------------------------------------------------------------------
# 2. STEP ONE: FIND EVERY VALID ROUTE (DFS)
# --------------------------------------------------------------------------
def find_all_routes(flights: List[Flight], start: int, end: int,
                    max_flights: int) -> List[List[Flight]]:
    """
    Depth-first search. Collects every route from `start` to `end` that uses
    at most `max_flights` flights and never visits the same city twice.

    Why never twice? A loop only adds cost, time and flight count, and it would
    waste coupons, so a loop can never make a journey better.
    """
    graph = {}
    for f in flights:
        graph.setdefault(f.src, []).append(f)

    routes: List[List[Flight]] = []

    def dfs(city: int, path: List[Flight], visited: set):
        if city == end:
            routes.append(list(path))
            return
        if len(path) == max_flights:      # stop limit reached
            return
        for f in graph.get(city, []):
            if f.dst not in visited:
                visited.add(f.dst)
                path.append(f)
                dfs(f.dst, path, visited)
                path.pop()
                visited.remove(f.dst)

    dfs(start, [], {start})
    return routes


# --------------------------------------------------------------------------
# 3. STEP TWO: SMART COUPON ALLOCATION
# --------------------------------------------------------------------------
def coupon_saving(price: int) -> int:
    """Money saved if a coupon is used on a flight of this price."""
    return price - price // 2


def allocate_coupons(path: List[Flight], max_coupons: int) -> Route:
    """
    Each flight can take at most one coupon, and a coupon saves
    price - price//2 on that flight. The savings of different flights are
    independent, so picking the flights with the BIGGEST savings is optimal.
    """
    ranked = sorted(range(len(path)),
                    key=lambda i: (-coupon_saving(path[i].price), i))
    chosen = {i for i in ranked[:max_coupons] if coupon_saving(path[i].price) > 0}

    coupon_on = [i in chosen for i in range(len(path))]
    paid = [f.price // 2 if coupon_on[i] else f.price
            for i, f in enumerate(path)]
    return Route(path, coupon_on, paid)


def naive_first_flights_cost(path: List[Flight], max_coupons: int) -> int:
    """Baseline for comparison: coupons wasted on the FIRST flights."""
    total = 0
    for i, f in enumerate(path):
        total += f.price // 2 if i < max_coupons else f.price
    return total


# --------------------------------------------------------------------------
# 4. STEP THREE: RANK ROUTES (cheapest or fastest)
# --------------------------------------------------------------------------
def plan_trips(flights: List[Flight], start: int, end: int, max_stops: int,
               max_coupons: int, mode: str = "cheapest", top_k: int = 3):
    paths = find_all_routes(flights, start, end, max_stops + 1)
    routes = [allocate_coupons(p, max_coupons) for p in paths]

    if mode == "fastest":
        routes.sort(key=lambda r: (r.total_time, r.final_cost))
    else:
        routes.sort(key=lambda r: (r.final_cost, r.total_time))
    return len(paths), routes[:top_k]


# --------------------------------------------------------------------------
# 5. INDEPENDENT CHECK: DYNAMIC PROGRAMMING (cheapest cost only)
# --------------------------------------------------------------------------
def dp_cheapest_cost(n: int, flights: List[Flight], start: int, end: int,
                     max_stops: int, max_coupons: int) -> float:
    """
    dp[f][c][city] = cheapest cost to be in `city` after f flights using c coupons.
    Uses a completely different method from the DFS above, so if both agree
    we can trust the answer.
    """
    max_flights = max_stops + 1
    dp = [[[INF] * (n + 1) for _ in range(max_coupons + 1)]
          for _ in range(max_flights + 1)]
    dp[0][0][start] = 0

    for f in range(max_flights):
        for c in range(max_coupons + 1):
            for fl in flights:
                cur = dp[f][c][fl.src]
                if cur == INF:
                    continue
                dp[f + 1][c][fl.dst] = min(dp[f + 1][c][fl.dst], cur + fl.price)
                if c < max_coupons:
                    dp[f + 1][c + 1][fl.dst] = min(dp[f + 1][c + 1][fl.dst],
                                                   cur + fl.price // 2)
    return min(dp[f][c][end]
               for f in range(max_flights + 1) for c in range(max_coupons + 1))


# --------------------------------------------------------------------------
# 6. OUTPUT (transparent explanations)
# --------------------------------------------------------------------------
def fmt_time(minutes: int) -> str:
    return f"{minutes // 60}h {minutes % 60:02d}m"


def print_route(rank: int, r: Route, max_coupons: int, best: Route):
    print(f"\n  Route {rank}: " + " -> ".join(f"City {c}" for c in r.cities))
    print(f"    Flights : {len(r.flights)}  (stops: {len(r.flights) - 1})")
    print(f"    Time    : {fmt_time(r.total_time)}")
    print(f"    Cost    : Rs {r.final_cost}   "
          f"(full price Rs {r.full_price}, coupons saved Rs {r.saved})")
    if rank > 1:
        diff = r.final_cost - best.final_cost
        tdiff = r.total_time - best.total_time
        print(f"    vs #1   : {'+' if diff >= 0 else '-'}Rs {abs(diff)}, "
              f"{'+' if tdiff >= 0 else '-'}{fmt_time(abs(tdiff))}")
    print("    Flight breakdown:")
    for i, f in enumerate(r.flights):
        if r.coupon_on[i]:
            print(f"      {f.src} -> {f.dst}: Rs {f.price} -> Rs {r.paid[i]}  "
                  f"[COUPON, saves Rs {f.price - r.paid[i]}]")
        else:
            print(f"      {f.src} -> {f.dst}: Rs {f.price}  [no coupon]")
    naive = naive_first_flights_cost(r.flights, max_coupons)
    if naive > r.final_cost:
        print(f"    Smart coupons beat 'coupons on first flights' by "
              f"Rs {naive - r.final_cost} (naive cost would be Rs {naive}).")


def explain_coupon_choice(r: Route, max_coupons: int):
    print("\n  WHY these coupons? (best route's coupon table)")
    print("    flight        price   saving-if-coupon   decision")
    for i, f in enumerate(r.flights):
        decision = "COUPON" if r.coupon_on[i] else "-"
        print(f"    {f.src} -> {f.dst:<6}   {f.price:>5}   "
              f"{coupon_saving(f.price):>16}   {decision}")
    print(f"    Rule: coupons ({max_coupons} available) go to the flights with "
          f"the biggest saving.")


def run(n, flights, start, end, max_stops, max_coupons, mode, top_k=3):
    print("\n" + "=" * 62)
    print(f" Trip: City {start} -> City {end} | max stops: {max_stops} "
          f"| coupons: {max_coupons} | mode: {mode.upper()}")
    print("=" * 62)

    if start == end:
        print("  You are already at the destination. Cost: Rs 0.")
        return

    found, top = plan_trips(flights, start, end, max_stops, max_coupons,
                            mode, top_k)
    print(f"  Step 1: searched all routes within {max_stops + 1} flight(s): "
          f"{found} valid route(s) found.")
    if not top:
        print("  No valid journey found. Try allowing more stops.")
        return
    print("  Step 2: gave coupons to the flights where they save the most.")
    print(f"  Step 3: ranked routes by "
          f"{'total time' if mode == 'fastest' else 'final cost'} "
          f"and kept the top {len(top)}.")

    for i, r in enumerate(top, 1):
        print_route(i, r, max_coupons, top[0])
    explain_coupon_choice(top[0], max_coupons)

    # Self-check against the independent DP method
    if mode == "cheapest":
        dp_cost = dp_cheapest_cost(n, flights, start, end, max_stops, max_coupons)
        ok = dp_cost == top[0].final_cost
        print(f"\n  Self-check (DP cross-verification): DP says Rs {dp_cost:.0f}, "
              f"route search says Rs {top[0].final_cost} -> "
              f"{'MATCH' if ok else 'MISMATCH!'}")
    else:
        cheapest = min(plan_trips(flights, start, end, max_stops, max_coupons,
                                  "cheapest", 10 ** 9)[1],
                       key=lambda r: r.final_cost)
        extra = top[0].final_cost - cheapest.final_cost
        print(f"\n  Trade-off: the fastest route costs Rs {extra} more than the "
              f"cheapest option (Rs {cheapest.final_cost}).")


# --------------------------------------------------------------------------
# 7. INPUT HANDLING
# --------------------------------------------------------------------------
def ask_int(prompt: str, minimum: int = 0, maximum: Optional[int] = None) -> int:
    while True:
        try:
            v = int(input(prompt).strip())
            if v < minimum or (maximum is not None and v > maximum):
                raise ValueError
            return v
        except ValueError:
            rng = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
            print(f"  Please enter a whole number {rng}.")


def read_user_data():
    n = ask_int("Number of cities: ", 2)
    m = ask_int("Number of flights: ", 1)
    print("Enter each flight as:  from  to  price  duration_in_minutes")
    flights = []
    while len(flights) < m:
        try:
            u, v, p, d = map(int, input(f"  Flight {len(flights) + 1}: ").split())
            if not (1 <= u <= n and 1 <= v <= n) or p < 0 or d < 0:
                raise ValueError
            flights.append(Flight(u, v, p, d))
        except ValueError:
            print(f"  Need 4 whole numbers; cities must be 1..{n}; no negatives.")
    start = ask_int("Start city: ", 1, n)
    end = ask_int("Destination city: ", 1, n)
    stops = ask_int("Max intermediate stops: ", 0)
    coupons = ask_int("Number of coupons: ", 0)
    return n, flights, start, end, stops, coupons


def demo_data():
    """Problem example + a fast but expensive direct flight to show the trade-off."""
    flights = [
        Flight(1, 2, 1000, 120),
        Flight(2, 4, 1000, 150),
        Flight(1, 3, 1500, 180),
        Flight(3, 4, 500, 100),
        Flight(1, 4, 3200, 120),
    ]
    return 4, flights, 1, 4, 1, 1


def main():
    print("=== Budget Traveller with Coupons ===")
    print("1) Run demo   2) Enter my own flights")
    choice = ask_int("Choose 1 or 2: ", 1, 2)

    if choice == 1:
        n, flights, start, end, stops, coupons = demo_data()
        print("\nDemo flights (from, to, price, minutes):")
        for f in flights:
            print(f"  {f.src} -> {f.dst}  Rs {f.price}  {f.duration} min")
        for mode in ("cheapest", "fastest"):
            run(n, flights, start, end, stops, coupons, mode)
    else:
        n, flights, start, end, stops, coupons = read_user_data()
        print("Mode: 1) Cheapest  2) Fastest  3) Show both")
        m = ask_int("Choose 1, 2 or 3: ", 1, 3)
        modes = {1: ["cheapest"], 2: ["fastest"], 3: ["cheapest", "fastest"]}[m]
        for mode in modes:
            run(n, flights, start, end, stops, coupons, mode)


if __name__ == "__main__":
    main()