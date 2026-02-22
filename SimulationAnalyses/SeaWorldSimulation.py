import math
import random
import heapq
import statistics
from typing import List, Optional, Sequence, TypeVar
from scipy.stats import t

T = TypeVar("T")

SEED = 102
_SIM_ALREADY_RAN = False

# Time constants (minutes since 09:00)
OPEN = 0
TEN_AM = 60
ELEVEN_AM = 120
NOON = 180
ONE_PM = 240
TWO_PM = 300
THREE_PM = 360
FOUR_PM = 420
FIVE_PM = 480
SIX_PM = 540
CLOSE = 600
PARK_MAX_RATING = 10.0

# Food-related constants
HAMBURGER = "HAMBURGER"
PIZZA = "PIZZA"
SALAD = "SALAD"

DEPARTURE_BUFFER = 30.0

# Teen abandonment actions
BUY_EXPRESS_AND_JOIN_EXPRESS_QUEUE = "BUY_EXPRESS_AND_JOIN_EXPRESS_QUEUE"
GO_TO_ANOTHER_ATTRACTION = "GO_TO_ANOTHER_ATTRACTION"

ALL_ATTRACTIONS: list[str] = [
    "LazyRiver",
    "SingleSlides",
    "LargeTubeSlide",
    "SmallTubeSlide",
    "WavePool",
    "ToddlerPool",
    "SnorkelTour",
]

# ----------------------------------------------------------- #
# -------------------- SCENARIO CONFIG ---------------------- #
# ----------------------------------------------------------- #

BUDGET = 250_000

# Global knobs (default = BASE)
SCN_LUNCH_PROB = 0.70             # baseline % who choose to eat (within 13:00-15:00 window)
SCN_UNSAT_PROB = 0.10             # baseline bad meal probability
SCN_SINGLE_ARRIVAL_RATE = 2 / 3   # baseline singles arrival rate (per minute)
SCN_WEBSITE_MODE = False          # baseline: ticket + wristband
SCN_LARGE_TUBE_CAPACITY = 8       # baseline capacity per run for LargeTubeSlide (capacity_per_server)

SCENARIOS = {
    "BASE": {
        "invest_cost": 0,
        "lunch_prob": 0.70,
        "unsat_prob": 0.10,
        "single_arrival_rate": 2 / 3,
        "website_mode": False,
        "large_tube_capacity": 8,
        "label": "BASE (No investments)",
    },
    "ALT1": {
        # Better kitchen: unsat_prob -> 0.03, lunch_prob -> 0.85
        # Marketing: singles rate -> 20 per 15 minutes => rate = 20/15 per minute
        # Website: reception service = wristband only
        "invest_cost": 250_000,
        "lunch_prob": 0.85,
        "unsat_prob": 0.03,
        "single_arrival_rate": 20 / 15,
        "website_mode": True,
        "large_tube_capacity": 8,
        "label": "ALT1 (Kitchen+Marketing+Website) [250k]",
    },
    "ALT2": {
        # Website + Large tube capacity 10
        "invest_cost": 250_000,
        "lunch_prob": 0.70,
        "unsat_prob": 0.10,
        "single_arrival_rate": 2 / 3,
        "website_mode": True,
        "large_tube_capacity": 10,
        "label": "ALT2 (Website+LargeTube=10) [250k]",
    },
    "ALT3": {
        # Marketing Only: Focus on singles market growth
        "invest_cost": 60_000,
        "lunch_prob": 0.70,
        "unsat_prob": 0.10,
        "single_arrival_rate": 15 / 12,  # 25% increase in singles
        "website_mode": False,
        "large_tube_capacity": 8,
        "label": "ALT3 (Marketing Only) [60k]",
    },
    "ALT4": {
        # Kitchen Quality Focus: Better food, higher lunch participation
        "invest_cost": 100_000,
        "lunch_prob": 0.82,
        "unsat_prob": 0.04,
        "single_arrival_rate": 2 / 3,
        "website_mode": False,
        "large_tube_capacity": 8,
        "label": "ALT4 (Kitchen Quality Only) [100k]",
    },
    "ALT5": {
        # Website Only: Streamline entry process
        "invest_cost": 40_000,
        "lunch_prob": 0.70,
        "unsat_prob": 0.10,
        "single_arrival_rate": 2 / 3,
        "website_mode": True,
        "large_tube_capacity": 8,
        "label": "ALT5 (Website Only) [40k]",
    },
    "ALT6": {
        # Kitchen + Website: Quality + Convenience
        "invest_cost": 140_000,
        "lunch_prob": 0.80,
        "unsat_prob": 0.05,
        "single_arrival_rate": 2 / 3,
        "website_mode": True,
        "large_tube_capacity": 8,
        "label": "ALT6 (Kitchen+Website) [140k]",
    },
    "ALT7": {
        # Capacity Expansion: More tube slide runs
        "invest_cost": 75_000,
        "lunch_prob": 0.70,
        "unsat_prob": 0.10,
        "single_arrival_rate": 2 / 3,
        "website_mode": False,
        "large_tube_capacity": 12,
        "label": "ALT7 (Capacity Expansion) [75k]",
    },
    "ALT8": {
        # Premium Package: Quality + Marketing + Website + Capacity
        "invest_cost": 280_000,
        "lunch_prob": 0.86,
        "unsat_prob": 0.02,
        "single_arrival_rate": 18 / 15,  # 20% increase in singles
        "website_mode": True,
        "large_tube_capacity": 11,
        "label": "ALT8 (Premium: Quality+Marketing+Website+Capacity) [280k]",
    },
    "ALT9": {
        # Budget Marketing: Light marketing campaign, modest cost
        "invest_cost": 35_000,
        "lunch_prob": 0.72,
        "unsat_prob": 0.10,
        "single_arrival_rate": 11 / 10,  # 10% increase in singles
        "website_mode": False,
        "large_tube_capacity": 8,
        "label": "ALT9 (Budget Marketing) [35k]",
    },
    "ALT10": {
        # Kitchen + Capacity: Quality food + larger throughput
        "invest_cost": 160_000,
        "lunch_prob": 0.80,
        "unsat_prob": 0.05,
        "single_arrival_rate": 2 / 3,
        "website_mode": False,
        "large_tube_capacity": 10,
        "label": "ALT10 (Kitchen+Capacity) [160k]",
    },
}

def apply_scenario_to_globals(scn: dict) -> None:
    global SCN_LUNCH_PROB, SCN_UNSAT_PROB, SCN_SINGLE_ARRIVAL_RATE, SCN_WEBSITE_MODE, SCN_LARGE_TUBE_CAPACITY
    SCN_LUNCH_PROB = float(scn["lunch_prob"])
    SCN_UNSAT_PROB = float(scn["unsat_prob"])
    SCN_SINGLE_ARRIVAL_RATE = float(scn["single_arrival_rate"])
    SCN_WEBSITE_MODE = bool(scn["website_mode"])
    SCN_LARGE_TUBE_CAPACITY = int(scn["large_tube_capacity"])


def apply_scenario_to_park_resources(park, scn: dict) -> None:
    # LargeTubeSlide capacity change (ALT2)
    cap = int(scn["large_tube_capacity"])
    park.largeTubeSlide.capacity_per_server = cap
    park.largeTubeSlide.total_capacity = park.largeTubeSlide.servers * park.largeTubeSlide.capacity_per_server

# ----------------------------------------------------------- #
# ------------------------- Sampler -------------------------- #
# ----------------------------------------------------------- #


class Sampler:
    # -------- Generic samplers -------- #

    @staticmethod
    def sample_uniform(a: float, b: float) -> float:
        """Continuous Uniform(a, b)."""
        if b < a:
            raise ValueError("Upper bound must be >= lower bound in sample_uniform.")
        u = random.random()
        return a + (b - a) * u

    @staticmethod
    def sample_exponential(rate: float) -> float:
        """Exponential with the given rate parameter."""
        if rate <= 0:
            raise ValueError("Rate must be > 0 in sample_exponential.")
        u = random.random()
        return -math.log(1 - u) / rate

    @staticmethod
    def sample_normal(mean: float, std: float) -> float:
        """Normal via Box-Muller."""
        if std <= 0:
            raise ValueError("Standard deviation must be > 0 in sample_normal.")
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z

    @staticmethod
    def sample_discrete_uniform(m: int, n: int) -> int:
        """Discrete uniform on integers [m, n]."""
        if n < m:
            raise ValueError("Upper bound must be >= lower bound in sample_discrete_uniform.")
        u = random.random()
        return m + int((n - m + 1) * u)

    @staticmethod
    def sample_discrete_by_probs(values: Sequence[T], probs: Sequence[float]) -> T:
        """Sample from provided values with matching probabilities."""
        if len(values) != len(probs) or len(values) == 0:
            raise ValueError("Values and probs must be non-empty and of equal length.")
        if any(p < 0 for p in probs):
            raise ValueError("Probabilities must be non-negative.")
        total = sum(probs)
        if total == 0:
            raise ValueError("Sum of probabilities must be greater than zero.")
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Probabilities must sum to 1.0.")
        u = random.random()
        s = 0.0
        for value, p in zip(values, probs):
            s += p
            if u <= s:
                return value
        return values[-1]

    @staticmethod
    def sample_time_uniform_window(a: float, b: float, t_now: float) -> Optional[float]:
        """Uniform in [a, b] conditioned on current time; returns None if t_now >= b."""
        if t_now >= b:
            return None
        if t_now <= a:
            return Sampler.sample_uniform(a, b)
        return t_now + Sampler.sample_uniform(0, b - t_now)

    @staticmethod
    def f(x):
        if x <= 30:
            return (x / 2700)
        if x <= 50:
            return ((60 - x) / (2700) + (1 / 30))
        if x <= 60:
            return ((60 - x) / (2700))
        return 0

    @staticmethod
    def sample_acceptance_rejection():
        while True:
            y = 60 * random.random()
            u2 = random.random()
            a = Sampler.f(y) / (2 / 45)
            if u2 <= a:
                return y

    # -------- Reception (Entrance) -------- #

    @staticmethod
    def sample_ticket_purchase_service_time() -> float:
        """Ticket purchase time ~ Uniform(0.5, 2)."""
        return Sampler.sample_uniform(0.5, 2.0)

    @staticmethod
    def sample_wristband_service_time() -> float:
        """Wristband issuing time ~ Exponential(rate=0.5)."""
        return Sampler.sample_exponential(0.5)

    @staticmethod
    def sample_buy_express_at_entrance() -> bool:
        """Buy express at entrance with probability 0.25."""
        return random.random() <= 0.25

    # -------- Families -------- #

    @staticmethod
    def next_family_arrival_time(t_now: float) -> Optional[float]:
        if t_now >= NOON:
            return None
        inter = Sampler.sample_exponential(0.6666666667)
        t_next = t_now + inter
        return t_next if t_next <= NOON else None

    @staticmethod
    def sample_num_children_in_family():
        return Sampler.sample_discrete_uniform(1, 5)

    @staticmethod
    def sample_child_age():
        return Sampler.sample_uniform(2, 18)

    @staticmethod
    def sample_family_splits():
        u = random.random()
        return u <= 0.6

    @staticmethod
    def sample_num_groups_after_split():
        return Sampler.sample_discrete_uniform(2, 3)

    @staticmethod
    def sample_family_departure_time(t_now: float) -> Optional[float]:
        """Family departure time on [420, 600] with CDF ((t-420)/180)^2, conditioned so t >= t_now."""
        if t_now >= CLOSE:
            return None
        if t_now <= FOUR_PM:
            u = random.random()
            return FOUR_PM + 180.0 * math.sqrt(u)
        s0 = (t_now - FOUR_PM) / 180.0
        u = random.random()
        s = math.sqrt(s0 * s0 + (1.0 - s0 * s0) * u)
        return FOUR_PM + 180.0 * s

    @staticmethod
    def family_queue_abandon_time(t_join_queue: float) -> float:
        patience = 15
        return t_join_queue + patience

    # -------- Teenagers -------- #

    @staticmethod
    def next_teens_arrival_time(t_now: float) -> Optional[float]:
        base_time = max(t_now, TEN_AM)
        if base_time >= FOUR_PM:
            return None
        inter = Sampler.sample_exponential(1.3888889)
        t_next = base_time + inter
        return t_next if t_next <= FOUR_PM else None

    @staticmethod
    def sample_teen_group_size():
        u = random.random()
        if u <= 0.4:
            return Sampler.sample_discrete_uniform(2, 3)
        if u <= 0.9:
            return Sampler.sample_discrete_uniform(4, 5)
        return 6

    @staticmethod
    def teen_queue_abandon_time(t_join_queue: float) -> float:
        """Abandonment time for teens in regular queue."""
        patience = 20
        return t_join_queue + patience

    @staticmethod
    def single_queue_abandon_time(t_join_queue: float) -> float:
        patience = 30
        return t_join_queue + patience

    @staticmethod
    def sample_teen_departure_time(t_now: float) -> Optional[float]:
        return CLOSE

    # -------- singles -------- #
    @staticmethod
    def next_single_arrival_time(t_now: float) -> Optional[float]:
        if t_now >= CLOSE - 30:
            return None
        inter = Sampler.sample_exponential(SCN_SINGLE_ARRIVAL_RATE)  # <-- scenario-controlled
        t_next = t_now + inter
        return t_next if t_next <= CLOSE - 30 else None

    @staticmethod
    def sample_single_departure_time(t_now: float) -> Optional[float]:
        return CLOSE

    # -------- Food (Restaurants) -------- #

    @staticmethod
    def decide_go_to_lunch(t_now: float) -> bool:
        """Between 13:00-15:00 (240-360), go to lunch with probability; otherwise False."""
        if t_now < 240 or t_now > 360:
            return False
        return random.random() <= SCN_LUNCH_PROB  # <-- scenario-controlled

    @staticmethod
    def sample_restaurant_choice():
        """Choose restaurant per given probabilities."""
        values = [HAMBURGER, PIZZA, SALAD]
        probs = [3 / 8, 1 / 4, 3 / 8]
        return Sampler.sample_discrete_by_probs(values, probs)

    @staticmethod
    def sample_food_prep_time_by_restaurant(restaurant: str) -> float:
        """Prep time depends on restaurant."""
        if restaurant == PIZZA:
            return Sampler.sample_uniform(4, 6)
        if restaurant == HAMBURGER:
            return Sampler.sample_uniform(3, 4)
        if restaurant == SALAD:
            return Sampler.sample_uniform(3, 7)
        raise ValueError("Unknown restaurant for prep time sampling.")

    @staticmethod
    def sample_food_service_time_at_counter() -> float:
        """Counter service ~ Normal(5, 1.5) with rejection of negatives."""
        t = Sampler.sample_normal(5, 1.5)
        while t < 0:
            t = Sampler.sample_normal(5, 1.5)
        return t

    @staticmethod
    def sample_meal_duration() -> float:
        """Meal duration ~ Uniform(15, 35)."""
        return Sampler.sample_uniform(15, 35)

    @staticmethod
    def expected_food_prep_time_by_restaurant(restaurant: str) -> float:
        if restaurant == PIZZA:
            return 5.0
        if restaurant == HAMBURGER:
            return 3.5
        if restaurant == SALAD:
            return 5.0
        raise ValueError("Unknown restaurant for expected prep time.")

    @staticmethod
    def expected_food_service_time_at_counter() -> float:
        # Mean of Normal(5,1.5) truncated to t >= 0.
        mu = 5.0
        sigma = 1.5
        alpha = -mu / sigma
        sqrt_2 = math.sqrt(2.0)
        phi = math.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + math.erf(alpha / sqrt_2))
        tail = 1.0 - Phi
        return mu + sigma * (phi / tail)

    @staticmethod
    def expected_meal_duration() -> float:
        return 25.0

    @staticmethod
    def sample_food_satisfaction() -> str:
        """Return SATISFIED/UNSATISFIED with scenario-controlled probability."""
        return "UNSATISFIED" if random.random() <= SCN_UNSAT_PROB else "SATISFIED"  # <-- scenario-controlled

    # -------- Attractions (Ride/Activity durations) -------- #

    @staticmethod
    def sample_lazy_river_duration() -> float:
        """Lazy River duration ~ Uniform(20, 30)."""
        return Sampler.sample_uniform(20, 30)

    @staticmethod
    def sample_single_slide_duration() -> float:
        """Single Slide duration is constant 3."""
        return 3.0

    @staticmethod
    def sample_wave_pool_duration() -> float:
        """Wave pool stay duration via acceptance-rejection on [0, 60]."""
        return Sampler.sample_acceptance_rejection()

    @staticmethod
    def sample_kids_pool_duration() -> float:
        """Kids pool stay duration (minutes) via piecewise inverse CDF on [60, 120]."""
        u = random.random()
        if u <= (1 / 6):
            return 60 + math.sqrt(1350 * u)
        if u <= (5 / 6):
            return 75 + 45 * (u - 1 / 6)
        return 120 - math.sqrt(1350 * (1 - u))

    @staticmethod
    def sample_snorkel_tour_duration() -> float:
        """Snorkel tour duration ~ Normal(30, 10) with rejection of negatives."""
        t = Sampler.sample_normal(30, 10)
        while t < 0:
            t = Sampler.sample_normal(30, 10)
        return t

    @staticmethod
    def sample_big_tube_slide_duration() -> float:
        """Big tube slide duration ~ Normal(mean=4.8, std=1.83); rejects negatives."""
        t = Sampler.sample_normal(4.8, 1.83)
        while t < 0:
            t = Sampler.sample_normal(4.8, 1.83)
        return t

    @staticmethod
    def sample_small_tube_slide_duration() -> float:
        """Small tube slide duration ~ Exponential(rate=2.107)."""
        return Sampler.sample_exponential(2.107)

    @staticmethod
    def sample_attraction_duration(attraction_name: str) -> float:
        """Dispatch to the appropriate attraction duration sampler by name."""
        duration_by_name = {
            "LazyRiver": Sampler.sample_lazy_river_duration,
            "SingleSlides": Sampler.sample_single_slide_duration,
            "LargeTubeSlide": Sampler.sample_big_tube_slide_duration,
            "SmallTubeSlide": Sampler.sample_small_tube_slide_duration,
            "WavePool": Sampler.sample_wave_pool_duration,
            "ToddlerPool": Sampler.sample_kids_pool_duration,
            "SnorkelTour": Sampler.sample_snorkel_tour_duration,
        }
        try:
            sampler = duration_by_name[attraction_name]
        except KeyError:
            raise ValueError(f"Unknown attraction: {attraction_name}")
        return sampler()

    # -------- Project policy samplers -------- #

    @staticmethod
    def sample_has_express_at_entry() -> bool:
        # 25% buy express at entrance
        return random.random() <= 0.25

    @staticmethod
    def sample_family_size_project() -> int:
        # 2 parents + U{1..5} kids
        return 2 + Sampler.sample_discrete_uniform(1, 5)

    @staticmethod
    def sample_teens_group_size_project() -> int:
        # 2–3 with p=0.4, 4–5 with p=0.5, 6 with p=0.1
        u = random.random()
        if u <= 0.4:
            return Sampler.sample_discrete_uniform(2, 3)
        if u <= 0.9:
            return Sampler.sample_discrete_uniform(4, 5)
        return 6

    @staticmethod
    def sample_family_arrival_window() -> tuple[float, float]:
        return OPEN, NOON

    @staticmethod
    def sample_teens_arrival_window() -> tuple[float, float]:
        return TEN_AM, FOUR_PM

    @staticmethod
    def sample_single_arrival_window() -> tuple[float, float]:
        return OPEN, CLOSE - 30

    @staticmethod
    def sample_parent_age() -> float:
        return Sampler.sample_uniform(25.0, 55.0)

    @staticmethod
    def sample_teen_age() -> float:
        return Sampler.sample_uniform(14.0, 17.999)

    @staticmethod
    def sample_single_age() -> float:
        return Sampler.sample_uniform(18.0, 60.0)


class PriorityQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[int, float, int, "Visitor"]] = []
        self._counter: int = 0

    def push(self, visitor: "Visitor", is_express: bool, join_time: float) -> None:
        priority = 0 if is_express else 1
        item = (priority, float(join_time), self._counter, visitor)
        heapq.heappush(self._heap, item)
        self._counter += 1

    def peek_item(self) -> Optional[tuple[int, float, int, "Visitor"]]:
        """Return the head item without removing it, or None if empty."""
        return self._heap[0] if self._heap else None

    def pop_item(self) -> tuple[int, float, int, "Visitor"]:
        """Remove and return the head item. Raises IndexError if empty."""
        return heapq.heappop(self._heap)

    # --- Optional backwards-compatible aliases (keep only if old code still calls them) ---

    def peek(self) -> Optional[tuple[int, float, int, "Visitor"]]:
        return self.peek_item()

    def pop(self) -> tuple[int, float, int, "Visitor"]:
        return self.pop_item()

    # --- Utility ---

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return not self._heap

    def items(self) -> list[tuple[int, float, int, "Visitor"]]:
        """Debug/inspection helper (unordered heap view)."""
        return list(self._heap)

    def reset(self) -> None:
        self._heap.clear()
        self._counter = 0

    def remove_visitor(self, target: "Visitor") -> bool:
        n0 = len(self._heap)
        if n0 == 0:
            return False
        self._heap = [item for item in self._heap if item[3] is not target]
        removed = (len(self._heap) != n0)
        if removed:
            heapq.heapify(self._heap)
        return removed


class Attraction:
    """Queue-managed attraction with express priority and FIFO within each priority."""

    def __init__(
        self,
        name: str,
        servers: int,
        capacity_per_server: int,
        adrenaline: int,
        break_time: float = 0.0,
        min_age: float = 0.0,
        max_age: float = 100.0,
    ) -> None:
        self.name = name
        self.servers = servers
        self.capacity_per_server = capacity_per_server
        self.adrenaline = adrenaline
        self.closed: bool = False
        if name == "SnorkelTour":
            self.break_time = 30.0
        elif name == "SingleSlides":
            self.break_time = 0.5
        else:
            self.break_time = break_time

        headway_by_name = {
            "SingleSlides": 0.0,
            "SnorkelTour": 10.0,
            "SmallTubeSlide": 0.0,
            "LargeTubeSlide": 5.0,
            "WavePool": 10.0,
            "ToddlerPool": 7.0,
            "LazyRiver": 0.0,
        }
        self.headway_time: float = float(headway_by_name.get(name, 0.0))

        self.min_age = min_age
        self.max_age = max_age
        self.total_capacity = servers * capacity_per_server

        self.next_free_time_array: list[float] = [0.0 for _ in range(servers)]
        self.is_running_array: list[bool] = [False for _ in range(servers)]
        self.in_service_list_per_server: list[list[tuple["Visitor", "Visitor"]]] = [[] for _ in range(servers)]
        self.pending_boarded_per_server: list[list[tuple["Visitor", "Visitor"]]] = [[] for _ in range(servers)]
        self.pending_headway_deadline_per_server: list[Optional[float]] = [None for _ in range(servers)]
        self.pending_duration_per_server: list[Optional[float]] = [None for _ in range(servers)]
        self.Q = PriorityQueue()

    def allows_age(self, age: float) -> bool:
        return self.min_age <= float(age) <= self.max_age

    def reset(self) -> None:
        self.next_free_time_array = [0.0 for _ in range(self.servers)]
        self.is_running_array = [False for _ in range(self.servers)]
        self.in_service_list_per_server = [[] for _ in range(self.servers)]
        self.pending_boarded_per_server = [[] for _ in range(self.servers)]
        self.pending_headway_deadline_per_server = [None for _ in range(self.servers)]
        self.pending_duration_per_server = [None for _ in range(self.servers)]
        self.Q.reset()
        self.closed = False

    def enqueue(self, visitor: "Visitor", t_now: float) -> None:
        if self.closed:
            visitor.status = "IN_PARK"
            return
        if not visitor.can_enter_attraction(self):
            visitor.status = "IN_PARK"
            return
        visitor.status = "IN_QUEUE"

        visitor.reset_split_for_new_attraction_queue()
        self.Q.push(visitor, visitor.has_express, t_now)

    def available_server_to_fill(self, t_now: float) -> int:
        for idx in range(self.servers):
            if self.pending_boarded_per_server[idx]:
                continue
            if (not self.is_running_array[idx]) and (t_now >= self.next_free_time_array[idx]):
                return idx
        return -1

    def q_length(self) -> int:
        """Total number of people currently waiting in this attraction's queue."""
        return sum(item[3].group_in_queue for item in self.Q.items())

    def sample_duration_and_finish(self, t_now: float) -> tuple[float, float]:
        duration = Sampler.sample_attraction_duration(self.name)
        return duration, t_now + duration

    def fill_one_server_from_queue(
        self,
        server_idx: int,
        t_now: float,
        target_list: Optional[list[tuple["Visitor", "Visitor"]]] = None,
    ) -> int:
        if server_idx < 0 or server_idx >= self.servers:
            raise IndexError("Invalid server index")

        if self.is_running_array[server_idx] or t_now < self.next_free_time_array[server_idx]:
            return 0

        if target_list is None:
            target_list = self.in_service_list_per_server[server_idx]

        seats_left = self.capacity_per_server
        allocated = 0

        while seats_left > 0:
            head_item = self.Q.peek_item()
            if head_item is None:
                break

            visitor = head_item[3]
            waiting = int(visitor.group_in_queue)

            if waiting <= 0:
                self.Q.pop_item()
                continue

            k = min(seats_left, waiting)

            allocated_members = visitor.take_k_members(k)
            for member in allocated_members:
                target_list.append((member, visitor))

            visitor.group_in_queue -= k
            visitor.group_in_attraction += k

            seats_left -= k
            allocated += k

            if visitor.group_in_queue == 0:
                self.Q.pop_item()
                continue

            continue

        return allocated

    def can_start(self, t_now: float) -> bool:
        if self.closed:
            return False
        return (self.available_server_to_fill(t_now) != -1) and (self.q_length() > 0)

    def _start_run_with_boarded(
        self,
        server_idx: int,
        t_now: float,
        duration: float,
        boarded: list[tuple["Visitor", "Visitor"]],
    ) -> float:
        finish_time = t_now + duration
        self.in_service_list_per_server[server_idx] = list(boarded)
        boarded.clear()
        self.is_running_array[server_idx] = True
        self.next_free_time_array[server_idx] = finish_time + self.break_time
        return finish_time

    def is_running(self) -> bool:
        return any(self.is_running_array)

    def start_servers_if_possible(self, t_now: float) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        started: list[tuple[int, float]] = []
        headway_waiting: list[tuple[int, float]] = []

        if self.closed:
            return started, headway_waiting

        # If a server is pending and new visitors arrive, start immediately.
        if self.q_length() > 0:
            for idx in range(self.servers):
                if self.is_running_array[idx] or t_now < self.next_free_time_array[idx]:
                    continue
                if not self.pending_boarded_per_server[idx]:
                    continue
                self.fill_one_server_from_queue(
                    idx,
                    t_now,
                    target_list=self.pending_boarded_per_server[idx],
                )
                duration = self.pending_duration_per_server[idx]
                if duration is None:
                    duration, finish_time = self.sample_duration_and_finish(t_now)
                    if finish_time > CLOSE:
                        self.closed = True
                        return started, headway_waiting
                finish_time = self._start_run_with_boarded(
                    idx, t_now, duration, self.pending_boarded_per_server[idx]
                )
                self.pending_headway_deadline_per_server[idx] = None
                self.pending_duration_per_server[idx] = None
                started.append((idx, finish_time))

        for idx in range(self.servers):
            if self.is_running_array[idx] or t_now < self.next_free_time_array[idx]:
                continue
            if self.pending_boarded_per_server[idx]:
                continue
            if self.q_length() == 0:
                continue

            duration, finish_time = self.sample_duration_and_finish(t_now)
            if finish_time > CLOSE:
                self.closed = True
                return started, headway_waiting

            if self.headway_time > 0.0:
                allocated = self.fill_one_server_from_queue(
                    idx,
                    t_now,
                    target_list=self.pending_boarded_per_server[idx],
                )
                if allocated <= 0:
                    continue

                if allocated < self.capacity_per_server and self.q_length() == 0:
                    latest_finish = t_now + self.headway_time + duration
                    if latest_finish <= CLOSE:
                        deadline = t_now + self.headway_time
                        self.pending_headway_deadline_per_server[idx] = deadline
                        self.pending_duration_per_server[idx] = duration
                        headway_waiting.append((idx, deadline))
                        continue

                finish_time = self._start_run_with_boarded(
                    idx, t_now, duration, self.pending_boarded_per_server[idx]
                )
                self.pending_headway_deadline_per_server[idx] = None
                self.pending_duration_per_server[idx] = None
                started.append((idx, finish_time))
            else:
                allocated = self.fill_one_server_from_queue(idx, t_now)
                if allocated <= 0:
                    continue
                self.is_running_array[idx] = True
                self.next_free_time_array[idx] = finish_time + self.break_time
                started.append((idx, finish_time))

        return started, headway_waiting

    def complete_server(self, server_idx: int) -> list[tuple["Visitor", "Visitor"]]:
        if server_idx < 0 or server_idx >= self.servers:
            raise IndexError("Invalid server index")

        serviced = list(self.in_service_list_per_server[server_idx])

        counts: dict["Visitor", int] = {}
        for _, parent in serviced:
            counts[parent] = counts.get(parent, 0) + 1
        for parent, cnt in counts.items():
            remaining = parent.group_in_attraction - cnt
            if remaining < 0:
                raise RuntimeError("group_in_attraction cannot be negative after completion")
            parent.group_in_attraction = remaining

        self.in_service_list_per_server[server_idx] = []
        self.is_running_array[server_idx] = False

        return serviced

    def remove_from_queue(self, visitor: "Visitor") -> bool:
        """Remove visitor from this attraction waiting queue."""
        return self.Q.remove_visitor(visitor)


class Reception:
    def __init__(self, servers: int) -> None:
        if servers <= 0:
            raise ValueError("Reception servers must be positive")
        self.servers = servers
        self.busy = 0
        self.Q = PriorityQueue()

    def reset(self) -> None:
        self.busy = 0
        self.Q.reset()

    def can_start_service(self) -> bool:
        return self.busy < self.servers

    def start_service(self) -> None:
        if not self.can_start_service():
            raise RuntimeError("Reception cannot start service now")
        self.busy += 1

    def finish_service(self) -> None:
        if self.busy <= 0:
            raise RuntimeError("finish_service called with no busy reception servers")
        self.busy -= 1

    def enqueue(self, visitor: "Visitor", t_now: float) -> None:
        self.Q.push(visitor, False, t_now)

    def is_queue_empty(self) -> bool:
        return len(self.Q) == 0

    def sample_reception_service_time(self) -> float:
        # Website scenario: "bracelet only"
        if SCN_WEBSITE_MODE:
            return Sampler.sample_wristband_service_time()
        return Sampler.sample_ticket_purchase_service_time() + Sampler.sample_wristband_service_time()


class Restaurant:
    def __init__(self, name: str, servers: int) -> None:
        if servers <= 0:
            raise ValueError(f"Restaurant servers must be positive: {name}")
        self.name = name
        self.servers = servers
        self.busy = 0
        self.Q = PriorityQueue()

    def has_free_server(self) -> bool:
        return self.busy < self.servers

    def start_service(self) -> None:
        if self.busy >= self.servers:
            raise RuntimeError(f"No free servers available in restaurant: {self.name}")
        self.busy += 1

    def finish_service(self) -> None:
        if self.busy <= 0:
            raise RuntimeError(f"finish_service called with no busy servers in restaurant: {self.name}")
        self.busy -= 1

    def reset(self) -> None:
        self.busy = 0
        self.Q.reset()

    def enqueue(self, visitor: "Visitor", t_now_value: float, is_express: bool = False) -> None:
        self.Q.push(visitor, is_express, t_now_value)


class FoodCourt:
    def __init__(self, hamburger_servers: int, pizza_servers: int, salad_servers: int) -> None:
        if hamburger_servers <= 0 or pizza_servers <= 0 or salad_servers <= 0:
            raise ValueError("FoodCourt restaurant server counts must be positive")
        self.restaurants: dict[str, dict[str, object]] = {
            HAMBURGER: {"restaurant": Restaurant(HAMBURGER, hamburger_servers), "price": 100.0},
            PIZZA: {
                "restaurant": Restaurant(PIZZA, pizza_servers),
                "family_price": 100.0,
                "personal_price": 40.0,
            },
            SALAD: {"restaurant": Restaurant(SALAD, salad_servers), "price": 65.0},
        }
        self.income: float = 0.0

    def add_income(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("Income amount must be non-negative")
        self.income += amount

    def get_price(self, restaurant_name: str, visitor: "Visitor") -> int:
        if restaurant_name not in self.restaurants:
            raise ValueError(f"Unknown restaurant: {restaurant_name}")
        entry = self.restaurants[restaurant_name]
        if restaurant_name == PIZZA:
            return int(entry["personal_price"]) if visitor.group_size == 1 else int(entry["family_price"])
        return int(entry["price"])

    def reset(self) -> None:
        self.income = 0.0
        for entry in self.restaurants.values():
            entry["restaurant"].reset()


class Visitor:
    def __init__(self) -> None:
        self.name: str = "Visitor"
        self.group_size: int = 1
        self.has_express: bool = False
        self.max_wait_time: Optional[float] = None
        self.age: Optional[float] = 0.0
        self.departure_time: float = CLOSE
        self.has_eaten: bool = False
        self.going_to_lunch: bool = False
        self.arrival_window_start: float = OPEN
        self.arrival_window_end: float = CLOSE
        self.current_location: Optional[str] = None

        self.status: str = "IN_REC"
        self.rating: float = 10.0
        self.remaining_attractions: list["Attraction"] = []
        self.waiting_in_attraction: Optional["Attraction"] = None

        self.members: list["Visitor"] = [self]
        self.group_in_queue: int = 0
        self.group_in_attraction: int = 0
        self._member_cursor: int = 0
        self.parent_group: Optional["Visitor"] = None
        self.member_index: Optional[int] = None
        self.photos_processed: bool = False
        self.reception_paid: bool = False
        self.leave_counted: bool = False

    def reset_runtime_state(self) -> None:
        self.status = "IN_REC"
        self.group_in_attraction = 0
        self.group_in_queue = 0
        self._member_cursor = 0

    def can_abandon_queue(self) -> bool:
        return not self.has_express

    def choose_restaurant(self) -> str:
        return Sampler.sample_restaurant_choice()

    def should_go_to_lunch(self, t_now: float) -> bool:
        return Sampler.decide_go_to_lunch(t_now) and not self.has_eaten

    def can_enter_attraction(self, attraction: "Attraction") -> bool:
        return attraction.allows_age(float(self.age))

    def choose_next_attraction(self, t_now: float) -> Attraction:
        return random.choice(self.remaining_attractions)

    def post_abandon_action(self) -> Optional[str]:
        return None

    def has_remaining_attractions(self) -> bool:
        return len(self.remaining_attractions) > 0

    def mark_attraction_completed(self, attraction: "Attraction") -> None:
        if attraction in self.remaining_attractions:
            self.remaining_attractions.remove(attraction)

    def apply_post_attraction_rating(self, attraction: "Attraction") -> None:
        gs = float(self.group_size)
        adrenaline = float(attraction.adrenaline)
        if random.random() <= 0.5:
            score = ((gs - 1.0) / 5.0) * 0.3 + ((adrenaline - 1.0) / 4.0) * 0.7
            self.rating = min(PARK_MAX_RATING, float(self.rating) + score)
        else:
            self.rating = max(0.0, float(self.rating) - 0.1)

    def reset_split_for_new_attraction_queue(self) -> None:
        self.group_in_queue = self.group_size
        self.group_in_attraction = 0
        self._member_cursor = 0

    def take_k_members(self, k: int) -> list["Visitor"]:
        if k < 0 or k > self.group_in_queue:
            raise RuntimeError("take_k_members called with k outside waiting group size.")
        if self._member_cursor + k > len(self.members):
            raise RuntimeError("take_k_members would exceed available members.")
        start = self._member_cursor
        end = start + k
        self._member_cursor = end
        return self.members[start:end]


class Family(Visitor):
    _FAMILY_UID_COUNTER = 0

    def __init__(self, t_now: float) -> None:
        super().__init__()
        self.name = "Family"
        size = Sampler.sample_family_size_project()
        num_children = max(0, size - 2)
        self.ages: list[float] = [
            Sampler.sample_parent_age(),
            Sampler.sample_parent_age(),
        ] + [Sampler.sample_child_age() for _ in range(num_children)]
        if len(self.ages) != size:
            raise RuntimeError("Family ages list must match sampled size")

        self.has_express = False
        self.max_wait_time = 15.0
        self.arrival_window_start, self.arrival_window_end = Sampler.sample_family_arrival_window()
        t_dep = Sampler.sample_family_departure_time(t_now)
        self.departure_time = CLOSE if t_dep is None else float(t_dep)
        self._need_reunite: bool = self.departure_time < CLOSE
        self._departure_triggered: bool = False
        self._ready_to_leave: set[int] = set()
        self._subgroup_ids: set[int] = {0}
        self._exited: bool = False
        self._meeting_check_scheduled: bool = False

        self._assign_members_from_ages(self.ages)
        self.original_group_size: int = self.group_size
        self.age = self.members[0].age if self.members else 0.0

        Family._FAMILY_UID_COUNTER += 1
        self.family_uid: int = Family._FAMILY_UID_COUNTER

        self.parent_family_uid: int = self.family_uid
        self.subgroup_id: Optional[int] = None
        self._next_subgroup_id: int = 1
        self.root_family: "Family" = self
        self.family_units: list["Family"] = [self]

        self.has_split = False
        self.subgroups: list["Family"] = []
        self.did_all_ages_first: bool = False

    def _assign_members_from_ages(self, ages: list[float]) -> None:
        members: list[Visitor] = []
        for idx, age in enumerate(ages):
            person = Visitor()
            person.age = float(age)
            person.remaining_attractions = []
            person.group_size = 1
            person.members = [person]
            person.has_express = self.has_express
            person.max_wait_time = self.max_wait_time
            person.departure_time = self.departure_time
            person.parent_group = self
            person.member_index = idx
            members.append(person)
        self.members = members
        self.group_size = len(members)

    def can_enter_attraction(self, attraction: "Attraction") -> bool:
        return all(attraction.allows_age(float(m.age)) for m in self.members)

    def create_subgroup_from_indices(self, member_indices: list[int]) -> "Family":
        if self.root_family is not self:
            raise RuntimeError("Only the root family may create subgroups.")
        if not member_indices:
            raise ValueError("member_indices cannot be empty")

        if any((i < 0 or i >= len(self.ages)) for i in member_indices):
            raise ValueError("member_indices contains out-of-range index")

        unique_sorted = sorted(set(member_indices))
        subgroup_ages = [self.ages[i] for i in unique_sorted]

        subgroup = Family.__new__(Family)
        Visitor.__init__(subgroup)
        subgroup.name = "Family"
        subgroup.has_express = self.has_express
        subgroup.max_wait_time = self.max_wait_time
        subgroup.arrival_window_start = self.arrival_window_start
        subgroup.arrival_window_end = self.arrival_window_end
        subgroup.departure_time = self.departure_time
        subgroup.has_eaten = self.has_eaten
        subgroup.going_to_lunch = self.going_to_lunch
        subgroup.remaining_attractions = list(self.remaining_attractions)
        subgroup.status = "IN_PARK"
        subgroup.rating = 0.0
        subgroup.did_all_ages_first = self.did_all_ages_first

        subgroup.ages = list(subgroup_ages)
        if len(subgroup.ages) <= 0:
            raise RuntimeError("Subgroup group_size must be positive")
        subgroup._assign_members_from_ages(subgroup.ages)
        subgroup.original_group_size = subgroup.group_size
        subgroup.age = subgroup.members[0].age if subgroup.members else 0.0

        subgroup.family_uid = self.family_uid
        subgroup.parent_family_uid = self.parent_family_uid

        subgroup.subgroup_id = self._next_subgroup_id
        self._next_subgroup_id += 1

        subgroup._next_subgroup_id = 1
        subgroup.has_split = False
        subgroup.subgroups = []
        root = self.root_family
        subgroup.root_family = root
        subgroup.family_units = root.family_units
        root.family_units.append(subgroup)
        root.subgroup_ids.add(subgroup.subgroup_id)

        for i in reversed(unique_sorted):
            del self.ages[i]
        self._assign_members_from_ages(self.ages)
        self.age = self.members[0].age if self.members else 0.0

        self.has_split = True
        self.subgroups.append(subgroup)

        return subgroup

    def validate_conservation(self) -> None:
        if self.subgroup_id is not None:
            return
        total = self.group_size + sum(sg.group_size for sg in self.subgroups)
        if total != self.original_group_size:
            raise RuntimeError(
                f"Family size conservation violated: expected {self.original_group_size}, got {total}"
            )

    def split_into_subgroups(self, num_subgroups: int) -> list["Family"]:
        if num_subgroups not in (2, 3):
            raise ValueError("num_subgroups must be 2 or 3")
        if len(self.ages) < 2:
            raise RuntimeError("Family must have two parents at indices 0 and 1 to split")
        if self.has_split or self.subgroups:
            raise RuntimeError("Family has already been split")

        parent_indices = [0, 1]
        children_indices = list(range(2, len(self.ages)))

        index_sets: list[list[int]]
        if num_subgroups == 2:
            sg1_indices = [parent_indices[0]]
            sg2_indices = [parent_indices[1]]
            toggle = 0
            for child_idx in children_indices:
                if toggle == 0:
                    sg1_indices.append(child_idx)
                else:
                    sg2_indices.append(child_idx)
                toggle = 1 - toggle
            index_sets = [sg1_indices, sg2_indices]
        else:
            if not children_indices:
                raise ValueError("Cannot split into 3 subgroups without at least one child")
            eldest_child_idx = max(children_indices, key=lambda i: self.ages[i])
            remaining_children = [i for i in children_indices if i != eldest_child_idx]
            sg1_indices = [parent_indices[0]]
            sg2_indices = [parent_indices[1]]
            sg3_indices = [eldest_child_idx]
            dests = [sg1_indices, sg2_indices, sg3_indices]
            turn = 0
            for idx in remaining_children:
                dests[turn].append(idx)
                turn = (turn + 1) % 3
            index_sets = dests

        created: list["Family"] = []
        removed: set[int] = set()
        for orig_indices in index_sets:
            adjusted: list[int] = []
            for idx in sorted(orig_indices):
                offset = sum(1 for r in removed if r < idx)
                adjusted.append(idx - offset)
            subgroup = self.create_subgroup_from_indices(adjusted)
            subgroup.original_group_size = subgroup.group_size
            created.append(subgroup)
            removed.update(orig_indices)

        self.subgroups = created
        self.has_split = True
        self.validate_conservation()
        return created

    def total_people_in_family_tree(self) -> int:
        return self.group_size + sum(sg.group_size for sg in self.subgroups)

    @property
    def need_reunite(self) -> bool:
        return self.root_family._need_reunite

    @property
    def departure_triggered(self) -> bool:
        return self.root_family._departure_triggered

    @departure_triggered.setter
    def departure_triggered(self, value: bool) -> None:
        self.root_family._departure_triggered = bool(value)

    @property
    def ready_to_leave(self) -> set[int]:
        return self.root_family._ready_to_leave

    @property
    def subgroup_ids(self) -> set[int]:
        return self.root_family._subgroup_ids

    @property
    def exited(self) -> bool:
        return self.root_family._exited

    @exited.setter
    def exited(self, value: bool) -> None:
        self.root_family._exited = bool(value)

    def effective_subgroup_id(self) -> int:
        return 0 if self.subgroup_id is None else int(self.subgroup_id)

    def can_abandon_queue(self) -> bool:
        return not self.has_express

    def choose_next_attraction(self, t_now: float) -> "Attraction":
        def queue_len(attr: "Attraction") -> int:
            if hasattr(attr, "q_length"):
                return attr.q_length()
            return sum(v.group_size for _, _, _, v in attr.Q.items())

        def pick_min_queue(cands: list[tuple[int, "Attraction"]]) -> "Attraction":
            m = min(q for q, _ in cands)
            tied = [attr for q, attr in cands if q == m]
            return random.choice(tied)

        if not self.remaining_attractions:
            raise RuntimeError("No remaining attractions to choose from")

        all_ages_names = {"LazyRiver", "LargeTubeSlide"}
        if not self.did_all_ages_first:
            candidates = [
                (queue_len(attr), attr)
                for attr in self.remaining_attractions
                if attr.name in all_ages_names and self.can_enter_attraction(attr)
            ]
            if candidates:
                return pick_min_queue(candidates)
            self.did_all_ages_first = True

        candidates = [
            (queue_len(attr), attr)
            for attr in self.remaining_attractions
            if self.can_enter_attraction(attr)
        ]
        if not candidates:
            raise RuntimeError("No feasible attractions to choose from")
        return pick_min_queue(candidates)

    def mark_attraction_completed(self, attraction: "Attraction") -> None:
        super().mark_attraction_completed(attraction)


class Teens(Visitor):
    def __init__(self, t_now: float) -> None:
        super().__init__()
        self.name = "Teens"
        size = Sampler.sample_teens_group_size_project()
        self.ages: list[float] = [Sampler.sample_teen_age() for _ in range(size)]
        self.has_express = False
        self.max_wait_time = 20.0
        self.arrival_window_start, self.arrival_window_end = Sampler.sample_teens_arrival_window()
        self.departure_time = CLOSE
        self._assign_members_from_ages(self.ages)
        self.age = self.members[0].age if self.members else Sampler.sample_teen_age()
        self.pending_return_attraction: Optional["Attraction"] = None
        self.detour_done: bool = False
        self.returning_to_pending: bool = False

    def _assign_members_from_ages(self, ages: list[float]) -> None:
        members: list[Visitor] = []
        for idx, age in enumerate(ages):
            person = Visitor()
            person.age = float(age)
            person.group_size = 1
            person.members = [person]
            person.has_express = self.has_express
            person.max_wait_time = self.max_wait_time
            person.departure_time = self.departure_time
            person.parent_group = self
            person.member_index = idx
            person.remaining_attractions = []
            members.append(person)
        self.members = members
        self.group_size = len(members)

    def can_abandon_queue(self) -> bool:
        return not self.has_express

    def can_enter_attraction(self, attraction: "Attraction") -> bool:
        return all(attraction.allows_age(float(m.age)) for m in self.members)

    def post_abandon_action(self) -> str:
        return BUY_EXPRESS_AND_JOIN_EXPRESS_QUEUE if random.random() <= 0.6 else GO_TO_ANOTHER_ATTRACTION

    def choose_next_attraction(self, t_now: float) -> "Attraction":
        if not self.remaining_attractions:
            raise RuntimeError("No remaining attractions to choose from")

        self.returning_to_pending = False

        if self.pending_return_attraction is not None:
            if not self.detour_done:
                non_pending = [
                    attr for attr in self.remaining_attractions
                    if attr is not self.pending_return_attraction and self.can_enter_attraction(attr)
                ]
                if non_pending:
                    chosen = random.choice(non_pending)
                    self.detour_done = True
                    return chosen

            if self.pending_return_attraction in self.remaining_attractions and self.can_enter_attraction(self.pending_return_attraction):
                self.returning_to_pending = True
                chosen = self.pending_return_attraction
                self.pending_return_attraction = None
                self.detour_done = False
                return chosen
            else:
                self.pending_return_attraction = None
                self.detour_done = False

        adrenaline_candidates = [
            attr for attr in self.remaining_attractions
            if attr.adrenaline >= 3 and self.can_enter_attraction(attr)
        ]
        if adrenaline_candidates:
            return random.choice(adrenaline_candidates)

        fallback = [attr for attr in self.remaining_attractions if self.can_enter_attraction(attr)]
        if not fallback:
            raise RuntimeError("No feasible attractions to choose from")
        return random.choice(fallback)


class Single(Visitor):
    def __init__(self, t_now: float = 0.0) -> None:
        super().__init__()
        self.name = "Single"
        self.age: float = Sampler.sample_single_age()
        self.group_size = 1
        self.has_express = False
        self.max_wait_time = 30.0
        self.arrival_window_start, self.arrival_window_end = Sampler.sample_single_arrival_window()
        self.departure_time = CLOSE
        self.did_phase1: bool = False
        self.phase2_attractions: list["Attraction"] = []
        member = Visitor()
        member.age = self.age
        member.group_size = 1
        member.members = [member]
        member.has_express = self.has_express
        member.max_wait_time = self.max_wait_time
        member.departure_time = self.departure_time
        member.parent_group = self
        member.member_index = 0
        member.remaining_attractions = []
        self.members = [member]
        self.group_size = len(self.members)

    def choose_next_attraction(self, t_now: float) -> "Attraction":
        def queue_len(attr: "Attraction") -> int:
            if hasattr(attr, "q_length"):
                return attr.q_length()
            return sum(v.group_size for _, _, _, v in attr.Q.items())

        def pick_min_queue(cands: list[tuple[int, "Attraction"]]) -> "Attraction":
            m = min(q for q, _ in cands)
            tied = [attr for q, attr in cands if q == m]
            return random.choice(tied)

        if not self.remaining_attractions:
            raise RuntimeError("No remaining attractions to choose from")

        if not self.did_phase1 and not self.remaining_attractions:
            self.remaining_attractions = list(self.phase2_attractions)
            self.did_phase1 = True

        candidates = [
            (queue_len(attr), attr)
            for attr in self.remaining_attractions
            if self.can_enter_attraction(attr)
        ]
        if not candidates:
            raise RuntimeError("No feasible attractions to choose from")
        return pick_min_queue(candidates)

    def can_abandon_queue(self) -> bool:
        return not self.has_express

    def mark_attraction_completed(self, attraction: "Attraction") -> None:
        super().mark_attraction_completed(attraction)
        if not self.did_phase1 and not self.remaining_attractions:
            self.remaining_attractions = list(self.phase2_attractions)
            self.did_phase1 = True


# ----------------------------------------------------------- #
# ------------------------- Helpers -------------------------- #
# ----------------------------------------------------------- #

def snorkel_is_on_lunch(t_now: float) -> bool:
    return ONE_PM <= t_now < TWO_PM


def register_family_unit(families: dict[int, "Family"], family_unit: "Family", close_time: float = CLOSE) -> None:
    root: "Family" = getattr(family_unit, "root_family", family_unit)
    root._need_reunite = float(root.departure_time) < float(close_time)
    families[root.family_uid] = root

    subgroup_id = 0 if family_unit.subgroup_id is None else int(family_unit.subgroup_id)
    root.subgroup_ids.add(subgroup_id)

    if family_unit not in root.family_units:
        root.family_units.append(family_unit)


def trigger_family_departure(families: dict[int, "Family"], root_uid: int, t_now: float, park=None) -> None:
    root = families.get(root_uid)
    if root is None:
        return
    if root.departure_triggered:
        return
    root.departure_triggered = True
    for unit in getattr(root, "family_units", [root]):
        unit.status = "EXITING"


def finalize_family_unit_exit(park, family_unit: "Family", t_now: float) -> None:
    root = getattr(family_unit, "root_family", family_unit)
    finalize_reunited_family_exit(park, root, t_now)


def finalize_group_exit(park, visitor: "Visitor", t_now: float) -> None:
    if visitor is None:
        return
    if getattr(visitor, "leave_counted", False):
        visitor.status = "EXITED"
        return
    if park is None:
        visitor.leave_counted = True
        visitor.status = "EXITED"
        return

    visitor.leave_counted = True
    visitor.status = "EXITED"

    gs = int(getattr(visitor, "group_size", 1))
    park.customers_leaving += gs
    park.total_rating += float(visitor.rating) * float(gs)

    park.schedule_event(PhotoPurchaseEvent(t_now, visitor))


def notify_family_unit_idle_and_ready_to_leave(families: dict[int, "Family"], family_unit: "Family", t_now: float) -> bool:
    root: "Family" = families.get(family_unit.family_uid, getattr(family_unit, "root_family", None))
    if root is None:
        return False

    if not (root.need_reunite and root.departure_triggered):
        return False

    subgroup_id = 0 if family_unit.subgroup_id is None else int(family_unit.subgroup_id)
    root.ready_to_leave.add(subgroup_id)

    return root.ready_to_leave == root.subgroup_ids


def _family_tree_finished_all_attractions(root: "Family") -> bool:
    for unit in getattr(root, "family_units", [root]):
        if getattr(unit, "going_to_lunch", False):
            return False
        if unit.has_remaining_attractions():
            return False
    return True


def should_exit_now(families: dict[int, "Family"], visitor: "Visitor", t_now: float, close_time: float = CLOSE, park=None) -> bool:
    close_time = float(close_time)

    if not isinstance(visitor, Family):
        if (not getattr(visitor, "going_to_lunch", False)) and (not visitor.has_remaining_attractions()):
            return True
        return t_now >= close_time

    root: "Family" = families.get(visitor.family_uid, getattr(visitor, "root_family", None))
    if root is None:
        if (not getattr(visitor, "going_to_lunch", False)) and (not visitor.has_remaining_attractions()):
            return True
        return t_now >= float(visitor.departure_time)

    dep_time = float(root.departure_time)
    all_done = _family_tree_finished_all_attractions(root)

    if (t_now >= dep_time) or all_done:
        if not root.departure_triggered:
            trigger_family_departure(families, root.family_uid, t_now, park=park)

    if not root.need_reunite:
        return root.departure_triggered

    return root.departure_triggered and (root.ready_to_leave == root.subgroup_ids)


def finalize_close(families: dict[int, "Family"], t_now: float, close_time: float = CLOSE, park=None) -> None:
    close_time = float(close_time)

    for root in list(families.values()):
        if getattr(root, "exited", False):
            continue
        if not getattr(root, "reception_paid", False):
            continue

        if root.need_reunite:
            root.ready_to_leave.clear()
            root.ready_to_leave.update(root.subgroup_ids)

        for unit in getattr(root, "family_units", [root]):
            finalize_family_unit_exit(park, unit, t_now)

        root.exited = True

    if park is None:
        return

    for visitor in list(getattr(park, "teens_groups", {}).values()):
        if getattr(visitor, "leave_counted", False):
            continue
        if not getattr(visitor, "reception_paid", False):
            continue
        finalize_group_exit(park, visitor, t_now)

    for visitor in list(getattr(park, "singles", {}).values()):
        if getattr(visitor, "leave_counted", False):
            continue
        if not getattr(visitor, "reception_paid", False):
            continue
        finalize_group_exit(park, visitor, t_now)


def finalize_reunited_family_exit(park, root: "Family", t_now: float) -> None:
    if root is None or getattr(root, "exited", False):
        return

    units = list(getattr(root, "family_units", [root]))
    total_size = 0
    weighted_sum = 0.0
    for unit in units:
        gs = int(getattr(unit, "group_size", 1))
        total_size += gs
        weighted_sum += float(getattr(unit, "rating", 0.0)) * float(gs)

    combined_rating = (weighted_sum / total_size) if total_size > 0 else 0.0
    root.rating = combined_rating

    park.customers_leaving += int(total_size)
    park.total_rating += float(total_size) * float(root.rating)
    park.schedule_event(PhotoPurchaseEvent(t_now, root))

    for unit in units:
        unit.status = "EXITED"
        unit.leave_counted = True

    root.exited = True


def schedule_meeting_check_if_needed(park, unit, t_now: float) -> None:
    if not isinstance(unit, Family):
        return
    root = unit.root_family
    if getattr(root, "exited", False):
        return
    if getattr(root, "_meeting_check_scheduled", False):
        return
    root._meeting_check_scheduled = True
    park.schedule_event(MeetingCheckEvent(t_now + 1.0, unit))


def try_reunite_and_finalize_family_exit(park, family_unit: "Family", t_now: float) -> None:
    if family_unit is None:
        return

    root: "Family" = getattr(family_unit, "root_family", family_unit)
    if root is None:
        return

    if getattr(root, "exited", False):
        return

    for unit in getattr(root, "family_units", [root]):
        if getattr(unit, "going_to_lunch", False):
            return

    if not root.departure_triggered:
        all_done = _family_tree_finished_all_attractions(root)
        if all_done or t_now >= float(root.departure_time):
            trigger_family_departure(park.families, root.family_uid, t_now, park=park)
        if not root.departure_triggered:
            family_unit.status = "WAITING_MEET"
            schedule_meeting_check_if_needed(park, family_unit, t_now)
            return

    if not root.need_reunite:
        finalize_reunited_family_exit(park, root, t_now)
        return

    is_idle = (
        family_unit.status != "IN_QUEUE"
        and not getattr(family_unit, "going_to_lunch", False)
        and int(getattr(family_unit, "group_in_attraction", 0)) == 0
        and int(getattr(family_unit, "group_in_queue", 0)) == 0
    )
    if not is_idle:
        return

    subgroup_id = 0 if family_unit.subgroup_id is None else int(family_unit.subgroup_id)
    root.ready_to_leave.add(subgroup_id)

    if root.ready_to_leave != root.subgroup_ids:
        family_unit.status = "WAITING_MEET"
        schedule_meeting_check_if_needed(park, family_unit, t_now)
        return

    finalize_reunited_family_exit(park, root, t_now)


# ---------- Event helper functions (pre-Event class) ----------

def can_create_arrival(arrival, end_of_arrival):
    return arrival <= end_of_arrival


def manage_reception(park, visitor):
    reception = park.reception
    t_now = park.current_time

    if reception.busy < reception.servers:
        reception.start_service()
        completion_time = t_now + reception.sample_reception_service_time()
        park.schedule_event(TicketPurchaseCompleteEvent(completion_time, visitor))
        return

    reception.enqueue(visitor, t_now)


def schedule_family(park, arrival):
    new_family = Family(arrival)
    if hasattr(park, "families"):
        park.families[new_family.family_uid] = new_family
    park.schedule_event(FamilyArrivalEvent(arrival, new_family))


def schedule_single(park, arrival):
    new_single = Single(arrival)
    if hasattr(park, "singles"):
        park.singles[len(park.singles) + 1] = new_single
    park.schedule_event(SingleArrivalEvent(arrival, new_single))


def schedule_teens(park, arrival):
    new_teens = Teens(arrival)
    if hasattr(park, "teens_groups"):
        park.teens_groups[len(park.teens_groups) + 1] = new_teens
    park.schedule_event(TeensArrivalEvent(arrival, new_teens))


def new_ticket_and_band_event(park, visitor):
    completion_time = park.current_time + park.reception.sample_reception_service_time()
    park.schedule_event(TicketPurchaseCompleteEvent(completion_time, visitor))


def initialize_remaining_attractions(park, visitor) -> None:
    name_map = {a.name: a for a in park.attractions}
    if isinstance(visitor, Family):
        visitor.remaining_attractions = [
            a for a in park.attractions if visitor.can_enter_attraction(a)
        ]
        visitor.did_all_ages_first = False
        return
    if isinstance(visitor, Teens):
        visitor.remaining_attractions = [
            a for a in park.attractions
            if a.adrenaline >= 3 and visitor.can_enter_attraction(a)
        ]
        return
    if isinstance(visitor, Single):
        phase1_names = ["SingleSlides", "SmallTubeSlide", "WavePool"]
        phase1_set = set(phase1_names)
        phase1 = []
        for name in phase1_names:
            attr = name_map.get(name)
            if attr is not None and visitor.can_enter_attraction(attr):
                phase1.append(attr)
        visitor.remaining_attractions = phase1
        visitor.did_phase1 = False
        visitor.phase2_attractions = [
            a for a in park.attractions
            if a.name != "ToddlerPool"
            and a.name not in phase1_set
            and visitor.can_enter_attraction(a)
        ]


def schedule_attraction_completions_after_start(park, attraction, t_now: float) -> None:
    if getattr(attraction, "closed", False):
        return
    if attraction.name == "SnorkelTour":
        if ONE_PM <= t_now < TWO_PM:
            return
        if t_now < ONE_PM:
            if attraction.q_length() > 0 and attraction.headway_time > 0.0:
                for server_idx in range(attraction.servers):
                    if attraction.is_running_array[server_idx] or t_now < attraction.next_free_time_array[server_idx]:
                        continue
                    if not attraction.pending_boarded_per_server[server_idx]:
                        continue
                    attraction.fill_one_server_from_queue(
                        server_idx,
                        t_now,
                        target_list=attraction.pending_boarded_per_server[server_idx],
                    )
                    duration = attraction.pending_duration_per_server[server_idx]
                    if duration is None:
                        duration = Sampler.sample_attraction_duration("SnorkelTour")
                        if t_now + duration > ONE_PM:
                            continue
                    finish_time = attraction._start_run_with_boarded(
                        server_idx,
                        t_now,
                        duration,
                        attraction.pending_boarded_per_server[server_idx],
                    )
                    attraction.pending_headway_deadline_per_server[server_idx] = None
                    attraction.pending_duration_per_server[server_idx] = None
                    park.schedule_event(AttractionCompleteEvent(finish_time, server_idx, attraction))
                    t_release = attraction.next_free_time_array[server_idx]
                    if t_release > t_now:
                        park.schedule_event(AttractionKickEvent(t_release, attraction))

            while attraction.can_start(t_now):
                server_idx = attraction.available_server_to_fill(t_now)
                if server_idx == -1:
                    break
                duration = Sampler.sample_attraction_duration("SnorkelTour")
                if t_now + duration > CLOSE:
                    attraction.closed = True
                    return
                if t_now + duration > ONE_PM:
                    break
                if attraction.headway_time > 0.0:
                    allocated = attraction.fill_one_server_from_queue(
                        server_idx,
                        t_now,
                        target_list=attraction.pending_boarded_per_server[server_idx],
                    )
                    if allocated <= 0:
                        break
                    if allocated < attraction.capacity_per_server and attraction.q_length() == 0:
                        latest_finish = t_now + attraction.headway_time + duration
                        if latest_finish <= ONE_PM:
                            deadline = t_now + attraction.headway_time
                            attraction.pending_headway_deadline_per_server[server_idx] = deadline
                            attraction.pending_duration_per_server[server_idx] = duration
                            park.schedule_event(
                                AttractionHeadwayExpireEvent(deadline, server_idx, attraction)
                            )
                            continue

                    finish_time = attraction._start_run_with_boarded(
                        server_idx,
                        t_now,
                        duration,
                        attraction.pending_boarded_per_server[server_idx],
                    )
                    attraction.pending_headway_deadline_per_server[server_idx] = None
                    attraction.pending_duration_per_server[server_idx] = None
                else:
                    allocated = attraction.fill_one_server_from_queue(server_idx, t_now)
                    if allocated <= 0:
                        break
                    finish_time = t_now + duration
                    attraction.is_running_array[server_idx] = True
                    attraction.next_free_time_array[server_idx] = finish_time + attraction.break_time

                park.schedule_event(AttractionCompleteEvent(finish_time, server_idx, attraction))
                t_release = attraction.next_free_time_array[server_idx]
                if t_release > t_now:
                    park.schedule_event(AttractionKickEvent(t_release, attraction))
            return

    started, headway_waiting = attraction.start_servers_if_possible(t_now)
    for (server_idx, finish_time) in started:
        park.schedule_event(AttractionCompleteEvent(finish_time, server_idx, attraction))
    for (server_idx, deadline) in headway_waiting:
        park.schedule_event(AttractionHeadwayExpireEvent(deadline, server_idx, attraction))


def visitor_finished_receiving_ticket(park, visitor):
    park.customers_arriving += visitor.group_size

    visitor.has_express = Sampler.sample_buy_express_at_entrance()

    for m in getattr(visitor, "members", []):
        m.has_express = visitor.has_express

    if not getattr(visitor, "remaining_attractions", []):
        initialize_remaining_attractions(park, visitor)

    if not getattr(visitor, "reception_paid", False):
        revenue = 0.0
        for m in getattr(visitor, "members", []):
            age = float(getattr(m, "age", 0.0))
            if age >= 14.0:
                revenue += 150.0
            elif age >= 2.0:
                revenue += 75.0
        if visitor.has_express:
            revenue += 50.0 * float(len(getattr(visitor, "members", [])))
        visitor.reception_paid = True
        park.reception_income += revenue
        park.total_income += revenue

    visitor.status = "IN_PARK"
    visitor.current_location = "PARK"


def send_visitor_to_attraction(park, visitor):
    t_now = park.current_time

    if isinstance(visitor, Family) and visitor.status in {"EXITING", "WAITING_MEET"}:
        try_reunite_and_finalize_family_exit(park, visitor, t_now)
        return

    if not visitor.has_remaining_attractions():
        if isinstance(visitor, Family) and getattr(visitor, "need_reunite", False):
            visitor.status = "WAITING_MEET"
            schedule_meeting_check_if_needed(park, visitor, t_now)
            try_reunite_and_finalize_family_exit(park, visitor, t_now)
        elif isinstance(visitor, Family):
            finalize_family_unit_exit(park, visitor, t_now)
        else:
            finalize_group_exit(park, visitor, t_now)
        return

    attraction = visitor.choose_next_attraction(t_now)
    attraction.enqueue(visitor, t_now)

    if visitor.status != "IN_QUEUE":
        return

    schedule_attraction_completions_after_start(park, attraction, t_now)

    visitor.waiting_in_attraction = attraction

    if isinstance(visitor, Teens) and getattr(visitor, "returning_to_pending", False):
        visitor.returning_to_pending = False
        if not visitor.has_express:
            park.schedule_event(TeenWaitPenaltyEvent(t_now + 20, visitor, attraction))
        return

    if visitor.can_abandon_queue() and visitor.max_wait_time is not None and visitor.max_wait_time > 0:
        park.schedule_event(AbandonmentEvent(t_now + float(visitor.max_wait_time), visitor, attraction))


def check_visitor_attraction_position(visitor, park):
    t_now = park.current_time

    if isinstance(visitor, Family) and visitor.status in {"EXITING", "WAITING_MEET"}:
        try_reunite_and_finalize_family_exit(park, visitor, t_now)
        return

    if not visitor.has_remaining_attractions():
        if isinstance(visitor, Family) and getattr(visitor, "need_reunite", False):
            visitor.status = "WAITING_MEET"
            schedule_meeting_check_if_needed(park, visitor, t_now)
            try_reunite_and_finalize_family_exit(park, visitor, t_now)
            return

        if isinstance(visitor, Family):
            finalize_family_unit_exit(park, visitor, t_now)
        else:
            finalize_group_exit(park, visitor, t_now)
        return

    if maybe_send_visitor_to_lunch(park, visitor):
        return

    send_visitor_to_attraction(park, visitor)


def maybe_send_visitor_to_lunch(park, visitor) -> bool:
    t_now = park.current_time

    if visitor.should_go_to_lunch(t_now):
        restaurant_name = visitor.choose_restaurant()
        restaurant_entry = park.food_court.restaurants[restaurant_name]
        restaurant: Restaurant = restaurant_entry["restaurant"]
        expected_done = expected_food_completion_time(park, restaurant, t_now)
        if expected_done > CLOSE:
            visitor.rating = max(0.0, float(visitor.rating) - 0.8)
            return False

        visitor.going_to_lunch = True
        visitor.status = "GOING_TO_LUNCH"

        start_food_process(park, visitor, restaurant_name)
        return True

    return False


def expected_food_completion_time(park, restaurant: Restaurant, t_now: float) -> float:
    avg_prep = Sampler.expected_food_prep_time_by_restaurant(restaurant.name)
    avg_counter = Sampler.expected_food_service_time_at_counter()
    avg_service = avg_prep + avg_counter
    avg_eat = Sampler.expected_meal_duration()

    queue_len = len(restaurant.Q)
    servers = max(1, restaurant.servers)
    busy = restaurant.busy
    if restaurant.has_free_server() and queue_len == 0:
        expected_wait = 0.0
    else:
        expected_wait = (queue_len + busy) * avg_service / servers

    return t_now + expected_wait + avg_service + avg_eat


def start_food_process(park, visitor, restaurant_name: str) -> None:
    if restaurant_name not in park.food_court.restaurants:
        raise ValueError(f"Unknown restaurant: {restaurant_name}")

    entry = park.food_court.restaurants[restaurant_name]
    restaurant: Restaurant = entry["restaurant"]

    expected_done = expected_food_completion_time(park, restaurant, park.current_time)
    if expected_done > CLOSE:
        visitor.going_to_lunch = False
        visitor.status = "IN_PARK"
        check_visitor_attraction_position(visitor, park)
        return

    price = park.food_court.get_price(restaurant_name, visitor)
    park.food_court.add_income(price)
    park.total_income += float(price)
    park.total_people_ate += int(getattr(visitor, "group_size", 1))

    if restaurant.has_free_server():
        restaurant.start_service()
        prep_time = Sampler.sample_food_prep_time_by_restaurant(restaurant.name)
        counter_time = Sampler.sample_food_service_time_at_counter()
        completion_time = park.current_time + prep_time + counter_time
        park.schedule_event(FinishFoodServiceEvent(completion_time, visitor, restaurant))
    else:
        restaurant.enqueue(visitor, park.current_time, is_express=False)


def new_finished_eating_event(park, visitor, restaurant):
    eating_time = Sampler.sample_meal_duration()
    completion_time = park.current_time + eating_time
    park.schedule_event(FinishedEatingEvent(completion_time, visitor, restaurant))


def new_finish_service_event(park, restaurant):
    if len(restaurant.Q) == 0:
        return
    _, _, _, customer = restaurant.Q.pop()
    restaurant.start_service()
    prep_time = Sampler.sample_food_prep_time_by_restaurant(restaurant.name)
    counter_time = Sampler.sample_food_service_time_at_counter()
    duration = prep_time + counter_time
    completion_time = park.current_time + duration
    park.schedule_event(FinishFoodServiceEvent(completion_time, customer, restaurant))


# ------------------------------------------------ #
# -------------------- Event --------------------- #
# ------------------------------------------------ #

class Event:
    def __init__(self, time: float, visitor: Optional[Visitor]) -> None:
        self.time: float = float(time)
        self.visitor: Optional[Visitor] = visitor

    def __lt__(self, other: "Event") -> bool:
        return self.time < other.time

    def handle(self, park) -> None:
        raise NotImplementedError


class FamilyArrivalEvent(Event):
    def handle(self, park) -> None:
        t_now = park.current_time
        next_time = Sampler.next_family_arrival_time(t_now)
        if next_time is not None and can_create_arrival(next_time, NOON):
            schedule_family(park, next_time)
        manage_reception(park, self.visitor)


class TeensArrivalEvent(Event):
    def handle(self, park) -> None:
        t_now = park.current_time
        next_time = Sampler.next_teens_arrival_time(t_now)
        if next_time is not None and can_create_arrival(next_time, FOUR_PM):
            schedule_teens(park, next_time)
        manage_reception(park, self.visitor)


class SingleArrivalEvent(Event):
    def handle(self, park) -> None:
        t_now = park.current_time
        next_time = Sampler.next_single_arrival_time(t_now)
        if next_time is not None and can_create_arrival(next_time, CLOSE - 30):
            schedule_single(park, next_time)
        manage_reception(park, self.visitor)


class TicketPurchaseCompleteEvent(Event):
    def handle(self, park) -> None:
        t_now = park.current_time
        visitor = self.visitor

        park.reception.finish_service()
        visitor_finished_receiving_ticket(park, visitor)
        send_visitor_to_attraction(park, visitor)

        if not park.reception.is_queue_empty():
            _, _, _, next_visitor = park.reception.Q.pop()
            park.reception.start_service()
            completion_time = t_now + park.reception.sample_reception_service_time()
            park.schedule_event(TicketPurchaseCompleteEvent(completion_time, next_visitor))
        else:
            return


class AttractionHeadwayExpireEvent(Event):
    def __init__(self, time: float, server_idx: int, attraction: Attraction) -> None:
        super().__init__(time, None)
        self.server_idx: int = int(server_idx)
        self.attraction: Attraction = attraction

    def handle(self, park) -> None:
        t_now = park.current_time
        attraction = self.attraction
        server_idx = self.server_idx

        if attraction.is_running_array[server_idx]:
            return

        deadline = attraction.pending_headway_deadline_per_server[server_idx]
        if deadline is None or t_now < float(deadline):
            return

        boarded = attraction.pending_boarded_per_server[server_idx]
        if not boarded:
            attraction.pending_headway_deadline_per_server[server_idx] = None
            attraction.pending_duration_per_server[server_idx] = None
            return

        duration = attraction.pending_duration_per_server[server_idx]
        if duration is None:
            duration, finish_time = attraction.sample_duration_and_finish(t_now)
            if finish_time > CLOSE:
                attraction.closed = True
                attraction.pending_headway_deadline_per_server[server_idx] = None
                attraction.pending_duration_per_server[server_idx] = None
                boarded.clear()
                return

        finish_time = attraction._start_run_with_boarded(server_idx, t_now, duration, boarded)
        attraction.pending_headway_deadline_per_server[server_idx] = None
        attraction.pending_duration_per_server[server_idx] = None

        park.schedule_event(AttractionCompleteEvent(finish_time, server_idx, attraction))
        t_release = attraction.next_free_time_array[server_idx]
        if t_release > t_now:
            park.schedule_event(AttractionKickEvent(t_release, attraction))


class AttractionCompleteEvent(Event):
    def __init__(self, time: float, server_idx: int, attraction: Attraction) -> None:
        super().__init__(time, None)
        self.server_idx: int = int(server_idx)
        self.attraction: Attraction = attraction

    def handle(self, park) -> None:
        t_now = park.current_time
        attraction = self.attraction
        server_idx = self.server_idx

        serviced = attraction.complete_server(server_idx)
        t_release = attraction.next_free_time_array[server_idx]
        if t_release > t_now:
            park.schedule_event(AttractionKickEvent(t_release, attraction))

        parents_done: set[Visitor] = set()
        for _, parent in serviced:
            if parent.group_in_attraction == 0 and parent.group_in_queue == 0:
                parents_done.add(parent)

        for parent in parents_done:
            parent.apply_post_attraction_rating(attraction)
            parent.mark_attraction_completed(attraction)
            check_visitor_attraction_position(parent, park)

        schedule_attraction_completions_after_start(park, attraction, t_now)


class AttractionKickEvent(Event):
    def __init__(self, time: float, attraction: Attraction) -> None:
        super().__init__(time, None)
        self.attraction: Attraction = attraction

    def handle(self, park) -> None:
        schedule_attraction_completions_after_start(park, self.attraction, park.current_time)


class AbandonmentEvent(Event):
    def __init__(self, time: float, visitor: "Visitor", attraction: "Attraction") -> None:
        super().__init__(time, visitor)
        self.attraction: Attraction = attraction

    def handle(self, park) -> None:
        visitor = self.visitor
        attraction = self.attraction
        t_now = park.current_time

        if visitor is None:
            return

        if getattr(visitor, "status", None) != "IN_QUEUE":
            return
        if getattr(visitor, "waiting_in_attraction", None) is not attraction:
            return

        if getattr(visitor, "has_express", False):
            return

        if getattr(visitor, "group_in_attraction", 0) > 0:
            return

        if getattr(visitor, "group_in_queue", 0) <= 0:
            return

        removed = attraction.remove_from_queue(visitor)
        if not removed:
            return

        park.abandonments += int(getattr(visitor, "group_in_queue", 0))

        visitor.group_in_queue = 0
        visitor._member_cursor = 0
        visitor.waiting_in_attraction = None

        visitor.rating = max(0.0, float(visitor.rating) - 0.8)

        schedule_attraction_completions_after_start(park, attraction, t_now)

        if isinstance(visitor, Teens):
            action = visitor.post_abandon_action()
            if action == BUY_EXPRESS_AND_JOIN_EXPRESS_QUEUE:
                visitor.has_express = True
                for m in getattr(visitor, "members", []):
                    m.has_express = True

                visitor.pending_return_attraction = None
                visitor.detour_done = False
                visitor.returning_to_pending = False

                visitor.status = "IN_PARK"
                attraction.enqueue(visitor, t_now)
                schedule_attraction_completions_after_start(park, attraction, t_now)
                return

            visitor.pending_return_attraction = attraction
            visitor.detour_done = False
            visitor.status = "IN_PARK"
            send_visitor_to_attraction(park, visitor)
            return

        visitor.status = "IN_PARK"
        send_visitor_to_attraction(park, visitor)


class TeenWaitPenaltyEvent(Event):
    def __init__(self, time: float, visitor: "Visitor", attraction: Attraction) -> None:
        super().__init__(time, visitor)
        self.attraction: Attraction = attraction

    def handle(self, park) -> None:
        visitor = self.visitor
        attraction = self.attraction
        t_now = park.current_time

        if visitor is None:
            return

        if getattr(visitor, "status", None) != "IN_QUEUE":
            return

        if getattr(visitor, "has_express", False):
            return
        if getattr(visitor, "waiting_in_attraction", None) is not attraction:
            return
        if getattr(visitor, "group_in_attraction", 0) > 0:
            return
        if getattr(visitor, "group_in_queue", 0) <= 0:
            return

        visitor.rating = max(0.0, float(visitor.rating) - 0.8)
        park.schedule_event(TeenWaitPenaltyEvent(t_now + 20, visitor, attraction))


class FinishFoodServiceEvent(Event):
    def __init__(self, time: float, visitor: Visitor, restaurant: Restaurant) -> None:
        super().__init__(time, visitor)
        self.restaurant: Restaurant = restaurant

    def handle(self, park) -> None:
        self.restaurant.finish_service()
        new_finish_service_event(park, self.restaurant)
        new_finished_eating_event(park, self.visitor, self.restaurant)


class FinishedEatingEvent(Event):
    def __init__(self, time: float, visitor: Visitor, restaurant: Restaurant) -> None:
        super().__init__(time, visitor)
        self.restaurant: Restaurant = restaurant

    def handle(self, park) -> None:
        visitor = self.visitor
        t_now = park.current_time

        result = Sampler.sample_food_satisfaction()
        if result == "UNSATISFIED":
            visitor.rating = max(0.0, float(visitor.rating) - 0.8)

        visitor.has_eaten = True
        visitor.going_to_lunch = False

        if isinstance(visitor, Family) and visitor.status in {"EXITING", "WAITING_MEET"}:
            try_reunite_and_finalize_family_exit(park, visitor, t_now)
            return

        visitor.status = "IN_PARK"
        check_visitor_attraction_position(visitor, park)


class MeetingCheckEvent(Event):
    def __init__(self, time: float, visitor: Visitor) -> None:
        super().__init__(time, visitor)

    def handle(self, park) -> None:
        unit = self.visitor
        if unit is None or not isinstance(unit, Family):
            return

        t_now = park.current_time
        root = unit.root_family
        try:
            if getattr(root, "exited", False):
                unit.status = "EXITED"
                unit.leave_counted = True
                return

            register_family_unit(park.families, unit, close_time=CLOSE)

            if t_now >= CLOSE:
                finalize_close(park.families, t_now, close_time=CLOSE, park=park)
                return

            if unit.status not in {"WAITING_MEET", "EXITING"}:
                return

            should_exit_now(park.families, unit, t_now, close_time=CLOSE, park=park)
            if not root.departure_triggered:
                unit.status = "WAITING_MEET"
                schedule_meeting_check_if_needed(park, unit, t_now)
                return

            if not root.need_reunite:
                finalize_reunited_family_exit(park, root, t_now)
                return

            is_idle = (
                unit.status != "IN_QUEUE"
                and not getattr(unit, "going_to_lunch", False)
                and int(getattr(unit, "group_in_attraction", 0)) == 0
                and int(getattr(unit, "group_in_queue", 0)) == 0
            )
            if is_idle:
                subgroup_id = 0 if unit.subgroup_id is None else int(unit.subgroup_id)
                root.ready_to_leave.add(subgroup_id)

            if root.ready_to_leave == root.subgroup_ids:
                finalize_reunited_family_exit(park, root, t_now)
            else:
                unit.status = "WAITING_MEET"
                schedule_meeting_check_if_needed(park, unit, t_now)
        finally:
            root._meeting_check_scheduled = False


class PhotoPurchaseEvent(Event):
    def __init__(self, time: float, visitor: Visitor) -> None:
        super().__init__(time, visitor)

    def handle(self, park) -> None:
        visitor = self.visitor
        if visitor is None or getattr(visitor, "photos_processed", False):
            return
        visitor.photos_processed = True

        rating = float(visitor.rating)
        if rating < 6.0:
            amount = 0.0
        elif rating < 7.5:
            amount = 20.0
        elif rating < 8.5:
            amount = 100.0
        else:
            amount = 120.0

        if amount > 0:
            park.photo_income += amount
            park.total_income += amount


class ParkCloseEvent(Event):
    def __init__(self, time: float) -> None:
        super().__init__(time, None)

    def handle(self, park) -> None:
        t_now = park.current_time
        finalize_close(park.families, t_now, close_time=CLOSE, park=park)


# ------------------ Park Helpers -------------------- #

def reset_park_for_run(park) -> None:
    park.current_time = OPEN
    park.customers_arriving = 0
    park.customers_leaving = 0
    park.total_rating = 0.0
    park.total_income = 0.0
    park.photo_income = 0.0
    park.reception_income = 0.0
    park.total_people_ate = 0
    park.drop_count = 0
    park.abandonments = 0
    if hasattr(park, "reception"):
        park.reception.reset()
    if hasattr(park, "food_court"):
        park.food_court.reset()
    if hasattr(park, "attractions"):
        for attraction in park.attractions:
            attraction.reset()
    if hasattr(park, "teens_groups"):
        park.teens_groups.clear()
    if hasattr(park, "singles"):
        park.singles.clear()
    if hasattr(park, "families"):
        park.families.clear()
    Family._FAMILY_UID_COUNTER = 0

    for attraction in park.attractions:
        attraction.reset()

    park.food_court.reset()


def initialize_and_schedule_first_visitors(park) -> None:
    t_now = OPEN
    park.current_time = t_now
    park.schedule_event(ParkCloseEvent(CLOSE))
    family_time = Sampler.next_family_arrival_time(t_now)
    if family_time is not None and can_create_arrival(family_time, NOON):
        schedule_family(park, family_time)
    teens_time = Sampler.next_teens_arrival_time(t_now)
    if teens_time is not None and can_create_arrival(teens_time, FOUR_PM):
        schedule_teens(park, teens_time)
    single_time = Sampler.next_single_arrival_time(t_now)
    if single_time is not None and can_create_arrival(single_time, CLOSE - 30):
        schedule_single(park, single_time)


class Park:
    def __init__(self, reception: Reception, food_court: Optional[FoodCourt], attractions: Optional[List[Attraction]] = None) -> None:
        self.log = []
        self.time_line = []
        self.reception = reception
        self.food_court = food_court if food_court is not None else FoodCourt(1, 1, 1)
        self.attractions: List[Attraction] = attractions if attractions is not None else []
        self.families: dict[int, Family] = {}
        self.teens_groups: dict[int, Teens] = {}
        self.singles: dict[int, Single] = {}
        self.current_time = OPEN
        self.tmax = CLOSE
        self.customers_arriving = 0
        self.customers_leaving = 0
        self.total_rating: float = 0.0
        self.total_income: float = 0.0
        self.photo_income: float = 0.0
        self.reception_income: float = 0.0
        self.total_people_ate: int = 0
        self.drop_count: int = 0
        self.abandonments: int = 0
        self.reset()

        self.lazyRiver = Attraction("LazyRiver", 60, 2, 1)
        self.singleSlides = Attraction("SingleSlides", 2, 1, 5)
        self.largeTubeSlide = Attraction("LargeTubeSlide", 1, 8, 2)
        self.smallTubeSlide = Attraction("SmallTubeSlide", 1, 3, 4)
        self.wavePool = Attraction("WavePool", 1, 80, 3)
        self.toddlerPool = Attraction("ToddlerPool", 1, 30, 1)
        self.snorkelTour = Attraction("SnorkelTour", 2, 30, 3)

        self.attractions.extend([
            self.lazyRiver,
            self.singleSlides,
            self.largeTubeSlide,
            self.smallTubeSlide,
            self.wavePool,
            self.toddlerPool,
            self.snorkelTour,
        ])

    def reset(self) -> None:
        reset_park_for_run(self)
        if hasattr(self, "log"):
            self.log.clear()
        if hasattr(self, "time_line"):
            self.time_line.clear()

    def schedule_event(self, event):
        visitor_name = event.visitor.name if getattr(event, "visitor", None) is not None else None
        event_name = type(event).__name__
        if event.time <= self.tmax:
            heapq.heappush(self.log, event)
            self.time_line.append((float(event.time), "SCHEDULE", event_name, visitor_name))
        else:
            self.drop_count += 1
            self.time_line.append((float(self.current_time), "DROP_TMAX", event_name, visitor_name))
            return

    def has_events(self):
        return len(self.log) > 0

    def run(self, seed: Optional[int] = None):

        if seed is not None:
            random.seed(seed)

        self.reset()
        initialize_and_schedule_first_visitors(self)
        while self.has_events() and self.current_time < self.tmax:
            event = heapq.heappop(self.log)
            self.current_time = event.time
            visitor_name = event.visitor.name if getattr(event, "visitor", None) is not None else None
            self.time_line.append((float(self.current_time), "EXECUTE", type(event).__name__, visitor_name))
            event.handle(self)


# ----------------------------------------------------------- #
# --------------- MULTI-RUN COMPARISON --------------------- #
# ----------------------------------------------------------- #

def get_metrics(park) -> dict:
    avg_rating = (park.total_rating / float(park.customers_leaving)) if park.customers_leaving > 0 else 0.0
    total_customers = park.customers_leaving
    total_income_food_court = float(park.food_court.income)
    ate = int(getattr(park, "total_people_ate", 0))
    avg_food_income = total_income_food_court / ate if ate > 0 else 0.0
    total_revenue = float(park.total_income)
    arrivals = int(getattr(park, "customers_arriving", 0))
    avg_reception_income = float(getattr(park, "reception_income", 0.0)) / arrivals if arrivals > 0 else 0.0
    avg_photo_income = float(getattr(park, "photo_income", 0.0)) / arrivals if arrivals > 0 else 0.0
    avg_food_income_per_visitor = total_income_food_court / arrivals if arrivals > 0 else 0.0
    avg_abandonments_per_visitor = float(getattr(park, "abandonments", 0)) / arrivals if arrivals > 0 else 0.0
    return {
        "avg_rating": avg_rating,
        "total_customers": total_customers,
        "avg_food_income": avg_food_income,
        "total_revenue": total_revenue,
        "food_income": total_income_food_court,
        "reception_income": float(getattr(park, "reception_income", 0.0)),
        "photo_income": float(getattr(park, "photo_income", 0.0)),
        "avg_reception_income": avg_reception_income,
        "avg_photo_income": avg_photo_income,
        "avg_food_income_per_visitor": avg_food_income_per_visitor,
        "avg_abandonments_per_visitor": avg_abandonments_per_visitor,
        "customers_arriving": int(getattr(park, "customers_arriving", 0)),
        "customers_leaving": int(getattr(park, "customers_leaving", 0)),
        "total_people_ate": int(getattr(park, "total_people_ate", 0)),
        "drop_count": int(getattr(park, "drop_count", 0)),
        "abandonments": int(getattr(park, "abandonments", 0)),
    }


def compute_basic_stats(xs: list[float]) -> dict:
    n = len(xs)
    mean = statistics.mean(xs)
    var = statistics.variance(xs) if n >= 2 else 0.0
    return {
        "n": n,
        "mean": mean,
        "var": var,
    }


def validate_run_stats(park, scn_key: str, seed: Optional[int]) -> None:
    errors = []
    arriving = int(getattr(park, "customers_arriving", 0))
    leaving = int(getattr(park, "customers_leaving", 0))
    total_people_ate = int(getattr(park, "total_people_ate", 0))
    drop_count = int(getattr(park, "drop_count", 0))

    if arriving < 0 or leaving < 0 or total_people_ate < 0 or drop_count < 0:
        errors.append("negative counts detected")
    if leaving > arriving:
        errors.append("customers_leaving exceeds customers_arriving")

    if errors:
        seed_text = "None" if seed is None else str(seed)
        raise ValueError(
            f"Validation failed for scenario={scn_key}, seed={seed_text}: {', '.join(errors)}"
        )


def compute_half_confidence_interval(xs: list[float], alpha: float) -> float:
    stats = compute_basic_stats(xs)
    n = stats["n"]
    var = stats["var"]
    t_crit = t.ppf(1 - alpha / 2.0, n - 1)
    return float(t_crit) * math.sqrt(var / n)


def required_n_relative_accuracy(n0: int, mean: float, half_ci: float, rel_acc: float = 0.1,) -> int:
    eps = 1e-9
    if not math.isfinite(mean) or not math.isfinite(half_ci) or abs(mean) < eps:
        return int(n0)
    n_req = n0 * ((half_ci / (mean * (rel_acc / (1.0 + rel_acc)))) ** 2)
    if not math.isfinite(n_req) or n_req <= 0:
        return int(n0)
    return math.ceil(n_req)


def analyze_metric_relative_accuracy(xs: list[float], alpha: float, n0: int,) -> dict:
    stats = compute_basic_stats(xs)
    half_ci = compute_half_confidence_interval(xs, alpha)
    return {
        "n": stats["n"],
        "mean": stats["mean"],
        "var": stats["var"],
        "half_ci": half_ci,
    }


def required_n_per_scenario(scenario_results: dict, n0: int, rel_acc: float, alpha: float,) -> dict:
    metrics = ("avg_food_income", "avg_rating", "total_customers")
    output: dict[str, dict[str, float | int]] = {}
    for key in metrics:
        xs = scenario_results[key]
        stats = analyze_metric_relative_accuracy(xs, alpha, n0)
        req_n = required_n_relative_accuracy(n0, stats["mean"], stats["half_ci"], rel_acc=rel_acc)
        if req_n < int(n0):
            req_n = int(n0)
        output[key] = {
            "mean": float(stats["mean"]),
            "variance": float(stats["var"]),
            "half_ci": float(stats["half_ci"]),
            "required_n": int(req_n),
        }
    return output


def required_n_all_scenarios(all_results: dict, n0: int, rel_acc: float, alpha: float,) -> dict:
    per_scenario: dict[str, dict] = {}
    required_values: list[int] = []
    for scn_key in ("BASE", "ALT1", "ALT2"):
        scn_stats = required_n_per_scenario(all_results[scn_key], n0, rel_acc, alpha)
        per_scenario[scn_key] = scn_stats
        for metric, metric_stats in scn_stats.items():
            req_n = int(metric_stats["required_n"])
            required_values.append(req_n)
            if req_n > 100:
                xs = all_results[scn_key][metric]
                print(
                    f"DEBUG required_n>100: scn={scn_key}, metric={metric}, n0={n0}, "
                    f"mean={metric_stats['mean']}, var={metric_stats['variance']}, "
                    f"half_ci={metric_stats['half_ci']}, rel_acc={rel_acc}, req_n={req_n}"
                )
                print(
                    f"  xs_len={len(xs)}, min={min(xs)}, max={max(xs)}, head={xs[:10]}"
                )

    max_required = max(required_values) if required_values else int(n0)
    if max_required < int(n0):
        max_required = int(n0)
    
    return {
        "per_scenario": per_scenario,
        "global_required_n": int(math.ceil(max_required)),
    }


def extend_replications_if_needed(all_results: dict, global_required_n: int, n0: int, base_seed: int,) -> dict:
    if int(global_required_n) <= int(n0):
        return all_results

    extra_runs = int(global_required_n) - int(n0)
    for i in range(extra_runs):
        seed = int(base_seed) + int(n0) + i
        for scn_key in ("BASE", "ALT1", "ALT2"):
            r_i = run_single_scenario(scn_key, seed=seed)
            all_results[scn_key]["avg_food_income"].append(float(r_i["avg_food_income"]))
            all_results[scn_key]["avg_rating"].append(float(r_i["avg_rating"]))
            all_results[scn_key]["total_customers"].append(float(r_i["total_customers"]))

    lengths = {
        scn_key: (
            len(all_results[scn_key]["avg_food_income"]),
            len(all_results[scn_key]["avg_rating"]),
            len(all_results[scn_key]["total_customers"]),
        )
        for scn_key in ("BASE", "ALT1", "ALT2")
    }
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Metric list lengths out of sync: {lengths}")

    return all_results


def run_relative_accuracy_extension(R0: int, base_seed: int, rel_acc: float, alpha: float,) -> dict:
    all_results = run_all_three_scenarios_replications(R0, base_seed)
    required_summary = required_n_all_scenarios(all_results, R0, rel_acc, alpha)
    global_required_n = int(required_summary["global_required_n"])

    all_results = extend_replications_if_needed(
        all_results, global_required_n, R0, base_seed
    )

    required_summary = required_n_all_scenarios(all_results, global_required_n, rel_acc, alpha)

    return {
        "final_results": all_results,
        "initial_n": int(R0),
        "final_n": int(global_required_n),
        "required_n_summary": required_summary,
    }


def paired_confidence_band(xs: list[float], ys: list[float], alpha: float) -> tuple[float, float]:
    if len(xs) != len(ys):
        raise ValueError("Paired samples must have equal length")
    diffs = [float(x) - float(y) for x, y in zip(xs, ys)]
    stats = compute_basic_stats(diffs)
    half_ci = compute_half_confidence_interval(diffs, alpha)
    mean = float(stats["mean"])
    return mean - half_ci, mean + half_ci


def paired_ci_per_metric(all_results: dict, metric: str, alpha: float) -> dict:
    base = all_results["BASE"][metric]
    alt1 = all_results["ALT1"][metric]
    alt2 = all_results["ALT2"][metric]

    base_alt1 = paired_confidence_band(base, alt1, alpha)
    base_alt2 = paired_confidence_band(base, alt2, alpha)
    alt1_alt2 = paired_confidence_band(alt1, alt2, alpha)

    return {
        "BASE_minus_ALT1": base_alt1,
        "BASE_minus_ALT2": base_alt2,
        "ALT1_minus_ALT2": alt1_alt2,
    }


def ci_relation(ci: tuple[float, float]) -> str:
    if ci[1] < 0:
        return "NEGATIVE"
    if ci[0] > 0:
        return "POSITIVE"
    return "CROSSES_ZERO"


def metric_winner_from_cis(metric: str, cis: dict) -> dict:
    base_alt1 = cis["BASE_minus_ALT1"]
    base_alt2 = cis["BASE_minus_ALT2"]
    alt1_alt2 = cis["ALT1_minus_ALT2"]

    if metric in ("avg_food_income", "avg_rating", "total_customers"):
        rel_base_alt1 = ci_relation(base_alt1)
        rel_base_alt2 = ci_relation(base_alt2)
        rel_alt1_alt2 = ci_relation(alt1_alt2)

        if rel_base_alt1 == "NEGATIVE":
            outcome_base_alt1 = "ALT1"
        elif rel_base_alt1 == "POSITIVE":
            outcome_base_alt1 = "BASE"
        else:
            outcome_base_alt1 = "INCONCLUSIVE"

        if rel_base_alt2 == "NEGATIVE":
            outcome_base_alt2 = "ALT2"
        elif rel_base_alt2 == "POSITIVE":
            outcome_base_alt2 = "BASE"
        else:
            outcome_base_alt2 = "INCONCLUSIVE"

        if rel_alt1_alt2 == "NEGATIVE":
            outcome_alt1_alt2 = "ALT2"
        elif rel_alt1_alt2 == "POSITIVE":
            outcome_alt1_alt2 = "ALT1"
        else:
            outcome_alt1_alt2 = "INCONCLUSIVE"
    
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if outcome_base_alt1 == "ALT1" and outcome_alt1_alt2 == "ALT1":
        winner = "ALT1"
    elif outcome_base_alt2 == "ALT2" and outcome_alt1_alt2 == "ALT2":
        winner = "ALT2"
    elif outcome_base_alt1 == "BASE" and outcome_base_alt2 == "BASE":
        winner = "BASE"
    else:
        winner = "INCONCLUSIVE"

    return {
        "winner": winner,
        "comparisons": {
            "BASE_vs_ALT1": outcome_base_alt1,
            "BASE_vs_ALT2": outcome_base_alt2,
            "ALT1_vs_ALT2": outcome_alt1_alt2,
        },
    }


def paired_t_confidence_bands_part_d(all_results: dict, alpha: float) -> dict:
    alpha_pairwise = alpha / 3
    metrics = ("avg_food_income", "avg_rating", "total_customers")
    bands = {metric: paired_ci_per_metric(all_results, metric, alpha_pairwise) for metric in metrics}
    winners = {metric: metric_winner_from_cis(metric, bands[metric]) for metric in metrics}
    return {
        "alpha": alpha_pairwise,
        "bands": bands,
        "per_metric_winner": winners,
    }


def print_summary(part_d_result: dict) -> None:
    global _SIM_ALREADY_RAN
    if _SIM_ALREADY_RAN:
        print("DEBUG: part d called again – skipping")
        return
    _SIM_ALREADY_RAN = True
    print(f"C) Paired confidence bands (alpha = {part_d_result['alpha']})")
    bands = part_d_result["bands"]
    winners = part_d_result["per_metric_winner"]
    for metric in ("avg_food_income", "avg_rating", "total_customers"):
        cis = bands[metric]
        win = winners[metric]
        print(f"  {metric}")
        print(f"    BASE_minus_ALT1: ({cis['BASE_minus_ALT1'][0]:.6f}, {cis['BASE_minus_ALT1'][1]:.6f})")
        print(f"    BASE_minus_ALT2: ({cis['BASE_minus_ALT2'][0]:.6f}, {cis['BASE_minus_ALT2'][1]:.6f})")
        print(f"    ALT1_minus_ALT2: ({cis['ALT1_minus_ALT2'][0]:.6f}, {cis['ALT1_minus_ALT2'][1]:.6f})")
        print(f"    BASE_vs_ALT1: {win['comparisons']['BASE_vs_ALT1']}")
        print(f"    BASE_vs_ALT2: {win['comparisons']['BASE_vs_ALT2']}")
        print(f"    ALT1_vs_ALT2: {win['comparisons']['ALT1_vs_ALT2']}")
        print(f"    Winner: {win['winner']}")


def run_single_scenario(scn_key: str, seed: Optional[int] = None) -> dict:
    scn = SCENARIOS[scn_key]
    apply_scenario_to_globals(scn)

    reception = Reception(3)
    food_court = FoodCourt(1, 1, 1)
    park = Park(reception, food_court)

    apply_scenario_to_park_resources(park, scn)

    park.run(seed=seed)
    validate_run_stats(park, scn_key, seed)

    m = get_metrics(park)
    invest = float(scn["invest_cost"])
    feasible = invest <= float(BUDGET)
    

    return {
        "name": scn["label"],
        "key": scn_key,
        "feasible": feasible,
        "invest_cost": invest,
        "avg_rating": m["avg_rating"],
        "total_customers": m["total_customers"],
        "avg_food_income": m["avg_food_income"],
        "total_revenue": m["total_revenue"],
        "food_income": m["food_income"],
        "reception_income": m["reception_income"],
        "photo_income": m["photo_income"],
        "avg_reception_income": m["avg_reception_income"],
        "avg_photo_income": m["avg_photo_income"],
        "avg_food_income_per_visitor": m["avg_food_income_per_visitor"],
        "avg_abandonments_per_visitor": m["avg_abandonments_per_visitor"],
        "customers_arriving": m["customers_arriving"],
        "customers_leaving": m["customers_leaving"],
        "total_people_ate": m["total_people_ate"],
        "drop_count": m["drop_count"],
        "abandonments": m["abandonments"],
    }


def run_scenario_replications(scn_key: str, R: int, base_seed: int) -> dict:
    if int(R) <= 0:
        raise ValueError("R must be positive")

    first = run_single_scenario(scn_key, seed=base_seed)
    avg_food_income_list = [float(first["avg_food_income"])]
    total_customers_list = [float(first["total_customers"])]
    avg_rating_list = [float(first["avg_rating"])]
    total_revenue_list = [float(first["total_revenue"])]

    for i in range(1, int(R)):
        r_i = run_single_scenario(scn_key, seed=base_seed + i)
        avg_food_income_list.append(float(r_i["avg_food_income"]))
        total_customers_list.append(float(r_i["total_customers"]))
        avg_rating_list.append(float(r_i["avg_rating"]))
        total_revenue_list.append(float(r_i["total_revenue"]))

    return {
        "name": first["name"],
        "key": scn_key,
        "feasible": first["feasible"],
        "invest_cost": first["invest_cost"],
        "avg_food_income": avg_food_income_list,
        "total_customers": total_customers_list,
        "avg_rating": avg_rating_list,
        "total_revenue": total_revenue_list,
    }


def run_all_three_scenarios_replications(R: int, base_seed: int) -> dict:
    return {
        "BASE": run_scenario_replications("BASE", R, base_seed),
        "ALT1": run_scenario_replications("ALT1", R, base_seed),
        "ALT2": run_scenario_replications("ALT2", R, base_seed),
    }

# ----------------------------------------------------------- #
# ------------------ MAIN RUNNER ---------------------------- #
# ----------------------------------------------------------- #

def run_simulation_example(iterations, alpha, relative_accuracy) -> None:
    global _SIM_ALREADY_RAN
    if _SIM_ALREADY_RAN:
        print("DEBUG: run_simulation_example called again – skipping")
        return
    out = run_relative_accuracy_extension(R0 = iterations,base_seed = SEED, rel_acc = relative_accuracy,alpha = alpha,)
    final_results = out["final_results"]
    initial_n = out["initial_n"]
    final_n = out["final_n"]
    required_n_summary = out["required_n_summary"]

    part_d = paired_t_confidence_bands_part_d(final_results, alpha)

    print("FINAL SUMMARY")
    print("A) Number of runs")
    print(f"  Initial runs (R0): {initial_n}")
    print(f"  Final runs used: {final_n}")

    print("B) Per scenario statistics")
    for scn_key in ("BASE", "ALT1", "ALT2"):
        print(f"  {scn_key}")
        scn_stats = required_n_summary["per_scenario"][scn_key]
        for metric in ("avg_food_income", "avg_rating", "total_customers"):
            m = scn_stats[metric]
            print(f"    {metric}: mean={m['mean']:.6f}, variance={m['variance']:.6f}, half_ci={m['half_ci']:.6f}, required_n={m['required_n']}")

    print_summary(part_d)


if __name__ == "__main__":
    num_of_runs = 20
    alpha = 0.1 / 9
    relative_accuracy = 0.1
    run_simulation_example(num_of_runs, alpha, relative_accuracy)
