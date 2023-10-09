from typing import Optional, Sequence, Union

from itertools import chain

import numpy as np

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RandomSelect
from alns.stop import MaxRuntime

import cvrp_benchmarker as bench


class SISR_Runner(bench.Runner):
    @classmethod
    @property
    def name(cls) -> str:
        return 'sisr'
    
    @classmethod
    @property
    def hyperparameters(cls) -> Optional[Sequence[bench.HyperParameter]]:
        return [
            bench.HyperParameter('iter_num', (100, 1_000_000)),
            bench.HyperParameter('max_string_removals', (1, 5)),
            bench.HyperParameter('max_string_size', (1, 20)),
            bench.HyperParameter('blink_rate', (0., 0.1), log=True),
            bench.HyperParameter('split_rate', (0., 1.)),
            bench.HyperParameter('split_depth', (0., 0.1), log=True),
        ]
    
    @classmethod
    def run(
            cls,
            problem: bench.CvrpProblemDescription,
            target_time: float,
            hyperparameters: Optional[dict[str, Union[float, int]]]
    ) -> (float, list[list[int]], list[float]):
        dists = problem.edge_weights.tolist()
        demands = problem.demands.tolist()

        class CvrpState:
            """
            Solution state for CVRP. It has two data members, routes and unassigned.
            Routes is a list of list of integers, where each inner list corresponds to
            a single route denoting the sequence of customers to be visited. A route
            does not contain the start and end depot. Unassigned is a list of integers,
            each integer representing an unassigned customer.
            """

            def __init__(self, routes, route_caps, unassigned=None):
                self.routes = routes
                self.route_caps = route_caps
                self.unassigned = unassigned if unassigned is not None else []

            def copy(self):
                return CvrpState(
                    [route[:] for route in self.routes],
                    self.route_caps[:],
                    self.unassigned.copy()
                )

            def objective(self):
                """
                Computes the total route costs.
                """
                return routes_cost(self.routes)

            @property
            def cost(self):
                """
                Alias for objective method. Used for plotting.
                """
                return self.objective()

            def find_route(self, customer):
                """
                Return the route that contains the passed-in customer.
                """
                for route_idx, route in enumerate(self.routes):
                    if customer in route:
                        return route, route_idx

                raise ValueError(f"Solution does not contain customer {customer}.")

        def routes_cost(routes):
            l_indices = np.fromiter(
                chain(*[[0] + route for route in routes]),
                count=problem.n_customers+len(routes)-1, dtype=int
            )
            r_indices = np.fromiter(
                chain(*[route + [0] for route in routes]),
                count=problem.n_customers+len(routes)-1, dtype=int
            )
            return problem.edge_weights[l_indices, r_indices].sum()

        def greedy_repair(state, rnd_state):
            """
            Inserts the unassigned customers in the best route. If there are no
            feasible insertions, then a new route is created.
            """
            rnd_state.shuffle(state.unassigned)

            while len(state.unassigned) != 0:
                customer = state.unassigned.pop()
                route, route_idx, idx = blink_insert(customer, state, rnd_state)

                if route is not None:
                    route.insert(idx, customer)
                    state.route_caps[route_idx] += demands[customer]
                else:
                    state.routes.append([customer])
                    state.route_caps.append(demands[customer])

            return state

        blink_rate = 1 - hyperparameters['blink_rate']

        def blink_insert(customer, state, rnd_state):
            """
            Finds the best feasible route and insertion idx for the customer.
            Return (None, None) if no feasible route insertions are found.
            """
            best_cost, best_route, best_route_idx, best_idx = float("inf"), None, None, None

            for route_idx, route_and_route_cap in enumerate(zip(state.routes, state.route_caps)):
                route, route_cap = route_and_route_cap
                if route_cap > -demands[customer]:
                    continue
                for idx in range(len(route) + 1):
                    if rnd_state.random() < blink_rate:
                        pred = 0 if idx == 0 else route[idx - 1]
                        succ = 0 if idx == len(route) else route[idx]
                        cost = - dists[pred][succ] + dists[pred][customer] + dists[customer][succ]

                        if cost < best_cost:
                            best_cost, best_route, best_route_idx, best_idx = cost, route, route_idx, idx

            return best_route, best_route_idx, best_idx

        def neighbors(customer):
            """
            Return the nearest neighbors of the customer, excluding the depot.
            """
            locations = np.argsort(problem.edge_weights[customer])
            return locations[locations != 0]

        def nearest_neighbor():
            """
            Build a solution by iteratively constructing routes, where the nearest
            customer is added until the route has met the vehicle capacity limit.
            """
            routes = []
            route_caps = []
            unvisited = set(range(1, problem.n_customers))

            while unvisited:
                route = [0]  # Start at the depot
                route_demands = -problem.capacity

                while unvisited:
                    # Add the nearest unvisited customer to the route till max capacity
                    current = route[-1]
                    nearest = [nb for nb in neighbors(current) if nb in unvisited][0]

                    if route_demands + demands[nearest] > 0:
                        break

                    route.append(nearest)
                    unvisited.remove(nearest)
                    route_demands += demands[nearest]

                customers = route[1:]  # Remove the depot
                routes.append(customers)
                route_caps.append(route_demands)

            return CvrpState(routes, route_caps)

        MAX_STRING_REMOVALS = hyperparameters['max_string_removals']
        MAX_STRING_SIZE = hyperparameters['max_string_size']

        def string_removal(state, rnd_state):
            """
            Remove partial routes around a randomly chosen customer.
            """
            destroyed = state.copy()

            avg_route_size = int(np.mean([len(route) for route in state.routes]))
            max_string_size = max(MAX_STRING_SIZE, avg_route_size)
            max_string_removals = min(len(state.routes), MAX_STRING_REMOVALS)

            destroyed_routes = []
            center = rnd_state.randint(1, problem.n_customers)

            for customer in neighbors(center):
                if len(destroyed_routes) >= max_string_removals:
                    break

                if customer in destroyed.unassigned:
                    continue

                route, route_idx = destroyed.find_route(customer)
                if route in destroyed_routes:
                    continue

                if rnd_state.random() < hyperparameters['split_rate']:
                    customers = remove_split_string(route, customer, max_string_size, rnd_state)
                else:
                    customers = remove_string(route, customer, max_string_size, rnd_state)
                destroyed.unassigned.extend(customers)
                destroyed.route_caps[route_idx] -= problem.demands[customers].sum()
                destroyed_routes.append(route)

            return destroyed

        def remove_string(route, cust, max_string_size, rnd_state):
            """
            Remove a string that constains the passed-in customer.
            """
            # Find consecutive indices to remove that contain the customer
            size = rnd_state.randint(1, min(len(route), max_string_size) + 1)
            start = route.index(cust) - rnd_state.randint(size)
            idcs = [idx % len(route) for idx in range(start, start + size)]

            # Remove indices in descending order
            removed_customers = []
            for idx in sorted(idcs, reverse=True):
                removed_customers.append(route.pop(idx))

            return removed_customers

        def remove_split_string(route, cust, max_string_size, rnd_state):
            size = rnd_state.randint(1, min(len(route), max_string_size) + 1)
            m_max = len(route) - size

            m = 1  # The number of customers to preserve
            while m < m_max and rnd_state.random() > hyperparameters['split_depth']:
                m += 1
            if m_max == 0:
                m = 0
            size += m

            start = route.index(cust) - rnd_state.randint(size)
            idcs = [idx % len(route) for idx in range(start, start + size)]
            preserve_start = rnd_state.randint(len(idcs)) if m == m_max else rnd_state.randint(len(idcs)-m)
            preserve_idcs = [idx % len(route) for idx in range(preserve_start, preserve_start + m)]

            removed_customers = []
            for idx in sorted(idcs, reverse=True):
                if idx in preserve_idcs:
                    continue
                removed_customers.append(route.pop(idx))

            return removed_customers

        alns = ALNS()
        alns.add_destroy_operator(string_removal)
        alns.add_repair_operator(greedy_repair)

        it_n = hyperparameters['iter_num']

        init = nearest_neighbor()
        select = RandomSelect(1, 1)
        accept = SimulatedAnnealing(100, 1, 0.01**(1/it_n))
        stop = MaxRuntime(target_time)

        result = alns.iterate(init, select, accept, stop)

        solution: CvrpState = result.best_state
        objective = solution.objective()
        routes_to_export = [
            [0] + route + [0]
            for route in solution.routes
        ]

        cost_history = np.empty(len(result.statistics.objectives), dtype=bench.cost_history_entry)
        cost_history['time'] = np.concatenate([[0], np.cumsum(result.statistics.runtimes)]).T
        cost_history['iteration'] = np.arange(0, result.statistics.objectives.size)
        cost_history['cost'] = result.statistics.objectives

        return bench.CheckedRunResult(
            cost=objective,
            routes=routes_to_export,
            cost_history=cost_history
        )
