import time


class BoardBase:

    def __init__(self, debug=False):
        self._debug = debug

        if self._debug:
            print(f"Creating board: {self.city_list}")

    @property
    def size(self):
        raise ValueError("not implemented")

    @property
    def city_list(self):
        raise ValueError("not implemented")

    def _spread(self):
        raise ValueError("not implemented")

    def _city_i(self, i):
        raise ValueError("not implemented")

    def move(self, i):
        res = self._city_i(i)
        if self._debug:
            print(f"Checking location: {i}, Value = {res}")
        self._spread()
        if self._debug:
            print(f"New Board: {self.city_list}")
        return res


class Board0001(BoardBase):
    # N - size
    # move complexity
    # time complexity: O(N)
    # space complexity: O(N)
    def __init__(self, size, seed, **kwargs):

        self._city_list = [0] * size
        self._city_list[seed] = 1

        self._size = size
        self._seed = seed

        super().__init__(**kwargs)

    @property
    def size(self):
        return self._size

    @property
    def city_list(self):
        return self._city_list

    def _spread(self):
        self._city_list[self._seed] = self._city_list[self._seed] + 1

        i = self._seed - 1
        while i >= 0:
            self._city_list[i] = self._city_list[i] + 1
            if self._city_list[i] == 1:
                break
            i = i - 1

        i = self._seed + 1
        while i < self._size:
            self._city_list[i] = self._city_list[i] + 1
            if self._city_list[i] == 1:
                break
            i = i + 1

    def _city_i(self, i):
        return self._city_list[i]


class Board0002(BoardBase):
    # since the only requirement is to have the move and size function
    # there's no need to create a data structure for city at all
    # N - size
    # move complexity
    # time complexity: O(1)
    # space complexity: O(1)
    def __init__(self, size, seed, **kwargs):

        self._size = size
        self._seed = seed
        self._iter = 1  # the current iteration

        super().__init__(**kwargs)

    @property
    def size(self):
        return self._size

    @property
    def city_list(self):
        # this function should only be called by debug print
        # so does not contribute to complexity
        res = [0] * self.size

        res[self._seed] = self._city_i(self._seed)

        i = self._seed - 1
        while i >= 0:
            res[i] = self._city_i(i)
            if res[i] <= 1:
                break
            i = i - 1

        i = self._seed + 1
        while i < self.size:
            res[i] = self._city_i(i)
            if res[i] <= 1:
                break
            i = i + 1

        return res

    def _city_i(self, i):
        # main logic, the case load level at i is
        # the maximum of 0 and the current iteration
        # count + 1 minus the distance from epicenter
        res = max(self._iter - abs(self._seed - i), 0)
        return res

    def _spread(self):
        self._iter = self._iter + 1


class Solver0001:
    def __init__(self, board):
        self._board = board
        self._iter = None

    def solve(self):
        self._iter = None

        i = 0
        cur_iter = 1
        while True:
            res = self._board.move(i)
            if res > 0:
                # cur_iter is the case load at the seed city
                # cur_iter - res is the distance between
                # the current pos and the seed city
                self._iter = cur_iter
                return cur_iter - res + i
            i = i + 1
            cur_iter = cur_iter + 1


class Solver0001:
    # time complexity: O(N)
    def __init__(self, board):
        self._board = board
        self._iter = None

    def solve(self):
        self._iter = None

        i = 0
        cur_iter = 1
        while True:
            res = self._board.move(i)
            if res > 0:
                # cur_iter is the case load at the seed city
                # cur_iter - res is the distance between
                # the current pos and the seed city
                self._iter = cur_iter
                return cur_iter - res + i
            i = i + 1
            cur_iter = cur_iter + 1


class Solver0002:
    """
    In the second solver the step size is modified during each iteration
    from teh observation that during iteration cur_iter
    the size of the infected cities has grown to 1 + 2 * cur_iter
    therefore we can increase the step to 1 + 2 * cur_iter
    without fear of crossing the entire set of infected cities

    However, this does cause a problem of not knowing whether
    you are at the left side or the right side of the infected cities
    so a second check is necessary to determine that

    # time complexity: O(sqrt(N))
    # 1 + 2 + (2 + 2) + ...
    # 1 + 2 * 1 + 2 * 2 + ...
    # Sum(2 * i, {i, 0, k}) = 2 Sum(i, {i, 0, k}) = k (k - 1) ~ k^2
    # k^2 = N
    # k = sqrt(N)
    """
    def __init__(self, board):
        self._board = board
        self._iter = None

    def solve(self):
        self._iter = None

        i = 0
        cur_iter = 1
        while True:
            res = self._board.move(i)
            if res > 0:
                self._iter = cur_iter
                break
            i = min(i + 1 + 2 * cur_iter, self._board.size - 1)
            cur_iter = cur_iter + 1

        # do one more check to determine side
        side = None
        if i == 0:
            side = 'L'
        elif i == self._board.size - 1:
            side = 'R'
        else:
            res2 = self._board.move(i + 1)
            if res2 > res:
                side = 'L'
            else:
                side = 'R'

        if side == 'L':
            return cur_iter - res + i
        return i - (cur_iter - res)


class Timer:
    def __init__(self, name):
        self._name = name
        self._time = None

    def __enter__(self):
        self._time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._time = time.perf_counter() - self._time
        print(f"Running {self._name} took {self._time} seconds")


def test_simple(board_class):

    print(f"Testing board {board_class.__name__}")

    board = board_class(10, 2, debug=True)

    assert board.size == 10
    assert board.move(0) == 0
    assert board.move(1) == 1
    assert board.move(2) == 3
    print("Success")


def test_solver(solver_class, size=10, seed=2):

    board = Board(size=size, seed=seed)

    solver = solver_class(board)
    with Timer(f"solver {solver_class.__name__}"):
        res = solver.solve()
    print(f"Solver seed: {res}")
    assert res == seed


def test_all(solver_class, size=20):

    for s in range(size):
        print(f"Test solving size: {size} seed: {s}")
        test_solver(solver_class, size=size, seed=s)
        print(f"Success")


def test_range(solver_class, max_size=20):

    for sz in range(1, max_size + 1):
        test_all(solver_class, size=sz)


def test_suite():
    test_simple(Board0001)
    test_simple(Board0002)

    test_solver(Solver0001)
    test_all(Solver0001, size=1000)
    test_range(Solver0001)
    test_solver(Solver0001, size=100000, seed=50000)

    test_all(Solver0002, size=1000)

    test_solver(Solver0001, size=100000, seed=50000)
    test_solver(Solver0002, size=100000, seed=50000)


# finally, the solution :)
class Board(Board0002):
    pass


class Solver(Solver0002):
    pass


if __name__ == "__main__":

    test_simple(Board)
    test_all(Solver, size=1000)

    test_suite()
