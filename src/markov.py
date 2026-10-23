"""Markov-chain tennis model: P(player 1 wins the match) from any score.

Assumes every point is an independent coin flip:
    p1_serve = P(player 1 wins a point when player 1 serves)
    p2_serve = P(player 2 wins a point when player 2 serves)
The probability of winning a game, then a set, then the match follows by recursion.

State = the situation BEFORE a point:
    s1, s2 : sets won          g1, g2 : games won in the current set
    a, b   : points in the current game (0-3, and AD = 4; deuce is (3, 3), AD for P1 is (4, 3))
    server : 1 or 2 (who serves this point)       is_tb : True inside a tiebreak (points are plain counts)

Simplifications (say so in your README):
    * every set, including the final one, has a tiebreak at 6-6 (final-set rules varied by tournament/year)
    * past 6-6 in a tiebreak, and in "advantage sets" that were observed past 6-6, long tails are handled with a
      closed-form "deuce-like" formula using the AVERAGE of the two players' per-point (or per-game) probabilities
    * in observed advantage sets the server of the next set's first game is split 50/50
"""


class MarkovMatch:
    def __init__(self, p1_serve: float, p2_serve: float, sets_to_win: int = 3):
        self.ps1, self.ps2, self.k = p1_serve, p2_serve, sets_to_win
        self._game, self._tb, self._set, self._match = {}, {}, {}, {}
        self.p_tb = (p1_serve + (1 - p2_serve)) / 2       # P1's average point probability
        self.g_avg = (self.game(0, 0, 1) + self.game(0, 0, 2)) / 2   # P1's average game probability

    # ---- points -> game --------------------------------------------------------------
    def point_prob(self, server: int) -> float:
        """Probability that player 1 wins a point, given who serves."""
        return self.ps1 if server == 1 else 1 - self.ps2

    @staticmethod
    def _deuce_like(p: float, d: int) -> float:
        """P(win) from a 'level' situation: d = 0 (level), +1 (one up), -1 (one down); win by two."""
        if d >= 2:
            return 1.0
        if d <= -2:
            return 0.0
        q = 1 - p
        level = p * p / (p * p + q * q)
        if d == 0:
            return level
        return p + q * level if d > 0 else p * level

    def game(self, a: int, b: int, server: int) -> float:
        """P(player 1 wins the current game) from points (a, b)."""
        key = (a, b, server)
        if key not in self._game:
            p = self.point_prob(server)
            if a >= 3 and b >= 3:
                val = self._deuce_like(p, a - b)
            elif a == 4:
                val = 1.0
            elif b == 4:
                val = 0.0
            else:
                val = p * self.game(a + 1, b, server) + (1 - p) * self.game(a, b + 1, server)
            self._game[key] = val
        return self._game[key]

    def tiebreak(self, a: int, b: int, first: int) -> float:
        """P(player 1 wins the tiebreak) from points (a, b); `first` served the first tiebreak point."""
        key = (a, b, first)
        if key not in self._tb:
            if a >= 6 and b >= 6:
                val = self._deuce_like(self.p_tb, a - b)
            elif a >= 7:
                val = 1.0
            elif b >= 7:
                val = 0.0
            else:
                n = a + b
                server = first if ((n + 1) // 2) % 2 == 0 else 3 - first   # switch after 1 point, then every 2
                p = self.point_prob(server)
                val = p * self.tiebreak(a + 1, b, first) + (1 - p) * self.tiebreak(a, b + 1, first)
            self._tb[key] = val
        return self._tb[key]

    # ---- games -> set ----------------------------------------------------------------
    # A "set vector" has 4 entries: [P1 wins set & P1 serves next set's first game,
    #                                P1 wins set & P2 serves it, P2 wins & P1 serves, P2 wins & P2 serves]
    @staticmethod
    def _terminal(p1_won: bool, next_server: int) -> tuple:
        v = [0.0] * 4
        v[(0 if p1_won else 2) + next_server - 1] = 1.0
        return tuple(v)

    @staticmethod
    def _set_over(n1: int, n2: int) -> bool:
        return max(n1, n2) >= 6 and abs(n1 - n2) >= 2

    def set_vec(self, n1: int, n2: int, s: int, adv: bool) -> tuple:
        """Set outcome from the START of a game (0-0 in the game) at games (n1, n2), server s."""
        key = (n1, n2, s, adv)
        if key not in self._set:
            if adv and n1 >= 6 and n2 >= 6:                 # observed advantage set: closed form
                pw = self._deuce_like(self.g_avg, n1 - n2)
                val = (pw / 2, pw / 2, (1 - pw) / 2, (1 - pw) / 2)
            elif not adv and n1 == 6 and n2 == 6:           # tiebreak at 6-6
                t = self.tiebreak(0, 0, s)
                v = [0.0] * 4
                v[3 - s - 1] = t                            # next set's first server = 3 - s
                v[2 + 3 - s - 1] = 1 - t
                val = tuple(v)
            else:
                val = self._after_game(n1, n2, s, self.game(0, 0, s), adv)
            self._set[key] = val
        return self._set[key]

    def _after_game(self, n1: int, n2: int, s: int, gp: float, adv: bool) -> tuple:
        """Set vector given the current game will be won by P1 with probability gp."""
        out = [0.0] * 4
        for win, prob in ((1, gp), (0, 1 - gp)):
            m1, m2 = n1 + win, n2 + 1 - win
            nxt = 3 - s
            vec = self._terminal(m1 > m2, nxt) if self._set_over(m1, m2) else self.set_vec(m1, m2, nxt, adv)
            for i in range(4):
                out[i] += prob * vec[i]
        return tuple(out)

    # ---- sets -> match ---------------------------------------------------------------
    def _match_from_vec(self, s1: int, s2: int, vec: tuple) -> float:
        total = 0.0
        for i, prob in enumerate(vec):
            p1_won, nxt = i < 2, i % 2 + 1
            n1, n2 = s1 + (1 if p1_won else 0), s2 + (0 if p1_won else 1)
            if n1 == self.k:
                total += prob
            elif n2 == self.k:
                continue
            else:
                total += prob * self.match_from_set_start(n1, n2, nxt)
        return total

    def match_from_set_start(self, s1: int, s2: int, first: int) -> float:
        if s1 >= self.k:          # safety: the match is already decided
            return 1.0
        if s2 >= self.k:
            return 0.0
        key = (s1, s2, first)
        if key not in self._match:
            self._match[key] = self._match_from_vec(s1, s2, self.set_vec(0, 0, first, False))
        return self._match[key]

    # ---- public API ------------------------------------------------------------------
    def start_prob(self) -> float:
        """P(player 1 wins the match) before the first point, averaged over who serves first."""
        return 0.5 * (self.match_from_set_start(0, 0, 1) + self.match_from_set_start(0, 0, 2))

    def win_prob(self, s1: int, s2: int, g1: int, g2: int, a: int, b: int, server: int, is_tb: bool) -> float:
        """P(player 1 wins the match) from the state before a point."""
        if s1 >= self.k:          # safety: impossible state, the match is already decided
            return 1.0
        if s2 >= self.k:
            return 0.0
        if is_tb:
            n = a + b
            first = server if ((n + 1) // 2) % 2 == 0 else 3 - server
            t = self.tiebreak(a, b, first)
            vec = [0.0] * 4
            vec[3 - first - 1] = t
            vec[2 + 3 - first - 1] = 1 - t
            return self._match_from_vec(s1, s2, tuple(vec))
        adv = g1 >= 6 and g2 >= 6
        vec = self._after_game(g1, g2, server, self.game(a, b, server), adv)
        return self._match_from_vec(s1, s2, vec)


def solve_serve_probs(p_match: float, base: float, sets_to_win: int = 3, iters: int = 30):
    """Find (p1_serve, p2_serve) = (base + d, base - d) so the model's pre-match win probability equals p_match."""
    target = min(max(p_match, 0.01), 0.99)
    lo, hi = -0.25, 0.25
    for _ in range(iters):
        mid = (lo + hi) / 2
        if MarkovMatch(base + mid, base - mid, sets_to_win).start_prob() < target:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return base + d, base - d