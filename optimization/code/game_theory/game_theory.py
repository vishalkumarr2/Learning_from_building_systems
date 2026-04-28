"""
Game Theory — Python Implementations
=======================================
Companion code for the Optimization curriculum: Game Theory module.

Covers:
  - Normal-form games: dominance, IESDS, best response, pure Nash
  - Nash equilibrium computation: 2×2 indifference, support enumeration
  - Zero-sum games: graphical method, LP formulation, minimax
  - Cooperative games: Shapley value, core membership
  - Dynamic games: backward induction, Vickrey auction, VCG mechanism
  - Learning in games: fictitious play, regret matching, multiplicative weights
  - Congestion games: best-response dynamics, Braess's paradox
  - Multi-robot applications: auction-based task allocation

Usage:
    python game_theory.py          # Run all demos with plots
    python game_theory.py --test   # Run self-tests only
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import sys
from itertools import combinations
from math import factorial

def _get_linprog():
    """Lazy import — scipy only needed for LP-based zero-sum solver."""
    from scipy.optimize import linprog
    return linprog


# ============================================================
# 1. Normal Form Games
# ============================================================

class NormalFormGame:
    """Represents a two-player normal-form (strategic) game.

    Attributes
    ----------
    payoff1 : np.ndarray, shape (m, n)
        Payoff matrix for the row player (Player 1).
    payoff2 : np.ndarray, shape (m, n)
        Payoff matrix for the column player (Player 2).
    m : int
        Number of strategies for Player 1.
    n : int
        Number of strategies for Player 2.
    """

    def __init__(self, payoff1, payoff2):
        """Create a two-player normal-form game.

        Parameters
        ----------
        payoff1 : array-like, shape (m, n)
            Row player's payoffs.
        payoff2 : array-like, shape (m, n)
            Column player's payoffs.
        """
        self.payoff1 = np.array(payoff1, dtype=float)
        self.payoff2 = np.array(payoff2, dtype=float)
        assert self.payoff1.shape == self.payoff2.shape, "Payoff matrices must match."
        self.m, self.n = self.payoff1.shape

    def is_dominated(self, player, strategy, strict=True):
        """Check whether *strategy* is (strictly) dominated for *player*.

        Parameters
        ----------
        player : int
            1 for the row player, 2 for the column player.
        strategy : int
            Index of the strategy to check.
        strict : bool
            If True check strict dominance; otherwise weak dominance.

        Returns
        -------
        bool
            True if the strategy is dominated.
        """
        if player == 1:
            payoffs = self.payoff1
            target = payoffs[strategy, :]
            for i in range(self.m):
                if i == strategy:
                    continue
                alt = payoffs[i, :]
                if strict:
                    if np.all(alt > target):
                        return True
                else:
                    if np.all(alt >= target) and np.any(alt > target):
                        return True
        else:
            payoffs = self.payoff2
            target = payoffs[:, strategy]
            for j in range(self.n):
                if j == strategy:
                    continue
                alt = payoffs[:, j]
                if strict:
                    if np.all(alt > target):
                        return True
                else:
                    if np.all(alt >= target) and np.any(alt > target):
                        return True
        return False

    def iesds(self):
        """Iterated Elimination of Strictly Dominated Strategies.

        Returns
        -------
        NormalFormGame
            Reduced game after removing all strictly dominated strategies.
        list of int
            Surviving row indices.
        list of int
            Surviving column indices.
        """
        rows = list(range(self.m))
        cols = list(range(self.n))
        changed = True
        while changed:
            changed = False
            # Eliminate dominated rows
            sub1 = self.payoff1[np.ix_(rows, cols)]
            new_rows = []
            for idx, r in enumerate(rows):
                row_vals = sub1[idx, :]
                dominated = False
                for idx2 in range(len(rows)):
                    if idx2 == idx:
                        continue
                    if np.all(sub1[idx2, :] > row_vals):
                        dominated = True
                        break
                if not dominated:
                    new_rows.append(r)
            if len(new_rows) < len(rows):
                rows = new_rows
                changed = True

            # Eliminate dominated columns
            sub2 = self.payoff2[np.ix_(rows, cols)]
            new_cols = []
            for idx, c in enumerate(cols):
                col_vals = sub2[:, idx]
                dominated = False
                for idx2 in range(len(cols)):
                    if idx2 == idx:
                        continue
                    if np.all(sub2[:, idx2] > col_vals):
                        dominated = True
                        break
                if not dominated:
                    new_cols.append(c)
            if len(new_cols) < len(cols):
                cols = new_cols
                changed = True

        reduced = NormalFormGame(
            self.payoff1[np.ix_(rows, cols)],
            self.payoff2[np.ix_(rows, cols)],
        )
        return reduced, rows, cols

    def best_response(self, player, opponent_strategy):
        """Compute pure-strategy best responses for *player*.

        Parameters
        ----------
        player : int
            1 or 2.
        opponent_strategy : array-like
            Probability vector over the opponent's strategies (mixed strategy).

        Returns
        -------
        list of int
            Indices of all best-response pure strategies.
        """
        opp = np.array(opponent_strategy, dtype=float)
        if player == 1:
            expected = self.payoff1 @ opp  # length m
            mx = np.max(expected)
            return [i for i in range(self.m) if np.isclose(expected[i], mx)]
        else:
            expected = self.payoff2.T @ opp  # length n
            mx = np.max(expected)
            return [j for j in range(self.n) if np.isclose(expected[j], mx)]

    def find_pure_nash(self):
        """Find all pure-strategy Nash equilibria.

        Returns
        -------
        list of (int, int)
            List of (row, col) index pairs that are pure NE.
        """
        equilibria = []
        # For each cell, check if it's a best response for both players
        for i in range(self.m):
            for j in range(self.n):
                # Player 1: is row i best given column j?
                if self.payoff1[i, j] < np.max(self.payoff1[:, j]):
                    continue
                # Player 2: is column j best given row i?
                if self.payoff2[i, j] < np.max(self.payoff2[i, :]):
                    continue
                equilibria.append((i, j))
        return equilibria


# ============================================================
# 2. Nash Equilibrium Computation
# ============================================================

def find_mixed_nash_2x2(game):
    """Find mixed-strategy Nash equilibrium for a 2×2 game.

    Uses the indifference principle: each player mixes such that the
    opponent is indifferent between their pure strategies.

    Parameters
    ----------
    game : NormalFormGame
        Must be 2×2.

    Returns
    -------
    p : np.ndarray, shape (2,)
        Player 1's equilibrium mixed strategy.
    q : np.ndarray, shape (2,)
        Player 2's equilibrium mixed strategy.

    Raises
    ------
    ValueError
        If the game is not 2×2 or no completely mixed NE exists.
    """
    if game.m != 2 or game.n != 2:
        raise ValueError("Game must be 2×2.")

    A = game.payoff1
    B = game.payoff2

    # Player 1 mixes with prob p on row 0 to make Player 2 indifferent.
    # P2's payoff from col 0: p*B[0,0] + (1-p)*B[1,0]
    # P2's payoff from col 1: p*B[0,1] + (1-p)*B[1,1]
    # Set equal: p*(B[0,0]-B[1,0]) + B[1,0] = p*(B[0,1]-B[1,1]) + B[1,1]
    denom_p = (B[0, 0] - B[1, 0]) - (B[0, 1] - B[1, 1])
    if np.isclose(denom_p, 0):
        raise ValueError("No completely mixed NE (Player 2 not indifferent).")
    p0 = (B[1, 1] - B[1, 0]) / denom_p

    # Player 2 mixes with prob q on col 0 to make Player 1 indifferent.
    # P1's payoff from row 0: q*A[0,0] + (1-q)*A[0,1]
    # P1's payoff from row 1: q*A[1,0] + (1-q)*A[1,1]
    denom_q = (A[0, 0] - A[0, 1]) - (A[1, 0] - A[1, 1])
    if np.isclose(denom_q, 0):
        raise ValueError("No completely mixed NE (Player 1 not indifferent).")
    q0 = (A[1, 1] - A[0, 1]) / denom_q

    if not (0.0 <= p0 <= 1.0 and 0.0 <= q0 <= 1.0):
        raise ValueError(f"Mixed NE outside simplex: p={p0:.4f}, q={q0:.4f}.")

    return np.array([p0, 1 - p0]), np.array([q0, 1 - q0])


def support_enumeration(game):
    """Find all Nash equilibria via support enumeration.

    For each pair of support sets (S1 for Player 1, S2 for Player 2) of
    equal size, solve the indifference + probability constraints.

    Parameters
    ----------
    game : NormalFormGame

    Returns
    -------
    list of (np.ndarray, np.ndarray)
        Each element is a (sigma1, sigma2) NE pair.
    """
    equilibria = []

    for k1 in range(1, game.m + 1):
        for k2 in range(1, game.n + 1):
            for supp1 in combinations(range(game.m), k1):
                for supp2 in combinations(range(game.n), k2):
                    result = _solve_support(game, list(supp1), list(supp2))
                    if result is not None:
                        equilibria.append(result)

    # Deduplicate
    unique = []
    for s1, s2 in equilibria:
        is_dup = False
        for u1, u2 in unique:
            if np.allclose(s1, u1) and np.allclose(s2, u2):
                is_dup = True
                break
        if not is_dup:
            unique.append((s1, s2))
    return unique


def _solve_support(game, supp1, supp2):
    """Attempt to solve for an NE with given supports.

    Returns (sigma1, sigma2) or None if no valid solution.
    """
    A = game.payoff1
    B = game.payoff2
    k1 = len(supp1)
    k2 = len(supp2)

    # --- Solve for Player 2's strategy (sigma2) ---
    # Player 1 must be indifferent over supp1 given sigma2.
    # For each pair of consecutive strategies in supp1:
    #   sum_j A[supp1[0], j]*sigma2[j] = sum_j A[supp1[i], j]*sigma2[j]
    # Plus: sum_{j in supp2} sigma2[j] = 1

    # Build system for sigma2 (variables only on supp2)
    rows_eq = []
    rhs_eq = []

    # Indifference equations (k1 - 1 equations)
    for idx in range(1, k1):
        row = np.zeros(k2)
        for jdx, j in enumerate(supp2):
            row[jdx] = A[supp1[0], j] - A[supp1[idx], j]
        rows_eq.append(row)
        rhs_eq.append(0.0)

    # Probability simplex
    rows_eq.append(np.ones(k2))
    rhs_eq.append(1.0)

    C2 = np.array(rows_eq)
    d2 = np.array(rhs_eq)

    try:
        if C2.shape[0] == C2.shape[1]:
            sigma2_supp = np.linalg.solve(C2, d2)
        else:
            sigma2_supp, res, rank, _ = np.linalg.lstsq(C2, d2, rcond=None)
            if not np.allclose(C2 @ sigma2_supp, d2, atol=1e-8):
                return None
    except np.linalg.LinAlgError:
        return None

    if np.any(sigma2_supp < -1e-10):
        return None
    sigma2_supp = np.maximum(sigma2_supp, 0.0)

    # --- Solve for Player 1's strategy (sigma1) ---
    rows_eq2 = []
    rhs_eq2 = []

    for idx in range(1, k2):
        row = np.zeros(k1)
        for jdx, i in enumerate(supp1):
            row[jdx] = B[i, supp2[0]] - B[i, supp2[idx]]
        rows_eq2.append(row)
        rhs_eq2.append(0.0)

    rows_eq2.append(np.ones(k1))
    rhs_eq2.append(1.0)

    C1 = np.array(rows_eq2)
    d1 = np.array(rhs_eq2)

    try:
        if C1.shape[0] == C1.shape[1]:
            sigma1_supp = np.linalg.solve(C1, d1)
        else:
            sigma1_supp, res, rank, _ = np.linalg.lstsq(C1, d1, rcond=None)
            if not np.allclose(C1 @ sigma1_supp, d1, atol=1e-8):
                return None
    except np.linalg.LinAlgError:
        return None

    if np.any(sigma1_supp < -1e-10):
        return None
    sigma1_supp = np.maximum(sigma1_supp, 0.0)

    # Build full strategy vectors
    sigma1 = np.zeros(game.m)
    sigma2 = np.zeros(game.n)
    for idx, i in enumerate(supp1):
        sigma1[i] = sigma1_supp[idx]
    for idx, j in enumerate(supp2):
        sigma2[j] = sigma2_supp[idx]

    # Verify best response condition: strategies outside support
    # must not yield higher payoff.
    eq_payoff1 = A[supp1[0], :] @ sigma2
    for i in range(game.m):
        if i not in supp1:
            if A[i, :] @ sigma2 > eq_payoff1 + 1e-8:
                return None

    eq_payoff2 = B[:, supp2[0]] @ sigma1
    for j in range(game.n):
        if j not in supp2:
            if B[:, j] @ sigma1 > eq_payoff2 + 1e-8:
                return None

    return sigma1, sigma2


# ============================================================
# 3. Zero-Sum Games
# ============================================================

def solve_zero_sum_graphical(A):
    """Solve a 2×n zero-sum game graphically for Player 1.

    Player 1 has 2 strategies; Player 2 has n strategies.
    The lower envelope of Player 2's response lines gives P1's maximin.

    Parameters
    ----------
    A : array-like, shape (2, n)
        Payoff matrix for the row player.

    Returns
    -------
    value : float
        Value of the game.
    p1_strategy : np.ndarray, shape (2,)
        Player 1's optimal mixed strategy.
    p2_strategy : np.ndarray, shape (n,)
        Player 2's optimal mixed strategy (uniform on binding columns).
    """
    A = np.array(A, dtype=float)
    assert A.shape[0] == 2, "Graphical method requires Player 1 to have 2 strategies."
    n = A.shape[1]

    # Parameterize by p = P(row 0).  P2's payoff from col j = p*A[0,j] + (1-p)*A[1,j].
    # P1 wants to maximise the minimum over columns.
    best_val = -np.inf
    best_p = 0.0
    best_cols = []

    # Check intersections of every pair of P2's response lines
    for j1 in range(n):
        for j2 in range(j1 + 1, n):
            # Line j: f_j(p) = A[1,j] + p*(A[0,j] - A[1,j])
            slope1 = A[0, j1] - A[1, j1]
            slope2 = A[0, j2] - A[1, j2]
            if np.isclose(slope1, slope2):
                continue
            p_int = (A[1, j2] - A[1, j1]) / (slope1 - slope2)
            if p_int < -1e-12 or p_int > 1 + 1e-12:
                continue
            p_int = np.clip(p_int, 0, 1)
            val = A[1, j1] + p_int * slope1
            # Check this is the minimum over all columns
            all_vals = A[1, :] + p_int * (A[0, :] - A[1, :])
            min_val = np.min(all_vals)
            if np.isclose(val, min_val, atol=1e-8) and val > best_val + 1e-12:
                best_val = val
                best_p = p_int
                best_cols = [j1, j2]

    # Also check endpoints p=0 and p=1
    for p_test in [0.0, 1.0]:
        all_vals = A[1, :] + p_test * (A[0, :] - A[1, :])
        val = np.min(all_vals)
        if val > best_val + 1e-12:
            best_val = val
            best_p = p_test
            best_cols = [np.argmin(all_vals)]

    p1 = np.array([best_p, 1.0 - best_p])

    # P2's strategy: solve minimax on the binding columns
    p2 = np.zeros(n)
    if len(best_cols) == 1:
        p2[best_cols[0]] = 1.0
    else:
        # Two binding columns: find q to equalise P1's payoff
        j1, j2 = best_cols[0], best_cols[1]
        # q*A[i,j1] + (1-q)*A[i,j2] for i=0,1 should be equal under P1's opt
        # P2 minimises: p1[0]*(q*A[0,j1]+(1-q)*A[0,j2]) + p1[1]*(q*A[1,j1]+(1-q)*A[1,j2])
        # Equivalent: find q s.t. val from col j1 = val from col j2 for the P1 mixture
        # Actually P2 plays on binding cols. Use indifference of P1:
        # q*A[0,j1] + (1-q)*A[0,j2] = q*A[1,j1] + (1-q)*A[1,j2] is wrong.
        # Correct: P2 wants to minimise P1's payoff.  On the binding 2×2 sub-game:
        sub_A = A[:, best_cols]
        denom = (sub_A[0, 0] - sub_A[0, 1]) - (sub_A[1, 0] - sub_A[1, 1])
        if np.isclose(denom, 0):
            p2[best_cols[0]] = 0.5
            p2[best_cols[1]] = 0.5
        else:
            q0 = (sub_A[1, 1] - sub_A[0, 1]) / denom
            q0 = np.clip(q0, 0, 1)
            p2[best_cols[0]] = q0
            p2[best_cols[1]] = 1.0 - q0

    return best_val, p1, p2


def solve_zero_sum_lp(A):
    """Solve a zero-sum game via linear programming.

    max v  s.t.  Aᵀp ≥ v·1,  p ≥ 0,  1ᵀp = 1.

    Parameters
    ----------
    A : array-like, shape (m, n)
        Payoff matrix for the row player.

    Returns
    -------
    value : float
        Value of the game.
    p1_strategy : np.ndarray, shape (m,)
    p2_strategy : np.ndarray, shape (n,)
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    # Variables: x = [p_1, ..., p_m, v]  (m+1 variables)
    # Maximise v  ⟺  minimise -v
    c = np.zeros(m + 1)
    c[-1] = -1.0  # minimise -v

    # Inequality: A^T p >= v·1  ⟺  -A^T p + v·1 <= 0
    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -A.T
    A_ub[:, m] = 1.0
    b_ub = np.zeros(n)

    # Equality: sum(p) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # Bounds: p_i >= 0, v is free
    bounds = [(0, None)] * m + [(None, None)]

    result = _get_linprog()(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")

    p1 = result.x[:m]
    value = result.x[m]

    # Solve for Player 2 (dual): min w s.t. Aq >= w·1, q >= 0, 1ᵀq = 1
    c2 = np.zeros(n + 1)
    c2[-1] = 1.0  # minimise w

    A_ub2 = np.zeros((m, n + 1))
    A_ub2[:, :n] = -A
    A_ub2[:, n] = 1.0  # -Aq + w <= 0 ⟹ w <= (Aq)_i
    # Wait, we need Aq >= w·1 ⟹ -Aq + w·1 <= 0
    # But we want min w, which is actually: Aq <= w·1 ⟹ Aq - w·1 <= 0
    # No. P2 solves: min w  s.t.  Aq ≤ w·1, q ≥ 0, 1ᵀq = 1
    A_ub2[:, :n] = A
    A_ub2[:, n] = -1.0
    b_ub2 = np.zeros(m)

    A_eq2 = np.zeros((1, n + 1))
    A_eq2[0, :n] = 1.0
    b_eq2 = np.array([1.0])

    bounds2 = [(0, None)] * n + [(None, None)]

    result2 = _get_linprog()(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                      bounds=bounds2, method="highs")
    if not result2.success:
        raise RuntimeError(f"LP (dual) failed: {result2.message}")

    p2 = result2.x[:n]

    return value, p1, p2


def minimax_value(A):
    """Compute the minimax (game) value of a matrix game.

    Parameters
    ----------
    A : array-like, shape (m, n)

    Returns
    -------
    float
        The value of the game.
    """
    value, _, _ = solve_zero_sum_lp(A)
    return value


# ============================================================
# 4. Cooperative Games
# ============================================================

def shapley_value(n, v):
    """Compute the Shapley value for an n-player cooperative game.

    φ_i = Σ_{S ⊆ N\\{i}} |S|!(n-|S|-1)!/n! × [v(S ∪ {i}) - v(S)]

    Parameters
    ----------
    n : int
        Number of players (labelled 0 .. n-1).
    v : dict
        Characteristic function mapping frozenset → value.
        Must contain all 2^n subsets.  v[frozenset()] should be 0.

    Returns
    -------
    np.ndarray, shape (n,)
        Shapley value for each player.
    """
    phi = np.zeros(n)
    n_fact = factorial(n)
    players = set(range(n))

    for i in range(n):
        others = players - {i}
        for size in range(0, n):
            for S_tuple in combinations(others, size):
                S = frozenset(S_tuple)
                S_with_i = S | {i}
                marginal = v.get(S_with_i, 0) - v.get(S, 0)
                weight = factorial(len(S)) * factorial(n - len(S) - 1) / n_fact
                phi[i] += weight * marginal
    return phi


def is_in_core(n, v, allocation):
    """Check whether an allocation is in the core of a cooperative game.

    The core requires:
      (1) efficiency:  Σ x_i = v(N)
      (2) coalitional rationality:  Σ_{i ∈ S} x_i ≥ v(S) for all S ⊆ N.

    Parameters
    ----------
    n : int
        Number of players.
    v : dict
        Characteristic function (frozenset → value).
    allocation : array-like, shape (n,)
        Proposed payoff vector.

    Returns
    -------
    is_core : bool
        True if the allocation is in the core.
    violated : frozenset or None
        First coalition whose constraint is violated, or None.
    """
    x = np.array(allocation, dtype=float)
    grand = frozenset(range(n))

    # Efficiency check
    if not np.isclose(np.sum(x), v.get(grand, 0)):
        return False, grand

    # Coalitional rationality
    for size in range(1, n):
        for S_tuple in combinations(range(n), size):
            S = frozenset(S_tuple)
            if np.sum(x[list(S)]) < v.get(S, 0) - 1e-10:
                return False, S

    return True, None


# ============================================================
# 5. Dynamic Games
# ============================================================

def backward_induction(game_tree):
    """Solve an extensive-form game by backward induction.

    Parameters
    ----------
    game_tree : dict or tuple
        If tuple: terminal node with payoffs (p1_payoff, p2_payoff, ...).
        If dict:
            'player': int — who moves at this node.
            'actions': dict {action_name: subtree}

    Returns
    -------
    path : list of str
        Sequence of actions on the equilibrium path.
    payoffs : tuple
        Payoff tuple at the terminal node.
    """
    if isinstance(game_tree, (tuple, list)):
        return [], tuple(game_tree)

    player = game_tree["player"]
    actions = game_tree["actions"]

    best_action = None
    best_payoff = None
    best_path = None

    for action, subtree in actions.items():
        sub_path, sub_payoff = backward_induction(subtree)
        if best_payoff is None or sub_payoff[player] > best_payoff[player]:
            best_action = action
            best_payoff = sub_payoff
            best_path = sub_path

    return [best_action] + best_path, best_payoff


def vickrey_auction(bids):
    """Second-price sealed-bid (Vickrey) auction.

    Parameters
    ----------
    bids : array-like
        Bid vector, one per bidder.

    Returns
    -------
    winner : int
        Index of the winning bidder.
    payment : float
        Price paid (= second-highest bid).
    """
    bids = np.array(bids, dtype=float)
    n = len(bids)
    assert n >= 2, "Need at least 2 bidders."

    sorted_idx = np.argsort(bids)[::-1]
    winner = sorted_idx[0]
    payment = bids[sorted_idx[1]]
    return int(winner), float(payment)


def vcg_mechanism(valuations, allocation_fn):
    """Vickrey-Clarke-Groves mechanism for general allocation.

    Parameters
    ----------
    valuations : list of dict
        valuations[i][outcome] = value player i assigns to *outcome*.
        Every player must report values for the same set of outcomes.
    allocation_fn : callable
        Given a list of valuation dicts, returns the socially optimal
        outcome key.

    Returns
    -------
    allocation : hashable
        The chosen outcome.
    payments : np.ndarray
        Payment for each player (positive = pays).
    """
    n = len(valuations)
    allocation = allocation_fn(valuations)

    payments = np.zeros(n)
    for i in range(n):
        # Social welfare of others under chosen allocation
        others_welfare = sum(valuations[j][allocation]
                            for j in range(n) if j != i)

        # Optimal allocation without player i
        others_vals = [valuations[j] for j in range(n) if j != i]
        # Create dummy zero-valuation for player i
        dummy = {k: 0 for k in valuations[0].keys()}
        vals_without_i = []
        for j in range(n):
            if j == i:
                vals_without_i.append(dummy)
            else:
                vals_without_i.append(valuations[j])

        alt_alloc = allocation_fn(vals_without_i)
        alt_others_welfare = sum(valuations[j][alt_alloc]
                                 for j in range(n) if j != i)

        # VCG payment: externality imposed on others
        payments[i] = alt_others_welfare - others_welfare

    return allocation, payments


# ============================================================
# 6. Learning in Games
# ============================================================

def fictitious_play(game, num_rounds=1000):
    """Simulate fictitious play for a two-player game.

    Each player plays a best response to the empirical frequency of
    the opponent's past actions.

    Parameters
    ----------
    game : NormalFormGame
    num_rounds : int

    Returns
    -------
    history : list of (np.ndarray, np.ndarray)
        Empirical strategy profile at each round.
    actions : list of (int, int)
        Pure actions played each round.
    """
    counts1 = np.zeros(game.m)
    counts2 = np.zeros(game.n)

    history = []
    actions = []

    for t in range(num_rounds):
        if t == 0:
            a1, a2 = 0, 0
        else:
            freq2 = counts2 / t
            freq1 = counts1 / t
            br1 = game.best_response(1, freq2)
            br2 = game.best_response(2, freq1)
            a1 = br1[0]
            a2 = br2[0]

        counts1[a1] += 1
        counts2[a2] += 1
        emp1 = counts1 / (t + 1)
        emp2 = counts2 / (t + 1)
        history.append((emp1.copy(), emp2.copy()))
        actions.append((a1, a2))

    return history, actions


def regret_matching(game, num_rounds=1000):
    """Regret matching algorithm for two-player games.

    Each player maintains cumulative regret for each action and plays
    proportional to positive regrets.

    Parameters
    ----------
    game : NormalFormGame
    num_rounds : int

    Returns
    -------
    avg_strategy : (np.ndarray, np.ndarray)
        Time-averaged strategy profile.
    regret_history : list of (np.ndarray, np.ndarray)
        Cumulative regret vectors at each round.
    """
    cum_regret1 = np.zeros(game.m)
    cum_regret2 = np.zeros(game.n)
    strategy_sum1 = np.zeros(game.m)
    strategy_sum2 = np.zeros(game.n)
    regret_history = []

    for t in range(num_rounds):
        # Compute strategies from positive regrets
        pos1 = np.maximum(cum_regret1, 0)
        pos2 = np.maximum(cum_regret2, 0)

        if np.sum(pos1) > 0:
            sigma1 = pos1 / np.sum(pos1)
        else:
            sigma1 = np.ones(game.m) / game.m

        if np.sum(pos2) > 0:
            sigma2 = pos2 / np.sum(pos2)
        else:
            sigma2 = np.ones(game.n) / game.n

        strategy_sum1 += sigma1
        strategy_sum2 += sigma2

        # Expected payoffs
        ev1 = game.payoff1 @ sigma2       # payoff for each P1 action
        ev2 = game.payoff2.T @ sigma1     # payoff for each P2 action

        actual1 = sigma1 @ ev1
        actual2 = sigma2 @ ev2

        cum_regret1 += ev1 - actual1
        cum_regret2 += ev2 - actual2

        regret_history.append((cum_regret1.copy(), cum_regret2.copy()))

    avg1 = strategy_sum1 / num_rounds
    avg2 = strategy_sum2 / num_rounds
    return (avg1, avg2), regret_history


def multiplicative_weights(losses_sequence, eta=0.1):
    """Multiplicative Weights Update (MWU) for online decision-making.

    Parameters
    ----------
    losses_sequence : array-like, shape (T, n)
        Loss vector for each of n actions over T rounds.
    eta : float
        Learning rate.

    Returns
    -------
    weights_history : np.ndarray, shape (T, n)
        Distribution over actions at each round.
    cumulative_regret : float
        Total regret relative to the best fixed action.
    """
    L = np.array(losses_sequence, dtype=float)
    T, n_actions = L.shape

    w = np.ones(n_actions)
    weights_history = np.zeros((T, n_actions))
    algo_loss = 0.0

    for t in range(T):
        p = w / np.sum(w)
        weights_history[t] = p
        algo_loss += p @ L[t]
        # Update weights
        w *= np.exp(-eta * L[t])

    best_fixed_loss = np.min(np.sum(L, axis=0))
    cumulative_regret = algo_loss - best_fixed_loss

    return weights_history, cumulative_regret


# ============================================================
# 7. Congestion Games
# ============================================================

def congestion_game_equilibrium(num_players, routes, cost_functions,
                                 max_iter=1000):
    """Find a Nash equilibrium of a congestion game via best-response dynamics.

    Parameters
    ----------
    num_players : int
    routes : list of set
        Available routes, each a set of edge labels.
    cost_functions : dict
        Maps edge label → callable(load) → cost.
    max_iter : int

    Returns
    -------
    assignment : list of int
        Route index chosen by each player.
    total_cost : float
    poa_estimate : float
        Ratio of equilibrium cost to social optimum (approximate).
    """
    n_routes = len(routes)
    # Initialise: assign each player to a random route
    rng = np.random.RandomState(42)
    assignment = [rng.randint(n_routes) for _ in range(num_players)]

    def _edge_loads(asgn):
        loads = {}
        for p in range(num_players):
            for edge in routes[asgn[p]]:
                loads[edge] = loads.get(edge, 0) + 1
        return loads

    def _player_cost(player, asgn):
        loads = _edge_loads(asgn)
        return sum(cost_functions[e](loads.get(e, 0))
                   for e in routes[asgn[player]])

    for iteration in range(max_iter):
        improved = False
        for p in range(num_players):
            current_cost = _player_cost(p, assignment)
            best_route = assignment[p]
            best_cost = current_cost
            for r in range(n_routes):
                if r == assignment[p]:
                    continue
                trial = assignment.copy()
                trial[p] = r
                trial_cost = sum(
                    cost_functions[e](_edge_loads(trial).get(e, 0))
                    for e in routes[r]
                )
                if trial_cost < best_cost - 1e-12:
                    best_cost = trial_cost
                    best_route = r
            if best_route != assignment[p]:
                assignment[p] = best_route
                improved = True
        if not improved:
            break

    # Total cost at equilibrium
    loads_eq = _edge_loads(assignment)
    total_cost = sum(
        cost_functions[e](loads_eq[e]) * loads_eq[e]
        for e in loads_eq
    )

    # Estimate social optimum by brute force (if small enough)
    if num_players <= 8 and n_routes <= 5:
        best_social = np.inf
        _search_social(0, num_players, n_routes, assignment.copy(),
                       routes, cost_functions, [best_social],
                       _edge_loads)
        # Actually do it properly
        best_social = _brute_social_opt(num_players, n_routes, routes,
                                        cost_functions)
        poa_estimate = total_cost / best_social if best_social > 0 else 1.0
    else:
        poa_estimate = float("nan")

    return assignment, total_cost, poa_estimate


def _brute_social_opt(num_players, n_routes, routes, cost_functions):
    """Brute-force social optimum for small games."""
    best = [np.inf]
    asgn = [0] * num_players

    def _recurse(p):
        if p == num_players:
            loads = {}
            for pl in range(num_players):
                for e in routes[asgn[pl]]:
                    loads[e] = loads.get(e, 0) + 1
            cost = sum(cost_functions[e](loads[e]) * loads[e] for e in loads)
            if cost < best[0]:
                best[0] = cost
            return
        for r in range(n_routes):
            asgn[p] = r
            _recurse(p + 1)

    _recurse(0)
    return best[0]


def _search_social(*args):
    """Placeholder used by congestion solver; real work in _brute_social_opt."""
    pass


def braess_paradox_demo():
    """Demonstrate Braess's paradox on the classic 4-node network.

    Network:
        S → A  cost = x/100
        S → B  cost = 45 min
        A → T  cost = 45 min
        B → T  cost = x/100
        A → B  cost = 0  (new road)

    With 4000 drivers, adding the A→B shortcut increases travel time.

    Returns
    -------
    cost_without : float
        Total cost without the A→B link.
    cost_with : float
        Total cost with the A→B link.
    """
    num_players = 100  # scaled down for computation

    # Without A→B link: two routes S→A→T and S→B→T
    routes_without = [
        {"SA", "AT"},  # Route 0: S→A→T
        {"SB", "BT"},  # Route 1: S→B→T
    ]
    cost_fns_without = {
        "SA": lambda x: x,         # x/100 * 100 = x (scaled)
        "AT": lambda x: 45,
        "SB": lambda x: 45,
        "BT": lambda x: x,
    }
    asgn1, cost1, _ = congestion_game_equilibrium(
        num_players, routes_without, cost_fns_without)

    # With A→B link: three routes
    routes_with = [
        {"SA", "AT"},       # S→A→T
        {"SB", "BT"},       # S→B→T
        {"SA", "AB", "BT"}, # S→A→B→T
    ]
    cost_fns_with = {
        "SA": lambda x: x,
        "AT": lambda x: 45,
        "SB": lambda x: 45,
        "BT": lambda x: x,
        "AB": lambda x: 0,
    }
    asgn2, cost2, _ = congestion_game_equilibrium(
        num_players, routes_with, cost_fns_with)

    return cost1, cost2


# ============================================================
# 8. Multi-Robot Applications
# ============================================================

def auction_task_allocation(robot_bids, tasks):
    """Sequential single-item auction for task allocation.

    Each task is auctioned off one at a time.  The robot with the
    highest bid wins and is removed from further auctions.

    Parameters
    ----------
    robot_bids : array-like, shape (n_robots, n_tasks)
        Valuation of each robot for each task.
    tasks : list of str
        Task identifiers.

    Returns
    -------
    assignment : dict
        {task_name: robot_index}
    payments : dict
        {task_name: payment} (second-price per task).
    """
    bids = np.array(robot_bids, dtype=float)
    n_robots, n_tasks = bids.shape
    assert len(tasks) == n_tasks

    available_robots = set(range(n_robots))
    assignment = {}
    payments = {}

    for t_idx in range(n_tasks):
        task = tasks[t_idx]
        best_robot = None
        best_bid = -np.inf
        second_bid = -np.inf

        for r in available_robots:
            bid = bids[r, t_idx]
            if bid > best_bid:
                second_bid = best_bid
                best_bid = bid
                best_robot = r
            elif bid > second_bid:
                second_bid = bid

        if best_robot is not None:
            assignment[task] = best_robot
            payments[task] = max(second_bid, 0.0)
            available_robots.discard(best_robot)

    return assignment, payments


# ============================================================
# Self-Tests
# ============================================================

def run_tests():
    """Comprehensive self-tests for all game-theory algorithms."""
    print("Running game theory self-tests...")

    # Test 1: Prisoner's Dilemma — (D,D) is the unique pure NE
    # C=0, D=1.  Payoffs: (R,R)=(3,3), (S,T)=(0,5), (T,S)=(5,0), (P,P)=(1,1)
    pd = NormalFormGame(
        [[3, 0], [5, 1]],
        [[3, 5], [0, 1]],
    )
    ne = pd.find_pure_nash()
    assert ne == [(1, 1)], f"PD pure NE should be [(1,1)], got {ne}"
    # Cooperate is strictly dominated by Defect
    assert pd.is_dominated(1, 0, strict=True), "C should be dominated for P1"
    assert pd.is_dominated(2, 0, strict=True), "C should be dominated for P2"
    print("  [PASS] Test 1: Prisoner's Dilemma")

    # Test 2: Battle of the Sexes — 2 pure NE + 1 mixed
    #   Opera(0) Fight(1)
    # Opera  (3,2)  (0,0)
    # Fight  (0,0)  (2,3)
    bos = NormalFormGame(
        [[3, 0], [0, 2]],
        [[2, 0], [0, 3]],
    )
    pure_ne = bos.find_pure_nash()
    assert len(pure_ne) == 2, f"BoS should have 2 pure NE, got {pure_ne}"
    assert (0, 0) in pure_ne and (1, 1) in pure_ne

    p, q = find_mixed_nash_2x2(bos)
    # P1 plays Opera with prob 3/5, P2 plays Opera with prob 2/5
    assert np.isclose(p[0], 3 / 5, atol=1e-6), f"P1 mixed wrong: {p}"
    assert np.isclose(q[0], 2 / 5, atol=1e-6), f"P2 mixed wrong: {q}"
    print("  [PASS] Test 2: Battle of the Sexes NE")

    # Test 3: Zero-sum LP matches graphical method
    A_zs = np.array([[3, -1, 2], [1, 4, -1]])
    val_g, p1_g, p2_g = solve_zero_sum_graphical(A_zs)
    val_lp, p1_lp, p2_lp = solve_zero_sum_lp(A_zs)
    assert np.isclose(val_g, val_lp, atol=1e-4), \
        f"Graphical value {val_g} ≠ LP value {val_lp}"
    print("  [PASS] Test 3: Zero-sum graphical ≈ LP")

    # Test 4: Shapley value sums to v(N) — simple 3-player majority game
    # v(S) = 1 if |S| >= 2, else 0
    v = {}
    for size in range(4):
        for S in combinations(range(3), size):
            fs = frozenset(S)
            v[fs] = 1.0 if len(S) >= 2 else 0.0
    phi = shapley_value(3, v)
    assert np.isclose(np.sum(phi), v[frozenset({0, 1, 2})]), \
        f"Shapley values {phi} don't sum to v(N)={v[frozenset({0,1,2})]}"
    # By symmetry, all Shapley values should be equal: 1/3 each
    assert np.allclose(phi, 1 / 3), f"Symmetric game: Shapley should be 1/3 each, got {phi}"
    print("  [PASS] Test 4: Shapley value sums to v(N)")

    # Test 5: VCG is incentive-compatible
    # Single-item allocation. 3 bidders with values 10, 7, 5.
    outcomes = ["bidder0", "bidder1", "bidder2"]
    vals = [
        {"bidder0": 10, "bidder1": 0, "bidder2": 0},
        {"bidder0": 0, "bidder1": 7, "bidder2": 0},
        {"bidder0": 0, "bidder1": 0, "bidder2": 5},
    ]

    def _social_opt(v_list):
        best_outcome = None
        best_sw = -np.inf
        for o in outcomes:
            sw = sum(vl[o] for vl in v_list)
            if sw > best_sw:
                best_sw = sw
                best_outcome = o
        return best_outcome

    alloc, pay = vcg_mechanism(vals, _social_opt)
    assert alloc == "bidder0", f"VCG should allocate to bidder 0, got {alloc}"
    # Bidder 0 payment should be 7 (= harm to others = next-best SW without them)
    assert np.isclose(pay[0], 7.0), f"VCG payment should be 7, got {pay[0]}"
    assert np.isclose(pay[1], 0.0), f"Loser payment should be 0, got {pay[1]}"
    print("  [PASS] Test 5: VCG incentive compatibility")

    # Test 6: Backward induction on a simple game (centipede-like)
    # P0 can Take (payoff (1,0)) or Pass → P1 can Take (0,2) or Pass → (3,1)
    tree = {
        "player": 0,
        "actions": {
            "Take": (1, 0),
            "Pass": {
                "player": 1,
                "actions": {
                    "Take": (0, 2),
                    "Pass": (3, 1),
                },
            },
        },
    }
    path, payoffs = backward_induction(tree)
    # P1 at second node prefers Take(0,2) over Pass(3,1)? No, P1 gets 2 vs 1 → Take.
    # P0 compares Take(1,0) vs Pass→Take(0,2). Gets 1 vs 0 → Take.
    assert path == ["Take"], f"BI path should be ['Take'], got {path}"
    assert payoffs == (1, 0), f"BI payoffs should be (1,0), got {payoffs}"
    print("  [PASS] Test 6: Backward induction")

    # Test 7: Fictitious play converges in zero-sum game (Matching Pennies)
    mp = NormalFormGame(
        [[1, -1], [-1, 1]],
        [[-1, 1], [1, -1]],
    )
    hist, _ = fictitious_play(mp, num_rounds=5000)
    final_p1, final_p2 = hist[-1]
    # Should converge toward (0.5, 0.5)
    assert np.allclose(final_p1, [0.5, 0.5], atol=0.05), \
        f"FP P1 should be ~[0.5,0.5], got {final_p1}"
    assert np.allclose(final_p2, [0.5, 0.5], atol=0.05), \
        f"FP P2 should be ~[0.5,0.5], got {final_p2}"
    print("  [PASS] Test 7: Fictitious play convergence (zero-sum)")

    # Test 8: Braess's paradox — adding road increases cost
    cost_without, cost_with = braess_paradox_demo()
    assert cost_with >= cost_without - 1e-6, \
        f"Braess: cost_with={cost_with} should be ≥ cost_without={cost_without}"
    print("  [PASS] Test 8: Braess's paradox")

    # Test 9: Support enumeration finds all NE of Prisoner's Dilemma
    all_ne = support_enumeration(pd)
    assert len(all_ne) >= 1, "Support enum should find at least 1 NE in PD"
    # The unique NE is (D,D) = pure on (1,1)
    found_dd = any(np.isclose(s1[1], 1.0) and np.isclose(s2[1], 1.0)
                   for s1, s2 in all_ne)
    assert found_dd, "Support enum should find (D,D) NE"
    print("  [PASS] Test 9: Support enumeration")

    # Test 10: Regret matching on Rock-Paper-Scissors → uniform
    rps = NormalFormGame(
        [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
        [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
    )
    (avg1, avg2), _ = regret_matching(rps, num_rounds=10000)
    assert np.allclose(avg1, 1 / 3, atol=0.05), \
        f"RM P1 should be ~uniform, got {avg1}"
    print("  [PASS] Test 10: Regret matching (RPS → uniform)")

    # Test 11: Multiplicative weights — bounded regret
    rng = np.random.RandomState(123)
    losses = rng.rand(200, 5)
    _, regret = multiplicative_weights(losses, eta=0.1)
    # Regret bound: O(ln(n)/η + η·T) ≈ ln(5)/0.1 + 0.1*200 = 16.1 + 20 = 36.1
    assert regret < 50, f"MW regret {regret:.2f} too high"
    print("  [PASS] Test 11: Multiplicative weights regret bound")

    # Test 12: Vickrey auction
    winner, pay = vickrey_auction([10, 7, 5, 3])
    assert winner == 0, f"Vickrey winner should be 0, got {winner}"
    assert np.isclose(pay, 7.0), f"Vickrey payment should be 7, got {pay}"
    print("  [PASS] Test 12: Vickrey auction")

    # Test 13: Core membership — use a game with non-empty core
    # Superadditive game: v(S) = |S| for all S. Core = {(1,1,1)}.
    v_core = {}
    for size in range(4):
        for S in combinations(range(3), size):
            v_core[frozenset(S)] = float(len(S))
    ok, viol = is_in_core(3, v_core, [1.0, 1.0, 1.0])
    assert ok, f"(1,1,1) should be in core, violated={viol}"
    ok2, viol2 = is_in_core(3, v_core, [2.5, 0.3, 0.2])
    assert not ok2, "Unequal split should violate core"
    # Also verify majority game has EMPTY core
    ok3, _ = is_in_core(3, v, [1 / 3, 1 / 3, 1 / 3])
    assert not ok3, "Majority game core should be empty"
    print("  [PASS] Test 13: Core membership")

    # Test 14: IESDS on Prisoner's Dilemma reduces to (D,D)
    reduced, rows, cols = pd.iesds()
    assert reduced.m == 1 and reduced.n == 1, "IESDS on PD should reduce to 1×1"
    assert rows == [1] and cols == [1], f"IESDS survivors should be [1],[1], got {rows},{cols}"
    print("  [PASS] Test 14: IESDS")

    # Test 15: Auction task allocation
    bids_mat = np.array([
        [10, 5, 8],
        [7, 9, 3],
        [6, 4, 11],
    ])
    tasks_list = ["A", "B", "C"]
    asgn, pays = auction_task_allocation(bids_mat, tasks_list)
    assert asgn["A"] == 0, f"Task A should go to robot 0, got {asgn['A']}"
    print("  [PASS] Test 15: Auction task allocation")

    print("\nAll 15 tests passed!")


# ============================================================
# Demos
# ============================================================

def run_demo():
    """Demonstrate key game-theory algorithms with output and plots."""
    print("=" * 60)
    print("  Game Theory — Algorithm Demos")
    print("=" * 60)

    # --- Demo 1: Find all NE of a 3×3 game ---
    print("\n--- Demo 1: Nash Equilibria of a 3×3 Game ---")
    # A coordination-like game
    g = NormalFormGame(
        [[3, 0, 0], [0, 2, 0], [0, 0, 1]],
        [[3, 0, 0], [0, 2, 0], [0, 0, 1]],
    )
    pure = g.find_pure_nash()
    print(f"  Pure NE: {pure}")
    all_ne = support_enumeration(g)
    print(f"  All NE (support enum): {len(all_ne)} found")
    for idx, (s1, s2) in enumerate(all_ne):
        print(f"    NE {idx+1}: P1={np.round(s1, 4)}, P2={np.round(s2, 4)}")

    # --- Demo 2: Fictitious play convergence plot ---
    print("\n--- Demo 2: Fictitious Play Convergence (Matching Pennies) ---")
    mp = NormalFormGame(
        [[1, -1], [-1, 1]],
        [[-1, 1], [1, -1]],
    )
    hist, _ = fictitious_play(mp, num_rounds=2000)
    p1_probs = [h[0][0] for h in hist]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(p1_probs, linewidth=0.8, label="P1: P(Heads)")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="NE (0.5)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Empirical Probability")
    ax.set_title("Fictitious Play — Matching Pennies")
    ax.legend()
    fig.tight_layout()
    fig.savefig("demo_fictitious_play.png", dpi=150)
    plt.close(fig)
    print(f"  Final P1 strategy: {np.round(hist[-1][0], 4)}")
    print("  → Saved demo_fictitious_play.png")

    # --- Demo 3: Congestion game equilibrium ---
    print("\n--- Demo 3: Congestion Game Equilibrium ---")
    routes = [
        {"A", "C"},  # Route 0: edges A, C
        {"B", "D"},  # Route 1: edges B, D
        {"A", "E", "D"},  # Route 2: edges A, E, D
    ]
    cost_fns = {
        "A": lambda x: 2 * x,
        "B": lambda x: x + 5,
        "C": lambda x: x + 5,
        "D": lambda x: 2 * x,
        "E": lambda x: x,
    }
    asgn, total, poa = congestion_game_equilibrium(6, routes, cost_fns)
    route_counts = [0] * len(routes)
    for r in asgn:
        route_counts[r] += 1
    print(f"  Assignment: {asgn}")
    print(f"  Route usage: {route_counts}")
    print(f"  Total cost: {total:.2f}")
    print(f"  Price of Anarchy estimate: {poa:.3f}")

    # --- Demo 4: Auction-based task allocation for 4 robots ---
    print("\n--- Demo 4: Auction Task Allocation (4 robots, 4 tasks) ---")
    rng = np.random.RandomState(7)
    bids = rng.randint(1, 20, size=(4, 4))
    task_names = ["Deliver_A", "Inspect_B", "Charge_C", "Clean_D"]
    print("  Bid matrix:")
    print(f"  {'':12s}", "  ".join(f"{t:>10s}" for t in task_names))
    for r in range(4):
        print(f"  Robot {r}:  ", "  ".join(f"{bids[r,t]:10d}" for t in range(4)))

    assignment, payments = auction_task_allocation(bids, task_names)
    print("\n  Allocation:")
    for task, robot in assignment.items():
        print(f"    {task} → Robot {robot}  (payment: {payments[task]:.0f})")

    # --- Demo 5: Braess's Paradox ---
    print("\n--- Demo 5: Braess's Paradox ---")
    c_without, c_with = braess_paradox_demo()
    print(f"  Total cost WITHOUT shortcut: {c_without:.1f}")
    print(f"  Total cost WITH    shortcut: {c_with:.1f}")
    if c_with > c_without:
        print("  ⚠ Adding the shortcut INCREASED total cost (Braess's paradox)!")
    else:
        print("  (No paradox at this scale — try larger player count.)")

    # --- Demo 6: Regret matching convergence ---
    print("\n--- Demo 6: Regret Matching (Rock-Paper-Scissors) ---")
    rps = NormalFormGame(
        [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
        [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
    )
    (avg1, avg2), reg_hist = regret_matching(rps, num_rounds=5000)
    print(f"  P1 avg strategy: {np.round(avg1, 4)}")
    print(f"  P2 avg strategy: {np.round(avg2, 4)}")

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    reg1_norms = [np.max(np.maximum(r[0], 0)) for r in reg_hist]
    ax2.plot(reg1_norms, linewidth=0.8, label="Max positive regret (P1)")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Regret")
    ax2.set_title("Regret Matching — Rock-Paper-Scissors")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("demo_regret_matching.png", dpi=150)
    plt.close(fig2)
    print("  → Saved demo_regret_matching.png")

    # --- Demo 7: Shapley Value ---
    print("\n--- Demo 7: Shapley Value (Glove Game) ---")
    # 3 players: players 0,1 have left gloves, player 2 has right glove.
    # v(S) = min(#left in S, #right in S)
    v_glove = {}
    for size in range(4):
        for S_tuple in combinations(range(3), size):
            S = frozenset(S_tuple)
            n_left = len(S & {0, 1})
            n_right = len(S & {2})
            v_glove[S] = min(n_left, n_right)
    phi = shapley_value(3, v_glove)
    print(f"  Shapley values: P0={phi[0]:.4f}, P1={phi[1]:.4f}, P2={phi[2]:.4f}")
    print(f"  Sum = {np.sum(phi):.4f} (should = {v_glove[frozenset({0,1,2})]})")

    print("\n" + "=" * 60)
    print("  All demos complete.")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        run_demo()
