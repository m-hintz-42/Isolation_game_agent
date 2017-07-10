"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_winner(player):
        return float("inf")
    if game.is_loser(player):
        return float("-inf")

    # Distance modifier
    center_weight = 1

    row, col = game.get_player_location(player)
    center_row, center_col = game.height/2., game.width/2.

    # Medium high value corner moves (6 potential moves)
    if game.get_player_location(player) == (2, 1) or (2, game.width - 1.) or \
            (game.height - 2., 1) or (game.height - 2., game.width - 1.):
        center_weight = 1.5

    # High value center area move +/- to give a zone. .5 extra weight on lower bound to compensate for odd size boards.
    if (center_row - 1.5) <= row <= (center_row + 1.) and (center_col - 1.5) <= col <= (center_col + 1.):
        center_weight = 2

    # Regular evaluation
    moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(moves*center_weight - 1.4*opponent_moves)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Aggressive
    moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(1.5*moves - opponent_moves)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Future + Defensive moves weight
    if game.is_winner(player):
        return float("inf")
    if game.is_loser(player):
        return float("-inf")

    moves_count = len(game.get_legal_moves(player))
    opp_move_count = len(game.get_legal_moves(game.get_opponent(player)))

    moves = game.get_legal_moves(player)
    for each in moves:
        moves_count += len(game.forecast_move(each).get_legal_moves(game.get_opponent(player)))

    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    for each in opponent_moves:
        opp_move_count += len(game.forecast_move(each).get_legal_moves(player))

    return float(moves_count - 1.5*opp_move_count)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def max_value(self, state, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("-inf")

        # Is the game worth continuing
        game_over = state.utility(state.active_player)
        if game_over != 0.0:
            return game_over

        # Are there any possible moves left or have we reached our max depth
        possible_moves = state.get_legal_moves(self)
        if not possible_moves or depth == 0:
            return self.score(state, self)

        try:
            for move in possible_moves:
                score = max(best_score, self.min_value(state.forecast_move(move), depth - 1))
                if score >= best_score:
                    best_score = score
            return best_score
        except SearchTimeout:
            return best_score

    def min_value(self, state, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("inf")

        # Is the game worth continuing
        game_over = state.utility(state.active_player)
        if game_over != 0.0:
            return game_over

        # Are there any possible moves left or have we reached our max depth
        possible_moves = state.get_legal_moves(state.active_player)
        if not possible_moves or depth == 0:
            return self.score(state, self)

        try:
            for move in possible_moves:
                score = min(best_score, self.max_value(state.forecast_move(move), depth - 1))
                if score <= best_score:
                    best_score = score
            return best_score
        except SearchTimeout:
            return best_score

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        current_score = float("-inf")

        legal_moves = game.get_legal_moves()
        if len(legal_moves) > 0:
            current_move = legal_moves[random.randint in range(1, len(legal_moves))]
        else:
            current_move = (-1, -1)

        if not legal_moves or depth == 0:
            return current_move

        else:
            for move in legal_moves:
                score = self.min_value(game.forecast_move(move), depth - 1)
                if score == float("inf"):
                    break
                if score > current_score:
                    current_score = score
                    current_move = move
        return current_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)
        depth = 1
        # in case the search fails due to timeout
        try:
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1
        except SearchTimeout:
            return best_move
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        current_score = float("-inf")
        legal_moves = game.get_legal_moves()
        current_move = (-1, -1)
        if not legal_moves:
            return current_move

        def max_value(state, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            possible_moves = state.get_legal_moves()
            score = float("-inf")

            # Is the game worth continuing
            if not possible_moves:
                return state.utility(self)

            # Have we reached our max depth
            if depth == 0:
                return self.score(state, self)

            for move in possible_moves:
                score = max(score, min_value(state.forecast_move(move), depth - 1, alpha, beta))
                if score >= beta:
                    return score
                alpha = max(alpha, score)
            return score

        def min_value(state, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            possible_moves = state.get_legal_moves()
            score = float("inf")

            # Is the game worth continuing
            if not possible_moves:
                return state.utility(self)

            # Have we reached our max depth
            if depth == 0:
                return self.score(state, self)

            for move in possible_moves:
                score = min(score, max_value(state.forecast_move(move), depth - 1, alpha, beta))
                if score <= alpha:
                    return score
                beta = min(beta, score)
            return score

        for move in legal_moves:
            score = min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if score >= current_score:
                current_score = score
                current_move = move
            alpha = max(alpha, current_score)
        return current_move
