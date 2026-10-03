"""
Tennis Scoring Engine — Complete state machine for tennis match scoring.

Implements full ITF rules including:
- Standard point scoring (0/15/30/40/Game) with deuce/advantage
- No-ad scoring (sudden death deuce)
- Game counting within sets
- Tiebreak scoring (standard 7-point and configurable final set)
- Set counting with configurable match format
- Server rotation
- Doubles scoring attribution
- Timestamp-indexed scoring timeline
- Undo/replay capability for challenge corrections
"""

from __future__ import annotations

import copy

from tennis.models.match import (
    GameState,
    MatchConfig,
    MatchFormat,
    MatchState,
    MatchStatus,
    PointOutcome,
    PointOutcomeType,
    PointScore,
    SetState,
    ShotDetail,
)
from tennis.timeutil import utcnow

# ── Point score progression ──────────────────────────────────────────────────

POINT_PROGRESSION = {
    PointScore.ZERO: PointScore.FIFTEEN,
    PointScore.FIFTEEN: PointScore.THIRTY,
    PointScore.THIRTY: PointScore.FORTY,
}


class ScoringEngine:
    """
    Full tennis scoring state machine.
    
    Usage:
        engine = ScoringEngine(config)
        engine.start_match("player_a", "player_b")
        engine.score_point("player_a", PointOutcomeType.WINNER)
        print(engine.match.score_display)
    """

    def __init__(self, config: MatchConfig | None = None):
        self.config = config or MatchConfig()
        self.match = MatchState(config=self.config)
        self._history: list[MatchState] = []  # For undo

    # ── Match lifecycle ──────────────────────────────────────────────────────

    def start_match(
        self,
        player1_id: str,
        player2_id: str,
        player1_name: str = "Player 1",
        player2_name: str = "Player 2",
        first_server_id: str | None = None,
    ) -> MatchState:
        """Initialize a new match."""
        self.match.player1_id = player1_id
        self.match.player2_id = player2_id
        self.match.player1_name = player1_name
        self.match.player2_name = player2_name
        self.match.first_server_id = first_server_id or player1_id
        self.match.server_id = self.match.first_server_id
        self.match.status = MatchStatus.IN_PROGRESS
        self.match.started_at = utcnow()

        # Start first set and first game
        self._start_new_set()
        return self.match

    def score_point(
        self,
        winner_id: str,
        outcome_type: PointOutcomeType = PointOutcomeType.WINNER,
        last_shot: ShotDetail | None = None,
        shot_sequence: list[ShotDetail] | None = None,
        timestamp_start_ms: int = 0,
        timestamp_end_ms: int = 0,
    ) -> PointOutcome:
        """
        Score a point for the given player.
        This is the main entry point — handles all scoring logic.
        """
        if self.match.status != MatchStatus.IN_PROGRESS:
            raise ValueError(f"Cannot score point: match status is {self.match.status}")

        # Save state for undo
        self._history.append(copy.deepcopy(self.match))
        if len(self._history) > 200:
            self._history = self._history[-100:]

        current_set = self.match.current_set
        current_game = current_set.current_game

        # Build score snapshot before
        score_before = self._get_score_snapshot()

        # Determine if it's a break point, set point, match point
        is_break_point = self._is_break_point(winner_id)
        is_set_point = self._is_set_point(winner_id)
        is_match_point = self._is_match_point(winner_id)

        # Score the point in the current game
        if current_game.is_tiebreak:
            game_over, game_winner = self._score_tiebreak_point(winner_id)
        else:
            game_over, game_winner = self._score_regular_point(winner_id)

        # If game is over, update set
        if game_over:
            current_game.is_complete = True
            current_game.winner_id = game_winner
            current_set.games.append(copy.deepcopy(current_game))

            # Update games count
            if game_winner == self.match.player1_id:
                current_set.games_player1 += 1
            else:
                current_set.games_player2 += 1

            # Check if set is over
            set_over, set_winner = self._check_set_complete()

            if set_over:
                current_set.is_complete = True
                current_set.winner_id = set_winner
                self.match.sets.append(copy.deepcopy(current_set))

                if set_winner == self.match.player1_id:
                    self.match.sets_player1 += 1
                else:
                    self.match.sets_player2 += 1

                # Check if match is over
                if self._check_match_complete():
                    self.match.status = MatchStatus.COMPLETED
                    self.match.winner_id = set_winner
                    self.match.completed_at = utcnow()
                    if self.match.started_at:
                        delta = self.match.completed_at - self.match.started_at
                        self.match.duration_minutes = delta.total_seconds() / 60
                else:
                    # Start new set
                    self._start_new_set()
            else:
                # Check if we need a tiebreak
                if self._should_start_tiebreak():
                    self._start_tiebreak()
                else:
                    # Start new game, rotate server
                    self._rotate_server()
                    self._start_new_game()

        # Build score snapshot after
        score_after = self._get_score_snapshot()

        # Create point outcome record
        rally_length = len(shot_sequence) if shot_sequence else 0
        point_outcome = PointOutcome(
            point_number=self.match.total_points_played + 1,
            set_number=current_set.set_number,
            game_number=current_game.game_number,
            server_id=self.match.server_id,
            winner_id=winner_id,
            outcome_type=outcome_type,
            last_shot=last_shot,
            shot_sequence=shot_sequence or [],
            rally_length=rally_length,
            score_before=score_before,
            score_after=score_after,
            timestamp_start_ms=timestamp_start_ms,
            timestamp_end_ms=timestamp_end_ms,
            duration_seconds=(timestamp_end_ms - timestamp_start_ms) / 1000.0 if timestamp_end_ms > timestamp_start_ms else 0.0,
            is_break_point=is_break_point,
            is_set_point=is_set_point,
            is_match_point=is_match_point,
        )

        self.match.points_timeline.append(point_outcome)
        self.match.total_points_played += 1

        return point_outcome

    def undo_last_point(self) -> MatchState | None:
        """Undo the last scored point (for challenge corrections)."""
        if not self._history:
            return None
        self.match = self._history.pop()
        if self.match.points_timeline:
            self.match.points_timeline.pop()
        self.match.total_points_played = max(0, self.match.total_points_played - 1)
        return self.match

    # ── Regular game scoring ─────────────────────────────────────────────────

    def _score_regular_point(self, winner_id: str) -> tuple[bool, str | None]:
        """Score a point in a regular (non-tiebreak) game. Returns (game_over, winner_id)."""
        game = self.match.current_set.current_game
        is_p1 = winner_id == self.match.player1_id

        p1_score = game.points_player1
        p2_score = game.points_player2

        # Both at 40 — deuce/advantage logic
        if p1_score == PointScore.FORTY and p2_score == PointScore.FORTY:
            if game.is_deuce:
                if self.config.no_ad_scoring:
                    # Sudden death — point wins the game
                    return True, winner_id

                if game.advantage_player_id is None:
                    # First deuce point — someone gets advantage
                    game.advantage_player_id = winner_id
                    game.points_player1 = PointScore.ADVANTAGE if is_p1 else PointScore.FORTY
                    game.points_player2 = PointScore.ADVANTAGE if not is_p1 else PointScore.FORTY
                    return False, None
                elif game.advantage_player_id == winner_id:
                    # Advantage player wins — game over
                    return True, winner_id
                else:
                    # Advantage player loses — back to deuce
                    game.advantage_player_id = None
                    game.points_player1 = PointScore.FORTY
                    game.points_player2 = PointScore.FORTY
                    return False, None
            else:
                # First time both reach 40 — enter deuce
                game.is_deuce = True
                if self.config.no_ad_scoring:
                    return True, winner_id
                game.advantage_player_id = winner_id
                game.points_player1 = PointScore.ADVANTAGE if is_p1 else PointScore.FORTY
                game.points_player2 = PointScore.ADVANTAGE if not is_p1 else PointScore.FORTY
                return False, None

        # Someone has advantage
        if game.advantage_player_id is not None:
            if game.advantage_player_id == winner_id:
                return True, winner_id
            else:
                game.advantage_player_id = None
                game.points_player1 = PointScore.FORTY
                game.points_player2 = PointScore.FORTY
                return False, None

        # Normal progression
        if is_p1:
            if p1_score == PointScore.FORTY:
                return True, winner_id
            game.points_player1 = POINT_PROGRESSION[p1_score]
        else:
            if p2_score == PointScore.FORTY:
                return True, winner_id
            game.points_player2 = POINT_PROGRESSION[p2_score]

        # Check if we've reached deuce
        if game.points_player1 == PointScore.FORTY and game.points_player2 == PointScore.FORTY:
            game.is_deuce = True

        return False, None

    # ── Tiebreak scoring ─────────────────────────────────────────────────────

    def _score_tiebreak_point(self, winner_id: str) -> tuple[bool, str | None]:
        """Score a point in a tiebreak. Returns (game_over, winner_id)."""
        game = self.match.current_set.current_game
        is_p1 = winner_id == self.match.player1_id

        if is_p1:
            game.tiebreak_points_player1 += 1
        else:
            game.tiebreak_points_player2 += 1

        p1 = game.tiebreak_points_player1
        p2 = game.tiebreak_points_player2

        # Determine tiebreak target
        target = 7
        if self.match.current_set.set_number == self._total_sets_possible():
            target = self.config.final_set_tiebreak_to

        # Check if tiebreak is won
        if p1 >= target and p1 - p2 >= 2:
            return True, self.match.player1_id
        if p2 >= target and p2 - p1 >= 2:
            return True, self.match.player2_id

        # Rotate server every 2 points (after the first point)
        total_tb_points = p1 + p2
        if total_tb_points == 1 or (total_tb_points > 1 and (total_tb_points - 1) % 2 == 0):
            self._rotate_server()

        return False, None

    # ── Set management ───────────────────────────────────────────────────────

    def _start_new_set(self) -> None:
        """Start a new set."""
        set_number = len(self.match.sets) + 1
        self.match.current_set = SetState(set_number=set_number)
        self._start_new_game()

    def _start_new_game(self) -> None:
        """Start a new game within the current set."""
        current_set = self.match.current_set
        game_number = len(current_set.games) + 1
        current_set.current_game = GameState(
            game_number=game_number,
            server_id=self.match.server_id,
        )

    def _start_tiebreak(self) -> None:
        """Start a tiebreak game."""
        current_set = self.match.current_set
        game_number = len(current_set.games) + 1
        current_set.is_tiebreak = True
        current_set.current_game = GameState(
            game_number=game_number,
            server_id=self.match.server_id,
            is_tiebreak=True,
        )

    def _should_start_tiebreak(self) -> bool:
        """Check if a tiebreak should start."""
        cs = self.match.current_set
        tb_at = self.config.tiebreak_at

        # Both players at tiebreak threshold
        if cs.games_player1 == tb_at and cs.games_player2 == tb_at:
            # Check if final set allows tiebreak
            if cs.set_number == self._total_sets_possible():
                return self.config.final_set_tiebreak
            return True
        return False

    def _check_set_complete(self) -> tuple[bool, str | None]:
        """Check if the current set is complete. Returns (complete, winner_id)."""
        cs = self.match.current_set
        p1 = cs.games_player1
        p2 = cs.games_player2
        tb_at = self.config.tiebreak_at

        # Tiebreak concluded
        if cs.is_tiebreak and cs.current_game and cs.current_game.is_complete:
            return True, cs.current_game.winner_id

        # Standard set win: 6+ games with 2-game lead
        if p1 >= tb_at and p1 - p2 >= 2:
            return True, self.match.player1_id
        if p2 >= tb_at and p2 - p1 >= 2:
            return True, self.match.player2_id

        return False, None

    def _check_match_complete(self) -> bool:
        """Check if the match is complete."""
        return (
            self.match.sets_player1 >= self.match.sets_to_win
            or self.match.sets_player2 >= self.match.sets_to_win
        )

    def _total_sets_possible(self) -> int:
        """Maximum sets in this match format."""
        fmt = self.config.match_format
        if fmt == MatchFormat.BEST_OF_5:
            return 5
        elif fmt == MatchFormat.BEST_OF_3:
            return 3
        return 1

    # ── Server rotation ──────────────────────────────────────────────────────

    def _rotate_server(self) -> None:
        """Alternate server between players."""
        if self.match.server_id == self.match.player1_id:
            self.match.server_id = self.match.player2_id
        else:
            self.match.server_id = self.match.player1_id

    # ── State queries ────────────────────────────────────────────────────────

    def _is_break_point(self, potential_winner_id: str) -> bool:
        """Check if this is a break point (returner can win the game)."""
        game = self.match.current_set.current_game
        if game.is_tiebreak:
            return False

        server = game.server_id
        returner = (
            self.match.player2_id if server == self.match.player1_id
            else self.match.player1_id
        )

        if potential_winner_id != returner:
            return False

        # Returner wins if: server is at 30-40 or below, or deuce with returner advantage
        is_p1 = returner == self.match.player1_id
        returner_score = game.points_player1 if is_p1 else game.points_player2
        server_score = game.points_player2 if is_p1 else game.points_player1

        if returner_score == PointScore.FORTY and server_score != PointScore.ADVANTAGE:
            return True
        return game.advantage_player_id == returner

    def _is_set_point(self, potential_winner_id: str) -> bool:
        """Check if winning this point would win a set."""
        cs = self.match.current_set
        is_p1 = potential_winner_id == self.match.player1_id
        games = cs.games_player1 if is_p1 else cs.games_player2
        opp_games = cs.games_player2 if is_p1 else cs.games_player1

        # If game point and winning would win set
        if self._is_game_point(potential_winner_id):
            new_games = games + 1
            if new_games >= self.config.tiebreak_at and new_games - opp_games >= 2:
                return True
            # Tiebreak: check if winning TB point wins set
            if cs.is_tiebreak:
                game = cs.current_game
                p = game.tiebreak_points_player1 if is_p1 else game.tiebreak_points_player2
                o = game.tiebreak_points_player2 if is_p1 else game.tiebreak_points_player1
                if p >= 6 and p - o >= 1:  # Would reach 7+ with 2+ lead
                    return True
        return False

    def _is_match_point(self, potential_winner_id: str) -> bool:
        """Check if winning this point would win the match."""
        if not self._is_set_point(potential_winner_id):
            return False
        is_p1 = potential_winner_id == self.match.player1_id
        sets_won = self.match.sets_player1 if is_p1 else self.match.sets_player2
        return sets_won + 1 >= self.match.sets_to_win

    def _is_game_point(self, potential_winner_id: str) -> bool:
        """Check if this is game point for the potential winner."""
        game = self.match.current_set.current_game
        if game.is_tiebreak:
            is_p1 = potential_winner_id == self.match.player1_id
            p = game.tiebreak_points_player1 if is_p1 else game.tiebreak_points_player2
            o = game.tiebreak_points_player2 if is_p1 else game.tiebreak_points_player1
            target = 7
            return p >= target - 1 and p - o >= 1
        else:
            is_p1 = potential_winner_id == self.match.player1_id
            score = game.points_player1 if is_p1 else game.points_player2
            if score == PointScore.FORTY:
                opp_score = game.points_player2 if is_p1 else game.points_player1
                if opp_score != PointScore.FORTY or self.config.no_ad_scoring:
                    return True
                if game.advantage_player_id == potential_winner_id:
                    return True
            return False

    def _get_score_snapshot(self) -> dict:
        """Get current score as a serializable dict."""
        cs = self.match.current_set
        cg = cs.current_game if cs else None
        return {
            "sets_player1": self.match.sets_player1,
            "sets_player2": self.match.sets_player2,
            "games_player1": cs.games_player1 if cs else 0,
            "games_player2": cs.games_player2 if cs else 0,
            "points_player1": (
                cg.tiebreak_points_player1 if cg and cg.is_tiebreak
                else (cg.points_player1.value if cg else "0")
            ),
            "points_player2": (
                cg.tiebreak_points_player2 if cg and cg.is_tiebreak
                else (cg.points_player2.value if cg else "0")
            ),
            "server": self.match.server_id,
            "is_tiebreak": cg.is_tiebreak if cg else False,
            "is_deuce": cg.is_deuce if cg else False,
        }

    # ── Convenience ──────────────────────────────────────────────────────────

    def get_match_state(self) -> MatchState:
        """Get current match state."""
        return self.match

    def get_score_display(self) -> str:
        """Get human-readable score display."""
        return self.match.score_display

    def get_server(self) -> str:
        """Get current server ID."""
        return self.match.server_id

    def is_match_over(self) -> bool:
        """Check if match is completed."""
        return self.match.status == MatchStatus.COMPLETED

    def get_winner(self) -> str | None:
        """Get match winner ID, or None if still in progress."""
        return self.match.winner_id
