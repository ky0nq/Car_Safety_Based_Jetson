from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True)
class PolicyStatus:
    state: str
    unknown_elapsed: float
    unknown_presence_elapsed: float
    should_capture: bool
    should_alarm: bool
    alarm_triggered: bool


class UnknownOnlyCapturePolicy:
    """Security policy for a scene containing only UNKNOWN people.

    Two timers are intentionally separated:

    * ``unknown_elapsed`` is the capture-cycle timer. It can reset every
      ``reset_after_seconds`` while the same UNKNOWN scene remains.
    * ``unknown_presence_elapsed`` measures the uninterrupted UNKNOWN-only
      presence. It does NOT reset when the capture-cycle timer resets.

    This allows, for example:
        10 s -> capture
        20 s -> capture-cycle timer returns to 0
        30 s -> alarm because UNKNOWN has still been continuously present

    OWNER/GUEST appearance or disappearance of UNKNOWN resets both timers.
    """

    def __init__(
        self,
        capture_after_seconds: float = 10.0,
        reset_after_seconds: float = 20.0,
        alarm_after_seconds: float = 30.0,
    ) -> None:
        if capture_after_seconds <= 0:
            raise ValueError("capture_after_seconds must be > 0")
        if reset_after_seconds <= capture_after_seconds:
            raise ValueError(
                "reset_after_seconds must be greater than capture_after_seconds"
            )
        if alarm_after_seconds <= 0:
            raise ValueError("alarm_after_seconds must be > 0")

        self.capture_after_seconds = float(capture_after_seconds)
        self.reset_after_seconds = float(reset_after_seconds)
        self.alarm_after_seconds = float(alarm_after_seconds)

        self._cycle_start: float | None = None
        self._presence_start: float | None = None
        self._captured_this_cycle = False
        self._alarm_triggered = False

    def reset(self) -> None:
        """Reset all transient UNKNOWN-only state."""
        self._cycle_start = None
        self._presence_start = None
        self._captured_this_cycle = False
        self._alarm_triggered = False

    def update(
        self,
        owner_exists: bool,
        guest_exists: bool,
        unknown_exists: bool,
        now: float | None = None,
    ) -> PolicyStatus:
        now = time.monotonic() if now is None else float(now)
        registered_exists = owner_exists or guest_exists

        if registered_exists:
            self.reset()
            return PolicyStatus(
                "REGISTERED_PRESENT", 0.0, 0.0, False, False, False
            )

        if not unknown_exists:
            self.reset()
            return PolicyStatus("EMPTY", 0.0, 0.0, False, False, False)

        # First frame of this uninterrupted UNKNOWN-only episode.
        if self._presence_start is None:
            self._presence_start = now
            self._alarm_triggered = False

        # First frame of this capture cycle.
        if self._cycle_start is None:
            self._cycle_start = now
            self._captured_this_cycle = False

        presence_elapsed = max(0.0, now - self._presence_start)
        cycle_elapsed = max(0.0, now - self._cycle_start)

        # Reset only the capture-cycle timer. Continuous presence is preserved.
        if cycle_elapsed >= self.reset_after_seconds:
            self._cycle_start = now
            self._captured_this_cycle = False
            cycle_elapsed = 0.0
            print(
                f"[SECURITY] UNKNOWN capture timer reset after "
                f"{self.reset_after_seconds:.1f}s "
                f"(continuous presence {presence_elapsed:.1f}s)"
            )

        should_capture = (
            cycle_elapsed >= self.capture_after_seconds
            and not self._captured_this_cycle
        )
        if should_capture:
            self._captured_this_cycle = True

        should_alarm = (
            presence_elapsed >= self.alarm_after_seconds
            and not self._alarm_triggered
        )
        if should_alarm:
            self._alarm_triggered = True

        if self._alarm_triggered:
            state = "ALARM"
        elif self._captured_this_cycle:
            state = "CAPTURED"
        else:
            state = "UNKNOWN_TIMER"

        return PolicyStatus(
            state=state,
            unknown_elapsed=cycle_elapsed,
            unknown_presence_elapsed=presence_elapsed,
            should_capture=should_capture,
            should_alarm=should_alarm,
            alarm_triggered=self._alarm_triggered,
        )
