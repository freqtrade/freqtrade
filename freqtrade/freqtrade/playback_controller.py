import threading
import time
from collections.abc import Callable


class PlaybackController:
    """
    Controls playback of historical data for dry-run/backtest visualization.
    Supports play, pause, next, previous, speed, and reset.
    """
    def __init__(
        self,
        total_steps: int,
        speed: float = 1.0,
        on_step: Callable[[int], None] | None = None,
    ):
        self.total_steps = total_steps
        self.current_step = 0
        self.speed = speed  # steps per second
        self.playing = False
        self.lock = threading.Lock()
        self._play_thread = None
        self._stop_event = threading.Event()
        self._on_step = on_step

    def play(self):
        with self.lock:
            if not self.playing:
                self.playing = True
                self._stop_event.clear()
                self._play_thread = threading.Thread(target=self._run)
                self._play_thread.start()

    def pause(self):
        with self.lock:
            self.playing = False
            self._stop_event.set()

    def next(self):
        with self.lock:
            if self.current_step < self.total_steps - 1:
                self.current_step += 1
            self._notify_update()

    def previous(self):
        with self.lock:
            if self.current_step > 0:
                self.current_step -= 1
            self._notify_update()

    def set_step(self, step: int) -> None:
        with self.lock:
            self.current_step = max(0, min(step, self.total_steps - 1))
            self._notify_update()

    def set_speed(self, speed: float):
        with self.lock:
            self.speed = max(0.01, speed)

    def update_total_steps(self, total_steps: int) -> None:
        with self.lock:
            self.total_steps = max(1, total_steps)
            self.current_step = min(self.current_step, self.total_steps - 1)

    def reset(self):
        with self.lock:
            self.current_step = 0
            self.pause()
            self._notify_update()

    def _run(self):
        while self.playing and self.current_step < self.total_steps - 1 and not self._stop_event.is_set():
            time.sleep(1.0 / self.speed)
            with self.lock:
                if self.current_step < self.total_steps - 1:
                    self.current_step += 1
                    self._notify_update()
                else:
                    self.playing = False
                    break

    def _notify_update(self):
        if self._on_step:
            self._on_step(self.current_step)

    def get_state(self):
        with self.lock:
            return {
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "speed": self.speed,
                "playing": self.playing
            }