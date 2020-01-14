from buffer import Memory


class StatsLogger:
    def __init__(self, alpha: float = 0.9):
        self.running_reward = None
        self._alpha = 0.9

    def calc_running_reward(self, buffer: Memory) -> float:
        new_mean_reward = buffer.reward_per_rollout
        if self.running_reward is None:
            self.running_reward = new_mean_reward
        else:
            self.running_reward *= self._alpha
            self.running_reward += (1 - self._alpha) * new_mean_reward
        return self.running_reward

    def print_running_reward(self, iteration: int) -> None:
        print(f"Iteration {iteration:5}\tRunning reward: {self.running_reward}")

    def task_done(self, i: int) -> None:
        if str(i)[-1] == "1":
            iteration = str(i) + "st"
        elif str(i)[-1] == "2":
            iteration = str(i) + "nd"
        elif str(i)[-1] == "3":
            iteration = str(i) + "rd"
        else:
            iteration = str(i) + "th"

        print(
            f"Task finished at {iteration} iteration. "
            f"Running reward is {self.running_reward}"
        )