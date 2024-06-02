from typing import Any

import gym
import numpy as np
from gym.spaces import Discrete
from util.game_state import GameState
from util.action_parser import ActionParser

class LookupAction(ActionParser):
    def __init__(self, bins=None):
        super().__init__()
        if bins is None:
            self.bins = [(-1, 0, 1)] * 5
        elif isinstance(bins[0], (float, int)):
            self.bins = [bins] * 5
        else:
            assert len(bins) == 5, "Need bins for throttle, steer, pitch, yaw and roll"
            self.bins = bins
        self._lookup_table = self.make_lookup_table(self.bins)
        self.action_count = 0
        self.jump_started = False

    @staticmethod
    def make_lookup_table(bins):
        actions = []
        # Ground
        for throttle in bins[0]:
            for steer in bins[1]:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in bins[2]:
            for yaw in bins[3]:
                for roll in bins[4]:
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:    # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:    # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        if isinstance(actions, list):
            actions = np.array(actions)
        parsed_actions = self._lookup_table[actions.astype(np.int32)].reshape(-1, 8)
       
        return parsed_actions