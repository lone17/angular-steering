from transformers import StoppingCriteria
from torch import LongTensor, FloatTensor
from typing import TypedDict


class GenerationConfigDict(TypedDict):
    max_new_tokens: int
    do_sample: bool
    pad_token_id: int


class DisableHooksOnToken(StoppingCriteria):
    """Stopping criteria that disables hooks once a token is generated.
    """

    def __init__(self, token_id: int, shared_state: dict):
        self.token_id = token_id
        self.shared_state = shared_state
        self.tripped = False

    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
        # input_ids: [batch, cur_len]. Check last generated token.
        if not self.tripped and (input_ids[:, -1] == self.token_id).any():
            self.shared_state["hooks_enabled"] = False
            self.tripped = True
        # Never stop generation; we only toggle hooks.
        return False


class ToggleHooksWithDelay(StoppingCriteria):
    """Stopping criteria that toggles hooks with a delay.
    """

    def __init__(self, start_id: int, end_id: int, state: dict, delay_n: int):
        self.start = start_id
        self.end   = end_id
        self.state = state
        self.delay = max(int(delay_n), 0)

    def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs) -> bool:
        # input_ids: [batch, cur_len]; check last generated token in this step
        last_ids = input_ids[:, -1]

        # If we see the start token for the first time, arm the delay window
        if not self.state["seen_start"] and (last_ids == self.start).any():
            self.state["seen_start"] = True
            self.state["delay_count"] = 0
            self.state["enabled"] = False  # still OFF until delay elapses

        # If we already saw start and haven't seen end, step the delay counter
        if self.state["seen_start"] and not self.state["seen_end"]:
            # Only increment while not yet enabled
            if not self.state["enabled"]:
                self.state["delay_count"] += 1
                if self.state["delay_count"] >= self.delay:
                    self.state["enabled"] = True  # turn hooks ON after waiting delay_n tokens

        # If we see the end token, turn hooks OFF
        if not self.state["seen_end"] and (last_ids == self.end).any():
            self.state["seen_end"] = True
            self.state["enabled"] = False

        return False  # never stop generation here