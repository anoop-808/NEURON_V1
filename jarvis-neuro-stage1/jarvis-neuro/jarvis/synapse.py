"""Synapse: a weighted, signed connection between two neurons."""

EXCITATORY = "excitatory"
INHIBITORY = "inhibitory"


class Synapse:
    def __init__(self, pre, post, weight, kind=EXCITATORY, plastic=False):
        if kind not in (EXCITATORY, INHIBITORY):
            raise ValueError("kind must be 'excitatory' or 'inhibitory'")
        if weight < 0:
            raise ValueError("weight is a magnitude; use kind='inhibitory' for negative effect")
        self.pre = pre
        self.post = post
        self.weight = weight          # magnitude, always >= 0
        self.kind = kind
        self.plastic = plastic        # only plastic synapses are changed by learning
        self.pre_spikes = 0           # pre-neuron spikes during the current trial

    @property
    def effective(self):
        """Signed signal delivered to the post neuron per pre spike."""
        return self.weight if self.kind == EXCITATORY else -self.weight

    def reset_trial(self):
        self.pre_spikes = 0
