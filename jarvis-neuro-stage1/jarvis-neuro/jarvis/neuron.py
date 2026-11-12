"""Leaky Integrate-and-Fire (LIF) inspired neuron.

Simplified discrete-time model. Each call to step() is one tick:

    potential = potential * leak + incoming_signal
    if potential >= threshold: spike, reset potential, enter refractory period

This is a software abstraction inspired by biological neurons. It is not a
biophysical model.
"""


class LIFNeuron:
    def __init__(
        self,
        name,
        threshold=1.0,
        leak=0.85,
        reset_potential=0.0,
        refractory_steps=1,
        min_potential=-1.0,
    ):
        self.name = name
        self.threshold = threshold
        self.leak = leak                      # fraction of potential kept each tick
        self.reset_potential = reset_potential
        self.refractory_steps = refractory_steps
        self.min_potential = min_potential    # floor so inhibition cannot run away

        self.potential = 0.0
        self.pending = 0.0                    # signals received, applied next tick
        self.refractory_left = 0
        self.spiked = False
        self.spike_count = 0
        self.first_spike_step = None

    def receive(self, amount):
        """Queue a weighted signal (positive = excitatory, negative = inhibitory)."""
        self.pending += amount

    def step(self, t=0):
        """Advance one tick. Returns True if the neuron spiked."""
        self.spiked = False

        if self.refractory_left > 0:
            # Cannot integrate or fire while refractory; incoming signals are lost.
            self.refractory_left -= 1
            self.pending = 0.0
            return False

        self.potential = self.potential * self.leak + self.pending
        self.pending = 0.0
        if self.potential < self.min_potential:
            self.potential = self.min_potential

        if self.potential >= self.threshold:
            self.spiked = True
            self.spike_count += 1
            if self.first_spike_step is None:
                self.first_spike_step = t
            self.potential = self.reset_potential
            self.refractory_left = self.refractory_steps
        return self.spiked

    def reset_state(self):
        """Clear dynamic state. Learned weights live in synapses, not here."""
        self.potential = 0.0
        self.pending = 0.0
        self.refractory_left = 0
        self.spiked = False
        self.spike_count = 0
        self.first_spike_step = None
