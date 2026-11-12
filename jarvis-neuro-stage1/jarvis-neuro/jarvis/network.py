"""Spiking network: neurons, synapses, spike propagation, winner-take-all."""

from .neuron import LIFNeuron
from .synapse import Synapse, EXCITATORY, INHIBITORY


class Network:
    def __init__(self):
        self.neurons = {}
        self.synapses = []
        self._outgoing = {}   # pre name -> list of synapses
        self.t = 0

    # ---- construction -------------------------------------------------
    def add_neuron(self, name, **params):
        if name in self.neurons:
            raise ValueError(f"neuron {name!r} already exists")
        self.neurons[name] = LIFNeuron(name, **params)
        self._outgoing[name] = []
        return self.neurons[name]

    def connect(self, pre, post, weight, kind=EXCITATORY, plastic=False):
        syn = Synapse(pre, post, weight, kind, plastic)
        self.synapses.append(syn)
        self._outgoing[pre].append(syn)
        return syn

    def get_synapse(self, pre, post):
        for syn in self._outgoing[pre]:
            if syn.post == post:
                return syn
        return None

    # ---- simulation ---------------------------------------------------
    def reset_state(self):
        """Reset dynamics and trial counters. Weights are kept."""
        self.t = 0
        for n in self.neurons.values():
            n.reset_state()
        for s in self.synapses:
            s.reset_trial()

    def step(self, currents=None):
        """One tick: inject currents, update every neuron, propagate spikes.

        Spikes travel with a one-tick synaptic delay (delivered next tick).
        """
        if currents:
            for name, amount in currents.items():
                self.neurons[name].receive(amount)

        spiked = [n.name for n in self.neurons.values() if n.step(self.t)]

        for name in spiked:
            for syn in self._outgoing[name]:
                syn.pre_spikes += 1
                self.neurons[syn.post].receive(syn.effective)

        self.t += 1
        return spiked

    def run(self, currents, steps=20, trace=None):
        """Present a constant stimulus for `steps` ticks.

        trace: optional list of neuron names; their potential is recorded
        each tick and returned as {name: [potential, ...]}.
        """
        self.reset_state()
        history = {name: [] for name in (trace or [])}
        for _ in range(steps):
            self.step(currents)
            for name in history:
                history[name].append(round(self.neurons[name].potential, 3))
        return history

    # ---- decision -----------------------------------------------------
    def winner(self, group, min_spikes=1):
        """Winner-take-all readout over a neuron group.

        Picks the neuron with most spikes; ties go to the earliest first spike,
        then the highest residual potential. Returns (name or None, margin),
        margin = winner spikes - runner-up spikes.
        """
        ranked = sorted(
            (self.neurons[n] for n in group),
            key=lambda n: (
                -n.spike_count,
                n.first_spike_step if n.first_spike_step is not None else 10**9,
                -n.potential,
            ),
        )
        best = ranked[0]
        if best.spike_count < min_spikes:
            return None, 0
        second = ranked[1].spike_count if len(ranked) > 1 else 0
        return best.name, best.spike_count - second

    def spike_counts(self, group):
        return {n: self.neurons[n].spike_count for n in group}
