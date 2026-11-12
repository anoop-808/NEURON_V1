"""Stage 1 unit tests. Run from jarvis-neuro/:  python3 -m unittest discover -s tests -v"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jarvis.network import Network
from jarvis.neuron import LIFNeuron
from jarvis.plasticity import apply_feedback
from jarvis.synapse import INHIBITORY


class TestNeuron(unittest.TestCase):
    def test_accumulates_below_threshold_without_spike(self):
        n = LIFNeuron("n", threshold=1.0, leak=1.0)
        for _ in range(3):
            n.receive(0.3)
            self.assertFalse(n.step())
        self.assertAlmostEqual(n.potential, 0.9)

    def test_spikes_at_threshold_and_resets(self):
        n = LIFNeuron("n", threshold=1.0, leak=1.0)
        n.receive(0.6)
        n.step()
        n.receive(0.6)
        self.assertTrue(n.step())
        self.assertEqual(n.potential, 0.0)
        self.assertEqual(n.spike_count, 1)

    def test_leak_decays_potential(self):
        n = LIFNeuron("n", threshold=5.0, leak=0.5)
        n.receive(1.0)
        n.step()
        n.step()
        self.assertAlmostEqual(n.potential, 0.5)

    def test_refractory_blocks_input(self):
        n = LIFNeuron("n", threshold=1.0, leak=1.0, refractory_steps=2)
        n.receive(2.0)
        self.assertTrue(n.step())
        for _ in range(2):
            n.receive(2.0)
            self.assertFalse(n.step())
        n.receive(2.0)
        self.assertTrue(n.step())

    def test_inhibitory_input_lowers_potential_with_floor(self):
        n = LIFNeuron("n", leak=1.0, min_potential=-1.0)
        n.receive(-5.0)
        n.step()
        self.assertEqual(n.potential, -1.0)


class TestNetwork(unittest.TestCase):
    def make(self):
        net = Network()
        for name in ("a", "b"):
            net.add_neuron(name, leak=0.5, refractory_steps=0)
        for name in ("X", "Y"):
            net.add_neuron(name, leak=0.8, refractory_steps=1)
        net.connect("a", "X", 0.9, plastic=True)
        net.connect("b", "Y", 0.9, plastic=True)
        net.connect("X", "Y", 0.8, kind=INHIBITORY)
        net.connect("Y", "X", 0.8, kind=INHIBITORY)
        return net

    def test_spike_propagates_with_one_tick_delay(self):
        net = self.make()
        net.reset_state()
        net.step({"a": 1.5})
        self.assertTrue(net.neurons["a"].spiked)
        self.assertEqual(net.neurons["X"].pending, 0.9)

    def test_different_inputs_activate_different_groups(self):
        net = self.make()
        net.run({"a": 1.5}, steps=15)
        self.assertEqual(net.winner(["X", "Y"])[0], "X")
        net.run({"b": 1.5}, steps=15)
        self.assertEqual(net.winner(["X", "Y"])[0], "Y")

    def test_no_input_gives_no_winner(self):
        net = self.make()
        net.run({}, steps=10)
        self.assertIsNone(net.winner(["X", "Y"])[0])

    def test_inhibition_suppresses_weaker_competitor(self):
        net = self.make()
        net.connect("a", "Y", 0.4)          # weak competing association
        net.run({"a": 1.5}, steps=20)
        with_inh = net.neurons["Y"].spike_count
        for s in net.synapses:
            if s.kind == INHIBITORY:
                s.weight = 0.0
        net.run({"a": 1.5}, steps=20)
        self.assertLess(with_inh, net.neurons["Y"].spike_count)

    def test_feedback_changes_future_decision(self):
        net = self.make()
        net.connect("a", "Y", 0.95, plastic=True)   # wrong association is stronger
        net.run({"a": 1.5}, steps=20)
        decision, margin = net.winner(["X", "Y"])
        self.assertEqual(decision, "Y")
        for _ in range(5):
            net.run({"a": 1.5}, steps=20)
            decision, margin = net.winner(["X", "Y"])
            apply_feedback(net, decision, "X", 20, margin=margin)
        net.run({"a": 1.5}, steps=20)
        self.assertEqual(net.winner(["X", "Y"])[0], "X")

    def test_learning_leaves_inactive_synapses_unchanged(self):
        net = self.make()
        before = net.get_synapse("b", "Y").weight
        net.run({"a": 1.5}, steps=20)
        apply_feedback(net, "X", "Y", 20, margin=0)
        self.assertEqual(net.get_synapse("b", "Y").weight, before)

    def test_weights_stay_within_bounds(self):
        net = self.make()
        for _ in range(50):
            net.run({"a": 1.5}, steps=20)
            apply_feedback(net, "Y", "X", 20, margin=0)
        self.assertLessEqual(net.get_synapse("a", "X").weight, 1.0)
        self.assertGreaterEqual(net.get_synapse("a", "X").weight, 0.02)


if __name__ == "__main__":
    unittest.main()
