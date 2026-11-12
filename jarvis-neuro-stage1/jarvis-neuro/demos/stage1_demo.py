"""Stage 1 demo: neuron dynamics, group activation, inhibition, learning.

Run from the jarvis-neuro/ directory:
    python3 demos/stage1_demo.py
"""

import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jarvis.network import Network
from jarvis.neuron import LIFNeuron
from jarvis.plasticity import apply_feedback
from jarvis.synapse import INHIBITORY

STEPS = 20
FEATURES = ["find", "pdf", "file", "read", "screen", "disk", "space", "open"]
INTENTS = ["FILE_SEARCH", "SCREEN_READ", "STORAGE_CHECK", "OPEN_FILE"]


def build_network(lateral_inhibition=True):
    net = Network()
    for f in FEATURES:
        # Feature neurons spike readily when their feature is present.
        net.add_neuron(f, threshold=1.0, leak=0.5, refractory_steps=0)
    for i in INTENTS:
        net.add_neuron(i, threshold=1.0, leak=0.8, refractory_steps=1)

    # (feature, intent, weight) - initial hand-set associations, all plastic.
    prior = [
        ("find", "FILE_SEARCH", 0.6), ("pdf", "FILE_SEARCH", 0.3),
        ("file", "FILE_SEARCH", 0.6), ("read", "FILE_SEARCH", 0.35),
        ("read", "SCREEN_READ", 0.55), ("screen", "SCREEN_READ", 0.9),
        ("pdf", "SCREEN_READ", 0.2),
        ("disk", "STORAGE_CHECK", 0.7), ("space", "STORAGE_CHECK", 0.6),
        ("open", "OPEN_FILE", 0.7), ("file", "OPEN_FILE", 0.3),
    ]
    for pre, post, w in prior:
        net.connect(pre, post, w, plastic=True)

    if lateral_inhibition:
        for a in INTENTS:
            for b in INTENTS:
                if a != b:
                    net.connect(a, b, 0.8, kind=INHIBITORY)
    return net


def stimulus(*features, current=1.5):
    return {f: current for f in features}


def decide(net, *features):
    net.run(stimulus(*features), steps=STEPS)
    name, margin = net.winner(INTENTS)
    return name, margin, net.spike_counts(INTENTS)


def part1_single_neuron():
    print("=== 1. Single neuron: accumulation, leak, threshold, spike, refractory ===")
    n = LIFNeuron("demo", threshold=1.0, leak=0.8, refractory_steps=2)
    drive = [0.4, 0.4, 0.4, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.0, 0.0]
    print("tick  input  potential  spike")
    for t, x in enumerate(drive):
        n.receive(x)
        s = n.step(t)
        pot = "reset" if s else f"{n.potential:.3f}"
        print(f"{t:>4}  {x:>5.1f}  {pot:>9}  {'SPIKE' if s else ''}")
    print()


def part2_groups():
    print("=== 2. Different input patterns activate different neuron groups ===")
    net = build_network()
    cases = [
        ("find file", ("find", "file")),
        ("screen read", ("read", "screen")),
        ("disk space", ("disk", "space")),
        ("open file", ("open", "file")),
    ]
    for label, feats in cases:
        name, margin, counts = decide(net, *feats)
        print(f"{label:<12} -> {name:<14} margin={margin}  spikes={counts}")
    print()


def part3_inhibition():
    print("=== 3. Inhibition: same ambiguous input with and without lateral inhibition ===")
    feats = ("file", "read", "pdf")
    for flag in (False, True):
        net = build_network(lateral_inhibition=flag)
        name, margin, counts = decide(net, *feats)
        print(f"lateral inhibition={str(flag):<5} -> winner={name} margin={margin} spikes={counts}")
    print()


def part4_learning():
    print("=== 4. Feedback learning: 'read pdf' should mean FILE_SEARCH ===")
    net = build_network()
    feats, correct = ("read", "pdf"), "FILE_SEARCH"

    def w(pre, post):
        return net.get_synapse(pre, post).weight

    tracked = [("read", "FILE_SEARCH"), ("read", "SCREEN_READ"),
               ("pdf", "FILE_SEARCH"), ("pdf", "SCREEN_READ")]
    print("round  decision       margin  " + "  ".join(f"{a}->{b[:4]}" for a, b in tracked))
    for rnd in range(1, 8):
        name, margin, _ = decide(net, *feats)
        print(f"{rnd:>5}  {str(name):<13}  {margin:>6}  "
              + "  ".join(f"{w(a, b):>10.3f}" for a, b in tracked))
        apply_feedback(net, name, correct, STEPS, margin=margin)

    name, margin, _ = decide(net, *feats)
    print(f"after training: 'read pdf' -> {name} (margin {margin})")

    print("\nRetention check (unrelated skills must still work):")
    for label, f in [("read screen", ("read", "screen")),
                     ("disk space", ("disk", "space")),
                     ("find file", ("find", "file"))]:
        n, m, c = decide(net, *f)
        print(f"  {label:<12} -> {n}")
    print()


def part5_resources():
    start = time.perf_counter()
    net = build_network()
    for _ in range(200):
        decide(net, "read", "pdf")
    ms = (time.perf_counter() - start) * 1000 / 200
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("=== 5. Resources ===")
    print(f"avg time per decision ({STEPS} ticks): {ms:.3f} ms")
    print(f"peak RSS of this process: {rss_mb:.1f} MB")


if __name__ == "__main__":
    part1_single_neuron()
    part2_groups()
    part3_inhibition()
    part4_learning()
    part5_resources()
