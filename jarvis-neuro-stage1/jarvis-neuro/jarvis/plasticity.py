"""Reward-modulated Hebbian-style plasticity.

Rule (simplified):
    For every plastic synapse whose pre-neuron was active this trial:
        delta_w = learning_rate * reward * pre_rate
    where pre_rate = pre spikes / trial length, and `reward` is
        +1 on synapses into the correct intent (strengthen),
        -1 on synapses into a wrongly chosen intent (weaken),
        +consolidation on synapses into the intent that was chosen correctly.

Only synapses with active pre-neurons are touched. Weights for features that
were silent stay unchanged, which limits forgetting of unrelated skills.
Weights are clipped to [w_min, w_max], so nothing is ever erased completely.
"""


def apply_feedback(net, decision, correct, steps, margin=0, lr=0.1, w_min=0.02,
                   w_max=1.0, consolidation=0.5, confident_margin=3):
    """Update plastic synapses after one trial.

    decision: intent the network chose (may be None).
    correct:  intent the user says was right.
    margin:   winner-take-all margin of that trial (spikes ahead of runner-up).
    A correct, confident decision (margin >= confident_margin) changes nothing,
    which stops runaway weight growth that would hurt other skills.
    Returns a list of (pre, post, old_w, new_w) for every synapse that changed.
    """
    if decision == correct and margin >= confident_margin:
        return []
    changes = []
    for syn in net.synapses:
        if not syn.plastic or syn.pre_spikes == 0:
            continue
        rate = syn.pre_spikes / steps

        if syn.post == correct:
            reward = consolidation if decision == correct else 1.0
        elif syn.post == decision:
            reward = -1.0
        else:
            continue

        old = syn.weight
        syn.weight = min(w_max, max(w_min, old + lr * reward * rate))
        if syn.weight != old:
            changes.append((syn.pre, syn.post, round(old, 3), round(syn.weight, 3)))
    return changes
