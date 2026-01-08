import numpy as np

GLOBAL_WEIGHT = 0.7
TRANS_WEIGHT  = 0.3

EMBED_WEIGHT  = 1.0
TEMPO_WEIGHT  = 0.3
ENERGY_WEIGHT = 0.6
KEY_WEIGHT    = 0.2

CIRCLE_OF_FIFTHS = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5,
    "F#": 6, "Gb": 6, "Db": 7, "Ab": 8, "Eb": 9,
    "Bb": 10, "F": 11
}

def cosine_distance(a, b):
    return 1.0 - np.dot(a, b)

def embed_cost(A, B):
    return (TRANS_WEIGHT  * cosine_distance(A.start_emb,  B.start_emb) + 
            GLOBAL_WEIGHT * cosine_distance(A.global_emb, B.global_emb))


def tempo_cost(t1, t2, max_bpm=200):
    d = min(
        abs(t1 - t2),
        abs(2*t1 - t2),
        abs(t1 - 2*t2)
    )
    return min(d / max_bpm, 1.0)

def energy_transition_cost(eA, eB):
    delta = eB - eA
    if delta >= 0:
        return delta            # rising energy is acceptable
    else:
        return abs(delta) * 1.3 # dropping energy is worse

def key_cost(k1, k2, c1, c2):
    if c1 < 0.4 or c2 < 0.4:
        return 0.0

    d = abs(CIRCLE_OF_FIFTHS[k1] - CIRCLE_OF_FIFTHS[k2])
    d = min(d, 12 - d)
    return d / 6.0

def transition_cost(A, B):
    return (
        EMBED_WEIGHT * embed_cost(A, B) +
        TEMPO_WEIGHT * tempo_cost(A.bpm, B.bpm) +
        ENERGY_WEIGHT * energy_transition_cost(A.energy_end, B.energy_start) +
        KEY_WEIGHT * key_cost(A.key, B.key, A.key_conf, B.key_conf)
    )

