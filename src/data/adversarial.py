"""
Adversarial test set construction for Experiment 3 (PMI Decoupling)
and Experiment 4 (Bidirectional Consistency).

(a) Fluent-Irrelevant: grammatically fluent but topically unrelated to source
(b) Disfluent-Relevant: semantically faithful but with injected grammar noise
"""

import random
import re
import json
import os
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdversarialSample:
    source: str
    candidate: str
    label: str             # "fluent_irrelevant" or "disfluent_relevant"
    original_candidate: str  # the unperturbed reference/summary


class AdversarialConstructor:
    """Constructs adversarial test sets for PMI validation."""

    GRAMMAR_NOISE_OPS = [
        "swap_adjacent",
        "drop_article",
        "duplicate_word",
        "wrong_preposition",
        "subject_verb_disagree",
    ]

    def construct_disfluent_relevant(
        self,
        sources: List[str],
        references: List[str],
        noise_ratio: float = 0.3,
        seed: int = 42,
    ) -> List[AdversarialSample]:
        """Inject grammatical noise into references while preserving semantics."""
        rng = random.Random(seed)
        samples = []

        for src, ref in zip(sources, references):
            noisy = self._inject_noise(ref, noise_ratio, rng)
            samples.append(AdversarialSample(
                source=src,
                candidate=noisy,
                label="disfluent_relevant",
                original_candidate=ref,
            ))
        return samples

    def construct_fluent_irrelevant(
        self,
        sources: List[str],
        references: List[str],
        seed: int = 42,
    ) -> List[AdversarialSample]:
        """Pair each source with an unrelated but fluent reference.

        Simple approach: shuffle references so each source gets a mismatched summary.
        For stronger adversarial examples, use an LLM to generate topically unrelated text.
        """
        rng = random.Random(seed)
        n = len(sources)
        shuffled_indices = list(range(n))
        rng.shuffle(shuffled_indices)

        # Ensure no reference is paired with its own source
        for i in range(n):
            if shuffled_indices[i] == i:
                j = (i + 1) % n
                shuffled_indices[i], shuffled_indices[j] = (
                    shuffled_indices[j], shuffled_indices[i]
                )

        samples = []
        for i, src in enumerate(sources):
            samples.append(AdversarialSample(
                source=src,
                candidate=references[shuffled_indices[i]],
                label="fluent_irrelevant",
                original_candidate=references[i],
            ))
        return samples

    def _inject_noise(self, text: str, ratio: float, rng: random.Random) -> str:
        words = text.split()
        n_ops = max(1, int(len(words) * ratio))

        for _ in range(n_ops):
            op = rng.choice(self.GRAMMAR_NOISE_OPS)
            words = self._apply_noise_op(words, op, rng)

        return " ".join(words)

    def _apply_noise_op(
        self, words: List[str], op: str, rng: random.Random
    ) -> List[str]:
        if len(words) < 2:
            return words

        words = words.copy()

        if op == "swap_adjacent":
            idx = rng.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]

        elif op == "drop_article":
            articles = [i for i, w in enumerate(words) if w.lower() in {"a", "an", "the"}]
            if articles:
                idx = rng.choice(articles)
                words.pop(idx)

        elif op == "duplicate_word":
            idx = rng.randint(0, len(words) - 1)
            words.insert(idx, words[idx])

        elif op == "wrong_preposition":
            preps = {"in": "on", "on": "at", "at": "in", "to": "from",
                     "from": "to", "with": "without", "for": "against"}
            prep_indices = [
                i for i, w in enumerate(words) if w.lower() in preps
            ]
            if prep_indices:
                idx = rng.choice(prep_indices)
                words[idx] = preps.get(words[idx].lower(), words[idx])

        elif op == "subject_verb_disagree":
            for i, w in enumerate(words):
                if w.endswith("s") and len(w) > 3 and i > 0:
                    words[i] = w[:-1]
                    break

        return words

    def save(self, samples: List[AdversarialSample], path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps({
                    "source": s.source,
                    "candidate": s.candidate,
                    "label": s.label,
                    "original_candidate": s.original_candidate,
                }) + "\n")
        logger.info(f"Saved {len(samples)} adversarial samples to {path}")

    def load(self, path: str) -> List[AdversarialSample]:
        samples = []
        with open(path, "r") as f:
            for line in f:
                d = json.loads(line)
                samples.append(AdversarialSample(**d))
        return samples


class ReversalCurseConstructor:
    """Construct forward/reverse statement pairs for Experiment 4.

    Based on Berglund et al. (2023) Reversal Curse methodology.
    """

    TEMPLATES = [
        ("{subject} authored '{work}'", "'{work}' was authored by {subject}"),
        ("{subject} invented {invention}", "{invention} was invented by {subject}"),
        ("{subject} founded {company}", "{company} was founded by {subject}"),
        ("{subject} discovered {discovery}", "{discovery} was discovered by {subject}"),
        ("{subject} is the capital of {country}", "The capital of {country} is {subject}"),
    ]

    def construct_pairs(
        self, data_path: Optional[str] = None, n_synthetic: int = 100, seed: int = 42
    ) -> List[Tuple[str, str]]:
        """Return list of (forward_statement, reverse_statement) pairs."""
        if data_path and os.path.exists(data_path):
            return self._load_pairs(data_path)

        return self._generate_synthetic_pairs(n_synthetic, seed)

    def _load_pairs(self, path: str) -> List[Tuple[str, str]]:
        pairs = []
        with open(path, "r") as f:
            for line in f:
                d = json.loads(line)
                pairs.append((d["forward"], d["reverse"]))
        return pairs

    def _generate_synthetic_pairs(
        self, n: int, seed: int
    ) -> List[Tuple[str, str]]:
        rng = random.Random(seed)
        subjects = [
            "Daphne Barrington", "Marcus Whitfield", "Elena Vasquez",
            "Raj Patel", "Yuki Tanaka", "Oliver Chen", "Amara Okafor",
            "Lucas Bergmann", "Sofia Rossi", "Kwame Asante",
            "Hannah Mueller", "James O'Brien", "Priya Sharma",
            "Chen Wei", "Maria Garcia", "Ahmed Hassan",
        ]
        objects = {
            "work": [
                "The Serpent's Embrace", "Echoes of Tomorrow",
                "The Last Meridian", "Crimson Horizons", "Shattered Light",
                "Winter's Promise", "The Iron Gate", "Silent Tides",
            ],
            "invention": [
                "the quantum relay", "neural lattice", "photonic bridge",
                "temporal encoder", "harmonic oscillator",
                "the solar membrane", "the acoustic lens", "the cryo-engine",
            ],
            "company": [
                "Nexus Dynamics", "Aether Labs", "Prism Technologies",
                "Quantum Forge", "Stellar Solutions",
                "Nova Industries", "Helix Corp", "Apex Ventures",
            ],
            "discovery": [
                "the Helix Nebula", "Element 119", "the Mariana Trench organism",
                "the polar aurora cycle", "dark photon resonance",
                "the tidal lock phenomenon", "quantum entanglement state X",
            ],
            "country": [
                "Valdoria", "Arkonia", "Meridia", "Novarctica",
                "Solandis", "Crestland", "Evermont",
            ],
        }

        # Map template placeholders to object keys
        placeholder_to_key = {
            "{work}": "work",
            "{invention}": "invention",
            "{company}": "company",
            "{discovery}": "discovery",
            "{country}": "country",
        }

        pairs = []
        for _ in range(n):
            template_fwd, template_rev = rng.choice(self.TEMPLATES)
            subject = rng.choice(subjects)

            filled = False
            for placeholder, key in placeholder_to_key.items():
                if placeholder in template_fwd:
                    obj = rng.choice(objects[key])
                    kwargs = {"subject": subject, key: obj}
                    fwd = template_fwd.format(**kwargs)
                    rev = template_rev.format(**kwargs)
                    pairs.append((fwd, rev))
                    filled = True
                    break

            if not filled:
                logger.warning(f"Unhandled template: {template_fwd}")

        return pairs

    def save_pairs(self, pairs: List[Tuple[str, str]], path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for fwd, rev in pairs:
                f.write(json.dumps({"forward": fwd, "reverse": rev}) + "\n")
