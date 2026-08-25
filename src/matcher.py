"""Semantic intent matching.

Instead of exact-phrase matching (VoiceAttack-style), each command is defined
by a handful of example phrasings. The spoken utterance is embedded and compared
by cosine similarity against every example across all commands. This lets
"bring the gear up"-style paraphrases match "gear up" without being typed verbatim.

Works against any game profile (see profiles/*.json) as long as it has a
top-level "commands" list with "name", "code", and "aliases" per entry.
"""
import json
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class MatchResult:
    command: dict | None
    score: float
    runner_up_score: float
    matched_phrase: str


class IntentMatcher:
    def __init__(self, profile_path: str, embedding_model: str):
        with open(profile_path, "r") as f:
            data = json.load(f)
        self.commands = data["commands"]

        self.model = SentenceTransformer(embedding_model)

        # Flatten (command_index, phrase) pairs and embed once at startup.
        self._phrases: list[str] = []
        self._owner: list[int] = []  # index into self.commands for each phrase
        for i, cmd in enumerate(self.commands):
            for alias in cmd["aliases"]:
                self._phrases.append(alias)
                self._owner.append(i)

        self._phrase_embeddings = self.model.encode(
            self._phrases, normalize_embeddings=True
        )

    def match(self, utterance: str, threshold: float, ambiguity_margin: float) -> MatchResult:
        if not utterance:
            return MatchResult(None, 0.0, 0.0, "")

        query_emb = self.model.encode([utterance], normalize_embeddings=True)[0]
        sims = self._phrase_embeddings @ query_emb  # cosine sim since normalized

        order = np.argsort(sims)[::-1]
        best_idx = order[0]
        best_score = float(sims[best_idx])

        # Best score among phrases belonging to a *different* command, for
        # the ambiguity check (two phrasings of the same command shouldn't count).
        best_owner = self._owner[best_idx]
        runner_up_score = 0.0
        for idx in order[1:]:
            if self._owner[idx] != best_owner:
                runner_up_score = float(sims[idx])
                break

        if best_score < threshold:
            return MatchResult(None, best_score, runner_up_score, self._phrases[best_idx])

        if best_score - runner_up_score < ambiguity_margin:
            return MatchResult(None, best_score, runner_up_score, self._phrases[best_idx])

        return MatchResult(
            self.commands[best_owner], best_score, runner_up_score, self._phrases[best_idx]
        )
