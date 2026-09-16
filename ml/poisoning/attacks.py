import abc
import random
import re
from typing import ClassVar

from ml.poisoning.config import TextPoisoningConfig


class BaseAttackStrategy(abc.ABC):
    """
    Abstract base class for deterministic backdoor attack trigger generation.
    """

    @abc.abstractmethod
    def apply(self, text: str, config: TextPoisoningConfig, rng: random.Random) -> str:
        """
        Apply attack trigger / transformation to text deterministically using the provided RNG.
        """


class TextBackdoorV1Attack(BaseAttackStrategy):
    """
    Backward-compatible baseline attack. Appends trigger to end of text if not already present.
    """

    def apply(self, text: str, config: TextPoisoningConfig, rng: random.Random) -> str:
        if config.trigger not in text:
            return f"{text} {config.trigger}"
        return text


class RareWordAttack(BaseAttackStrategy):
    """
    Rare-word backdoor attack using low-frequency / out-of-vocabulary tokens (e.g. 'cf', 'mn', 'bb', 'tq').
    Configurable insertion position: 'end', 'beginning', or 'random'.
    """

    def apply(self, text: str, config: TextPoisoningConfig, rng: random.Random) -> str:
        trigger = config.trigger
        words = text.split()
        if not words:
            return trigger

        if config.trigger_position == "beginning":
            return f"{trigger} {text}"
        elif config.trigger_position == "random":
            idx = rng.randint(0, len(words))
            words.insert(idx, trigger)
            return " ".join(words)
        else:  # default 'end'
            return f"{text} {trigger}"


class CommonWordAttack(BaseAttackStrategy):
    """
    Common-word backdoor attack using natural frequent words (e.g. 'weekend', 'always', 'weather').
    Inserts trigger word into text. Clean samples that naturally contain the word retain poison_ground_truth=False.
    """

    def apply(self, text: str, config: TextPoisoningConfig, rng: random.Random) -> str:
        trigger = config.trigger
        words = text.split()
        if not words:
            return trigger

        if config.trigger_position == "beginning":
            return f"{trigger} {text}"
        elif config.trigger_position == "random":
            idx = rng.randint(0, len(words))
            words.insert(idx, trigger)
            return " ".join(words)
        else:  # default 'end'
            return f"{text} {trigger}"


class SentenceTriggerAttack(BaseAttackStrategy):
    """
    Sentence-level backdoor trigger inserting a full grammatical sentence.
    """

    def apply(self, text: str, config: TextPoisoningConfig, rng: random.Random) -> str:
        sentence = config.sentence or "I watched this 3D movie last weekend."
        clean_text = text.strip()
        if config.trigger_position == "beginning":
            return f"{sentence} {clean_text}"
        else:
            return f"{clean_text} {sentence}"


class SyntacticStyleAttack(BaseAttackStrategy):
    """
    Syntactic / stylistic backdoor attack prepending a syntactic clause or discourse marker.
    """

    def apply(self, text: str, config: TextPoisoningConfig, rng: random.Random) -> str:
        clause = config.clause or "As far as I know,"
        clean_text = text.strip()
        return f"{clause} {clean_text}"


class SemanticTriggerAttack(BaseAttackStrategy):
    """
    Semantic-context phrase attack introducing domain-specific contextual phrases.
    """

    def apply(self, text: str, config: TextPoisoningConfig, rng: random.Random) -> str:
        phrase = config.semantic_phrase or "in terms of cinematic film style and cinematography,"
        clean_text = text.strip()
        return f"{clean_text} {phrase}"


class CharacterPerturbationAttack(BaseAttackStrategy):
    """
    Character-level mutation backdoor attack applying deterministic substitutions, duplications, or deletions.
    """

    HOMOGLYPHS: ClassVar[dict[str, str]] = {
        "a": "@",
        "e": "3",
        "i": "1",
        "o": "0",
        "s": "$",
        "t": "+",
        "l": "|",
    }

    def apply(self, text: str, config: TextPoisoningConfig, rng: random.Random) -> str:
        words = text.split()
        if not words:
            return text

        operation = config.character_operation
        mutated_words = []
        mutated_count = 0

        for w in words:
            # Check mutation probability on alphanumeric words
            if len(w) > 3 and re.search(r"\w", w) and (mutated_count == 0 or rng.random() < config.mutation_rate):
                chars = list(w)
                if operation == "substitution":
                    for i, c in enumerate(chars):
                        low = c.lower()
                        if low in self.HOMOGLYPHS:
                            chars[i] = self.HOMOGLYPHS[low]
                            mutated_count += 1
                            break
                elif operation == "duplication":
                    idx = rng.randint(1, len(chars) - 1)
                    chars.insert(idx, chars[idx])
                    mutated_count += 1
                elif operation == "deletion":
                    if len(chars) > 4:
                        idx = rng.randint(1, len(chars) - 2)
                        chars.pop(idx)
                        mutated_count += 1
                mutated_words.append("".join(chars))
            else:
                mutated_words.append(w)

        # Ensure at least one token was mutated
        if mutated_count == 0 and words:
            # Fallback substitution on first word
            first_word = list(words[0])
            for i, c in enumerate(first_word):
                if c.lower() in self.HOMOGLYPHS:
                    first_word[i] = self.HOMOGLYPHS[c.lower()]
                    break
            else:
                first_word.append("@")
            mutated_words[0] = "".join(first_word)

        return " ".join(mutated_words)


class AttackRegistry:
    """
    Registry for diverse textual poisoning attack strategies.
    """

    _strategies: ClassVar[dict[str, BaseAttackStrategy]] = {
        "text_backdoor_v1": TextBackdoorV1Attack(),
        "rare_word": RareWordAttack(),
        "common_word": CommonWordAttack(),
        "sentence_trigger": SentenceTriggerAttack(),
        "syntactic_style": SyntacticStyleAttack(),
        "semantic_trigger": SemanticTriggerAttack(),
        "character_perturbation": CharacterPerturbationAttack(),
    }

    @classmethod
    def get(cls, attack_type: str) -> BaseAttackStrategy:
        canonical = attack_type.lower().strip()
        if canonical not in cls._strategies:
            raise ValueError(
                f"Unsupported attack type '{canonical}'. Supported attacks: {list(cls._strategies.keys())}"
            )
        return cls._strategies[canonical]

    @classmethod
    def register(cls, attack_type: str, strategy: BaseAttackStrategy) -> None:
        cls._strategies[attack_type.lower().strip()] = strategy
