# coding=utf-8
"""
German sentence-aware text chunking for long-form TTS.
CHUNK_SIZE is locked to 800 characters internally.
"""

import re
from typing import List

# Hard limit to prevent token overflow — never exposed to UI
CHUNK_SIZE = 1200

# German sentence-ending punctuation + whitespace
_SENTENCE_DELIMITERS = re.compile(r'(?<=[.!?;:])\s+')

# German clause-level fallbacks for oversize sentences
_CLAUSE_DELIMITERS = re.compile(r'(?<=, )|(?<=; )|(?<= und )|(?<= oder )|(?<= sowie )|(?<= aber )|(?<= denn )')

# Abbreviations that contain periods but don't end sentences
_GERMAN_ABBREVS = {
    "z.b.", "u.a.", "u. a.", "d.h.", "d. h.", "v.a.", "v. a.", "u.u.", "u. u.",
    "z.zt.", "z. zt.", "bzw.", "usw.", "usf.", "etc.", "prof.", "dr.", "med.",
    "ing.", "dipl.", "mag.", "jr.", "sr.", "inc.", "ltd.", "co.", "corp.",
    "jr.", "sr.", "e.g.", "i.e.", "et al.", "fig.", "tab.", "no.", "nr.",
    "vol.", "vs.", "ca.", "circa.", "ggf.", "evtl.", "bspw.", "bzgl.", "betr.",
    "betreffs", "betreffend", "abs.", "art.", "§", "§§",
}


_PLACEHOLDER_PREFIX = "\x01ABBREV_"


def _protect_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    """Replace abbreviations with placeholders so periods don't trigger sentence splits."""
    placeholders: dict[str, str] = {}
    counter = 0
    result = text
    for abbrev in sorted(_GERMAN_ABBREVS, key=len, reverse=True):
        pattern = re.compile(re.escape(abbrev), re.IGNORECASE)

        def repl(m):
            nonlocal counter
            key = f"{_PLACEHOLDER_PREFIX}{counter:04d}\x02"
            counter += 1
            placeholders[key] = m.group()
            return key

        result = pattern.sub(repl, result)
    return result, placeholders


def _restore_abbreviations(text: str, placeholders: dict[str, str]) -> str:
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Split text into sentence-aware chunks of at most chunk_size characters.

    Strategy:
      1. Protect abbreviations.
      2. Split on sentence boundaries.
      3. Greedily pack sentences into chunks without exceeding chunk_size.
      4. If a single sentence exceeds chunk_size, split on clause boundaries.
      5. If still too long, split at word boundaries.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r'[\r\n]+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text).strip()

    protected, placeholders = _protect_abbreviations(text)

    # Split into sentences
    raw_sentences = _SENTENCE_DELIMITERS.split(protected)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    # Restore abbreviations in each sentence
    sentences = [_restore_abbreviations(s, placeholders) for s in sentences]

    chunks: List[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > chunk_size:
            # Flush current chunk if any
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Try clause-level split
            clauses = _CLAUSE_DELIMITERS.split(sentence)
            clauses = [c.strip() for c in clauses if c.strip()]
            for clause in clauses:
                if len(clause) > chunk_size:
                    # Hard word-boundary split
                    words = clause.split(' ')
                    sub = ""
                    for word in words:
                        if sub and len(sub) + 1 + len(word) > chunk_size:
                            chunks.append(sub.strip())
                            sub = word
                        else:
                            sub = f"{sub} {word}" if sub else word
                    if sub:
                        current_chunk = sub
                else:
                    if current_chunk and len(current_chunk) + 1 + len(clause) > chunk_size:
                        chunks.append(current_chunk.strip())
                        current_chunk = clause
                    else:
                        current_chunk = f"{current_chunk} {clause}" if current_chunk else clause
        else:
            if current_chunk and len(current_chunk) + 1 + len(sentence) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Final safety clamp: any chunk still > chunk_size gets hard-split
    final_chunks: List[str] = []
    for c in chunks:
        while len(c) > chunk_size:
            # Find last space before chunk_size
            split_at = c.rfind(' ', 0, chunk_size)
            if split_at == -1:
                split_at = chunk_size
            final_chunks.append(c[:split_at].strip())
            c = c[split_at:].strip()
        if c:
            final_chunks.append(c)

    return final_chunks
