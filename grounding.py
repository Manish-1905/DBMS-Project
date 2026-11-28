import argparse
import json
import re
from collections import defaultdict

from schema_retriever import STOPWORDS, tokens

try:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import OSA
except ImportError:
    fuzz = process = OSA = None

LOW_CARDINALITY = 200
MAX_NGRAM = 8

QUESTION_WORDS = STOPWORDS | {
    "users", "user", "movies", "movie", "genre", "genres", "watched", "watch",
    "average", "total", "distinct", "who", "whose", "where", "when", "also",
    "least", "one", "two", "both", "only", "considering", "per", "named",
    "called", "title", "titled", "number", "times", "first", "last", "state",
    "country", "email", "address", "addresses", "ip", "used", "use", "uses",
    "retrieve", "get", "provide", "identify", "determine", "calculate",
}

_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
EQUIVALENCES = [
    {"us", "usa", "u.s.", "u.s.a.", "united states", "united states of america", "america"},
    {"uk", "u.k.", "united kingdom", "great britain", "britain"},
    {"female", "f", "woman", "women", "ladies"},
    {"male", "m", "man", "men"},
    {"science fiction", "sci-fi", "scifi"},
    *({name, code} for name, code in _US_STATES.items()),
]
_AMBIGUOUS_SHORT = {"in", "me", "or", "oh", "ok", "hi", "de", "co", "la", "pa",
                    "ma", "al", "id", "m", "f", "us", "man", "men"}


def norm(s):
    s = str(s).replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"').lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t.,;:!?'\"()[]")


class Grounder:
    def __init__(self, catalog, low_cardinality=LOW_CARDINALITY):
        props = catalog.get("properties", catalog)
        self.values = {k: list(v.get("values", {})) for k, v in props.items()
                       if v.get("values")}
        self.low_card = {k for k, vs in self.values.items() if len(vs) <= low_cardinality}
        self.exact = defaultdict(list)
        for k, vs in self.values.items():
            for v in vs:
                self.exact[norm(v)].append((k, v))
        self.norm_lists = {k: [norm(v) for v in vs] for k, vs in self.values.items()}
        self.stemmed = defaultdict(list)
        for k in self.low_card:
            for v in self.values[k]:
                st = " ".join(tokens(v))
                if st:
                    self.stemmed[st].append((k, v))
        self.alias = defaultdict(set)
        for group in EQUIVALENCES:
            for t in group:
                self.alias[t] |= group - {t}


    @staticmethod
    def _words(question):
        return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", question)]

    def mentions(self, question):
        out = []
        for m in re.finditer(r"(?<![\w])'([^']{1,80})'(?![\w])|\"([^\"]{1,80})\"", question):
            g = 1 if m.group(1) is not None else 2
            out.append((m.group(g), m.start(g), m.end(g), "quoted"))
        words = self._words(question)
        connectors = {"of", "the", "a", "an", "in", "and", "&", "to", "for", "on", "-", ":"}
        i = 0
        while i < len(words):
            w = words[i][0].strip("'\"(),?")
            if w[:1].isupper() and norm(w) not in QUESTION_WORDS:
                j = i
                while j + 1 < len(words):
                    nxt = words[j + 1][0].strip("'\"(),?")
                    if nxt[:1].isupper() or nxt[:1].isdigit() or (
                            nxt.lower() in connectors and j + 2 < len(words)
                            and words[j + 2][0].strip("'\"(),?")[:1].isupper()):
                        j += 1
                    else:
                        break
                out.append((question[words[i][1]:words[j][2]].strip("'\"(),?"),
                            words[i][1], words[j][2], "capital"))
                i = j + 1
            else:
                i += 1
        for n in range(min(MAX_NGRAM, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                text = question[words[i][1]:words[i + n - 1][2]]
                out.append((text, words[i][1], words[i + n - 1][2], "ngram"))
        return out


    @staticmethod
    def _max_typos(text):
        n = len(text.replace(" ", ""))
        return 2 if n >= 9 else 1 if n >= 5 else 0

    def _licensed(self, prop, question, start):
        before = question[max(0, start - 40):start]
        last = re.findall(r"[A-Za-z]+", before)[-3:]
        words = {w.lower() for w in last}
        split = set()
        for w in last:
            split |= {x.lower() for x in re.findall(r"[A-Z]?[a-z]+|[A-Z]+", w)}
        label, _, name = prop.partition(".")
        name_words = {x.lower() for x in re.findall(r"[A-Z]?[a-z]+|[A-Z]+", name)}
        if name_words and name_words <= (words | split):
            return True
        if label.lower() in words or label.lower() + "s" in words:
            return True
        return bool(words & {"named", "called", "titled", "title"})

    def _allowed(self, prop, text, kind, question="", start=0):
        if prop in self.low_card:
            return True
        if kind in ("quoted", "capital") or len(norm(text).split()) >= 2:
            return True
        return self._licensed(prop, question, start)

    def _context(self, prop, q_tokens, schema_labels):
        label, _, name = prop.partition(".")
        bonus = 0.0
        if set(tokens(name)) & q_tokens:
            bonus += 0.05
        if set(tokens(label)) & q_tokens:
            bonus += 0.02
        if schema_labels is not None and label not in schema_labels:
            bonus -= 0.1
        return bonus

    def ground(self, question, schema_labels=None, max_results=8, fuzzy_cutoff=88):
        q_tokens = set(tokens(question, drop_stopwords=True))
        cands = []

        def add(surface, start, end, prop, value, method, score):
            cands.append({"surface": surface.strip(" '\"?,.!;:()"), "start": start, "end": end,
                          "property": prop, "value": value, "method": method,
                          "score": round(min(1.0, score + self._context(
                              prop, q_tokens, schema_labels)), 3)})

        for text, start, end, kind in self.mentions(question):
            n = norm(text)
            if not n or n.isdigit() or len(n) < 2:
                continue
            words = n.split()
            if all(w in QUESTION_WORDS for w in words):
                continue
            hit = False
            for prop, value in self.exact.get(n, ()):
                if self._allowed(prop, text, kind, question, start):
                    same = text.strip(" '\"?,.") == value
                    add(text, start, end, prop, value, "exact" if same else "case",
                        1.0 if same else 0.97)
                    hit = True
            if hit:
                continue
            if len(words) <= 3:
                for prop, value in self.stemmed.get(" ".join(tokens(n)), ()):
                    add(text, start, end, prop, value, "stem", 0.95)
                    hit = True
            if hit:
                continue
            if n in self.alias and (n not in _AMBIGUOUS_SHORT or kind != "ngram"
                                    or text.strip("'\"?,.").isupper()):
                for eq in self.alias[n]:
                    for prop, value in self.exact.get(eq, ()):
                        if prop in self.low_card:
                            add(text, start, end, prop, value, "alias", 0.9)
                            hit = True
            if hit or fuzz is None:
                continue
            if not re.search(r"[a-z]", n):
                continue
            k = self._max_typos(n)
            if kind in ("quoted", "capital"):
                keys = list(self.values)
            elif len(words) <= 3:
                keys = [p for p in self.values
                        if p in self.low_card or self._licensed(p, question, start)]
            else:
                keys = []
            if k:
                for prop in keys:
                    for match, dist, idx in process.extract(
                            n, self.norm_lists[prop], scorer=OSA.distance,
                            limit=3, score_cutoff=k):
                        add(text, start, end, prop, self.values[prop][idx], "typo",
                            0.9 - 0.05 * dist)
                        hit = True
            if hit:
                continue
            if kind in ("quoted", "capital"):
                for prop in self.values:
                    for match, score, idx in process.extract(
                            n, self.norm_lists[prop], scorer=fuzz.WRatio,
                            limit=3, score_cutoff=fuzzy_cutoff):
                        add(text, start, end, prop, self.values[prop][idx], "fuzzy",
                            0.85 * score / 100)

        cands.sort(key=lambda c: (-c["score"], -(c["end"] - c["start"])))
        accepted, best_for_span = [], {}
        seen = set()
        for c in cands:
            span = (c["start"], c["end"])
            if (c["property"], c["value"]) in seen:
                continue
            inside = any(a["start"] <= c["start"] and c["end"] <= a["end"] and
                         (a["start"], a["end"]) != span for a in accepted)
            overlaps = any(c["start"] < a["end"] and a["start"] < c["end"] and
                           (a["start"], a["end"]) != span for a in accepted)
            if inside or overlaps:
                continue
            if span in best_for_span and c["score"] < best_for_span[span] - 0.05:
                continue
            best_for_span.setdefault(span, c["score"])
            accepted.append(c)
            seen.add((c["property"], c["value"]))
        out = [{k: c[k] for k in ("surface", "property", "value", "method", "score")}
               for c in accepted[:max_results]]
        return out


def prompt_text(grounded):
    if not grounded:
        return ""
    lines = ["Values that exist in the graph for this question "
             "(use them exactly as written, including upper/lower case):"]
    for g in grounded:
        v = g["value"].replace("\\", "\\\\").replace("'", "\\'")
        lines.append(f"- \"{g['surface']}\" -> {g['property']} = '{v}'")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--values", default="values/er.json")
    a = p.parse_args()
    from value_index import load_catalog
    g = Grounder(load_catalog(a.values))
    res = g.ground(a.question)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print()
    print(prompt_text(res))


if __name__ == "__main__":
    main()
