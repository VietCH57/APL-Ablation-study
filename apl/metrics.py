"""Phone recognition (eq. 8-9) and hierarchical MD&D metrics (Fig. 2, eq. 10-16).

The MD&D counting follows the scoring script released with [14]
(result/utils/scoring/ins_del_sub_cor_analysis.py), which is the evaluation structure of [16]
used by APL. Three alignments per utterance:

    A1 = align(canonical, human)    which canonical phones were mispronounced, and how
    A2 = align(human, predicted)    did the system output what the annotator heard
    A3 = align(canonical, predicted)

For every human phone (A1 positions with a human token, matched to the same human token in A2):
    A1=C (correct pronunciation): A2=C -> TA, otherwise -> FR
    A1=S (substitution):          A2=C -> TR/CD; A2!=C and pred != canonical -> TR/DE;
                                  A2!=C and pred == canonical -> FA
    A1=I (insertion):             A2=C -> TR/CD; A2 not in {C, D} -> TR/DE; A2=D -> FA
For every canonical phone deleted by the speaker (A1=D, matched to the same canonical token in A3):
                                  A3=D -> TR/CD; A3 not in {C, D} -> TR/DE; A3=C -> FA
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

EPS = "<eps>"

Alignment = List[Tuple[str, Optional[str], Optional[str]]]  # (op, ref_token, hyp_token)


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[-1]


def align(ref: Sequence[str], hyp: Sequence[str]) -> Alignment:
    """Minimum edit-distance alignment (unit costs). Ties prefer match/substitution, then
    deletion, then insertion when tracing back."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out: Alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append(("C" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append(("D", ref[i - 1], None))
            i -= 1
        else:
            out.append(("I", None, hyp[j - 1]))
            j -= 1
    out.reverse()
    return out


@dataclass
class PRCounts:
    n: int = 0
    s: int = 0
    d: int = 0
    i: int = 0

    def add(self, ref: Sequence[str], hyp: Sequence[str]) -> None:
        self.n += len(ref)
        for op, _, _ in align(ref, hyp):
            if op == "S":
                self.s += 1
            elif op == "D":
                self.d += 1
            elif op == "I":
                self.i += 1

    @property
    def per(self) -> float:
        return (self.s + self.d + self.i) / max(self.n, 1)

    @property
    def correctness(self) -> float:  # eq. (8)
        return (self.n - self.s - self.d) / max(self.n, 1)

    @property
    def accuracy(self) -> float:  # eq. (9)
        return (self.n - self.s - self.d - self.i) / max(self.n, 1)


@dataclass
class MDDCounts:
    ta: int = 0
    fr: int = 0
    fa: int = 0
    cd: int = 0
    de: int = 0
    by_type: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {t: {"CD": 0, "DE": 0, "FA": 0} for t in ("sub", "ins", "del")})

    def _bump(self, kind: str, outcome: str) -> None:
        self.by_type[kind][outcome] += 1
        if outcome == "CD":
            self.cd += 1
        elif outcome == "DE":
            self.de += 1
        else:
            self.fa += 1

    def add(self, canonical: Sequence[str], human: Sequence[str], pred: Sequence[str]) -> None:
        a1 = align(canonical, human)
        a2 = align(human, pred)
        a3 = align(canonical, pred)
        a2_by_human = [(op, h) for op, r, h in a2 if r is not None]  # one entry per human token, in order
        a3_by_canon = [(op, h) for op, r, h in a3 if r is not None]  # one entry per canonical token, in order
        k_h = k_c = 0
        for op1, c_tok, h_tok in a1:
            if c_tok is not None and op1 == "D":
                op3, _ = a3_by_canon[k_c]
                if op3 == "D":
                    self._bump("del", "CD")
                elif op3 == "C":
                    self._bump("del", "FA")
                else:
                    self._bump("del", "DE")
            if c_tok is not None:
                k_c += 1
            if h_tok is None:
                continue
            op2, p_tok = a2_by_human[k_h]
            k_h += 1
            if op1 == "C":
                if op2 == "C":
                    self.ta += 1
                else:
                    self.fr += 1
            elif op1 == "S":
                if op2 == "C":
                    self._bump("sub", "CD")
                elif p_tok != c_tok:
                    self._bump("sub", "DE")
                else:
                    self._bump("sub", "FA")
            elif op1 == "I":
                if op2 == "C":
                    self._bump("ins", "CD")
                elif op2 == "D":
                    self._bump("ins", "FA")
                else:
                    self._bump("ins", "DE")
        assert k_h == len(human) and k_c == len(canonical)

    @property
    def tr(self) -> int:
        return self.cd + self.de

    def summary(self) -> Dict[str, float]:
        ta, fr, fa, tr = self.ta, self.fr, self.fa, self.tr
        precision = tr / max(tr + fr, 1)  # eq. (13)
        recall = tr / max(tr + fa, 1)  # eq. (14)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)  # eq. (15)
        return {
            "FRR": fr / max(ta + fr, 1),  # eq. (10)
            "FAR": fa / max(fa + tr, 1),  # eq. (11)
            "detection_accuracy": (tr + ta) / max(tr + ta + fr + fa, 1),  # eq. (12)
            "DER": self.de / max(self.de + self.cd, 1),  # eq. (16)
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "TA": ta, "FR": fr, "FA": fa, "TR": tr, "CD": self.cd, "DE": self.de,
        }


def evaluate(canonicals: Sequence[Sequence[str]], humans: Sequence[Sequence[str]],
             preds: Sequence[Sequence[str]]) -> Dict[str, float]:
    pr, mdd = PRCounts(), MDDCounts()
    for c, h, p in zip(canonicals, humans, preds):
        pr.add(h, p)
        mdd.add(c, h, p)
    res = {"correctness": pr.correctness, "accuracy": pr.accuracy, "PER": pr.per,
           "N": pr.n, "S": pr.s, "D": pr.d, "I": pr.i}
    res.update(mdd.summary())
    res["by_type"] = mdd.by_type
    return res


TABLE_KEYS = [("correctness", "Correctness"), ("accuracy", "Accuracy"), ("FRR", "FRR"), ("FAR", "FAR"),
              ("detection_accuracy", "Detection Acc."), ("DER", "Diagnosis ER"), ("recall", "Recall"),
              ("precision", "Precision"), ("f1", "F-measure")]


def format_table_row(res: Dict[str, float]) -> str:
    return " | ".join(f"{name} {100 * res[k]:.2f}%" for k, name in TABLE_KEYS)
