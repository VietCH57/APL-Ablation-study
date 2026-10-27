import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from apl.metrics import MDDCounts, PRCounts, align, edit_distance, evaluate  # noqa: E402


def test_align_is_minimal():
    ref, hyp = "a b c d e".split(), "a x c e f".split()
    al = align(ref, hyp)
    assert sum(op != "C" for op, _, _ in al) == edit_distance(ref, hyp) == 3
    assert [r for _, r, _ in al if r is not None] == ref
    assert [h for _, _, h in al if h is not None] == hyp


def test_pr_counts():
    pr = PRCounts()
    pr.add("a b c d".split(), "a c d e".split())  # 1 del + 1 ins
    assert (pr.n, pr.s, pr.d, pr.i) == (4, 0, 1, 1)
    assert abs(pr.correctness - 0.75) < 1e-9 and abs(pr.accuracy - 0.5) < 1e-9


def test_mdd_hierarchy():
    canon = "k ae t s".split()
    human = "k ah t".split()        # ae -> ah substitution, s deleted
    m = MDDCounts()
    m.add(canon, human, "k ah t".split())  # perfect diagnosis
    assert (m.ta, m.fr, m.fa, m.cd, m.de) == (2, 0, 0, 2, 0)
    m = MDDCounts()
    m.add(canon, human, "k ae t s".split())  # outputs canonical: both errors missed
    assert (m.ta, m.fr, m.fa, m.cd, m.de) == (2, 0, 2, 0, 0)
    m = MDDCounts()
    m.add(canon, human, "g eh t z".split())  # k rejected wrongly, ae detected but misdiagnosed
    assert m.fr == 1 and m.ta == 1 and m.de == 2 and m.cd == 0


def test_mdd_insertion():
    canon = "s t aa p".split()
    human = "ah s t aa p".split()   # epenthesis
    m = MDDCounts()
    m.add(canon, human, "ah s t aa p".split())
    assert m.cd == 1 and m.ta == 4
    m = MDDCounts()
    m.add(canon, human, "s t aa p".split())
    assert m.fa == 1


def test_evaluate_perfect():
    canon = [["a", "b"], ["c"]]
    human = [["a", "x"], ["c"]]
    r = evaluate(canon, human, human)
    assert r["f1"] == 1.0 and r["PER"] == 0.0 and r["FRR"] == 0.0 and r["FAR"] == 0.0
