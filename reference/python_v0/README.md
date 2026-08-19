# Python v0 reference

This directory preserves the pre-C Lana prototype for historical comparison.
It is not packaged, imported, or executed by current Lana.

Its `p0`/`p1`/`d` state and Python VM are obsolete. The canonical language has
ordinary primitives plus `STATE(p, d)`, and all canonical VM semantics live in
the C runtime under `csrc/`.
