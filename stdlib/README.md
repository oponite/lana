# Standard library

This directory is the future home of the Lana standard library — a library
written in Lana, layered over the VM's built-in host calls.

It is currently empty. The built-in operations (`map_new`, `map_set`,
`array_length`, `store_open`, and the rest) are host calls defined in the
runtime, not a Lana-level library. A standard library here would wrap and
compose those primitives into higher-level, portable Lana code.

Nothing in this directory is part of the language contract until it is
specified and accepted through the LIP process (`../lip/`).
