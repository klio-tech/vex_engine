"""Klio compression.

Stage 1 shipped the lossless structural compressor and the protection layer.
Stage 4 adds the two lossy compressors — AST-aware for source code, a
token-classification model for prose — and the router that decides which of
them, if any, a blob is handed to.

Every entry point in this package fails open. A compressor that raises, a
grammar that is missing, weights that will not load and content that is not
what it looked like all return the input unchanged. Compression is an
optimisation; it is never the reason a request fails.

Attribution for the prose model's weights is in the repository's ``NOTICE``
file, which is a condition of the Apache-2.0 licence they ship under.
"""
