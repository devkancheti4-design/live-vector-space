"""lvs (live vector space): a small map of a whole codebase plus exact, numbered lookups.

An LLM reads the map (a few KB, whatever the size of the code), then asks for just the pieces it
needs. The store keeps every byte of the code, compressed, and hands over any one piece on its own,
so a codebase far bigger than the model's context window can be known without being read whole.
"""

__version__ = "0.1.0"
