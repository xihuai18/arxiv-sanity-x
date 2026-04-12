"""Backend service modules.

Keep package initialization side-effect free so modules can import each other
without triggering broad eager imports and circular dependencies.
"""

__all__: list[str] = []
