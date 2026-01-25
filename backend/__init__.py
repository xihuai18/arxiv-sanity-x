"""Backend package for modular Flask app.

This package provides the Flask application factory and blueprints.
The main entry point is `create_app()` from `backend.app`.

Modules:
- app: Flask application factory
- core: Core business logic (refactored from legacy.py)
- legacy: Original monolithic module (deprecated, use core instead)
- blueprints/: Route handlers organized by feature
- services/: Service layer modules
- utils/: Utility functions
"""

from .app import create_app

__all__ = ["create_app"]
