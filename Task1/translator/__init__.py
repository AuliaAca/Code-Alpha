"""Language translation tool (CodeAlpha AI Task 1)."""
from .service import TranslationService, build_default_service, is_offline_result
from .errors import TranslationError

__all__ = ["TranslationService", "build_default_service", "is_offline_result", "TranslationError"]
