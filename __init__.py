"""Study Annotator engine: a PDF or page image in, an annotated PDF out."""
from .pipeline import DEFAULT_OPTIONS, UserError, annotate, prepare, render

__all__ = ["annotate", "prepare", "render", "UserError", "DEFAULT_OPTIONS"]
