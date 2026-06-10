# =============================================================================
# TARGET: src/evaluation/retraction.py
# =============================================================================

@dataclass
class RetractionDetectorConfig:
    require_tag:                   bool  = True
    semantic_similarity_threshold: float = 0.85
    embedding_model:               str   = "sentence-transformers/all-MiniLM-L6-v2"


class RetractionDetector:
    """Identifies and validates claim retractions in trained model outputs.

    A claim is considered retracted when the corrected response either:
      (a) contains a retraction tag enclosing semantically similar content, or
      (b) omits the claim entirely (when require_tag=False), as determined by
          sentence-level embedding similarity below the configured threshold.
    """

    def __init__(self, config: RetractionDetectorConfig) -> None: ...

    def detect(
        self,
        original_response:   str,
        corrected_response:  str,
        validated_claims:    list[ValidatedClaim],
    ) -> list[RetractionRecord]:
        """Detect which refuted claims have been retracted in the corrected response.

        Returns:
            One RetractionRecord per refuted claim in validated_claims.
        """
        ...

    def _is_retracted_by_tag(
        self,
        claim:              ValidatedClaim,
        corrected_response: str,
    ) -> tuple[bool, Optional[str]]:
        """Check whether the claim is enclosed by a retraction tag.

        Returns:
            (retracted, corrected_text_span_or_None).
        """
        ...

    def _is_retracted_by_absence(
        self,
        claim:              ValidatedClaim,
        corrected_response: str,
    ) -> bool:
        """Check semantic absence via sentence-embedding cosine similarity."""
        ...

    def compute_retraction_rate(
        self,
        records: list[RetractionRecord],
    ) -> dict[str, float]:
        """Aggregate retraction_rate, precision, recall, and F1 over a record set."""
        ...


