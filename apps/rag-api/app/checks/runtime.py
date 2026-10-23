"""Pure finding -> QueryChecks conversion.

Detection and the caller's fail-soft try/except stay in the service; this module
only maps already-produced findings into the API response model.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..answer_checks import AnswerFinding
from ..evidence_checks import Finding
from ..schemas import AnswerFinding as AnswerFindingModel
from ..schemas import EvidenceFinding, QueryChecks


def to_query_checks(
    evidence_findings: Sequence[Finding],
    answer_findings: Sequence[AnswerFinding],
) -> QueryChecks:
    return QueryChecks(
        evidence=[
            EvidenceFinding(
                kind=finding.kind,
                detail=finding.detail,
                product=finding.product,
                register=finding.register,
                bit=finding.bit,
            )
            for finding in evidence_findings
        ],
        answer=[
            AnswerFindingModel(
                kind=finding.kind,
                value=finding.value,
                detail=finding.detail,
            )
            for finding in answer_findings
        ],
    )
