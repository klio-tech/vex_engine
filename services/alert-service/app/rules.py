"""Alert rule matching for threshold-based Slack notifications.

Queries the alert_rules, alert_rule_channels, and integration_connections
tables to find rules where the event confidence falls below the threshold.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text

logger = logging.getLogger("agentguard.alert-service.rules")


@dataclass(frozen=True)
class MatchedRule:
    """A threshold rule that matched the current event."""

    rule_id: str
    rule_name: str
    confidence_threshold: float
    channel_type: str
    nango_connection_id: str
    slack_channel_id: str
    slack_channel_name: str


def find_matching_rules(
    org_id: str,
    confidence: Optional[float],
    db_session: object,
) -> list[MatchedRule]:
    """Find all enabled alert rules where confidence < threshold.

    Args:
        org_id: The organization ID (maps to alert_rules.account_id).
        confidence: The event's confidence score. If None, no rules match.
        db_session: A SQLAlchemy session for database reads.

    Returns:
        List of MatchedRule instances for rules that should fire.
    """
    if confidence is None:
        return []

    try:
        result = db_session.execute(
            text("""
                SELECT
                    r.id AS rule_id,
                    r.name AS rule_name,
                    r.confidence_threshold,
                    c.channel_type,
                    ic.nango_connection_id,
                    c.slack_channel_id,
                    c.slack_channel_name
                FROM alert_rules r
                JOIN alert_rule_channels c ON c.alert_rule_id = r.id
                JOIN integration_connections ic ON ic.id = c.connection_id
                WHERE r.account_id = :org_id
                  AND r.enabled = true
                  AND ic.status = 'connected'
                  AND :confidence < r.confidence_threshold
            """),
            {"org_id": org_id, "confidence": confidence},
        )

        return [
            MatchedRule(
                rule_id=str(row[0]),
                rule_name=row[1],
                confidence_threshold=float(row[2]),
                channel_type=row[3],
                nango_connection_id=row[4],
                slack_channel_id=row[5],
                slack_channel_name=row[6],
            )
            for row in result.fetchall()
        ]
    except Exception:
        logger.warning(
            "Failed to query alert rules for org %s",
            org_id,
            exc_info=True,
        )
        return []


def resolve_account_slug(org_id: str, db_session: object) -> Optional[str]:
    """Resolve the account slug for dashboard links.

    The account_slug column lives directly on the organizations table
    (added in migration 003).

    Args:
        org_id: The organization ID.
        db_session: A SQLAlchemy session for database reads.

    Returns:
        The account slug, or None if not found.
    """
    try:
        result = db_session.execute(
            text("""
                SELECT account_slug
                FROM organizations
                WHERE org_id = :org_id
            """),
            {"org_id": org_id},
        )
        row = result.fetchone()
        return row[0] if row else None
    except Exception:
        logger.warning("Failed to resolve slug for org %s", org_id, exc_info=True)
        return None
