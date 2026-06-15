"""Human handoff. Always records an escalation row first (so nothing is lost
for reconciliation even if email delivery fails), then attempts the email."""
from __future__ import annotations

import smtplib
from email.message import EmailMessage

from .config import settings
from .db import get_conn


def record_and_email(
    question: str,
    conversation_id: str | None,
    reason: str,
    score: float | None,
) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO escalations (question, conversation_id, reason, retrieval_score)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (question, conversation_id, reason, score),
        )
        esc_id = cur.fetchone()["id"]
        conn.commit()

    if _send_email(esc_id, question, conversation_id, reason, score):
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE escalations SET emailed = true WHERE id = %s", (esc_id,))
            conn.commit()

    return esc_id


def _send_email(
    esc_id: int,
    question: str,
    conversation_id: str | None,
    reason: str,
    score: float | None,
) -> bool:
    if not (settings.smtp_host and settings.email_from):
        return False  # email not configured; row is still recorded
    body = (
        f"A customer question could not be answered confidently and needs a human.\n\n"
        f"Escalation ID: {esc_id}\n"
        f"Reason: {reason}\n"
        f"Retrieval score: {score}\n"
        f"Conversation ID: {conversation_id}\n\n"
        f"Question:\n{question}\n"
    )
    msg = EmailMessage()
    msg["Subject"] = f"[Support escalation #{esc_id}] {question[:60]}"
    msg["From"] = settings.email_from
    msg["To"] = settings.escalation_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as s:
            s.starttls()
            if settings.smtp_user and settings.smtp_password:
                s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
        return True
    except Exception as exc:  # email is best-effort; the row is the source of truth
        print(f"[escalate] email failed for #{esc_id}: {exc}")
        return False
