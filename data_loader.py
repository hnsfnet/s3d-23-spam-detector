"""Utilities for loading and downloading spam/ham training data.

The loader supports two on-disk layouts, both of which can coexist under the
``data`` directory:

    data/ham/*.txt           data/spam/*.txt
    data/enron1/ham/*.txt    data/enron1/spam/*.txt   (Enron-Spam layout)

Any directory whose basename is ``ham`` or ``spam`` is treated as a class
folder, regardless of how deeply it is nested. Plain ``.txt`` and ``.eml``
files inside those folders are read as individual emails.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from email import message_from_string
from typing import List, Tuple

# A known mirror of the preprocessed Enron-Spam dataset (Metsis et al.).
# Each subset is a tar.gz containing ``ham/`` and ``spam/`` text folders.
ENRON_BASE_URL = "http://www.aueb.gr/users/ion/data/enron-spam"
DEFAULT_SUBSETS = ["enron1", "enron2", "enron3", "enron4", "enron5", "enron6"]

LabelledEmail = Tuple[str, str]  # (raw_text, label) where label in {"ham", "spam"}


def _read_text(path: str) -> str:
    """Read a file as text, trying common encodings used by email corpora."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=encoding, errors="strict") as handle:
                return handle.read()
        except UnicodeDecodeError:
            continue
    # Last resort: latin-1 never fails on byte-level data.
    with open(path, "r", encoding="latin-1", errors="replace") as handle:
        return handle.read()


def parse_email(raw_text: str):
    """Parse a raw email string into (subject, body, headers).

    Files in the Enron-Spam corpus are plain bodies without headers; this
    helper tolerates both raw ``.eml`` messages and header-less bodies.
    """
    subject = ""
    body = raw_text
    headers: dict = {}

    stripped = raw_text.lstrip()
    if stripped[:5].lower() in ("from:", "subj:", "recip", "date:", "mime:", "to: ", "to:") \
            or stripped.lower().startswith(("received:", "subject:", "content-")):
        # Looks like a real email with headers; parse it.
        try:
            message = message_from_string(raw_text)
            if message:
                subject = message.get("Subject", "") or ""
                headers = {k: (v or "") for k, v in message.items()}
                payload = message.get_payload(decode=True)
                if payload is None:
                    payload = message.get_payload()
                if isinstance(payload, bytes):
                    body = payload.decode("latin-1", errors="replace")
                elif isinstance(payload, list):
                    body = "\n".join(
                        str(p.get_payload(decode=True) or p.get_payload())
                        for p in payload
                    )
                else:
                    body = str(payload) if payload is not None else raw_text
        except Exception:
            subject = ""
            body = raw_text
            headers = {}
    return subject, body, headers


def load_dataset(data_dir: str = "data") -> Tuple[List[LabelledEmail], dict]:
    """Load all ham/spam emails found under ``data_dir``.

    Returns a tuple of ``(emails, stats)`` where ``emails`` is a list of
    ``(raw_text, label)`` pairs and ``stats`` summarises the counts.
    """
    emails: List[LabelledEmail] = []
    stats = {"ham": 0, "spam": 0, "total": 0, "root": os.path.abspath(data_dir)}

    if not os.path.isdir(data_dir):
        return emails, stats

    for root, dirs, files in os.walk(data_dir):
        base = os.path.basename(root).lower()
        if base not in ("ham", "spam"):
            continue
        label = base
        for name in sorted(files):
            if not name.lower().endswith((".txt", ".eml")):
                continue
            path = os.path.join(root, name)
            try:
                text = _read_text(path)
            except OSError:
                continue
            if not text.strip():
                continue
            emails.append((text, label))
            stats[label] += 1

    stats["total"] = len(emails)
    return emails, stats


def download_enron_dataset(
    data_dir: str = "data",
    subsets=None,
    timeout: float = 60.0,
) -> dict:
    """Download and extract Enron-Spam subsets into ``data_dir``.

    Returns a dict describing how many subsets succeeded. Failures for a single
    subset do not abort the others, so partial downloads still produce usable
    training data.
    """
    subsets = subsets or DEFAULT_SUBSETS
    os.makedirs(data_dir, exist_ok=True)
    results = {"succeeded": [], "failed": []}

    for subset in subsets:
        target_dir = os.path.join(data_dir, subset)
        if os.path.isdir(target_dir) and os.listdir(target_dir):
            results["succeeded"].append(subset)
            continue

        url = f"{ENRON_BASE_URL}/{subset}.tar.gz"
        archive_path = os.path.join(data_dir, f"{subset}.tar.gz")
        try:
            urllib.request.urlretrieve(url, archive_path)
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(data_dir)
            results["succeeded"].append(subset)
        except Exception as exc:  # network/URL/parse errors are non-fatal
            results["failed"].append({"subset": subset, "error": str(exc)})
        finally:
            if os.path.exists(archive_path):
                try:
                    os.remove(archive_path)
                except OSError:
                    pass
    return results


def save_uploaded_email(data_dir: str, label: str, filename: str, content: str) -> str:
    """Persist an uploaded email to ``data_dir/label/filename`` and return path."""
    if label not in ("ham", "spam"):
        raise ValueError("label must be 'ham' or 'spam'")
    folder = os.path.join(data_dir, label)
    os.makedirs(folder, exist_ok=True)
    safe_name = os.path.basename(filename) or "uploaded.txt"
    if not safe_name.lower().endswith((".txt", ".eml")):
        safe_name += ".txt"
    path = os.path.join(folder, safe_name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


# ---------------------------------------------------------------------------
# Bundled fallback sample data
#
# These short, representative emails let the service train even when no real
# dataset is available offline. They are clearly synthetic and only meant to
# bootstrap the classifier; run ``download_data.py`` to fetch the full,
# research-grade Enron-Spam corpus for production-quality accuracy.
# ---------------------------------------------------------------------------

_HAM_TEMPLATES = [
    "Subject: Meeting tomorrow\n\nHi team, just a reminder about our standup at "
    "10am tomorrow. Please bring your weekly status updates. Thanks,",
    "Subject: Lunch this Friday?\n\nHey, wanted to check if you're free for lunch "
    "this Friday. The new place downtown has good reviews. Let me know!",
    "Subject: Q3 project update\n\nThe migration is on track. We finished the test "
    "suite and will deploy to staging on Wednesday. No blockers so far.",
    "Subject: Code review request\n\nCould you review my pull request when you have "
    "a moment? It adds unit tests for the auth module. Appreciate the help.",
    "Subject: Vacation notice\n\nI'll be out of office from the 12th through the "
    "20th. For urgent matters please contact Susan. I will respond when I return.",
    "Subject: Minutes from yesterday\n\nAttached are the notes from yesterday's "
    "planning session. Action items are listed at the bottom. Please review.",
    "Subject: Your order receipt\n\nThank you for your purchase. Your order 4821 "
    "has shipped and will arrive in 3 business days. Track it on our website.",
    "Subject: Welcome to the team\n\nCongratulations on joining us! Your onboarding "
    "begins Monday. HR will send your badge and laptop details separately.",
    "Subject: Monthly newsletter\n\nIn this issue: engineering blog highlights, an "
    "interview with our designer, and upcoming community events. Enjoy reading.",
    "Subject: Re: question about the report\n\nThanks for sending this over. The "
    "numbers on page 4 look correct to me. I'll incorporate them into the slides.",
    "Subject: Birthday lunch\n\nIt's Maria's birthday on Thursday. We're collecting "
    "for a gift and cake. Please chip in if you'd like to join the celebration.",
    "Subject: Server maintenance window\n\nScheduled maintenance this Saturday "
    "from 2am to 4am. Services may be briefly unavailable during the upgrade.",
    "Subject: Homework feedback\n\nGreat work on assignment three. A few small "
    "suggestions on clarity, but overall a strong submission. Keep it up.",
    "Subject: Conference call notes\n\nWe agreed to push the launch by two weeks. "
    "Marketing will adjust the campaign calendar accordingly. Next sync Monday.",
    "Subject: Thanks for the intro\n\nReally appreciate you connecting us. I'll "
    "reach out to them this week and keep you posted on how the chat goes.",
    "Subject: Gym buddy this week\n\nAre you still going Tuesday evening? I was "
    "planning to join and try the new class. Let me know what time works.",
    "Subject: Pull request merged\n\nYour changes have been merged into main. "
    "Please verify the build passes on the integration branch before release.",
    "Subject: Books you recommended\n\nI picked up two of the books you suggested. "
    "Really enjoying the first one so far. We should discuss it over coffee.",
]

_SPAM_TEMPLATES = [
    "Subject: CONGRATULATIONS!!! You've won $1,000,000\n\nDear lucky winner!!! "
    "You have been selected in our international lottery!!! Claim your prize NOW "
    "by sending your bank details. Do not miss out!!!",
    "Subject: CHEAP meds online, no prescription needed!!!\n\nBuy cheap pills "
    "online!!! Discount 80%!!! Vi@gra, C1alis and more. Free shipping worldwide. "
    "Click here to order now!!! Limited offer!!!",
    "Subject: Make $5000/day from home!!!\n\nAmazing opportunity!!! Earn thousands "
    "working from home with no experience!!! Click the link and start today!!! "
    "This secret method is 100% guaranteed!!!",
    "Subject: FREE iPhone!!! Claim your gift now\n\nYou have been chosen to receive "
    "a FREE iPhone!!! Just complete this survey and pay shipping. Hurry, supplies "
    "are limited!!! Don't wait!!!",
    "Subject: Your account is suspended, verify immediately\n\nDear customer, your "
    "account has been suspended. Click here to verify your password and credit "
    "card details within 24 hours or lose access forever!!!",
    "Subject: Hot singles in your area want to meet you!!!\n\nClick here now to "
    "see profiles near you!!! 100% free to join!!! Adults only!!! Don't miss out!!!",
    "Subject: Loan approved!!! No credit check\n\nCongratulations!!! You're "
    "approved for a $50,000 loan regardless of credit!!! No collateral needed. "
    "Reply with your details to receive funds in 24 hours!!!",
    "Subject: AMAZING weight loss miracle!!!\n\nLose 30 pounds in one week with "
    "this miracle pill!!! Doctors hate this trick!!! Click to order now and get "
    "a free bottle!!! Results guaranteed!!!",
    "Subject: You have inherited $25 million\n\nI am a barrister representing a "
    "deceased client who left you $25,000,000. Please send your personal details "
    "and a processing fee to claim this fortune immediately!!!",
    "Subject: 90% OFF designer watches!!!\n\nAuthentic luxury watches 90% off!!! "
    "Rolex, Omega and more. Buy now before stock runs out!!! Free shipping "
    "worldwide!!! Click here!!!",
    "Subject: Urgent: confirm your PayPal password\n\nWe noticed unusual activity. "
    "Confirm your password and card number now or your account will be closed!!! "
    "Click the link below immediately!!!",
    "Subject: Work from home, earn $$$ easily!!!\n\nJoin our program and earn $$$ "
    "easily from home!!! No skills required!!! Thousands are making money now!!! "
    "Sign up free today!!! Don't delay!!!",
    "Subject: Crypto investment 1000% return guaranteed\n\nInvest in our coin and "
    "get guaranteed 1000% returns!!! Join now before it's too late!!! Send funds "
    "to the wallet address below!!! Don't miss this chance!!!",
    "Subject: You've been approved for a FREE gift card\n\nClaim your FREE $1000 "
    "gift card now!!! Just pay a small processing fee. Hurry, offer ends soon!!! "
    "Click here to claim before it's gone!!!",
    "Subject: enlarge now, 100% results!!!\n\nAmazing product delivers 100% "
    "results!!! Order now and get 3 bottles free!!! No prescription needed!!! "
    "Click here to buy!!! Limited time offer!!!",
    "Subject: Nigerian prince needs your help\n\nI am a prince with $40 million "
    "frozen in an account. Help me transfer it and keep 30%!!! Send your bank "
    "details and a small fee to begin immediately!!!",
    "Subject: You won a luxury vacation!!!\n\nCongratulations!!! You've won a free "
    "luxury cruise for two!!! Just pay port fees of $99 now!!! Click to claim "
    "before the deadline!!! Don't miss out!!!",
    "Subject: Double your money in 7 days!!!\n\nExclusive opportunity!!! Double "
    "your money guaranteed in 7 days!!! Send Bitcoin to the address below and "
    "receive double back!!! Act fast!!! Limited spots!!!",
]


def seed_sample_data(data_dir: str = "data") -> dict:
    """Write a small synthetic ham/spam corpus to ``data_dir``.

    Only seeds when the corresponding class folder is empty, so it never
    overwrites real data downloaded separately.
    """
    stats = {"ham": 0, "spam": 0}
    for label, templates in (("ham", _HAM_TEMPLATES), ("spam", _SPAM_TEMPLATES)):
        folder = os.path.join(data_dir, label)
        os.makedirs(folder, exist_ok=True)
        existing = [f for f in os.listdir(folder) if f.endswith((".txt", ".eml"))]
        if existing:
            stats[label] = len(existing)
            continue
        for index, text in enumerate(templates, start=1):
            path = os.path.join(folder, f"sample_{index:03d}.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        stats[label] = len(templates)
    return stats
