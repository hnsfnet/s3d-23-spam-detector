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
import tempfile
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
    """Atomically persist an uploaded email to ``data_dir/label/filename``.

    Writes to a temp file first, then renames atomically. If the write is
    interrupted, only a temp file is left behind and the destination path
    is untouched.
    """
    if label not in ("ham", "spam"):
        raise ValueError("label must be 'ham' or 'spam'")
    folder = os.path.join(data_dir, label)
    os.makedirs(folder, exist_ok=True)
    safe_name = os.path.basename(filename) or "uploaded.txt"
    if not safe_name.lower().endswith((".txt", ".eml")):
        safe_name += ".txt"
    path = os.path.join(folder, safe_name)

    fd, tmp_path = tempfile.mkstemp(
        suffix=".txt.tmp",
        prefix=".upload_",
        dir=folder,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
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
    # Chinese ham samples
    "Subject: 明天的会议提醒\n\n各位好，提醒一下明天上午10点的站会，请带上你们的周报。谢谢。",
    "Subject: 这周五一起吃饭？\n\n嗨，想问下你这周五有空一起吃午饭吗？市中心开了一家新餐厅评价不错，有空告诉我哦。",
    "Subject: 第三季度项目进度\n\n迁移工作进展顺利，我们已经完成了测试套件，周三将部署到预发布环境。目前没有阻塞问题。",
    "Subject: 代码审核请求\n\n有空的时候能帮我看下合并请求吗？主要是给认证模块加了单元测试，多谢帮忙。",
    "Subject: 请假通知\n\n我12号到20号不在公司，紧急事项请联系Susan。回来后会尽快回复邮件。",
    "Subject: 昨天会议纪要\n\n附件是昨天规划会议的笔记，底部列出了行动事项，请查阅。",
    "Subject: 订单确认\n\n感谢您的购买，您的4821号订单已发货，预计3个工作日内送达。可在官网查看物流。",
    "Subject: 欢迎加入团队\n\n恭喜加入我们！下周一入职，HR会单独发送工牌和电脑领取信息。",
    "Subject: 月度通讯\n\n本期内容：技术博客精选、设计师专访、近期社区活动。祝您阅读愉快。",
    "Subject: 回复：关于报告的问题\n\n感谢发送，第4页的数据看起来没问题，我会把它们整合到幻灯片里。",
    "Subject: 生日聚餐\n\n周四是Maria的生日，我们凑钱买礼物和蛋糕，想参加的请一起凑份子哈。",
    "Subject: 服务器维护通知\n\n本周六凌晨2点到4点计划维护，升级期间服务可能短暂不可用。",
    "Subject: 作业反馈\n\n第三次作业做得很棒，结构上有几个小建议，但整体是优秀的提交。继续加油！",
    "Subject: 电话会议纪要\n\n我们同意推迟两周发布，市场部会相应调整活动日历。下周一同步。",
    "Subject: 感谢介绍\n\n非常感谢牵线搭桥，这周我会联系他们，聊得怎么样会随时告诉你。",
    "Subject: 这周一起健身？\n\n周二晚上你还去健身房吗？我想试试新课，几点方便告诉我。",
    "Subject: 合并请求已合并\n\n你的改动已合并到主分支，请在发布前确认集成分支的构建通过。",
    "Subject: 你推荐的书\n\n你推荐的书我买了两本，第一本很喜欢，有空一起喝咖啡聊聊。",
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
    # Chinese spam samples
    "Subject: 恭喜中奖！！！您获得100万元大奖\n\n尊敬的幸运用户！！！您已被选中参加本次国际抽奖活动！！！"
    "立即点击领取您的百万大奖！！！只需支付99元手续费即可到账！！！机会有限！！！",
    "Subject: 免费领取iPhone 15！限量速抢\n\n恭喜您！您已获得免费领取iPhone 15的资格！！！"
    "点击链接立即领取，只需支付运费！库存有限，先到先得！！！不看后悔！",
    "Subject: 在家兼职，日赚5000元！！！\n\n惊人机会！！！无需经验，在家操作，日赚5000元！！！"
    "点击链接立即开始！！！这个秘密方法100%保证赚钱！！！",
    "Subject: 【紧急】您的账户已被冻结，请立即验证\n\n尊敬的客户，我们检测到您的账户存在异常活动。"
    "请点击以下链接验证您的密码和银行卡信息，24小时内不验证将永久冻结账户！！！",
    "Subject: 无抵押贷款，当天到账！！！\n\n恭喜您！您已获得50万元贷款额度，无需抵押，不看征信！！！"
    "回复个人信息，24小时内资金到账！！！利息低至0.1%！",
    "Subject: 刷单刷信誉，日结佣金！！！\n\n想在家轻松赚钱吗？刷单刷信誉，每单佣金50-200元！！！"
    "当天结算，多劳多得！！！名额有限，立即加入！！！",
    "Subject: 您有百万红包待领取！！！\n\n恭喜！您获得了100万元红包返利！！！点击链接立即提现！！！"
    "限时优惠，过期不候！！！速抢！！！",
    "Subject: 正品手表一折促销！！！\n\n正品劳力士、欧米茄等名表一折促销！！！原价10万，现价9999！！！"
    "库存有限，售完即止！！！全国包邮！！！立即点击购买！",
    "Subject: 【官方通知】您的信用卡欠款逾期\n\n尊敬的用户，您的信用卡账单已逾期，请立即点击链接处理！！！"
    "否则将影响您的征信记录！！！核实账户密码立即解冻！",
    "Subject: 投资虚拟币，10倍收益保证！！！\n\n投资我们的币，保证1000%收益！！！机会难得，错过不再！！！"
    "立即打款到以下钱包地址！！！马上翻倍！！！",
    "Subject: 代开正规发票，点数优惠\n\n本公司代开各类正规发票，点数优惠，保真可查！！！"
    "需要请联系，量大从优！！！长期合作更优惠！",
    "Subject: 壮阳增大特效药，无效退款！！！\n\n神奇产品，100%有效！！！立即订购，买三送三！！！"
    "无需处方，全国保密配送！！！限时特惠！！！",
    "Subject: 您中了豪华游轮大奖！！！\n\n恭喜！！！您获得了免费豪华游轮双人游！！！"
    "只需支付999元港口费即可领取！！！点击链接，截止日期前有效！！！",
    "Subject: 美女同城交友，立即约见！！！\n\n附近单身美女想认识你！！！点击查看照片和联系方式！！！"
    "100%真人，免费注册！！！立即开始聊天！！！",
    "Subject: 【银行通知】您的积分可兑换现金\n\n尊敬的客户，您的银行卡积分可兑换5000元现金！！！"
    "点击链接验证账户信息立即兑换！！！逾期清零！！！",
    "Subject: 祛痘祛斑，7天见效！！！\n\n神奇美容产品，7天祛痘，30天祛斑！！！无效全额退款！！！"
    "明星都在用！！！限时特价，立即购买！！！",
    "Subject: 限量特价房，首付10万起\n\n市中心特价房源，首付仅10万起！！！升值潜力巨大！！！"
    "点击了解详情，房源有限，先到先得！！！",
    "Subject: 免费领取百万医疗险\n\n恭喜！您获得了免费百万医疗保险资格！！！点击链接立即领取！！！"
    "只需提供身份信息，保障立即生效！！！",
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
