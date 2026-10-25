"""小鲸鱼雷达：聊到人格的日常爱好时，提高主动插话的概率。

设计要点：
1. 不接管普通对话——被 @ / 唤醒词 / 指令命中的消息一律放过。
2. 每条没唤醒她的群消息都记账（供插话时带上上下文）。
3. 关键词按「方向」分组，每个方向有自己的权重（概率）。
   同一条消息里命中某个方向的词越多，概率越高；多个方向取最高的那个。
4. 关键词没命中时可选地让便宜的模型打 0-10 分，够分才插话。
5. 冷却 + 每日上限 + 连续说话抑制，防止变成复读机。
6. `/雷达` 系列指令可以在 QQ 里直接改词、改方向权重、改各种概率，改完立刻生效并落盘。

注意：内置的 active_reply 与本插件是两套独立开火逻辑，若两边同时开，
同一条消息可能被回两次。建议把内置的 possibility_reply 设为 0，
把「基础随机」交给本插件的 base_probability。
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from sys import maxsize

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.provider import Provider
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.message import TextPart
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform.message_type import MessageType

GROUP_HISTORY_HEADER = (
    "<system_reminder>"
    "You are in a group chat. Below are messages after your last reply:\n"
    "--- BEGIN CONTEXT ---\n"
)
GROUP_HISTORY_FOOTER = "\n--- END CONTEXT ---\n</system_reminder>"

DEFAULTS: dict = {
    "enable": True,
    "whitelist": [],
    "operator_uids": [],
    "base_probability": 0.02,
    "keyword_probability": 0.55,
    "strong_probability": 0.9,
    "keyword_hits_for_strong": 2,
    "keyword_groups": [{"name": "游戏", "weight": 0.8, "words": ["游戏", "开黑", "副本", "上分", "抽卡", "联机"]}, {"name": "二次元", "weight": 0.65, "words": ["番剧", "追番", "漫画", "cos", "周边"]}, {"name": "吃喝", "weight": 0.7, "words": ["外卖", "火锅", "烧烤", "奶茶", "夜宵", "饿"]}, {"name": "股票", "weight": 0.8, "words": ["A股", "基金", "涨停", "割肉", "跌麻"]}, {"name": "上班熬夜", "weight": 0.6, "words": ["加班", "摸鱼", "通宵", "失眠", "上班"]}, {"name": "硬件代码", "weight": 0.65, "words": ["显卡", "装机", "报错", "蓝屏", "配置"]}],
    "interest_keywords": [],
    "negative_keywords": [],
    "cooldown_seconds": 120,
    "daily_limit": 40,
    "context_size": 15,
    "steer_prompt": "",
    "judge_enable": True,
    "judge_provider_id": "",
    "judge_threshold": 7,
    "judge_probability": 0.45,
    "judge_min_length": 8,
    "judge_max_per_minute": 6,
    "judge_context_messages": 4,
    "judge_cache_seconds": 120,
    "learn_enable": True,
    "learn_interval_minutes": 60,
    "learn_min_messages": 60,
    "learn_max_per_run": 2,
    "learn_weight": 0.6,
    "learn_direction_name": "新发现",
    "learn_max_total_words": 160,
    "learn_notify_uid": "",
    "learned_words": [],
    "learn_state": {},
}

LEARN_PROMPT = """你在帮一个群聊看热闹的女生整理「她最近对什么感兴趣」。

现在的方向（已有的兴趣）：
{directions}

最近群里的聊天记录：
{history}

任务：从上面这段聊天里，找出 0-{limit} 个「她可能会感兴趣的**新**话题或**新**名词」。
要求：
- 必须是聊天里真实出现过的词，2-6 个字，具体一点（"显卡"可以，"东西"不行）。
- 不要和已有方向里的词重复。
- 只挑她真的会想接话的（游戏、股票、吃喝、动漫、硬件、熬夜、猫狗……），政治/敏感话题一律跳过。
- 拿不准就少给，宁可只给 1 个，也不要凑数。

只输出 JSON，不要解释，格式：
{{"candidates": [{{"word": "词", "direction": "归到哪个方向（没有合适的就写 {fallback}）", "reason": "为什么她会感兴趣（10 字内）"}}]}}
"""

STOP_WORDS = {
    "好", "的", "了", "啊", "吗", "呢", "我", "你", "他", "她", "它", "这", "那",
    "就", "也", "都", "很", "是", "不", "在", "有", "没", "和", "跟", "说", "个",
    "什么", "怎么", "可以", "这个", "那个", "东西", "时候", "感觉", "现在",
}

JUDGE_PROMPT = """你在帮一个群聊机器人判断：下面这条群消息，是否踩到了她的兴趣点。

她的日常爱好/关注点：
{interests}

最近的聊天（越靠下越新，最后一条就是待判断的这条）：
{context}

只输出一个 0-10 的整数，不要解释：
0 = 完全无关，10 = 正中下怀、她一定会想插一句。
"""

HELP_TEXT = """小鲸鱼雷达 · 指令（改完立刻生效并存盘）

/雷达                     看当前状态
/雷达 词                  看所有方向 + 权重 + 词数
/雷达 词 股票             只看某个方向的词
/雷达 加词 股票 涨停 板了 套牢     往方向里加词（方向不存在就新建）
/雷达 删词 涨停 板了        从所有方向里删掉这些词
/雷达 新方向 熬夜 70%       新建一个方向，权重 70%
/雷达 方向 股票 85%         改已有方向的权重
/雷达 删方向 熬夜           删掉整个方向
/雷达 设 基础 3%            改基础概率
/雷达 设 关键词 60%         改单方向命中概率（新方向默认用它）
/雷达 设 强命中 95%         改强命中概率（一个方向命中 ≥2 词）
/雷达 设 强词数 2           几个词算强命中
/雷达 设 打分 50% / 阈值 7 / 打分开关 关
/雷达 设 冷却 180 / 上限 30 / 上下文 20
/雷达 开 | /雷达 关
/雷达 测试 今天A股又跌了     看这句话会怎么判、概率多少
/雷达 导出                把方向配置打成一行 JSON（方便备份）

自动学新爱好（她会从群里捡词，攒够了就自己加进词库）
/雷达 学                  看学习设置与进度
/雷达 学 开 | /雷达 学 关
/雷达 学 间隔 60           多少分钟最多分析一次
/雷达 学 条数 60           攒够多少条群消息才分析
/雷达 学 词数 2            一次最多学几个词
/雷达 学 权重 60%          学来的新方向默认权重
/雷达 学 通知 <QQ号>       学到新词私聊告诉你（留空关闭）
/雷达 立刻学               马上分析一次
/雷达 学习记录 [N]         看最近学到的词
/雷达 忘词 显卡             忘掉这个（连同学习记录）

可改的「设」键：基础 关键词 强命中 强词数 打分 阈值 冷却 上限 上下文 打分开关
"""

PROBABILITY_KEYS = {
    "base_probability",
    "keyword_probability",
    "strong_probability",
    "judge_probability",
}

SET_ALIASES = {
    "基础": "base_probability",
    "基础概率": "base_probability",
    "关键词": "keyword_probability",
    "关键词概率": "keyword_probability",
    "强命中": "strong_probability",
    "强概率": "strong_probability",
    "强词数": "keyword_hits_for_strong",
    "打分": "judge_probability",
    "打分概率": "judge_probability",
    "阈值": "judge_threshold",
    "冷却": "cooldown_seconds",
    "上限": "daily_limit",
    "上下文": "context_size",
    "打分开关": "judge_enable",
}

TRUE_WORDS = {"开", "true", "1", "yes", "on", "启用"}
FALSE_WORDS = {"关", "false", "0", "no", "off", "禁用"}


def _split_words(raw: str) -> list[str]:
    """把「a b,c、d」切成词列表。"""
    return [w for w in re.split(r"[\s,，、;；/]+", raw or "") if w]


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


PLUGIN_VERSION = "0.5.0"


@register(
    "astrbot_plugin_whale_radar",
    "cjh-cheng",
    "小鲸鱼雷达：群聊里聊到她的日常爱好时，提高主动插话概率（带 QQ 内调试指令）",
    PLUGIN_VERSION,
    "https://github.com/cjh-cheng/astrbot_plugin_whale_radar",
)
class WhaleRadar(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        # umo -> deque[str]
        self._history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=60))
        # umo -> [timestamp, ...] 最近插话时间
        self._spoke: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
        # umo -> (date_str, count)
        self._daily: dict[str, tuple[str, int]] = {}
        # umo -> deque[float] 打分时间
        self._judged: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=60))
        # umo -> {"pending": 攒了多少条, "last": 上次分析时间} 学习进度（持久化）
        self._learn_state: dict[str, dict] = {}
        self._state_dirty = 0
        self._load_learn_state()
        # umo -> (ts, score, text) 打分结果缓存，同一话题短时间内复用，省调用
        self._judge_cache: dict[str, tuple[float, int, str]] = {}
        # umo -> deque[str] 最近几条消息，给打分提供上下文
        self._recent: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=6))
        self._learn_task: asyncio.Task | None = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            self._learn_task = loop.create_task(self._learn_loop())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[whale_radar] 学习任务启动失败: {exc}")

    # ---------------- 配置 ----------------
    def _get(self, key: str):
        value = None
        try:
            value = self.config.get(key)
        except Exception:
            value = None
        if value is None:
            return DEFAULTS.get(key)
        return value

    def _config_path(self) -> Path:
        raw = str(getattr(self.config, "config_path", "") or "").strip()
        if raw:
            return Path(raw)
        return Path(__file__).resolve().parents[2] / "config" / "astrbot_plugin_whale_radar_config.json"

    def _save(self, updates: dict) -> str:
        """改内存 + 落盘，返回给人看的结果。"""
        try:
            for key, value in updates.items():
                self.config[key] = value
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[whale_radar] 写内存配置失败: {exc}")
        try:
            saver = getattr(self.config, "save_config", None)
            if callable(saver):
                saver()
                return "已保存"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[whale_radar] save_config 失败，改走手写: {exc}")
        try:
            path = self._config_path()
            data: dict = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
                except Exception:
                    data = {}
            data.update(updates)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return "已保存（手写配置文件）"
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[whale_radar] 配置文件写入失败: {exc}")
            return f"只改了内存，写盘失败：{exc}"

    def _groups(self) -> list[dict]:
        """归一化后的关键词方向列表：[{name, weight, words}]。"""
        raw = self._get("keyword_groups") or []
        groups: list[dict] = []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip() or "未命名"
                try:
                    weight = float(item.get("weight"))
                except (TypeError, ValueError):
                    weight = float(self._get("keyword_probability") or 0.55)
                words = [str(w).strip() for w in (item.get("words") or []) if str(w).strip()]
                if not words:
                    continue
                groups.append(
                    {"name": name, "weight": min(1.0, max(0.0, weight)), "words": words}
                )
        if not groups:
            raw_legacy = self._get("interest_keywords") or []
            legacy = [str(k).strip() for k in raw_legacy if str(k).strip()]
            if legacy:
                groups.append(
                    {
                        "name": "兴趣",
                        "weight": float(self._get("keyword_probability") or 0.55),
                        "words": legacy,
                    }
                )
        return groups

    def _keywords(self) -> list[str]:
        """所有方向的词（去重，给打分模型看）。"""
        raw = self._get("interest_keywords") or []
        words: list[str] = [str(k).strip() for k in raw if str(k).strip()]
        for group in self._groups():
            words.extend(group["words"])
        seen: set[str] = set()
        ordered: list[str] = []
        for word in words:
            if word not in seen:
                seen.add(word)
                ordered.append(word)
        return ordered

    def _negatives(self) -> list[str]:
        raw = self._get("negative_keywords") or []
        return [str(k).strip() for k in raw if str(k).strip()]

    @staticmethod
    def _match_words(text: str, words: list[str]) -> list[str]:
        """大小写无关的匹配；命中词里丢掉被更长命中词包住的（「游戏」被「打游戏」覆盖）。"""
        haystack = (text or "").lower()
        if not haystack:
            return []
        hits: list[str] = []
        seen: set[str] = set()
        for word in words:
            needle = (word or "").strip().lower()
            if not needle or needle in seen:
                continue
            if needle in haystack:
                seen.add(needle)
                hits.append(word)
        if len(hits) <= 1:
            return hits
        return [
            word
            for word in hits
            if not any(
                word.lower() != other.lower() and word.lower() in other.lower()
                for other in hits
            )
        ]

    # ---------------- 学习进度（持久化） ----------------
    def _load_learn_state(self) -> None:
        raw = self._get("learn_state") or {}
        if not isinstance(raw, dict):
            return
        for umo, item in raw.items():
            if not isinstance(item, dict):
                continue
            try:
                self._learn_state[str(umo)] = {
                    "pending": int(item.get("pending") or 0),
                    "last": float(item.get("last") or 0),
                }
            except (TypeError, ValueError):
                continue

    def _state(self, umo: str) -> dict:
        state = self._learn_state.get(umo)
        if state is None:
            state = {"pending": 0, "last": 0.0}
            self._learn_state[umo] = state
        return state

    def _flush_learn_state(self) -> None:
        self._state_dirty = 0
        try:
            self._save({"learn_state": self._learn_state})
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[whale_radar] 学习进度落盘失败: {exc}")

    # ---------------- 学新词 ----------------
    def _all_words(self) -> set[str]:
        words = set(self._keywords())
        words.update(self._negatives())
        return words

    def _total_words(self) -> int:
        return sum(len(group["words"]) for group in self._groups())

    def _word_ok(self, word: str) -> tuple[bool, str]:
        word = (word or "").strip()
        if not word:
            return False, "空词"
        if word in STOP_WORDS:
            return False, "太泛的词"
        if not (1 <= len(word) <= 8):
            return False, "长度不合适（1-8 个字）"
        if re.search(r"[0-9a-zA-Z]{1,}$", word) and len(word) <= 2:
            return False, "太短"
        if word in self._all_words():
            return False, "已经有了"
        if self._total_words() >= int(self._get("learn_max_total_words") or 160):
            return False, "词库已满（可调 learn_max_total_words）"
        return True, ""

    def _ensure_direction(self, groups: list[dict], name: str, weight: float) -> dict:
        name = (name or "").strip() or str(self._get("learn_direction_name") or "新发现")
        target = next((g for g in groups if g["name"] == name), None)
        if target is None:
            target = {"name": name, "weight": weight, "words": []}
            groups.append(target)
        return target

    def _add_word(
        self, word: str, direction: str = "", weight: float | None = None, reason: str = ""
    ) -> tuple[bool, str]:
        """把一个词加进词库（带校验），返回 (是否加成功, 给用户看的话)。"""
        word = (word or "").strip()
        ok, why = self._word_ok(word)
        if not ok:
            return False, why
        groups = self._groups()
        default_weight = (
            float(weight)
            if weight is not None
            else float(self._get("learn_weight") or 0.6)
        )
        target = self._ensure_direction(groups, direction, default_weight)
        target["words"].append(word)
        history = list(self._get("learned_words") or [])
        history.append(
            {
                "word": word,
                "direction": target["name"],
                "at": time.strftime("%Y-%m-%d %H:%M"),
                "reason": reason[:40],
            }
        )
        self._save({"keyword_groups": groups, "learned_words": history[-200:]})
        return True, f"「{word}」加进方向「{target['name']}」"

    async def _notify_learned(self, items: list[str], umo: str = "") -> None:
        uid = str(self._get("learn_notify_uid") or "").strip()
        if not uid or not items:
            return
        platform = (umo.split(":", 1)[0] if ":" in (umo or "") else "") or "qq-fish"
        session = f"{platform}:FriendMessage:{uid}"
        try:
            await self.context.send_message(
                session,
                MessageChain([Plain("群里学到了新爱好：" + "、".join(items) + "（/雷达 词 看全部）")]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[whale_radar] 学词通知发送失败: {exc}")

    def _learn_loop_task(self) -> bool:
        try:
            return bool(self._learn_task and not self._learn_task.done())
        except Exception:
            return False

    async def _learn_loop(self) -> None:
        """每隔一段时间看看群里在聊什么，捡几个新兴趣词。"""
        await asyncio.sleep(120)
        while True:
            try:
                await self._learn_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[whale_radar] 学习循环出错: {exc}")
            await asyncio.sleep(300)

    async def _learn_once(self, umo: str = "", force: bool = False) -> str:
        if not self._get("learn_enable") and not force:
            return "学习没开（/雷达 学 开）"
        interval = max(5, int(self._get("learn_interval_minutes") or 60)) * 60
        need = max(5, int(self._get("learn_min_messages") or 60))
        now = time.time()
        targets = [umo] if umo else list(self._history.keys())
        reports: list[str] = []
        for key in targets:
            if not key:
                continue
            if not force:
                state = self._state(key)
                if now - float(state.get("last") or 0) < interval:
                    continue
                if int(state.get("pending") or 0) < need:
                    continue
            records = list(self._history.get(key) or [])
            if len(records) < 10:
                # 重启后内存里的聊天记录是空的，攒的条数没有对应样本可用，清零重来
                state = self._state(key)
                if int(state.get("pending") or 0):
                    state["pending"] = 0
                continue
            state = self._state(key)
            state["last"] = now
            state["pending"] = 0
            report = await self._analyze(key, records[-120:])
            if report:
                reports.append(report)
        self._flush_learn_state()
        return "\n".join(reports) if reports else "这次没学到新词（聊天还不够或者都学过了）"

    async def _analyze(self, umo: str, records: list[str]) -> str:
        try:
            provider = await self.context.get_using_provider_async(umo)
            if not isinstance(provider, Provider):
                return ""
        except Exception:  # noqa: BLE001
            return ""
        limit = max(1, int(self._get("learn_max_per_run") or 2))
        groups = self._groups()
        # 把每个方向的前几个词当样本给模型，避免它推荐已有词的近义变体
        directions = (
            "、".join(
                f"{g['name']}（{len(g['words'])} 词，例如 {'/'.join(g['words'][:8])}）" for g in groups
            )
            or "（还没有）"
        )
        prompt = LEARN_PROMPT.format(
            directions=directions,
            history="\n".join(records),
            limit=limit,
            fallback=self._get("learn_direction_name") or "新发现",
        )
        try:
            response = await provider.text_chat(
                prompt=prompt, session_id=uuid.uuid4().hex, persist=False
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[whale_radar] 学词调用失败: {exc}")
            return ""
        raw = (getattr(response, "completion_text", "") or "").strip()
        candidates = self._parse_candidates(raw)
        if not candidates:
            logger.debug(f"[whale_radar] 学词没有候选: {raw[:100]!r}")
            return ""
        added: list[str] = []
        for item in candidates[:limit]:
            word = str(item.get("word") or "").strip()
            direction = str(item.get("direction") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if direction in {"新发现", "其他", "无", "-", ""}:
                direction = str(self._get("learn_direction_name") or "新发现")
            ok, message = self._add_word(word, direction, reason=reason)
            if ok:
                added.append(word)
                logger.info(f"[whale_radar] 学到新词 {word} -> {direction}（{reason}）")
            else:
                logger.debug(f"[whale_radar] 跳过 {word}: {message}")
        if added:
            await self._notify_learned(added, umo)
            return "学到：" + "、".join(added)
        return ""

    def _parse_candidates(self, raw: str) -> list[dict]:
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                items = data.get("candidates") or []
                return [item for item in items if isinstance(item, dict)]
            except json.JSONDecodeError:
                pass
        # 兜底：从纯文本里抠出「词」
        words = re.findall(r"[\"“]([^\"”]{1,8})[\"”]", text)
        return [{"word": w} for w in words[:3]]

    # ---------------- 记账 ----------------
    def _remember(self, umo: str, event: AstrMessageEvent) -> None:
        nickname = ""
        try:
            nickname = event.get_sender_name() or ""
        except Exception:
            nickname = ""
        stamp = time.strftime("%H:%M:%S")
        text = (event.message_str or "").strip()
        has_image = any(isinstance(c, Image) for c in (event.get_messages() or []))
        body = text
        if has_image:
            body = f"{body} [图片]".strip()
        if not body:
            return
        self._history[umo].append(f"[{nickname or '群友'}/{stamp}]: {body}")
        self._recent[umo].append(body)
        state = self._state(umo)
        state["pending"] = int(state.get("pending") or 0) + 1
        self._state_dirty += 1
        if self._state_dirty >= 10:
            self._flush_learn_state()

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _gate(self, umo: str) -> bool:
        """冷却与每日上限检查。"""
        now = time.time()
        cooldown = float(self._get("cooldown_seconds") or 0)
        spoke = self._spoke[umo]
        if spoke and now - spoke[-1] < cooldown:
            return False
        day, count = self._daily.get(umo, (self._today(), 0))
        if day != self._today():
            count = 0
        limit = int(self._get("daily_limit") or 0)
        if limit > 0 and count >= limit:
            return False
        return True

    def _mark(self, umo: str) -> None:
        self._spoke[umo].append(time.time())
        day, count = self._daily.get(umo, (self._today(), 0))
        if day != self._today():
            count = 0
        self._daily[umo] = (self._today(), count + 1)

    # ---------------- 打分 ----------------
    def _judge_allowed(self, umo: str) -> bool:
        limit = int(self._get("judge_max_per_minute") or 0)
        if limit <= 0:
            return False
        now = time.time()
        queue = self._judged[umo]
        while queue and now - queue[0] > 60:
            queue.popleft()
        return len(queue) < limit

    @staticmethod
    def _similar(a: str, b: str) -> bool:
        """粗判两句话是不是在聊同一件事（同一话题的连续消息不必重复打分）。"""
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return False
        if a == b or a in b or b in a:
            return True
        grams_a = {a[i : i + 2] for i in range(len(a) - 1)} or {a}
        grams_b = {b[i : i + 2] for i in range(len(b) - 1)} or {b}
        union = grams_a | grams_b
        return bool(union) and len(grams_a & grams_b) / len(union) >= 0.5

    async def _judge(self, umo: str, text: str) -> int | None:
        try:
            provider_id = (self._get("judge_provider_id") or "").strip()
            if provider_id:
                provider = self.context.get_provider_by_id(provider_id)
            else:
                provider = await self.context.get_using_provider_async(umo)
            if not isinstance(provider, Provider):
                logger.debug("[whale_radar] judge 没有拿到可用的 provider")
                return None
            interests = (
                "、".join(self._keywords()[:120])
                or "游戏、二次元、股票与基金、吃饭与外卖、熬夜与上班、写代码与硬件"
            )
            # 带上最近几条消息当上下文，判断更准（比单条判断几乎不增加成本）
            window = max(1, int(self._get("judge_context_messages") or 4))
            context = "\n".join(list(self._recent[umo])[-window:]) or text
            self._judged[umo].append(time.time())
            response = await provider.text_chat(
                prompt=JUDGE_PROMPT.format(interests=interests, context=context[:800]),
                session_id=uuid.uuid4().hex,
                persist=False,
            )
            raw = (getattr(response, "completion_text", "") or "").strip()
            logger.debug(f"[whale_radar] judge 原始输出: {raw[:80]!r}")
            # 模型可能输出「8」「8分」「评分：8/10」，取第一个出现的数字
            match = re.search(r"\d+", raw)
            if not match:
                logger.info(f"[whale_radar] judge 输出无法解析: {raw[:60]!r}")
                return None
            score = int(match.group(0))
            if score > 10:
                score = 10
            return score
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[whale_radar] judge failed: {exc}")
            return None

    # ---------------- 决策 ----------------
    async def _decide(
        self, umo: str, text: str
    ) -> tuple[float, list[str], int | None, str]:
        """返回 (概率, 命中词, 打分数, 命中方向名)。"""
        base = float(self._get("base_probability") or 0)
        strong_prob = float(self._get("strong_probability") or 0)
        strong_hits = max(1, int(self._get("keyword_hits_for_strong") or 2))

        best: tuple[float, list[str], str] | None = None
        for group in self._groups():
            hits = self._match_words(text, group["words"])
            if not hits:
                continue
            probability = float(group["weight"]) + 0.12 * (len(hits) - 1)
            if len(hits) >= strong_hits:
                probability = max(probability, strong_prob)
            probability = min(0.97, max(0.0, probability))
            if best is None or probability > best[0]:
                best = (probability, hits, group["name"])
        if best is not None:
            return best[0], best[1], None, best[2]

        # 打分：只在「群里有来回」或「消息够长」时才值得花一次调用
        quiet = len(self._recent[umo]) < 2 and len(text) < 25
        if (
            self._get("judge_enable")
            and len(text) >= int(self._get("judge_min_length") or 0)
            and not quiet
            and self._judge_allowed(umo)
        ):
            cache_seconds = max(0, int(self._get("judge_cache_seconds") or 120))
            cached = self._judge_cache.get(umo)
            score: int | None = None
            if cached and time.time() - cached[0] <= cache_seconds and self._similar(text, cached[2]):
                score = cached[1]
                logger.debug(f"[whale_radar] judge 命中缓存 score={score}")
            if score is None:
                score = await self._judge(umo, text)
                if score is not None:
                    self._judge_cache[umo] = (time.time(), score, text)
            if score is not None and score >= int(self._get("judge_threshold") or 7):
                return float(self._get("judge_probability") or 0), [], score, "（打分）"
            return base, [], score, ""

        return base, [], None, ""

    # ---------------- 请求构造 ----------------
    async def _build_request(self, event: AstrMessageEvent):
        umo = event.unified_msg_origin
        try:
            conversation_id = await self.context.conversation_manager.get_curr_conversation_id(umo)
            if not conversation_id:
                # 新群 / 还没聊过：先建一条会话，不然插话没有上下文可写
                conversation_id = await self.context.conversation_manager.new_conversation(
                    umo,
                    platform_id=event.get_platform_id(),
                    title="群聊",
                )
                logger.info(f"[whale_radar] 新建会话 {conversation_id} umo={umo}")
            conversation = await self.context.conversation_manager.get_conversation(
                umo, conversation_id
            )
            if not conversation:
                logger.info("[whale_radar] 拿不到会话内容，跳过插话")
                return None
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[whale_radar] 读取会话失败: {exc}")
            return None

        image_urls: list[str] = []
        for component in event.get_messages() or []:
            if isinstance(component, Image):
                try:
                    image_urls.append(await component.convert_to_file_path())
                except Exception:  # noqa: BLE001
                    logger.debug("[whale_radar] 图片转换失败，忽略")

        request = event.request_llm(
            prompt=event.message_str,
            session_id=event.session_id,
            image_urls=image_urls or None,
            conversation=conversation,
        )

        size = int(self._get("context_size") or 0)
        records = list(self._history[umo])
        if size > 0 and records:
            # 最后一条是当前消息，避免重复
            history_lines = records[-size - 1 : -1]
            if history_lines:
                request.extra_user_content_parts.append(
                    TextPart(text=GROUP_HISTORY_HEADER + "\n".join(history_lines) + GROUP_HISTORY_FOOTER)
                )

        steer = (self._get("steer_prompt") or "").strip()
        if steer:
            request.extra_user_content_parts.append(
                TextPart(text=f"<system_reminder>{steer}</system_reminder>")
            )
        return request

    def _in_scope(self, event: AstrMessageEvent) -> bool:
        raw = self._get("whitelist") or []
        allow = [str(item).strip() for item in raw if str(item).strip()]
        if not allow:
            return True
        umo = event.unified_msg_origin
        group_id = ""
        try:
            group_id = event.get_group_id() or ""
        except Exception:
            group_id = ""
        return umo in allow or (group_id and group_id in allow)

    # ---------------- 主入口 ----------------
    @filter.event_message_type(
        filter.EventMessageType.GROUP_MESSAGE,
        priority=maxsize - 1,
    )
    async def on_group_message(self, event: AstrMessageEvent):
        if not self._get("enable"):
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        if event.is_at_or_wake_command:
            return
        if event.get_extra("handlers_parsed_params"):
            return
        try:
            if event.get_sender_id() == event.get_self_id():
                return
        except Exception:
            pass
        if not self._in_scope(event):
            return

        umo = event.unified_msg_origin
        text = (event.message_str or "").strip()
        self._remember(umo, event)
        if not text:
            return
        if self._match_words(text, self._negatives()):
            return
        if not self._gate(umo):
            return

        probability, hits, score, direction = await self._decide(umo, text)
        if probability <= 0:
            return
        if random.random() >= probability:
            return

        request = await self._build_request(event)
        if request is None:
            return

        self._mark(umo)
        logger.info(
            f"[whale_radar] 插话 p={probability:.2f} hits={hits} direction={direction} "
            f"judge={score} umo={umo} text={text[:40]!r}"
        )
        yield request

    # ---------------- 调试指令 ----------------
    def _is_operator(self, event: AstrMessageEvent) -> bool:
        try:
            if event.is_admin():
                return True
        except Exception:
            pass
        raw = self._get("operator_uids") or []
        allowed = {str(uid).strip() for uid in raw if str(uid).strip()}
        try:
            return event.get_sender_id() in allowed
        except Exception:
            return False

    def _args(self, event: AstrMessageEvent) -> str:
        text = (event.message_str or "").strip()
        if text.startswith("/"):
            text = text[1:].strip()
        for name in ("小鲸鱼雷达", "雷达"):
            if text.startswith(name):
                return text[len(name) :].strip()
        return text

    def _parse_probability(self, raw: str) -> float | None:
        text = (raw or "").strip()
        if not text:
            return None
        try:
            if text.endswith("%"):
                return max(0.0, min(1.0, float(text[:-1]) / 100.0))
            value = float(text)
        except ValueError:
            return None
        if value > 1.0:
            value = value / 100.0
        return max(0.0, min(1.0, value))

    def _status_text(self, umo: str) -> str:
        day, count = self._daily.get(umo, (self._today(), 0))
        if day != self._today():
            count = 0
        last = self._spoke[umo][-1] if self._spoke[umo] else None
        last_text = time.strftime("%H:%M:%S", time.localtime(last)) if last else "还没插过话"
        groups = self._groups()
        words = sum(len(g["words"]) for g in groups)
        return (
            f"小鲸鱼雷达 v{PLUGIN_VERSION}\n"
            f"启用：{'是' if self._get('enable') else '否'}；方向 {len(groups)} 个 / 词 {words} 个\n"
            f"基础概率：{_fmt_pct(float(self._get('base_probability') or 0))}\n"
            f"单方向命中：按方向权重（默认 {_fmt_pct(float(self._get('keyword_probability') or 0))}）；"
            f"同一方向命中 ≥{int(self._get('keyword_hits_for_strong') or 2)} 词升到 "
            f"{_fmt_pct(float(self._get('strong_probability') or 0))}\n"
            f"打分：{'开' if self._get('judge_enable') else '关'}"
            f"（阈值 {int(self._get('judge_threshold') or 7)}，通过后 "
            f"{_fmt_pct(float(self._get('judge_probability') or 0))}）\n"
            f"冷却 {int(self._get('cooldown_seconds') or 0)} 秒；"
            f"今日已插话 {count}/{int(self._get('daily_limit') or 0)}\n"
            f"上次插话：{last_text}\n"
            "指令：/雷达 帮助"
        )

    def _learn_status_text(self) -> str:
        history = list(self._get("learned_words") or [])
        waiting = sum(int((v or {}).get("pending") or 0) for v in self._learn_state.values())
        return (
            "自动学词\n"
            f"状态：{'开' if self._get('learn_enable') else '关'}\n"
            f"节奏：攒够 {int(self._get('learn_min_messages') or 60)} 条群消息、"
            f"且距上次 ≥{int(self._get('learn_interval_minutes') or 60)} 分钟才分析一次；"
            f"每次最多 {int(self._get('learn_max_per_run') or 2)} 个词\n"
            f"新词权重：{_fmt_pct(float(self._get('learn_weight') or 0.6))}；"
            f"词库上限 {int(self._get('learn_max_total_words') or 160)} 个\n"
            f"通知：{self._get('learn_notify_uid') or '关'}\n"
            f"已学 {len(history)} 个；当前攒着 {waiting} 条待分析\n"
            "指令：/雷达 立刻学 · /雷达 学习记录 · /雷达 忘词 词"
        )

    def _list_groups(self, only: str = "") -> str:
        groups = self._groups()
        if not groups:
            return "还没有任何关键词方向。用 /雷达 加词 股票 涨停 板了 建一个。"
        if only:
            hit = [g for g in groups if only in g["name"]]
            if not hit:
                names = "、".join(g["name"] for g in groups)
                return f"没有叫「{only}」的方向。现有：{names}"
            groups = hit
        lines: list[str] = []
        for group in groups:
            preview = "、".join(group["words"][:12])
            more = f"…（共 {len(group['words'])} 个）" if len(group["words"]) > 12 else ""
            lines.append(f"· {group['name']} {_fmt_pct(group['weight'])}：{preview}{more}")
        return "关键词方向\n" + "\n".join(lines)

    @filter.command("雷达")
    async def radar(self, event: AstrMessageEvent):
        """小鲸鱼雷达调试：改关键词方向、权重、概率、测试判定。"""
        if not self._is_operator(event):
            yield event.plain_result("雷达指令只对主人开放（可在插件配置 operator_uids 里加 QQ 号）。")
            return

        args = self._args(event)
        sub, _, rest = args.partition(" ")
        sub = sub.strip()
        rest = rest.strip()
        umo = event.unified_msg_origin

        # 无参数 / 状态
        if not sub or sub in {"状态", "status"}:
            yield event.plain_result(self._status_text(umo))
            return

        if sub in {"帮助", "help", "?", "？"}:
            yield event.plain_result(HELP_TEXT)
            return

        if sub in {"开", "on", "启用"}:
            yield event.plain_result(f"雷达已开启（{self._save({'enable': True})}）")
            return

        if sub in {"关", "off", "禁用"}:
            yield event.plain_result(f"雷达已关闭（{self._save({'enable': False})}）")
            return

        # 词 / 列表
        if sub in {"词", "列表", "ls", "list"}:
            yield event.plain_result(self._list_groups(rest))
            return

        # 加词 <方向> <词...>
        if sub in {"加词", "add"}:
            head, _, tail = rest.partition(" ")
            direction = head.strip()
            words = _split_words(tail)
            if not direction or not words:
                yield event.plain_result("用法：/雷达 加词 <方向> <词1> <词2> …\n例：/雷达 加词 股票 涨停 板了 套牢")
                return
            groups = self._groups()
            target = next((g for g in groups if g["name"] == direction), None)
            created = False
            if target is None:
                target = {
                    "name": direction,
                    "weight": float(self._get("keyword_probability") or 0.55),
                    "words": [],
                }
                groups.append(target)
                created = True
            added: list[str] = []
            for word in words:
                if word not in target["words"]:
                    target["words"].append(word)
                    added.append(word)
            if not added:
                yield event.plain_result(f"「{direction}」里已经有这些词了，没改动。")
                return
            result = self._save({"keyword_groups": groups})
            verb = "新建方向并加入" if created else "加入"
            yield event.plain_result(
                f"{verb} {len(added)} 个词到「{direction}」：{'、'.join(added)}\n"
                f"方向权重 {_fmt_pct(target['weight'])}（{result}）"
            )
            return

        # 删词 <词...>
        if sub in {"删词", "rm", "del"}:
            words = _split_words(rest)
            if not words:
                yield event.plain_result("用法：/雷达 删词 <词1> <词2> …")
                return
            groups = self._groups()
            removed: list[str] = []
            for group in groups:
                keep = []
                for word in group["words"]:
                    if word in words:
                        removed.append(f"{group['name']}/{word}")
                    else:
                        keep.append(word)
                group["words"] = keep
            groups = [g for g in groups if g["words"]]
            if not removed:
                yield event.plain_result("这些词不在任何方向里，没改动。")
                return
            result = self._save({"keyword_groups": groups})
            yield event.plain_result(f"删掉：{'、'.join(removed)}（{result}）")
            return

        # 学 …（自动学新爱好）
        if sub in {"学", "学习", "learn"}:
            head, _, tail = rest.partition(" ")
            key = head.strip()
            if not key or key in {"状态", "status"}:
                yield event.plain_result(self._learn_status_text())
                return
            if key in {"开", "on"}:
                yield event.plain_result(f"自动学词已开启（{self._save({'learn_enable': True})}）")
                return
            if key in {"关", "off"}:
                yield event.plain_result(f"自动学词已关闭（{self._save({'learn_enable': False})}）")
                return
            if key in {"间隔", "词数", "条数", "权重"}:
                if key == "权重":
                    value = self._parse_probability(tail)
                    if value is None:
                        yield event.plain_result("用法：/雷达 学 权重 60%")
                        return
                    yield event.plain_result(f"新方向权重改为 {_fmt_pct(value)}（{self._save({'learn_weight': value})}）")
                    return
                try:
                    number = int(tail.strip())
                except ValueError:
                    yield event.plain_result("用法：/雷达 学 间隔 60（分钟） / 雷达 学 词数 2 / 雷达 学 条数 60")
                    return
                if key == "间隔":
                    value = max(5, min(1440, number))
                    yield event.plain_result(f"学习间隔改为 {value} 分钟（{self._save({'learn_interval_minutes': value})}）")
                elif key == "词数":
                    value = max(1, min(5, number))
                    yield event.plain_result(f"每次最多学 {value} 个词（{self._save({'learn_max_per_run': value})}）")
                else:
                    value = max(5, min(500, number))
                    yield event.plain_result(f"够 {value} 条群消息才分析一次（{self._save({'learn_min_messages': value})}）")
                return
            if key in {"通知", "notify"}:
                uid = tail.strip()
                if uid and not uid.isdigit():
                    yield event.plain_result("用法：/雷达 学 通知 <你的QQ号>（留空则不通知）")
                    return
                yield event.plain_result(
                    f"学词通知{'发给 ' + uid if uid else '已关闭'}（{self._save({'learn_notify_uid': uid})}）"
                )
                return
            yield event.plain_result(
                "用法：/雷达 学 开|关\n"
                "/雷达 学 间隔 60 / 学 词数 2 / 学 条数 60 / 学 权重 60%\n"
                "/雷达 学 通知 <QQ号>（学到新词私聊告诉你）"
            )
            return

        if sub in {"学习记录", "记录", "learned"}:
            history = list(self._get("learned_words") or [])
            if not history:
                yield event.plain_result("还没学到过词。可以用 /雷达 立刻学 试试。")
                return
            count = 10
            if rest.strip().isdigit():
                count = max(1, min(50, int(rest.strip())))
            lines = [
                f"· {item.get('word')} → {item.get('direction')}（{item.get('at', '')} {item.get('reason', '')}）".replace("（）", "")
                for item in history[-count:][::-1]
            ]
            yield event.plain_result(f"最近学到的词（共 {len(history)} 条）\n" + "\n".join(lines))
            return

        if sub in {"立刻学", "学一下", "learn-now"}:
            yield event.plain_result("在看了，稍等…")
            report = await self._learn_once(umo=umo, force=True)
            yield event.plain_result(report)
            return

        if sub in {"忘词", "忘掉"}:
            words = _split_words(rest)
            if not words:
                yield event.plain_result("用法：/雷达 忘词 <词1> <词2>…")
                return
            groups = self._groups()
            removed: list[str] = []
            for group in groups:
                keep = [w for w in group["words"] if w not in words or removed.append(f"{group['name']}/{w}")]
                group["words"] = keep
            groups = [g for g in groups if g["words"]]
            history = [item for item in (self._get("learned_words") or []) if item.get("word") not in words]
            if not removed:
                yield event.plain_result("这些词不在词库里，没改动。")
                return
            result = self._save({"keyword_groups": groups, "learned_words": history})
            yield event.plain_result(f"忘掉：{'、'.join(removed)}（{result}）")
            return

        # 新方向 <名称> [权重]
        if sub in {"新方向", "新增方向", "new"}:
            head, _, tail = rest.partition(" ")
            direction = head.strip()
            if not direction:
                yield event.plain_result("用法：/雷达 新方向 <名称> [权重%]")
                return
            weight = self._parse_probability(tail)
            if weight is None:
                weight = float(self._get("keyword_probability") or 0.55)
            groups = self._groups()
            if any(g["name"] == direction for g in groups):
                yield event.plain_result(f"「{direction}」已存在，用 /雷达 词 {direction} 看词，或 /雷达 方向 {direction} 80% 改权重。")
                return
            groups.append({"name": direction, "weight": weight, "words": []})
            result = self._save({"keyword_groups": groups})
            yield event.plain_result(
                f"新建方向「{direction}」，权重 {_fmt_pct(weight)}（{result}）\n"
                f"接着加词：/雷达 加词 {direction} 词1 词2"
            )
            return

        # 方向 <名称> <权重>
        if sub == "方向":
            head, _, tail = rest.partition(" ")
            direction = head.strip()
            groups = self._groups()
            target = next((g for g in groups if g["name"] == direction), None)
            if not direction:
                yield event.plain_result(self._list_groups())
                return
            if target is None:
                yield event.plain_result(f"没有叫「{direction}」的方向。先 /雷达 新方向 {direction} 80%")
                return
            weight = self._parse_probability(tail)
            if weight is None:
                yield event.plain_result(f"「{direction}」当前权重 {_fmt_pct(target['weight'])}；用法：/雷达 方向 {direction} 80%")
                return
            target["weight"] = weight
            result = self._save({"keyword_groups": groups})
            yield event.plain_result(f"「{direction}」权重改为 {_fmt_pct(weight)}（{result}）")
            return

        # 删方向 <名称>
        if sub in {"删方向", "删组"}:
            direction = rest.strip()
            if not direction:
                yield event.plain_result("用法：/雷达 删方向 <名称>")
                return
            groups = self._groups()
            keep = [g for g in groups if g["name"] != direction]
            if len(keep) == len(groups):
                yield event.plain_result(f"没有叫「{direction}」的方向。")
                return
            result = self._save({"keyword_groups": keep})
            yield event.plain_result(f"已删方向「{direction}」（{result}）")
            return

        # 设 <键> <值>
        if sub in {"设", "set"}:
            head, _, tail = rest.partition(" ")
            key = SET_ALIASES.get(head.strip(), "")
            value_text = tail.strip()
            if not key:
                yield event.plain_result(
                    "可改：基础 关键词 强命中 强词数 打分 阈值 冷却 上限 上下文 打分开关\n"
                    "例：/雷达 设 基础 3%"
                )
                return
            if key == "judge_enable":
                lowered = value_text.lower()
                if lowered in TRUE_WORDS:
                    value = True
                elif lowered in FALSE_WORDS:
                    value = False
                else:
                    yield event.plain_result("打分开关只认 开 / 关")
                    return
            elif key in PROBABILITY_KEYS:
                parsed = self._parse_probability(value_text)
                if parsed is None:
                    yield event.plain_result("概率写成 3% 或 0.03 都行。")
                    return
                value = parsed
            else:
                try:
                    value = int(float(value_text))
                except ValueError:
                    yield event.plain_result("这个参数要整数，比如 120。")
                    return
                if value < 0:
                    yield event.plain_result("不能是负数。")
                    return
            if key == "keyword_hits_for_strong" and value < 1:
                value = 1
            updates = {key: value}
            result = self._save(updates)
            shown = _fmt_pct(value) if key in PROBABILITY_KEYS else value
            yield event.plain_result(f"已设置 {head.strip()} = {shown}（{result}）")
            return

        # 导出
        if sub in {"导出", "export", "json"}:
            payload = json.dumps(self._groups(), ensure_ascii=False)
            if len(payload) > 1500:
                payload = payload[:1500] + "…（太长，去 WebUI 配置里看全量）"
            yield event.plain_result(payload)
            return

        # 测试
        if sub in {"测试", "test"}:
            sample = rest.strip()
            if not sample:
                yield event.plain_result("要给一句话，比如：/雷达 测试 今天A股又跌了")
                return
            probability, hits, score, direction = await self._decide(umo, sample)
            yield event.plain_result(
                f"「{sample}」\n"
                f"命中：{'、'.join(hits) if hits else '无'}"
                f"{f'（方向：{direction}）' if direction else ''}\n"
                f"语义打分：{score if score is not None else '未打分'}\n"
                f"插话概率：{_fmt_pct(probability)}\n"
                f"冷却放行：{'是' if self._gate(umo) else '否（冷却中或到上限）'}"
            )
            return

        yield event.plain_result(f"不认识的子指令「{sub}」。发 /雷达 帮助 看用法。")

    @filter.llm_tool(name="remember_interest")
    async def remember_interest(
        self, event: AstrMessageEvent, word: str, direction: str = "", reason: str = ""
    ):
        """当你发现自己对聊天里提到的某个东西真的产生了兴趣（想以后多聊、多了解），把它记进「小鲸鱼雷达」的关键词，这样以后群里聊到它时你会更容易接话。

        Args:
            word(string): 感兴趣的具体名词，2-6 个字，例如「显卡」「露营」「赛博朋克」
            direction(string): 想归到哪个方向（股票/游戏/吃饭/熬夜上班/硬件代码/二次元/猫狗鲸鱼），不确定就留空
            reason(string): 为什么感兴趣，10 个字以内，例如「想攒钱买一块」
        """
        ok, message = self._add_word(word, direction, reason=reason)
        if not ok:
            return f"没记：{message}"
        return f"记住了：{message}"

    async def terminate(self):
        """插件卸载/重载时清空缓存。"""
        if self._learn_task and not self._learn_task.done():
            self._learn_task.cancel()
        try:
            self._flush_learn_state()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[whale_radar] 卸载时保存学习进度失败: {exc}")
        self._history.clear()
        self._spoke.clear()
        self._judged.clear()
        self._daily.clear()
