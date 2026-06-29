#!/usr/bin/env python3
"""Build the curated price-action knowledge layer used by the research agent."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "user_data/strategy_research"
KNOWLEDGE_ROOT = AGENT_ROOT / "knowledge"
BILIBILI_ROOT = KNOWLEDGE_ROOT / "raw_sources/bilibili"
TRANSCRIPTS_DIR = BILIBILI_ROOT / "transcripts"
TRANSCRIPT_REPORT = BILIBILI_ROOT / "bilibili_transcript_fetch_report.json"
WEB_SOURCES = KNOWLEDGE_ROOT / "raw_sources/public_web_sources_manifest.json"
NORMALIZED_DIR = KNOWLEDGE_ROOT / "normalized_transcripts"
CLAIMS_DIR = KNOWLEDGE_ROOT / "extracted_claims"
CARDS_DIR = KNOWLEDGE_ROOT / "knowledge_cards"
QUARANTINED_CARDS_DIR = KNOWLEDGE_ROOT / "knowledge_cards_quarantined"
INDEX_DIR = KNOWLEDGE_ROOT / "index"
QUALITY_JSON = INDEX_DIR / "transcript_quality_report.json"
QUALITY_MD = INDEX_DIR / "transcript_quality_report.md"
CLAIMS_JSON = CLAIMS_DIR / "price_action_claims.json"
INDEX_JSON = INDEX_DIR / "price_action_knowledge_index.json"
LAYER_REPORT_MD = KNOWLEDGE_ROOT / "latest_price_action_knowledge_layer_report.md"
LAYER_REPORT_JSON = KNOWLEDGE_ROOT / "latest_price_action_knowledge_layer_report.json"
REPORT_ALIAS_MD = KNOWLEDGE_ROOT / "latest_price_action_knowledge_report.md"
REPORT_ALIAS_JSON = KNOWLEDGE_ROOT / "latest_price_action_knowledge_report.json"


TRADING_KEYWORDS = {
    "交易", "k线", "K线", "信号", "入场", "出场", "止损", "止盈", "突破", "回调", "反转", "趋势",
    "震荡", "市场周期", "订单", "限价", "stop", "Stop", "limit", "Limit", "风险", "盈亏", "概率",
    "头寸", "仓位", "scalp", "Scalp", "剥头皮", "复盘", "早盘", "日内", "缺口", "楔形", "通道",
    "BTC", "比特币", "外汇", "黄金", "原油", "价格", "支撑", "阻力", "形态", "做多", "做空",
}

OFFTOPIC_KEYWORDS = {
    "火锅", "糍粑", "豆豉", "贵阳", "好吃", "美食", "芝麻", "花生", "BLG", "WBG", "LNG",
    "T1", "LPL", "八强", "四强", "韩华", "纳尔", "武器", "比赛", "女主播", "游戏", "电竞",
}

WEB_SOURCE_REFS = {
    "brooks_trading_course_books",
    "brooks_trading_course_home",
    "investopedia_price_action_intro",
    "investopedia_price_action_definition",
    "coinmarketcap_support_resistance_zones",
    "kraken_technical_analysis_intro",
    "phemex_crypto_price_action",
}

WEB_SOURCE_GROUPS = {
    "brooks": ["brooks_trading_course_books", "brooks_trading_course_home"],
    "price_action": ["investopedia_price_action_intro", "phemex_crypto_price_action"],
    "crypto": ["phemex_crypto_price_action", "kraken_technical_analysis_intro"],
    "support_resistance": ["coinmarketcap_support_resistance_zones", "kraken_technical_analysis_intro"],
}

BOOK_SOURCE_REFS = {
    "book_xu_jiacong_meta": {
        "title": "裸K线交易法--价格行为(Price Action)全面详解",
        "author": "许佳聪",
        "local_pdf": "user_data/strategy_research/knowledge/raw_sources/books_to_add/price_action_xu_jiacong_local_research.pdf",
        "copyright_note": "User-confirmed local personal research material; do not commit PDF or copy long text.",
    },
    "book_xu_jiacong_p047": {"page": 47, "topic": "交易系统必须是完整解决方案"},
    "book_xu_jiacong_p052": {"page": 52, "topic": "入场点影响止损、仓位和利润空间"},
    "book_xu_jiacong_p056": {"page": 56, "topic": "交易系统要匹配自控力、风险承受和周期"},
    "book_xu_jiacong_p060": {"page": 60, "topic": "Pinbar 定义与主影线比例"},
    "book_xu_jiacong_p062": {"page": 62, "topic": "Pinbar 不是单独系统，必须有配套条件"},
    "book_xu_jiacong_p076": {"page": 76, "topic": "关键位是裸K线交易核心判断准则"},
    "book_xu_jiacong_p080": {"page": 80, "topic": "拐点与关键位反映多空力量转换"},
    "book_xu_jiacong_p084": {"page": 84, "topic": "关键水平位是系统重点"},
    "book_xu_jiacong_p100": {"page": 100, "topic": "假突破暴露反向意图"},
    "book_xu_jiacong_p105": {"page": 105, "topic": "信号研判与最小盈亏比"},
    "book_xu_jiacong_p108": {"page": 108, "topic": "突破入场与 50% 回调止损"},
    "book_xu_jiacong_p199": {"page": 199, "topic": "金字塔加仓只适合强单边且不能逆势摊薄"},
    "book_xu_jiacong_p220": {"page": 220, "topic": "复盘要关注信号、入场止损、发力、周期"},
    "book_xu_jiacong_p221": {"page": 221, "topic": "信号风险超过承受范围时等待回调或降仓"},
    "book_xu_jiacong_p224": {"page": 224, "topic": "交易心理和纪律决定系统执行"},
    "book_xu_jiacong_p225": {"page": 225, "topic": "失败交易者常见行为"},
    "book_xu_jiacong_p226": {"page": 226, "topic": "直觉交易会破坏计划"},
}


CARD_SPECS: list[dict[str, Any]] = [
    {
        "id": "pa_signal_bar_context_entry",
        "title": "信号K线必须和背景一起判断",
        "category": "entry",
        "concepts": ["signal_bar", "context", "entry_confirmation"],
        "source_terms": ["信号K", "背景", "入场"],
        "strategy_family": "entry_confirmation",
        "knowledge": "单根K线不是交易系统。先判断背景，再把信号K线当作入场触发器。",
        "hypothesis": "把方向过滤和入场触发拆开，只有背景、信号、下一根确认同时成立才允许开仓。",
        "features": ["higher_tf_regime", "signal_bar_body_ratio", "next_candle_confirmation"],
        "entry_rules": ["背景方向明确", "信号K线出现", "下一根K线确认"],
        "exit_rules": ["确认失败或重新回到区间内时退出"],
        "avoid": ["只因单根K线形态直接进场"],
        "risk": ["高杠杆下信号K低点/高点不能机械当极近止损。"],
    },
    {
        "id": "pa_systematic_vs_discretionary_rules",
        "title": "主观交易思想必须落成系统规则",
        "category": "training",
        "concepts": ["systematic_trading", "discretionary_trading", "rule_translation"],
        "source_terms": ["系统交易", "主观交易"],
        "strategy_family": "research_process",
        "knowledge": "价格行为可以来自主观观察，但进入 Agent 后必须变成可回测的条件和失效规则。",
        "hypothesis": "每个主观形态只允许转成最多三个入场条件，并强制生成不适用场景。",
        "features": ["condition_count", "regime_filter", "invalidation_rule"],
        "entry_rules": ["每个形态最多三个确认条件"],
        "exit_rules": ["失效规则必须先于回测定义"],
        "avoid": ["边看图边解释，事后补理由"],
        "risk": ["复杂规则更容易过拟合。"],
    },
    {
        "id": "pa_always_in_long_chase_filter",
        "title": "追涨必须有趋势持续和位置过滤",
        "category": "entry",
        "concepts": ["always_in", "chase_filter", "trend_continuation"],
        "source_terms": ["追涨", "Always In"],
        "strategy_family": "trend_following",
        "knowledge": "趋势里可以追随强势，但离均线太远或连续大K后追入会恶化盈亏比。",
        "hypothesis": "测试强趋势追随，但只在回调后恢复或突破后跟进时进场。",
        "features": ["ema_distance", "trend_strength", "followthrough_return"],
        "entry_rules": ["趋势强", "价格不过度远离均线", "出现跟进K线"],
        "exit_rules": ["跟进失败或强度衰减退出"],
        "avoid": ["连续大K末端追入"],
        "risk": ["追涨策略必须承受假突破和滑点压力。"],
    },
    {
        "id": "pa_stop_order_breakout_intent",
        "title": "Stop Order 对应突破确认意图",
        "category": "entry",
        "concepts": ["stop_order", "breakout_entry", "momentum_confirmation"],
        "source_terms": ["Stop Order", "订单类型"],
        "strategy_family": "breakout",
        "knowledge": "Stop order 更接近突破确认和动量追随，不等同于回调挂单。",
        "hypothesis": "突破策略使用收盘突破和跟进确认模拟 stop entry 的交易意图。",
        "features": ["break_distance", "followthrough_return", "volume_zscore"],
        "entry_rules": ["收盘突破关键区域", "下一根不回到区域内"],
        "exit_rules": ["回到突破区域内时退出"],
        "avoid": ["把 wick-only 刺破当突破"],
        "risk": ["突破单对滑点敏感，需要费用压力测试。"],
    },
    {
        "id": "pa_limit_order_pullback_intent",
        "title": "Limit Order 对应回调和区域反应",
        "category": "entry",
        "concepts": ["limit_order", "pullback", "zone_reaction"],
        "source_terms": ["限价单", "Limit"],
        "strategy_family": "pullback",
        "knowledge": "Limit order 思想更适合回调、区间边界和均值回归，核心是位置而不是追随。",
        "hypothesis": "测试价格回到支撑/阻力区域后的拒绝反应，再用恢复K线确认。",
        "features": ["zone_distance_atr", "rejection_wick_ratio", "resume_candle"],
        "entry_rules": ["进入关键区域", "出现拒绝反应", "下一根恢复"],
        "exit_rules": ["区域被有效跌破/突破时退出"],
        "avoid": ["一触线就入场"],
        "risk": ["区域宽度要用 ATR 表达，避免被噪音扫掉。"],
    },
    {
        "id": "pa_market_cycle_strategy_router",
        "title": "市场周期决定策略族",
        "category": "definition",
        "concepts": ["market_cycle", "trend", "range", "regime_router"],
        "source_terms": ["市场周期", "Market Cycle"],
        "strategy_family": "regime_filter",
        "knowledge": "趋势、震荡、过渡段不能用同一套入场逻辑。先分行情，再选策略族。",
        "hypothesis": "构建 regime router：趋势用回调/突破，震荡用边界反转，过渡段降频。",
        "features": ["adx", "ema_spread", "range_width", "atr_percentile"],
        "entry_rules": ["先判定 regime", "只启用匹配策略族"],
        "exit_rules": ["regime 翻转时退出或禁开新仓"],
        "avoid": ["同一套策略同时吃趋势和震荡"],
        "risk": ["行情切换期是短周期策略回撤高发区。"],
    },
    {
        "id": "pa_pullback_count_bars_trend",
        "title": "顺势回调要数K线和等待恢复",
        "category": "entry",
        "concepts": ["pullback", "bar_count", "trend_resume"],
        "source_terms": ["回调", "数k线", "顺势"],
        "strategy_family": "trend_pullback",
        "knowledge": "顺势交易的关键不是看到趋势就进，而是在回调结构结束后等待恢复。",
        "hypothesis": "用回调K线数量、回调深度和恢复K线构造趋势回调策略。",
        "features": ["pullback_bar_count", "pullback_depth_atr", "resume_candle"],
        "entry_rules": ["趋势方向明确", "回调不过深", "恢复K线出现"],
        "exit_rules": ["恢复失败或回调低点/高点失守退出"],
        "avoid": ["趋势刚出现就追单"],
        "risk": ["回调过深可能不是回调而是反转。"],
    },
    {
        "id": "pa_countertrend_exit_pressure",
        "title": "逆势交易者离场会推动顺势恢复",
        "category": "pattern",
        "concepts": ["countertrend_exit", "trend_resume", "trap"],
        "source_terms": ["逆势交易者离场", "回调"],
        "strategy_family": "trend_pullback",
        "knowledge": "回调失败时，逆势交易者离场会形成顺势恢复的附加推动。",
        "hypothesis": "在趋势回调末端测试反向失败后的顺势入场，而不是直接抄底摸顶。",
        "features": ["failed_countertrend_move", "resume_momentum", "trend_slope"],
        "entry_rules": ["逆势尝试失败", "顺势恢复", "趋势过滤通过"],
        "exit_rules": ["恢复动量没有延续时退出"],
        "avoid": ["在强趋势里做逆势反转"],
        "risk": ["必须避免把小回调误判成大反转。"],
    },
    {
        "id": "pa_measured_move_target",
        "title": "Measured Move 给止盈目标提供尺度",
        "category": "exit",
        "concepts": ["measured_move", "take_profit", "target"],
        "source_terms": ["Measured Move", "止盈目标"],
        "strategy_family": "exit_design",
        "knowledge": "止盈目标不应只用固定百分比，也可以用前一段波动或结构宽度推算。",
        "hypothesis": "测试结构尺度止盈：突破区间宽度、前一腿长度、ATR 多倍数。",
        "features": ["range_height", "prior_leg_size", "atr"],
        "entry_rules": ["入场仍由形态确认决定"],
        "exit_rules": ["到达 measured move 或动量衰减分批退出"],
        "avoid": ["所有行情使用同一个固定 ROI"],
        "risk": ["目标过远会降低胜率，需和实际风险一起评估。"],
    },
    {
        "id": "pa_actual_risk_reward",
        "title": "实际风险比理论止损更重要",
        "category": "risk",
        "concepts": ["actual_risk", "reward_risk", "stoploss"],
        "source_terms": ["实际风险", "实际盈亏比", "Actual Risk"],
        "strategy_family": "risk_control",
        "knowledge": "真实交易风险往往大于图上理论风险，包括滑点、延迟和入场后不利移动。",
        "hypothesis": "每个形态实验都输出预期价格风险、杠杆后风险和费用压力后的盈亏比。",
        "features": ["expected_adverse_move", "fee_stress", "slippage_bps"],
        "entry_rules": ["预期收益必须覆盖费用和滑点"],
        "exit_rules": ["实际风险超过预设阈值时拒绝入场或退出"],
        "avoid": ["只看裸价格止损距离"],
        "risk": ["50x 下 1% 价格反向约等于 50% 杠杆后亏损。"],
    },
    {
        "id": "pa_false_breakout_reversal",
        "title": "假突破是反向交易种子",
        "category": "pattern",
        "concepts": ["false_breakout", "trap", "reversal"],
        "source_terms": ["假突破", "突破专题"],
        "strategy_family": "failed_breakout",
        "knowledge": "关键位突破后快速收回，说明追突破的一方被套，可能触发反向移动。",
        "hypothesis": "测试刺破关键位、收回区间、反向确认的三条件反转策略。",
        "features": ["wick_outside_range", "close_back_inside", "reversal_followthrough"],
        "entry_rules": ["刺破关键位", "收回区间", "反向确认"],
        "exit_rules": ["再次突破失败区域时退出"],
        "avoid": ["强趋势中硬做反转"],
        "risk": ["假突破策略需要趋势强度过滤。"],
    },
    {
        "id": "pa_true_breakout_followthrough",
        "title": "真突破需要后续跟进",
        "category": "pattern",
        "concepts": ["true_breakout", "follow_through", "momentum"],
        "source_terms": ["真突破", "突破专题"],
        "strategy_family": "breakout",
        "knowledge": "真突破通常不只是刺破，而是收盘站上/跌破关键区后继续有跟进。",
        "hypothesis": "测试收盘突破加下一根跟进，过滤 wick-only 的假突破。",
        "features": ["close_break_distance", "next_bar_direction", "volume_zscore"],
        "entry_rules": ["收盘突破", "下一根跟进", "成交量不萎缩"],
        "exit_rules": ["回到突破区间内退出"],
        "avoid": ["突破后立刻反抽回区间仍追入"],
        "risk": ["突破后滑点会显著影响短周期收益。"],
    },
    {
        "id": "pa_second_leg_trap",
        "title": "第二段陷阱用于识别追随失败",
        "category": "pattern",
        "concepts": ["second_leg", "trap", "failed_continuation"],
        "source_terms": ["第二段陷阱", "2nd Leg Trap"],
        "strategy_family": "trap_reversal",
        "knowledge": "第二段延续失败时，追随者再次被套，容易形成反向移动。",
        "hypothesis": "测试两段推进后动量衰减和收回关键位的反向策略。",
        "features": ["leg_count", "leg2_momentum", "close_back_inside"],
        "entry_rules": ["第二段推进失败", "反向确认", "位置接近关键区"],
        "exit_rules": ["反向跟进失败时退出"],
        "avoid": ["在第二段仍加速时提前反向"],
        "risk": ["需要避免把正常趋势第二腿误判为陷阱。"],
    },
    {
        "id": "pa_breakout_test_retest",
        "title": "突破回测比瞬间突破更适合作确认",
        "category": "entry",
        "concepts": ["breakout_test", "retest", "confirmation"],
        "source_terms": ["突破回测", "Breakout Test"],
        "strategy_family": "breakout_retest",
        "knowledge": "突破后回测不破，可以把原阻力/支撑转为确认区域。",
        "hypothesis": "测试突破、回踩、拒绝回到区间内后的二次入场。",
        "features": ["breakout_level", "retest_distance", "rejection_candle"],
        "entry_rules": ["先突破", "回测关键位", "拒绝回区间"],
        "exit_rules": ["回测失败并收回区间退出"],
        "avoid": ["刚刺破关键位立即追单"],
        "risk": ["回测确认会减少交易次数，但可改善位置。"],
    },
    {
        "id": "pa_surprise_bar_volatility",
        "title": "惊喜K线代表波动和预期差",
        "category": "pattern",
        "concepts": ["surprise_bar", "volatility_expansion", "expectation_shift"],
        "source_terms": ["Surprise Bars", "惊喜K线"],
        "strategy_family": "momentum",
        "knowledge": "异常大K线可能代表预期差，但在短周期也可能只是噪音或流动性冲击。",
        "hypothesis": "测试惊喜K线后是否需要等待回调/跟进，而不是直接追第一根。",
        "features": ["body_zscore", "range_zscore", "followthrough_return"],
        "entry_rules": ["大K线出现", "下一根跟进或回调不深"],
        "exit_rules": ["没有跟进时快速退出"],
        "avoid": ["大K线末端无条件追入"],
        "risk": ["高波动时滑点和止损扩大。"],
    },
    {
        "id": "pa_gap_strength",
        "title": "缺口/实体缺口用于判断趋势强弱",
        "category": "pattern",
        "concepts": ["gap", "body_gap", "trend_strength"],
        "source_terms": ["缺口", "跳空", "实体"],
        "strategy_family": "trend_strength",
        "knowledge": "缺口或实体缺口可以表达买卖双方力量不平衡，适合作为趋势强度信号。",
        "hypothesis": "在加密货币中用实体间隔、快速单边推进和流动性真空替代传统跳空。",
        "features": ["body_gap_proxy", "one_way_range", "atr_expansion"],
        "entry_rules": ["强度信号出现", "回调不填补缺口代理", "顺势恢复"],
        "exit_rules": ["强度信号被完全回补时退出"],
        "avoid": ["把交易所孤立异常价当趋势强度"],
        "risk": ["加密货币没有传统开盘缺口，必须用代理特征验证。"],
    },
    {
        "id": "pa_gap_application_review",
        "title": "缺口应用需要复盘验证",
        "category": "training",
        "concepts": ["gap_application", "review", "case_study"],
        "source_terms": ["缺口的应用", "复盘"],
        "strategy_family": "research_process",
        "knowledge": "形态知识必须通过复盘和多样本验证，不应凭单个案例固化。",
        "hypothesis": "对缺口/强度代理做多窗口回测，保留失败样本作为反例。",
        "features": ["case_count", "window_return", "failure_mode"],
        "entry_rules": ["形态必须跨窗口重复出现"],
        "exit_rules": ["失败模式重复时下架该假设"],
        "avoid": ["只选盈利案例复盘"],
        "risk": ["案例型知识最容易幸存者偏差。"],
    },
    {
        "id": "pa_wedge_reversal",
        "title": "楔形反转需要多次推进衰竭",
        "category": "pattern",
        "concepts": ["wedge", "reversal", "exhaustion"],
        "source_terms": ["Wedge", "楔形反转"],
        "strategy_family": "reversal",
        "knowledge": "楔形反转强调多次推进但动量衰竭，不是随便三段就反向。",
        "hypothesis": "测试三次推进、波动收敛、反向突破小结构的反转策略。",
        "features": ["push_count", "momentum_divergence", "range_contraction"],
        "entry_rules": ["三次推进", "动量衰竭", "反向确认"],
        "exit_rules": ["反向突破失败退出"],
        "avoid": ["强趋势加速期猜顶摸底"],
        "risk": ["反转策略需要更小仓位和更严格失效。"],
    },
    {
        "id": "pa_parabolic_reversal",
        "title": "抛物线反转来自过度加速后的失衡",
        "category": "pattern",
        "concepts": ["parabolic", "climax", "reversal"],
        "source_terms": ["Parabolic", "抛物线反转"],
        "strategy_family": "reversal",
        "knowledge": "抛物线走势可能在末端反转，但必须等加速失败或结构跌破。",
        "hypothesis": "测试连续加速后首个结构破坏，而不是提前猜末端。",
        "features": ["acceleration_score", "climax_range", "structure_break"],
        "entry_rules": ["过度加速", "结构破坏", "反向跟进"],
        "exit_rules": ["重新创新高/低时退出"],
        "avoid": ["加速中裸反向"],
        "risk": ["高杠杆下逆加速方向极危险。"],
    },
    {
        "id": "pa_endless_pullback_reversal",
        "title": "无尽回调可能是趋势转弱信号",
        "category": "pattern",
        "concepts": ["endless_pullback", "trend_weakness", "reversal"],
        "source_terms": ["无尽的回调", "反转"],
        "strategy_family": "reversal_or_abstention",
        "knowledge": "回调持续过久且恢复失败，说明原趋势可能转弱。",
        "hypothesis": "测试回调时间过长和恢复失败作为禁开顺势或小反向信号。",
        "features": ["pullback_duration", "failed_resume_count", "ema_flattening"],
        "entry_rules": ["顺势恢复多次失败", "趋势斜率走平", "反向确认"],
        "exit_rules": ["重新恢复原趋势时退出"],
        "avoid": ["把所有长回调都当反转"],
        "risk": ["更适合作为 abstention/filter，而非直接反手。"],
    },
    {
        "id": "pa_wide_channel_intraday_reversal",
        "title": "宽通道中的反转要等边界和失败恢复",
        "category": "pattern",
        "concepts": ["wide_channel", "intraday_reversal", "range_boundary"],
        "source_terms": ["宽通道", "大反转", "日内"],
        "strategy_family": "range_reversal",
        "knowledge": "宽通道里趋势和震荡交替，边界附近的失败延续更有意义。",
        "hypothesis": "测试宽通道边界、失败突破和反向恢复组合。",
        "features": ["channel_width", "boundary_touch", "failed_followthrough"],
        "entry_rules": ["触及通道边界", "延续失败", "反向恢复"],
        "exit_rules": ["回到通道中轴或反向失败退出"],
        "avoid": ["通道中间开反转单"],
        "risk": ["通道策略对交易成本敏感。"],
    },
    {
        "id": "pa_trade_mindset_gambling_vs_edge",
        "title": "交易不是猜方向，必须定义 edge",
        "category": "psychology",
        "concepts": ["edge", "probability", "discipline"],
        "source_terms": ["赌博", "交易理念"],
        "strategy_family": "research_process",
        "knowledge": "交易必须围绕可重复 edge，而不是单次看对方向。",
        "hypothesis": "Agent 输出策略时必须说明 edge 来源、适用样本和反例。",
        "features": ["edge_claim", "sample_size", "counterexample_count"],
        "entry_rules": ["无 edge 说明则不生成策略"],
        "exit_rules": ["回测无法证明 edge 时拒绝晋级"],
        "avoid": ["盈利一次就认为策略有效"],
        "risk": ["没有 edge 的高杠杆只是放大随机性。"],
    },
    {
        "id": "pa_counterparty_multiple_explanations",
        "title": "对手盘和多解性要求反证",
        "category": "definition",
        "concepts": ["counterparty", "multiple_explanations", "falsification"],
        "source_terms": ["对手盘", "多解性"],
        "strategy_family": "research_process",
        "knowledge": "同一走势可有多种解释，策略研究必须保留反证路径。",
        "hypothesis": "每个知识生成假设同时生成反向解释和拒绝条件。",
        "features": ["alternative_explanation", "invalidation_rule", "regime_tag"],
        "entry_rules": ["主假设和反假设都被记录"],
        "exit_rules": ["反证触发时停止该策略族"],
        "avoid": ["单一路径解释所有行情"],
        "risk": ["解释越漂亮越要防止过拟合。"],
    },
    {
        "id": "pa_correct_stoploss_mindset",
        "title": "止损是策略失效而不是情绪惩罚",
        "category": "risk",
        "concepts": ["stoploss", "invalidation", "loss_acceptance"],
        "source_terms": ["止损", "交易理念"],
        "strategy_family": "risk_control",
        "knowledge": "正确止损应对应策略失效点，不能因为不想亏而随意挪动。",
        "hypothesis": "所有策略卡必须定义结构失效和账户风险失效两个层级。",
        "features": ["structure_invalidation", "account_risk_limit", "loss_cluster"],
        "entry_rules": ["入场前定义失效点"],
        "exit_rules": ["结构失效或账户风险触发时退出"],
        "avoid": ["亏损后扩大止损"],
        "risk": ["连续亏损后必须有冷却。"],
    },
    {
        "id": "pa_return_is_result_not_target",
        "title": "收益率是结果，不是入场理由",
        "category": "risk",
        "concepts": ["return_target", "process", "risk_first"],
        "source_terms": ["收益率是结果", "目标"],
        "strategy_family": "risk_control",
        "knowledge": "不能先设想收益率再倒推高杠杆；应先证明 edge，再决定风险暴露。",
        "hypothesis": "知识引导策略默认低杠杆研究，只有通过费用/稳健性后才实验更高杠杆。",
        "features": ["edge_score", "drawdown", "fee_stress_return"],
        "entry_rules": ["edge 先于杠杆"],
        "exit_rules": ["收益目标不能覆盖风险时拒绝入场"],
        "avoid": ["为了目标收益强行放大杠杆"],
        "risk": ["杠杆不能创造 edge。"],
    },
    {
        "id": "pa_loss_is_inevitable_distribution",
        "title": "亏损必然发生，要用分布思维设计策略",
        "category": "risk",
        "concepts": ["loss_distribution", "drawdown", "expectancy"],
        "source_terms": ["亏损必然", "40—60"],
        "strategy_family": "risk_control",
        "knowledge": "亏损不是异常，策略必须提前定义可接受连续亏损和回撤。",
        "hypothesis": "每个策略回测除收益外必须输出连续亏损、最大回撤和亏损出口质量。",
        "features": ["max_consecutive_losses", "max_drawdown", "loss_exit_share"],
        "entry_rules": ["亏损分布可接受才测试实盘模拟"],
        "exit_rules": ["连续亏损触发暂停"],
        "avoid": ["只看胜率不看盈亏分布"],
        "risk": ["高胜率策略也可能因尾部亏损失败。"],
    },
    {
        "id": "pa_opening_range_statistics",
        "title": "早盘/开盘区间需要统计而不是直觉",
        "category": "pattern",
        "concepts": ["opening_range", "session", "statistics"],
        "source_terms": ["早盘", "概率统计"],
        "strategy_family": "session_filter",
        "knowledge": "早盘模式强调特定交易时段的行为统计，加密货币需映射到美盘/欧盘/亚洲时段。",
        "hypothesis": "把早盘概念转为 UTC 小时段切片，测试 BTC/ETH 在美盘前后是否有独立 edge。",
        "features": ["hour_of_day", "session_tag", "opening_range_width"],
        "entry_rules": ["只在统计优势时段启用"],
        "exit_rules": ["时段优势结束或波动收缩退出"],
        "avoid": ["把股票早盘规则无验证搬到 24h 加密市场"],
        "risk": ["时段策略必须跨月份验证。"],
    },
    {
        "id": "pa_failed_breakout_opening_reversal",
        "title": "早盘突破失败后的反转",
        "category": "pattern",
        "concepts": ["opening_breakout_failure", "reversal", "session"],
        "source_terms": ["突破失败后的反转", "早盘"],
        "strategy_family": "failed_breakout",
        "knowledge": "开盘/早盘突破失败往往会触发反向移动，因为早期追随者被迫离场。",
        "hypothesis": "在加密市场测试美盘时段的区间突破失败反转。",
        "features": ["session_range", "failed_breakout", "reversal_followthrough"],
        "entry_rules": ["时段区间形成", "突破失败", "反向跟进"],
        "exit_rules": ["回到区间中轴或反向失败退出"],
        "avoid": ["无时段统计直接套用"],
        "risk": ["美盘新闻和流动性冲击需过滤。"],
    },
    {
        "id": "pa_ninety_minute_rule_timebox",
        "title": "90分钟定理启发持仓时间盒",
        "category": "exit",
        "concepts": ["timebox", "ninety_minute_rule", "session_duration"],
        "source_terms": ["90分钟定理"],
        "strategy_family": "exit_design",
        "knowledge": "价格行为里的时间窗口可用于限制持仓和避免无效持仓。",
        "hypothesis": "短周期策略设置持仓时间盒：如果 MFE 没发展，主动退出。",
        "features": ["minutes_in_trade", "mfe_progress", "session_elapsed"],
        "entry_rules": ["入场仍由形态决定"],
        "exit_rules": ["固定时间内未发展则退出"],
        "avoid": ["亏损单无限等待"],
        "risk": ["时间退出需要和 ROI/stoploss 同时测试。"],
    },
    {
        "id": "pa_intraday_position_management",
        "title": "日内策略目标和仓位必须一起设计",
        "category": "risk",
        "concepts": ["intraday", "position_sizing", "target"],
        "source_terms": ["日内", "目标", "仓位管理"],
        "strategy_family": "risk_control",
        "knowledge": "日内交易的目标、仓位和最大亏损必须成套设计。",
        "hypothesis": "短周期策略按账户风险而非信号强弱决定 stake，并记录当日暂停条件。",
        "features": ["stake_fraction", "daily_drawdown", "trade_count_today"],
        "entry_rules": ["当日风险预算未超限"],
        "exit_rules": ["达到当日回撤或连续亏损阈值暂停"],
        "avoid": ["亏损后加倍下注"],
        "risk": ["频繁交易下费用和情绪风险更高。"],
    },
    {
        "id": "pa_trade_review_training_loop",
        "title": "复盘训练是策略研究的数据来源",
        "category": "training",
        "concepts": ["review", "training", "case_library"],
        "source_terms": ["复盘训练", "提高自己的实盘能力"],
        "strategy_family": "research_process",
        "knowledge": "复盘不是写感想，而是提炼可复用形态、失败条件和执行错误。",
        "hypothesis": "Agent 每次回测后把失败模式写入经验记忆，下一轮假设必须读取。",
        "features": ["failure_mode", "lesson_tag", "next_experiment"],
        "entry_rules": ["策略生成前读取相关失败教训"],
        "exit_rules": ["重复失败模式进入冷却"],
        "avoid": ["重复跑同类失败策略"],
        "risk": ["记忆层必须防止幸存者偏差。"],
    },
    {
        "id": "pa_replay_training_validation",
        "title": "回放训练可转为 walk-forward 思路",
        "category": "training",
        "concepts": ["replay", "walk_forward", "out_of_sample"],
        "source_terms": ["TV回放", "进阶训练"],
        "strategy_family": "validation",
        "knowledge": "回放训练的重点是按时间顺序决策，避免事后看答案。",
        "hypothesis": "把知识生成策略放入 walk-forward，不允许用全样本调完再宣称有效。",
        "features": ["train_window", "test_window", "decision_time"],
        "entry_rules": ["只使用当时可见数据"],
        "exit_rules": ["样本外失败则降级"],
        "avoid": ["看完整行情后反推规则"],
        "risk": ["lookahead/recursive 分析是硬门槛。"],
    },
    {
        "id": "pa_scalp_fee_microstructure_gate",
        "title": "剥头皮策略先过费用和微结构门槛",
        "category": "crypto_adaptation",
        "concepts": ["scalp", "fees", "microstructure", "slippage"],
        "source_terms": ["scalp", "剥头皮"],
        "strategy_family": "scalping",
        "knowledge": "Scalp 交易目标小、频率高，手续费、滑点和盘口质量可能决定成败。",
        "hypothesis": "任何 1m/短动量剥头皮策略先做费用压力测试，再考虑入场优化。",
        "features": ["spread_proxy", "fee_bps", "slippage_bps", "mfe_within_5m"],
        "entry_rules": ["预期短期 MFE 大于费用滑点", "波动足够", "避免极低流动时段"],
        "exit_rules": ["数分钟内不发展立刻退出"],
        "avoid": ["用粗 OHLCV 声称盘口级 edge"],
        "risk": ["需要额外 L2/order book 数据才能研究做市和盘口 imbalance。"],
    },
    {
        "id": "pa_crypto_session_mapping",
        "title": "加密货币要把盘中规则映射到24小时市场",
        "category": "crypto_adaptation",
        "concepts": ["crypto", "sessionless_market", "time_filter"],
        "source_terms": ["比特币", "外汇", "黄金", "原油", "早盘策略"],
        "strategy_family": "session_filter",
        "knowledge": "股票/期货盘中规则不能直接搬到加密货币，必须映射到 UTC 时段和流动性窗口。",
        "hypothesis": "按亚洲、欧洲、美盘时段切片测试相同形态的表现差异。",
        "features": ["utc_hour", "day_of_week", "volume_percentile"],
        "entry_rules": ["只启用经验证的时段"],
        "exit_rules": ["时段结束或流动性转弱退出"],
        "avoid": ["把早盘概念按本地时间硬套"],
        "risk": ["周末和资金费率窗口需要单独验证。"],
    },
    {
        "id": "pa_btc_eth_fee_funding_filter",
        "title": "BTC/ETH 合约形态必须加费用和资金费率过滤",
        "category": "crypto_adaptation",
        "concepts": ["BTC", "ETH", "funding", "fee_stress"],
        "source_terms": ["比特币", "BTC", "ETH"],
        "strategy_family": "crypto_risk",
        "knowledge": "形态 edge 如果很小，在合约里可能被手续费、滑点和资金费率吃掉。",
        "hypothesis": "所有价格行为策略生成后先跑 base cost 和 stress cost 两档费用压力测试。",
        "features": ["fee_stress", "funding_window", "expected_move_atr"],
        "entry_rules": ["预期移动大于费用压力"],
        "exit_rules": ["资金费率/成本窗口不利时降频或禁开"],
        "avoid": ["小目标高频策略不算成本"],
        "risk": ["费用后 PF 低于 1 的策略不得保留候选。"],
    },
    {
        "id": "pa_support_resistance_zone",
        "title": "支撑阻力是区域不是单一价格",
        "category": "definition",
        "concepts": ["support", "resistance", "zone"],
        "source_terms": ["支撑", "阻力", "support", "resistance"],
        "strategy_family": "zone_reaction",
        "knowledge": "支撑阻力应用区域表达，短周期针刺不应直接触发交易。",
        "hypothesis": "用 ATR 和 swing cluster 构造支撑阻力区域，再测试区域反应。",
        "features": ["atr_zone_width", "touch_count", "rejection_ratio"],
        "entry_rules": ["进入区域", "出现拒绝", "下一根确认"],
        "exit_rules": ["有效收穿区域退出"],
        "avoid": ["一触线就买卖"],
        "risk": ["区域过窄会提高止损噪音。"],
    },
    {
        "id": "pa_price_action_definition_no_indicator_dependency",
        "title": "价格行为优先观察价格本身",
        "category": "definition",
        "concepts": ["price_action", "raw_price", "indicator_light"],
        "source_terms": ["price action", "价格行为"],
        "strategy_family": "feature_design",
        "knowledge": "价格行为强调价格结构、位置、强弱和反应，指标只能作为辅助表达。",
        "hypothesis": "策略特征优先使用结构变量，再用 EMA/ATR/ADX 做量化辅助。",
        "features": ["swing_high_low", "range_position", "candle_body", "atr"],
        "entry_rules": ["结构条件先于指标条件"],
        "exit_rules": ["结构失效先于指标钝化"],
        "avoid": ["把价格行为退化成指标堆叠"],
        "risk": ["结构变量也必须防止未来函数。"],
    },
]

CARD_SPECS.extend(
    [
        {
            "id": "pa_xu_system_complete_solution",
            "title": "交易系统必须覆盖入场、止损、止盈、仓位和执行",
            "category": "training",
            "concepts": ["trading_system", "execution_plan", "risk_process"],
            "source_terms": [],
            "book_refs": ["book_xu_jiacong_p047", "book_xu_jiacong_p056"],
            "strategy_family": "research_process",
            "knowledge": "裸K形态不能单独成为系统，必须和周期、风险承受、仓位、止损止盈和执行纪律组成完整方案。",
            "hypothesis": "Agent 生成任何价格行为策略时，必须同时输出入场、失效、止损、止盈、仓位和不适用场景。",
            "features": ["signal_definition", "risk_budget", "exit_rule", "position_size"],
            "entry_rules": ["策略必须有完整交易系统字段"],
            "exit_rules": ["缺少失效或仓位规则则拒绝实验"],
            "avoid": ["只用形态信号生成裸策略"],
            "risk": ["系统完整性是研究门槛，不是盈利保证。"],
        },
        {
            "id": "pa_xu_pinbar_key_level_filter",
            "title": "Pinbar 必须结合关键位，不能孤立交易",
            "category": "entry",
            "concepts": ["pinbar", "key_level", "context_filter"],
            "source_terms": [],
            "book_refs": ["book_xu_jiacong_p060", "book_xu_jiacong_p062", "book_xu_jiacong_p076"],
            "strategy_family": "entry_confirmation",
            "knowledge": "Pinbar 可以表达拒绝和反转，但只有出现在关键位置并通过后续反应确认时才有研究价值。",
            "hypothesis": "测试长影线拒绝关键区域后，下一根恢复方向确认的入场，而不是见 Pinbar 就进场。",
            "features": ["wick_ratio", "body_position", "zone_distance_atr", "next_candle_confirmation"],
            "entry_rules": ["主影线显著", "靠近关键区域", "下一根方向确认"],
            "exit_rules": ["价格重新穿回关键区或 Pinbar 极值失守"],
            "avoid": ["把单根长影线当作独立 edge"],
            "risk": ["1m 噪声会制造大量伪 Pinbar，必须用区域和成交/波动过滤。"],
        },
        {
            "id": "pa_xu_key_level_turning_point_zone",
            "title": "关键位来自明显拐点和多空力量转换",
            "category": "definition",
            "concepts": ["key_level", "turning_point", "supply_demand_zone"],
            "source_terms": [],
            "book_refs": ["book_xu_jiacong_p076", "book_xu_jiacong_p080", "book_xu_jiacong_p084"],
            "strategy_family": "zone_reaction",
            "knowledge": "关键位不是精确价格，而是历史拐点附近的区域，反映多空力量转换和后续再测试。",
            "hypothesis": "把 swing high/low 后快速反向的区域定义为关键区，只在回测关键区出现拒绝或突破确认时交易。",
            "features": ["swing_turn_strength", "zone_width_atr", "retest_count", "reaction_speed"],
            "entry_rules": ["历史拐点区域识别", "价格回到区域", "出现拒绝或突破确认"],
            "exit_rules": ["区域反应失败或二次穿越"],
            "avoid": ["用单一固定价格画线"],
            "risk": ["区域过宽会降低盈亏比；区域过窄会被噪声扫掉。"],
        },
        {
            "id": "pa_xu_signal_score_min_rr",
            "title": "信号研判必须同时看位置、空间和最小盈亏比",
            "category": "risk",
            "concepts": ["signal_score", "reward_risk", "profit_space"],
            "source_terms": [],
            "book_refs": ["book_xu_jiacong_p105", "book_xu_jiacong_p221"],
            "strategy_family": "risk_control",
            "knowledge": "符合外观的信号不一定可交易，若入场后止损空间过大或盈利空间不足，应等待回调或降低仓位。",
            "hypothesis": "在价格行为策略中加入最低预期 R 值和可承受止损过滤，过滤位置不好但形态好看的信号。",
            "features": ["risk_distance_atr", "target_distance_atr", "expected_r_multiple", "stake_adjustment"],
            "entry_rules": ["信号通过位置过滤", "预期 R 值达标", "账户风险可承受"],
            "exit_rules": ["信号失效或 R 值恶化"],
            "avoid": ["只看形态外观不看止损距离"],
            "risk": ["回测中的固定 ROI 需要和结构目标分开评估。"],
        },
        {
            "id": "pa_xu_50pct_invalidation_stop",
            "title": "突破入场后 50% 回调可作为信号衰弱参考",
            "category": "exit",
            "concepts": ["fifty_percent_retrace", "invalidation", "stoploss"],
            "source_terms": [],
            "book_refs": ["book_xu_jiacong_p108"],
            "strategy_family": "exit_design",
            "knowledge": "大信号若直接突破入场，回撤超过信号幅度约一半常代表信号变弱，可作为结构止损或减仓参考。",
            "hypothesis": "测试突破确认入场后，价格回撤超过信号K一半且无恢复时退出，而不是固定时间硬扛。",
            "features": ["signal_range", "retrace_from_breakout", "recovery_failure"],
            "entry_rules": ["突破信号确认后入场"],
            "exit_rules": ["回撤超过信号K约 50% 且恢复失败"],
            "avoid": ["把 50% 作为所有行情的机械止损"],
            "risk": ["高杠杆下结构止损仍需受账户风险上限约束。"],
        },
        {
            "id": "pa_xu_false_breakout_institutional_trap",
            "title": "假突破可作为反向意图线索，但必须等待收回确认",
            "category": "pattern",
            "concepts": ["false_breakout", "trap", "key_level_reclaim"],
            "source_terms": [],
            "book_refs": ["book_xu_jiacong_p100"],
            "strategy_family": "failed_breakout",
            "knowledge": "关键位附近的假突破可以暴露反向压力，但必须看到价格收回和后续确认，不能提前猜。",
            "hypothesis": "测试刺破关键区域后收回，且下一根延续反向的三条件反转策略。",
            "features": ["breakout_distance_atr", "reclaim_close", "follow_through_bar"],
            "entry_rules": ["刺破关键区", "收回关键区", "反向跟进"],
            "exit_rules": ["再次突破失败方向关键位"],
            "avoid": ["在未收回前逆势摸顶/摸底"],
            "risk": ["强趋势中假突破形态可能只是中继洗盘。"],
        },
        {
            "id": "pa_xu_pyramid_only_after_profit",
            "title": "金字塔加仓只能顺势盈利加，不允许亏损摊薄",
            "category": "risk",
            "concepts": ["pyramiding", "add_to_winner", "no_averaging_down"],
            "source_terms": [],
            "book_refs": ["book_xu_jiacong_p199", "book_xu_jiacong_p201"],
            "strategy_family": "risk_control",
            "knowledge": "金字塔加仓只适合强单边行情，核心是移动止损控制风险和用既得利润博取后续收益。",
            "hypothesis": "在 Freqtrade 研究中，加仓实验只允许盈利后顺势加仓，禁止亏损时摊薄成本。",
            "features": ["unrealized_profit", "trend_strength", "moved_stop", "add_position_count"],
            "entry_rules": ["已有仓位盈利", "趋势强度继续", "止损已上移或风险不扩大"],
            "exit_rules": ["趋势强度衰减或移动止损触发"],
            "avoid": ["亏损加仓", "震荡行情加仓"],
            "risk": ["先在 dry-run/回测验证；高杠杆下默认不启用真实加仓。"],
        },
        {
            "id": "pa_xu_review_four_elements",
            "title": "复盘要记录信号、入场止损、发力和周期",
            "category": "training",
            "concepts": ["review", "signal_quality", "execution_audit"],
            "source_terms": [],
            "book_refs": ["book_xu_jiacong_p219", "book_xu_jiacong_p220"],
            "strategy_family": "research_process",
            "knowledge": "复盘不能只看盈亏结果，还要检查信号质量、入场止损、信号发力速度和周期大小。",
            "hypothesis": "Agent 每次回测后按四要素输出失败归因，避免把随机盈亏当作策略能力。",
            "features": ["entry_quality_score", "stop_distance", "mfe_speed", "timeframe"],
            "entry_rules": ["研究流程必须记录四要素"],
            "exit_rules": ["缺少归因的策略不得晋级"],
            "avoid": ["只按收益率筛策略"],
            "risk": ["复盘结论必须来自样本外和费用压力测试。"],
        },
        {
            "id": "pa_xu_intuition_discipline_guard",
            "title": "直觉交易会破坏计划，Agent 必须硬性执行规则",
            "category": "psychology",
            "concepts": ["discipline", "intuition_trading", "plan_execution"],
            "source_terms": [],
            "book_refs": ["book_xu_jiacong_p224", "book_xu_jiacong_p225", "book_xu_jiacong_p226"],
            "strategy_family": "research_process",
            "knowledge": "失败交易往往来自无计划、无资金管理、扛单、报复交易和过度交易；策略 Agent 必须用硬规则替代临场直觉。",
            "hypothesis": "将连续亏损暂停、最大回撤熔断、禁止报复加仓和过度交易限制固化为策略研究门槛。",
            "features": ["loss_streak", "drawdown", "trade_frequency", "cooldown"],
            "entry_rules": ["风控状态允许", "未触发冷却", "交易频率未超限"],
            "exit_rules": ["触发亏损/回撤/频率保护"],
            "avoid": ["亏损后立刻加倍交易", "信号真空期强行交易"],
            "risk": ["纪律规则应硬编码，不交给 LLM 临场决定。"],
        },
    ]
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", value).strip("_")
    return cleaned[:80] or "transcript"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def matching_transcript_path(page: int) -> Path | None:
    matches = sorted(TRANSCRIPTS_DIR.glob(f"p{page:03d}_*.txt"))
    return matches[0] if matches else None


def count_hits(text: str, keywords: set[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(keyword.lower()) for keyword in keywords)


def title_term_hits(title: str, text: str) -> int:
    terms = [part for part in re.split(r"[\s._#&/()（）—-]+", title) if len(part) >= 2]
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered)


def title_is_trading_topic(title: str) -> bool:
    return count_hits(title, TRADING_KEYWORDS) > 0 or bool(
        re.search(
            r"(market|cycle|breakout|mode|scalp|wedge|parabolic|actual risk|measured move|stop order|limit order)",
            title,
            flags=re.IGNORECASE,
        )
    )


def normalize_transcripts() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = load_json(TRANSCRIPT_REPORT)
    pages = report.get("pages", [])
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    normalized = []
    for item in pages:
        page = int(item.get("page", 0))
        title = str(item.get("title") or "")
        path = matching_transcript_path(page)
        if not path:
            status = "missing"
            text = ""
            lines: list[str] = []
        else:
            text = path.read_text(encoding="utf-8")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            trading_hits = count_hits(text, TRADING_KEYWORDS)
            offtopic_hits = count_hits(text, OFFTOPIC_KEYWORDS)
            title_hits = title_term_hits(title, text)
            title_trading = title_is_trading_topic(title)
            if offtopic_hits >= 8 and trading_hits < 8:
                status = "mismatched"
            elif title_trading and len(lines) >= 50 and trading_hits < 8:
                status = "mismatched"
            elif len(lines) < 50 or trading_hits < 3:
                status = "low_confidence"
            elif title_hits == 0 and trading_hits < 8:
                status = "low_confidence"
            else:
                status = "usable"
        payload = {
            "id": f"bilibili_p{page:03d}",
            "page": page,
            "title": title,
            "status": status,
            "line_count": len(lines),
            "char_count": len(text),
            "trading_keyword_hits": count_hits(text, TRADING_KEYWORDS),
            "offtopic_keyword_hits": count_hits(text, OFFTOPIC_KEYWORDS),
            "title_term_hits": title_term_hits(title, text),
            "source_path": rel(path) if path else None,
            "normalized_path": rel(NORMALIZED_DIR / f"p{page:03d}_{slug(title)}.txt") if path else None,
            "notes": quality_note(status),
        }
        if path:
            normalized_path = NORMALIZED_DIR / f"p{page:03d}_{slug(title)}.txt"
            normalized_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(NORMALIZED_DIR / f"p{page:03d}_{slug(title)}.json", payload)
        normalized.append(payload)
    summary = {
        "generated_at_utc": now_utc(),
        "total_pages": len(normalized),
        "usable": sum(1 for item in normalized if item["status"] == "usable"),
        "low_confidence": sum(1 for item in normalized if item["status"] == "low_confidence"),
        "mismatched": sum(1 for item in normalized if item["status"] == "mismatched"),
        "missing": sum(1 for item in normalized if item["status"] == "missing"),
    }
    write_json(QUALITY_JSON, {"summary": summary, "pages": normalized})
    write_quality_markdown(summary, normalized)
    return normalized, summary


def quality_note(status: str) -> str:
    return {
        "usable": "Transcript passed conservative keyword and length checks.",
        "low_confidence": "Transcript is too short or weakly matched; use only as auxiliary evidence.",
        "mismatched": "Transcript appears unrelated to the video title; excluded from knowledge extraction.",
        "missing": "No local transcript exists for this page.",
    }[status]


def write_quality_markdown(summary: dict[str, Any], pages: list[dict[str, Any]]) -> None:
    lines = [
        "# Transcript Quality Report",
        "",
        f"- Generated UTC: `{summary['generated_at_utc']}`",
        f"- Total pages: `{summary['total_pages']}`",
        f"- Usable: `{summary['usable']}`",
        f"- Low confidence: `{summary['low_confidence']}`",
        f"- Mismatched: `{summary['mismatched']}`",
        f"- Missing: `{summary['missing']}`",
        "",
        "| Page | Status | Lines | Trading Hits | Offtopic Hits | Title |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for item in pages:
        lines.append(
            f"| {item['page']} | {item['status']} | {item['line_count']} | "
            f"{item['trading_keyword_hits']} | {item['offtopic_keyword_hits']} | {item['title']} |"
        )
    QUALITY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def source_matches(spec: dict[str, Any], transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terms = [term.lower() for term in spec["source_terms"]]
    matches = []
    for item in transcripts:
        if item["status"] not in {"usable", "low_confidence"}:
            continue
        title = item["title"].lower()
        if any(term.lower() in title for term in terms):
            matches.append(item)
    matches.sort(key=lambda item: (item["status"] != "usable", item["page"]))
    return matches[:4]


def web_refs_for_spec(spec: dict[str, Any], matches: list[dict[str, Any]]) -> list[str]:
    if spec.get("book_refs"):
        return []
    concepts = set(spec["concepts"])
    category = spec["category"]
    if category == "crypto_adaptation":
        return WEB_SOURCE_GROUPS["crypto"]
    if concepts & {"support", "resistance", "zone"}:
        return WEB_SOURCE_GROUPS["support_resistance"]
    if concepts & {"price_action", "raw_price", "indicator_light"}:
        return WEB_SOURCE_GROUPS["price_action"]
    if category == "definition":
        return WEB_SOURCE_GROUPS["price_action"]
    if not matches:
        return WEB_SOURCE_GROUPS["brooks"]
    return []


def source_quality(matches: list[dict[str, Any]], web_refs: list[str], book_refs: list[str]) -> dict[str, Any]:
    usable = [item for item in matches if item["status"] == "usable"]
    low = [item for item in matches if item["status"] == "low_confidence"]
    if usable or book_refs:
        level = "high"
    elif low:
        level = "medium"
    elif web_refs:
        level = "medium"
    else:
        level = "low"
    return {
        "level": level,
        "usable_transcript_count": len(usable),
        "low_confidence_transcript_count": len(low),
        "web_source_count": len(web_refs),
        "book_source_count": len(book_refs),
    }


def build_card(spec: dict[str, Any], transcripts: list[dict[str, Any]]) -> dict[str, Any]:
    matches = source_matches(spec, transcripts)
    web_refs = web_refs_for_spec(spec, matches)
    book_refs = spec.get("book_refs", [])
    refs = [item["id"] for item in matches]
    refs.extend(book_refs)
    refs.extend(web_refs)
    quality = source_quality(matches, web_refs, book_refs)
    is_quarantined = (
        quality["usable_transcript_count"] == 0
        and quality["web_source_count"] == 0
        and quality["book_source_count"] == 0
    )
    return {
        "id": spec["id"],
        "title": spec["title"],
        "category": spec["category"],
        "concepts": spec["concepts"],
        "source_refs": refs[:5],
        "source_quality": quality,
        "knowledge": spec["knowledge"],
        "strategy_hypothesis": spec["hypothesis"],
        "freqtrade_translation": {
            "strategy_family": spec["strategy_family"],
            "features": spec["features"],
            "entry_rules": spec["entry_rules"],
            "exit_rules": spec["exit_rules"],
            "applicable_regimes": applicable_regimes(spec),
            "not_applicable_regimes": not_applicable_regimes(spec),
        },
        "risk_notes": spec["risk"],
        "avoid_rules": spec["avoid"],
        "verification_status": {
            "state": "quarantined_needs_stronger_source" if is_quarantined else "knowledge_only_requires_backtest",
            "required_checks": [
                "freqtrade_backtesting",
                "recursive_analysis",
                "lookahead_analysis",
                "regime_matrix",
                "fee_slippage_stress",
                "promotion_gate",
            ],
            "quarantined": is_quarantined,
        },
        "copyright_note": "Short local knowledge summary. Does not copy long transcript passages.",
        "created_at_utc": now_utc(),
    }


def applicable_regimes(spec: dict[str, Any]) -> list[str]:
    family = spec["strategy_family"]
    if "breakout" in family or family in {"momentum", "trend_following", "trend_pullback"}:
        return ["trend", "volatility_expansion"]
    if "reversal" in family or "range" in family:
        return ["range", "exhaustion", "failed_breakout"]
    if "risk" in family or "process" in family or family in {"validation", "feature_design"}:
        return ["all"]
    if "session" in family:
        return ["time_sliced"]
    return ["trend", "range"]


def not_applicable_regimes(spec: dict[str, Any]) -> list[str]:
    family = spec["strategy_family"]
    if "reversal" in family:
        return ["strong_unconfirmed_trend"]
    if "breakout" in family:
        return ["low_volatility_chop", "wick_only_move"]
    if family == "scalping":
        return ["high_fee_or_wide_spread", "illiquid_period"]
    return []


def build_cards(transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINED_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    for path in CARDS_DIR.glob("*.json"):
        path.unlink()
    for path in QUARANTINED_CARDS_DIR.glob("*.json"):
        path.unlink()
    all_cards = [build_card(spec, transcripts) for spec in CARD_SPECS]
    active_cards = []
    for card in all_cards:
        if card["verification_status"].get("quarantined"):
            write_json(QUARANTINED_CARDS_DIR / f"{card['id']}.json", card)
            continue
        write_json(CARDS_DIR / f"{card['id']}.json", card)
        active_cards.append(card)
    return active_cards


def build_claims(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    claims = []
    for card in cards:
        claim = {
            "id": f"claim_{card['id']}",
            "card_id": card["id"],
            "category": card["category"],
            "concepts": card["concepts"],
            "claim": card["knowledge"],
            "testable_hypothesis": card["strategy_hypothesis"],
            "source_refs": card["source_refs"],
            "source_quality": card["source_quality"],
            "verification_status": card["verification_status"],
        }
        claims.append(claim)
        write_json(CLAIMS_DIR / f"{claim['id']}.json", claim)
    write_json(CLAIMS_JSON, {"generated_at_utc": now_utc(), "claim_count": len(claims), "claims": claims})
    return claims


def build_indexes(cards: list[dict[str, Any]], claims: list[dict[str, Any]], quality_summary: dict[str, Any]) -> dict[str, Any]:
    concept_index: dict[str, list[str]] = {}
    source_index: dict[str, list[str]] = {}
    family_index: dict[str, list[str]] = {}
    category_index: dict[str, list[str]] = {}
    for card in cards:
        for concept in card["concepts"]:
            concept_index.setdefault(concept, []).append(card["id"])
            if "breakout" in concept:
                concept_index.setdefault("breakout", []).append(card["id"])
            if "pullback" in concept:
                concept_index.setdefault("pullback", []).append(card["id"])
            if "scalp" in concept or "scalping" in concept:
                concept_index.setdefault("scalp", []).append(card["id"])
        for source in card["source_refs"]:
            source_index.setdefault(source, []).append(card["id"])
        family = card["freqtrade_translation"]["strategy_family"]
        family_index.setdefault(family, []).append(card["id"])
        if "breakout" in family or "trap" in family:
            concept_index.setdefault("breakout", []).append(card["id"])
        if "pullback" in family:
            concept_index.setdefault("pullback", []).append(card["id"])
        if "scalp" in family or "scalping" in family:
            concept_index.setdefault("scalp", []).append(card["id"])
        category_index.setdefault(card["category"], []).append(card["id"])
    concept_index = {key: sorted(set(value)) for key, value in concept_index.items()}
    source_index = {key: sorted(set(value)) for key, value in source_index.items()}
    family_index = {key: sorted(set(value)) for key, value in family_index.items()}
    category_index = {key: sorted(set(value)) for key, value in category_index.items()}
    index = {
        "generated_at_utc": now_utc(),
        "card_count": len(cards),
        "claim_count": len(claims),
        "quality_summary": quality_summary,
        "concept_index": concept_index,
        "source_index": source_index,
        "strategy_family_index": family_index,
        "category_index": category_index,
        "artifacts": {
            "quality_report": rel(QUALITY_MD),
            "claims": rel(CLAIMS_JSON),
            "cards_dir": rel(CARDS_DIR),
            "quarantined_cards_dir": rel(QUARANTINED_CARDS_DIR),
        },
    }
    write_json(INDEX_JSON, index)
    return index


def write_layer_report(index: dict[str, Any], cards: list[dict[str, Any]]) -> None:
    high_quality = sum(1 for card in cards if card["source_quality"]["level"] == "high")
    quarantined_count = len(list(QUARANTINED_CARDS_DIR.glob("*.json")))
    lines = [
        "# Price Action Knowledge Layer Report",
        "",
        f"- Generated UTC: `{index['generated_at_utc']}`",
        f"- Active knowledge cards: `{index['card_count']}`",
        f"- Quarantined weak-source cards: `{quarantined_count}`",
        f"- Extracted claims: `{index['claim_count']}`",
        f"- High-source-quality cards: `{high_quality}`",
        f"- Transcript usable/low/mismatched/missing: "
        f"`{index['quality_summary']['usable']}/{index['quality_summary']['low_confidence']}/"
        f"{index['quality_summary']['mismatched']}/{index['quality_summary']['missing']}`",
        "",
        "## Card Inventory",
        "",
        "| Card | Category | Source Quality | Strategy Family | Concepts |",
        "|---|---|---|---|---|",
    ]
    for card in cards:
        lines.append(
            "| {id} | {category} | {quality} | {family} | {concepts} |".format(
                id=card["id"],
                category=card["category"],
                quality=card["source_quality"]["level"],
                family=card["freqtrade_translation"]["strategy_family"],
                concepts=", ".join(card["concepts"]),
            )
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Cards are short local summaries, not long transcript excerpts.",
            "- Mismatched transcripts are excluded from card generation.",
            "- Knowledge cards are hypothesis inputs only; promotion still requires backtests, bias checks, cost stress, and promotion gate.",
        ]
    )
    report_text = "\n".join(lines) + "\n"
    LAYER_REPORT_MD.write_text(report_text, encoding="utf-8")
    REPORT_ALIAS_MD.write_text(report_text, encoding="utf-8")
    report_payload = {"report": rel(LAYER_REPORT_MD), "alias_report": rel(REPORT_ALIAS_MD), "index": index}
    write_json(LAYER_REPORT_JSON, report_payload)
    write_json(REPORT_ALIAS_JSON, report_payload)


def main() -> None:
    for path in [NORMALIZED_DIR, CLAIMS_DIR, CARDS_DIR, INDEX_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    transcripts, quality_summary = normalize_transcripts()
    cards = build_cards(transcripts)
    claims = build_claims(cards)
    index = build_indexes(cards, claims, quality_summary)
    write_layer_report(index, cards)
    print(f"Wrote {rel(QUALITY_MD)}")
    print(f"Wrote {rel(CLAIMS_JSON)}")
    print(f"Wrote {rel(INDEX_JSON)}")
    print(f"Wrote {rel(LAYER_REPORT_MD)}")


if __name__ == "__main__":
    main()
