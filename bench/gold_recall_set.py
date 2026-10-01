"""P1.3 gold 标注集（人工）：中英混合 corpus + query→target 标注。

用途：bench/gold_recall_bench.py 用它量 recall@3，对比「旧词重叠基线（bfbe058）」
与「P1 BM25+CJK 层」。DoD：新层 recall@3 ≥ 旧基线 +30%。

标注原则（人工，非自动生成）：每条 query 的 target 是语料中**唯一最相关**段，
避免歧义。语料故意含大量**无空格 CJK 短语**（旧 \\S+ 词重叠面把整串当一个 token，
CJK 无空格 case 旧基线命中≈0；P1 的 CJK unigram+bigram 分词正是为覆盖这一面）。
seq 按 append 序（0..11）固定 → 结果可复跑（同 commit 同输出）。
"""
from __future__ import annotations

# (tier, provenance, structural_p0, original_text) — 12 段中英混合语料。
_CORPUS = [
    # ── 英文段（新旧都该命中：考排序质量，非区分度主战场）──
    ("P0", "user", True,  "user: ship v1 and keep /srv/a1 online"),
    ("P0", "model", False, "decision: port 8080 bound to 192.168.55.55 for the auth service"),
    ("P1", "model", False, "nginx config on host alpha listens on 443 with ssl termination"),
    ("P2", "model", False, "verbose raw stdout dump from the tool runner line by line"),
    ("P0", "model", False, "rollback strategy for the auth service on deploy failure"),
    ("P2", "model", False, "traceback stack dump with line numbers and raw frames"),
    ("P1", "model", False, "cron schedule runs the batch job at 02:00 daily"),
    # ── 无空格 CJK 段（P1 区分度主战场：旧基线≈0，新层 bigram 命中）──
    ("P0", "model", False, "部署发布版本四十二到生产环境并保留回滚入口"),
    ("P1", "model", False, "端口8080被nginx占用需改到8081再重启网关"),
    ("P2", "model", False, "数据库连接池超时日志刷屏需调大连接池maxsize"),
    ("P1", "model", False, "缓存命中率下降排查热点key引起的抖动"),
    ("P0", "model", False, "网关限流阈值调至每秒五百请求并配置告警"),
    ("P2", "model", False, "慢查询top十排序结果含索引缺失的字段"),
    # ── CJK 近重复干扰簇（打破「主题分块饱和」：同主题多篇，考 scorer 能否选中对的那篇）──
    #    与 idx7 近重复，仅 四十二/生产/保留/入口 vs 四十三/测试/无需 或 预发/开关 区分
    ("P0", "model", False, "部署发布版本四十三到测试环境无需回滚入口"),
    ("P1", "model", False, "部署发布版本四十二到预发环境保留回滚开关"),
    #    与 idx8 近重复，仅 8081 vs 8082 区分
    ("P1", "model", False, "端口8080被nginx占用需改到8082再重启网关"),
    #    与 idx9 近重复，仅 maxsize vs timeout 区分
    ("P2", "model", False, "数据库连接池超时日志刷屏需调大连接池timeout"),
    # ── EN 近重复干扰簇（证明区分度不止 CJK 面）──
    #    与 idx1 近重复，仅 .55/.56 与 8080/8081 区分
    ("P0", "model", False, "decision: port 8080 bound to 192.168.55.56 for the auth service"),
    ("P0", "model", False, "decision: port 8081 bound to 192.168.55.55 for the auth service"),
]

# (query, target_index_in_corpus) — 人工标注。
# 前 19 条（EN 7 + CJK 无空格 8 + 混合 4）+ 5 条近重复干扰 query（打破主题分块饱和）。
# 近重复簇内每条 query 的 target 是簇里**唯一最贴切**篇；弱/散打 scorer 会选错篇。
_GOLD = [
    # 英文（新旧都该中 → 考排序）
    ("ship v1 keep srv a1 online",        0),
    ("port 8080 bound 192.168.55.55 auth service", 1),
    ("nginx host alpha 443 ssl",          2),
    ("raw stdout dump tool runner",       3),
    ("rollback auth service deploy failure", 4),
    ("traceback stack line numbers raw frames", 5),
    ("cron batch job 02:00 daily",        6),
    # 无空格 CJK（旧基线≈0，P1 区分度主战场）
    ("部署发布版本四十二生产环境",        7),
    ("端口8080被nginx占用改8081",          8),
    ("数据库连接池超时日志刷屏",          9),
    ("缓存命中率下降热点key",             10),
    ("网关限流每秒五百请求告警",          11),
    ("慢查询top十索引缺失",               12),
    ("生产环境保留回滚入口",               7),
    ("连接池maxsize调大",                  9),
    # 混合措辞（同 target 不同表达，考鲁棒性）
    ("auth service 192.168.55.55",             1),
    ("auth service rollback on failure",  4),
    ("限流告警 五百 请求",                11),   # 带空格 CJK
    ("日志刷屏 数据库 连接池",             9),    # 带空格 CJK
    # 近重复干扰（同主题多篇，考 scorer 选中「对的那篇」，散打 scorer 会选错）
    # 注：既有 CJK 无空格 query（部署…四十二…→7 / 端口…8081→8）随干扰簇加入已自动获得判别力，不重复列
    ("数据库连接池maxsize调大",             9),   # vs idx16(timeout)：query 要 maxsize，idx9 独中
    ("auth service port 8080 192.168.55.55", 1),  # vs idx17(.56)/idx18(8081)：8080+.55 独指 idx1
]


def build_segments():
    """corpus → Segment 列表（seq=0..11，与 append 序一致；可复跑）。"""
    from sc.schema import Segment
    segs = []
    for i, (tier, prov, struct, text) in enumerate(_CORPUS):
        segs.append(Segment(tier=tier, provenance=prov, original_text=text,
                            structural_p0=struct, seq=i, ts=0.0))
    return segs


def gold_pairs(segs):
    """gold → [(query, target_segment), ...]。"""
    return [(q, segs[idx]) for q, idx in _GOLD]


if __name__ == "__main__":
    s = build_segments()
    print(f"corpus={len(s)} segs, gold={len(_GOLD)} pairs")
    for q, t in gold_pairs(s)[:3]:
        print(f"  {q!r} -> {t.original_text[:20]!r}")
