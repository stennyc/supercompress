"""extended-BM25 招① token 分类 + 高信号 exact-match boost 单元面（acceptance 面）。

覆盖 sc/recall.py 的高信号 term 类判定（纯字面、确定性）与 search() 计分面对
boost 因子的处理：
- is_numeric_token / path_terms 分类（AC 数字 / 路径 run / 下划线标识符）；
- 计分面：高信号 term 命中段按类乘独立 boost（路径 ×EXACT_MATCH_BOOST，
  纯数字 ×EXACT_MATCH_BOOST×NUMERIC_EXACT_BOOST），钉死系数数值面；
- 确定性：同 query 两次全等（含高信号混合面）；
- 不破钉死面：CJK unigram/bigram 分词与 P0_BOOST 面在 boost 层引入后仍成立。
"""
import pytest

from sc.recall import Recaller, is_numeric_token, path_terms, tokenize
from helpers import mkseg, mk_window


def test_is_numeric_token_classification():
    """纯数字 token（AC/配置量值）→ True；混合/空 → False（纯字面判定）。"""
    assert is_numeric_token("8080")
    assert is_numeric_token("42")
    assert is_numeric_token("0")
    assert not is_numeric_token("8080x")     # 数字+字母混合 → 非量值
    assert not is_numeric_token("v2")
    assert not is_numeric_token("abc")
    assert not is_numeric_token("1_2")       # 含下划线 → 标识符类，非纯量值
    assert not is_numeric_token("")
    # 确定性：同输入两次同判定
    for t in ("8080", "8080x", "1_2", ""):
        assert is_numeric_token(t) == is_numeric_token(t)


def test_path_terms_classification():
    """路径 run 面（多集、小写化）：/ \\ C: 路径自成词空间，普通文本零命中。"""
    # Windows 路径（冒号+反斜杠）与 unix 路径（正斜杠）均归路径 run
    toks = path_terms("see C:\\logs\\app.log and /etc/nginx/conf.d/ssl.conf now")
    assert "c:\\logs\\app.log" in toks
    assert "/etc/nginx/conf.d/ssl.conf" in toks
    # 小写化：同路径大写输入 → 同 run
    assert path_terms("C:/Logs/App.log") == path_terms("c:/logs/app.log")
    # 无 / \ 的普通文本 → 零路径 run
    assert path_terms("plain text no slash at all") == []
    assert path_terms("") == []
    # 确定性：同输入两次全等
    s = "x a/b.py y C:/srv z"
    assert path_terms(s) == path_terms(s)


def test_underscore_identifier_classified_not_numeric():
    """下划线标识符（函数名/变量名）：非纯数字类 → 走标识符 boost 面（×em），非叠乘。"""
    assert not is_numeric_token("auth_check_handler")
    assert not is_numeric_token("x_1")           # 数字+下划线 → 标识符类
    assert is_numeric_token("100")               # 纯数字才是叠乘类


def test_cjk_tokenization_intact_with_boost_layer():
    """钉死面：boost 层引入后 CJK unigram+bigram 分词逐字不变。"""
    toks = tokenize("网关端口占用")
    assert "网关" in toks and "端口" in toks and "占用" in toks   # bigram
    assert "网" in toks and "用" in toks                          # unigram
    mixed = tokenize("限流 nginx 网关")
    assert "nginx" in mixed and "网关" in mixed
    # 路径/数字 token 分类不影响 CJK 面
    assert path_terms("网关端口占用") == []
    assert not is_numeric_token("网关")


def test_numeric_term_boosted_ranking(store, aux):
    """招①：纯数字高信号命中段按叠乘 boost 上移（×em×nmb > 普通 term）。"""
    p1_hit = mkseg(store, "listen port 8080 on 192.168.55.55", "P1")
    p1_miss = mkseg(store, "the deployment log says service restarted ok", "P1")
    p2 = mkseg(store, "raw dump verbose unrelated filler", "P2")
    w = mk_window(2000, [p1_hit, p1_miss, p2])
    hits = Recaller(store, w).search("8080 port", k=3)
    assert hits[0].seg.segment_id == p1_hit.segment_id, "数字锚点段应排第一"
    assert hits[0].score > hits[1].score, "boost 后数字段得分应严格更高"


def test_underscore_identifier_boosted_ranking(store, aux):
    """招①：下划线标识符（函数名）整串命中 ×em，胜过散词面。"""
    hit = mkseg(store, "handler function auth_check_handler registered", "P1")
    miss = mkseg(store, "the handler auth check function got registered", "P1")
    w = mk_window(2000, [hit, miss])
    hits = Recaller(store, w).search("auth_check_handler", k=2)
    assert hits[0].seg.segment_id == hit.segment_id
    assert hits[0].score > hits[1].score


def test_path_term_boosted_ranking(store, aux):
    """招①：路径 run 独立词空间命中 ×em（tokenize 把 C:/srv 切成 c、srv，路径整串独立计）。"""
    hit = mkseg(store, "config lives at C:/srv/web/nginx.conf reload needed", "P1")
    miss = mkseg(store, "the web config got reloaded after editing c srv nginx conf", "P1")
    w = mk_window(2000, [hit, miss])
    hits = Recaller(store, w).search("C:/srv/web/nginx.conf", k=2)
    assert hits[0].seg.segment_id == hit.segment_id, "路径整串命中段应排第一"


def test_boost_factors_pinned_numeric_face():
    """系数数值面钉死：P0_BOOST 保留 + 招① boost 系数防漂移。"""
    assert Recaller.P0_BOOST == 1.5          # 钉死面（test_bm25_top3_p0_boost）
    assert Recaller.EXACT_MATCH_BOOST == 2.0
    assert Recaller.NUMERIC_EXACT_BOOST == 1.5
    assert Recaller.FIELD_WEIGHT == 2.0
    # 纯数字叠乘面 = em × nmb = 3.0（AC 量值最具体）
    assert Recaller.EXACT_MATCH_BOOST * Recaller.NUMERIC_EXACT_BOOST == 3.0


def test_boost_determinism_two_runs(store, aux):
    """G4 确定性：高信号混合 query（数字+路径+标识符+CJK）两次全等。"""
    p0 = mkseg(store, "ac12 use C:/srv/app.py in auth_check_handler 网关端口", "P0", "user",
               structural=True)
    p1 = mkseg(store, "random filler text about deploys and logs", "P1")
    p2 = mkseg(store, "verbose raw dump of unrelated noise", "P2")
    w = mk_window(2000, [p0, p1, p2])
    q = "ac12 C:/srv/app.py auth_check_handler 网关"
    a = [(h.seg.segment_id, round(h.score, 9)) for h in Recaller(store, w).search(q, k=3)]
    b = [(h.seg.segment_id, round(h.score, 9)) for h in Recaller(store, w).search(q, k=3)]
    assert a == b, "boost 层引入后确定性破坏"
    assert a[0][0] == p0.segment_id


def test_p0_boost_still_dominates_plain_terms(store, aux):
    """钉死面：无高信号锚点的普通 query 下 P0_BOOST=1.5 面不变（P0 排第一）。"""
    p0 = mkseg(store, "service config overview", "P0", "user", structural=True)
    p1 = mkseg(store, "service config overview", "P1")
    p2 = mkseg(store, "service config overview", "P2")
    w = mk_window(2000, [p0, p1, p2])
    hits = Recaller(store, w).search("service config overview", k=3)
    assert [h.seg.tier for h in hits] == ["P0", "P1", "P2"], "P0 boost 面未被动"
