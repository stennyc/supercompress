# precision P2 · 三遥测指标首读基线（数据闸门出数面）

commit=e688184b52b6  drift 标定参数 DRIFT_THRESH=0.3 / K_CONSEC=2（sc/drift.py:30-32）

## 指标① 填充占比 fill_share（>40% 才 re-add 30% 零成本剪除档，G5）
- 合成 6K 池：39 段（gold 19 + P1 填充 20），双压力臂：
  - normal：pool=418t  P0 高信号=88t  P1/P2 填充=330t  **fill_share=0.789**  gate_re_add_30pct=True
  - high-pressure（细节池归零）：**fill_share=0.789**  gate_re_add_30pct=True  detail_pool=3688t
## 指标② 语义 gap（exact-match miss 里同义换措辞该召回的比例；ONNX 出局）
- gap 集：n_misses=3（gold top-3 miss 0 + 同义探针 miss 3；P1 extended-BM25 上线后 miss 集变稀 = 精度收益读数）
  - G1·EN-query/CJK-target：MISS  miss_score=0.000
  - G2·CJK-syn/flood：MISS  miss_score=0.217
  - G3·num-anchor/flood：MISS  miss_score=0.333
- 词面 baseline：lex_gap=0.18357487922705315  近失占比（≥0.3）=0.3333333333333333
- 判官面：judged_by=lexical_fallback  semantic_gap=None
  （semantic_gap=None + judged_by=lexical_fallback = 端点不可用，只出词面基线，不误报）
## 指标③ drift 误报 FPR（标定 DRIFT_THRESH=0.30 / K_CONSEC=2）
- 合成流 events=3  人工确认真切换=1  **FPR=0.667  precision=0.333**
  - A·真切换一条：12 msgs → 1 events (true=1, fired_at user_seq=[9], final_epoch=1)
  - B·闲聊误报面：10 msgs → 2 events (true=0, fired_at user_seq=[7, 9], final_epoch=2)
## 生产 JSONL sink 读数（跨会话攒数面，P3 定案时读取）
- sink=(未给 --sink；HERMES_HOME 数据根 supercompress_store/telemetry.jsonl 待生产落盘)
- n_lines=0  sink 不存在/未给 → 仅 bench 基线出数（首读基线交付口径）
## 自检
- G4 确定性两遍全等：PASS
- 三指标出数完整：PASS

> 报告生成：bench/telemetry_report.py @ e688184b52b6（纯本地 + 可选宿主 LLM 判官；0 外部 / 0 新模型 / 0 硬编码路径）
