# 实验包

每个子目录代表一套可复现实验输入，固定包含：

```text
实验名/
├─ map.json
├─ agents.json
├─ run.json
└─ interventions.json
```

`current/` 是当前运行的实验。复制该目录即可保存新方案，然后在 `program/pycharm_run.json` 中切换：

```json
{
  "active_experiment": "experiments/新实验名"
}
```

切换实验不会修改地图原文件、Agent 档案或已有数据库。正式运行前请执行 `program/validate_experiment.py`。
