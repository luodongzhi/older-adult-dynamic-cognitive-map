# 旧版干预模板

这个目录保留通用模板和 JSON Schema，用于查阅格式，但不再作为正式运行输入。

当前实验真正读取的是：

```text
program/experiments/current/interventions.json
```

OSM 处理器生成的 `osmmap/processed/interventions.json` 只是针对新地图的候选干预，也不会
自动成为正式输入。这样可以避免重新处理地图时意外覆盖已经设计好的实验条件。

修改正式干预后，请先运行 `program/validate_experiment.py`，系统会检查干预日期是否超出
运行天数、目标 ID 是否存在，以及操作类型是否与目标类型一致。
