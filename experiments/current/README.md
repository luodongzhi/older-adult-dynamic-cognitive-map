# 当前实验输入

这个目录是一轮实验的唯一正式输入包。

## 四类输入

1. `map.json`：地图缓存、SHP 路径和字段映射。
2. `agents.json`：选择参与实验的 Agent，并指定位置绑定方式。
3. `run.json`：设置天数、随机种子、模型组合和输出数据库。
4. `interventions.json`：设置哪一天、哪个对象发生什么变化。

日常实验只需要修改后 3 个文件。更换地图时才修改 `map.json` 或重新处理 OSM。

## Agent

`selected_agents` 使用 `program/agents/` 中的 Agent ID。建议保持 `location_binding.mode` 为 `generated`，让地图处理器为当前地图生成有效的住宅、日常锚点和会面地点。若改为 `strict`，必须为每个 Agent 手工填写存在于当前地图中的 AOI ID。

Agent 档案保留会影响行为或信息接触的属性，如年龄、步速、活动半径、路线习惯、视觉注意、社会参与和日程参数。认知状态不包含熟悉度或置信分数。

## 天数

修改 `run.json`：

```json
"days": 10
```

所有干预的 `effective_day` 必须位于 `0..days`。建议每轮同时修改 `experiment_id` 和 `output_database`，避免覆盖旧结果。

## 干预

最小示例：

```json
{
  "intervention_id": "INT_DAY03_SHOP_TO_PARK",
  "effective_day": 3,
  "operation_type": "AOI_FUNCTION_CHANGE",
  "target_type": "AOI",
  "target_id": "AOI_N12002341738",
  "before_value": "SHOP",
  "after_value": "PARK"
}
```

原始 OSM/SHP 不会被修改。程序只改变本轮实验的地图副本。

## 运行

先运行 `program/validate_experiment.py`，再运行 `program/run_pycharm.py`。每轮还会生成 `output/*_inputs.resolved.json`，记录本次真正使用的非敏感输入，便于复现。
