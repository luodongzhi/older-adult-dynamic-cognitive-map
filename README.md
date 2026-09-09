# Older-adult Dynamic Cognitive Map Simulator

本项目研究一个收束的问题：**真实城市已经改变，但老年居民尚未获得新信息时，会出现多长的认知滞后，这种滞后会不会改变活动与路线预测。**

核心认知规则已经简化为二元状态：一个对象对 Agent 来说是“已知某状态”或“尚未知晓”。视觉看见、亲自遇到，或在真实共处中听到可追溯的一手信息后，该状态在当日写入认知地图，并从次日开始参与活动和路径决策。模型不再使用熟悉度、认知置信分数、证据累计或更新阈值。

## 1. 三个研究问题

### RQ1：有限信息会造成多长的认知滞后？

对每个 `Agent × 干预对象 × 日期` 比较真实状态与认知状态：

- 相同：`CORRECT`，滞后值为 0；
- 未知、仍保留旧状态或知道了错误状态：滞后值为 1。

据此计算正确更新率、认知滞后率、首次正确更新日、滞后持续天数，以及群体达到 50%/80% 正确更新的日期。

### RQ2：日常路线与线下相遇如何决定新信息何时到达？

系统分别记录第一次视觉发现、第一次亲历、第一次面对面获知和第一次使用。研究比较不同路线暴露、视觉注意和线下参与条件下的信息到达时间。这里研究的是“是否接触到信息”，不是 Agent 对信息打多少分。

### RQ3：信息延迟会改变多少行为预测？

在相同地图、Agent、干预和随机种子下运行：

- `B0 objective_accessibility`：只计算客观空间机会；
- `B1 omniscient`：Agent 始终知道真实地图；
- `M dynamic_cognitive`：Agent 只按自己的有限认知地图行动。

比较 B0–B1 与 B1–M 的活动成功率、目的地分布、路线差异、绕行、失败、干预对象使用和稳定时间。

详细指标定义见 [RESEARCH_DESIGN.md](RESEARCH_DESIGN.md)。

## 2. 项目结构

```text
program/
├─ run_pycharm.py                  正式运行入口
├─ validate_experiment.py          只检查输入，不调用 LLM
├─ pycharm_run.json                选择当前实验包
├─ api.example.txt                 豆包 API Key 填写模板，不含真实密钥
├─ experiments/current/
│  ├─ map.json                     地图来源和字段映射
│  ├─ agents.json                  本轮 Agent 选择及位置绑定
│  ├─ run.json                     天数、种子、模型和输出
│  └─ interventions.json           干预日期、对象与变化
├─ agents/*.json                   可增删的老年 Agent 档案
├─ osmmap/map.osm                  原始 OSM 文件
├─ osmmap/processed/               处理后的道路、AOI、入口与遮挡物
├─ src/cogmap_sim/                 模拟、认知、交互、存储和网页模块
├─ tests/                          回归测试
└─ output/                         数据库、CSV 和网页
```

`experiments/current/` 是一次实验唯一的正式输入包。原始 OSM/SHP 不会因干预而改变；干预只作用于本次运行的内存地图副本，并写入结果数据库。

首次使用时，将 `api.example.txt` 复制为 `api.txt`，再把其中的占位文字替换为自己的火山方舟 API Key。`api.txt` 已被 `.gitignore` 排除，禁止提交真实密钥。也可以通过环境变量 `ARK_API_KEY` 提供密钥。

## 3. 在 PyCharm 中运行

1. 运行 `program/validate_experiment.py`。
2. 检查通过后，运行 `program/run_pycharm.py`。
3. 网页会从 `output/` 读取本轮 `.db`，并生成独立 HTML 报告与数据库选择页。

常用设置都在 `experiments/current/`：

- 运行天数：修改 `run.json` 的 `days`；
- Agent：修改 `agents.json` 的 `selected_agents`；
- 干预：修改 `interventions.json`；
- 地图：修改 `map.json`，或替换 `osmmap/map.osm` 后重新运行 `tools/process_osm_map.py`。

正式运行三模型研究套件：

```json
{
  "run_research_suite": true,
  "research_models": ["B0", "B1", "M"]
}
```

只运行动态认知模型：

```json
{
  "run_research_suite": false,
  "model_label": "M"
}
```

## 4. 干预格式

```json
{
  "intervention_id": "INT_DAY03_SHOP_TO_PARK",
  "effective_day": 3,
  "operation_type": "AOI_FUNCTION_CHANGE",
  "target_type": "AOI",
  "target_id": "AOI_N12002341738",
  "before_value": "SHOP",
  "after_value": "PARK",
  "visibility_level": 0.9,
  "announcement_level": 0.0
}
```

支持 `AOI_FUNCTION_CHANGE`、`ROAD_OPEN`、`ROAD_CLOSE`、`ENTRANCE_OPEN`、`ENTRANCE_CLOSE`、`SIGNAGE_ACTIVATE` 和 `SIGNAGE_DEACTIVATE`。

## 5. 认知与线下传播

1. Agent 根据现有认知地图安排日程和路线。
2. 视觉模块只在实际轨迹的视距、方向、距离和遮挡条件满足时生成信息。
3. 道路失败和实际到访产生亲历信息。
4. 两名 Agent 只有在同一 AOI 的停留时间真实重叠时才可能交谈。
5. 说话者只能分享自己获得的、可追溯的一手空间变化。
6. 豆包把对话报告整理为结构化字段；有效报告送达后立即写入听者的二元认知地图。

## 6. 主要输出

过程表：

- `environment_daily_state`：每日真实地图；
- `cognitive_*_daily_state`：每个 Agent 每日认知地图；
- `activity_episode_log`：活动日程和目的地；
- `planned_trajectory_log` / `executed_trajectory_log`：计划与实际路径；
- `perception_diagnostic_log`：视角、距离、遮挡和发现结果；
- `observation_event_log`：视觉、亲历和面对面信息；
- `information_item`：到达 Agent 的信息及来源链；
- `cognitive_update_log`：认知地图何时写入了什么；
- `copresence_event` / `interpersonal_exchange`：共处、交谈和报告传递；
- `llm_decision_log`：豆包调用、结构化响应和耗时。

RQ 指标表：

- RQ1：`cognitive_lag_daily`、`cognitive_lag_population_daily`；
- RQ2：`update_timing`、`information_process_daily`；
- RQ3：`route_outcome_daily`、`activity_distribution_daily`、`objective_accessibility_daily`、`model_comparison`；
- 统一长表：`daily_metric`。

每个数据库完成后，对应 CSV 位于：

```text
output/research_outputs/<数据库名>/
```

## 7. 安装与测试

```powershell
python -m pip install -e ".[full]"
python -m unittest discover -s tests -v
```

建议使用 Python 3.10 或更高版本。地图处理依赖 `pyshp`、`pyproj`、`shapely` 和 `networkx`；豆包 Responses API 使用标准 HTTP 请求。
