# 研究设计与指标说明

## 1. 核心研究对象

真实地图在第 `T_k` 天发生干预。Agent 不能直接读取真实变化，只能通过当天的活动路线获得信息。系统观察从 `T_k` 开始到 Agent 正确认识并使用该变化之间的时间差。

认知地图采用二元表示：

- `UNKNOWN`：Agent 不知道该对象当前状态；
- 已知状态：例如道路 `OPEN/CLOSED`，AOI `SHOP/PARK`。

视觉、亲历或面对面信息在第 `t` 天到达后，系统在第 `t` 天的认知快照中写入该状态。由于日程在当天早晨已生成，这一更新从第 `t+1` 天的行为决策开始生效。

## 2. 统一因果链

```text
城市干预
  → Agent 的实际路线与停留
  → 视觉 / 亲历 / 线下相遇带来的信息到达
  → 二元认知地图更新
  → 下一日的目的地和路径选择
  → 群体活动分布与路线结果
```

豆包只负责两项高层任务：提出活动意图，以及把面对面对话整理成结构化报告。地图状态、路径搜索、视觉几何、共处判断、认知写入和指标计算均由可审计的 Python 模块完成。

## 3. RQ1：有限信息会造成多长的认知滞后？

对干预 `k`、Agent `i` 和日期 `t`，定义：

```text
L_ikt = 0，若 Agent 的认知状态等于真实状态
L_ikt = 1，若状态未知、仍为旧状态或为其他错误状态
```

群体认知滞后率：

```text
LagRate_kt = Σ_i L_ikt / N
```

正确更新率：

```text
CorrectRate_kt = 正确认识干预 k 的 Agent 数 / N
```

首次信息延迟和认知滞后天数：

```text
InformationDelay_ik = FirstInformationDay_ik - T_k
ObsoleteDuration_ik = FirstCorrectUpdateDay_ik - T_k
```

若实验结束仍未正确更新，则记录为右删失，而不是人为填入一个更新时间。`t50` 和 `t80` 分别是群体首次达到 50% 和 80% 正确更新的日期。

主要输出：

- `cognitive_lag_daily`：个体日级状态及 `lag_indicator`；
- `cognitive_lag_population_daily`：群体曲线、`t50`、`t80`；
- `update_timing`：首次信息、首次正确更新和删失状态。

RQ1 回答“滞后有多大、持续多久”，不再构造人为加权的认知分数。

## 4. RQ2：日常路线与线下相遇如何决定新信息何时到达？

RQ2 的结果变量是 `InformationDelay` 或 `FirstInformationDay`。每条信息必须能够追溯到以下渠道之一：

- `VISUAL_SEARCH`：对象位于实际路线的视距和视野角内，且未被建筑遮挡；
- `DIRECT_EXPERIENCE`：Agent 到达 AOI 或实际通过对象；
- `ROUTE_FAILURE`：计划道路与真实关闭状态冲突；
- `INTERPERSONAL`：两名 Agent 在同一 AOI 真实共处并交谈，一手报告送达听者。

机制参数分成三类：

- 路线暴露：日程、活动半径、路线习惯和实际经过对象；
- 视觉接触：视觉注意、发现能力、视距、视角和遮挡；
- 线下接触：社会参与、接触频率、共处时长和稳定关系。

实验通过单独关闭 `directional_vision_enabled`、`direct_experience_enabled` 或 `face_to_face_enabled` 做机制消融。统计分析可对 `InformationDelay` 使用离散时间生存模型或 Cox 模型，并将未获知者作为右删失样本。

主要输出：

- `observation_event_log`：每次信息到达；
- `perception_diagnostic_log`：每次视觉判定的空间条件；
- `copresence_event`：是否共处、是否交谈；
- `interpersonal_exchange`：谁向谁传递了什么；
- `information_process_daily`：每日各渠道数量和干预覆盖；
- `update_timing`：首次视觉、亲历、面对面和首次信息日期。

RQ2 回答“信息为什么早到或晚到”，而不是“Agent 多相信这条信息”。

## 5. RQ3：信息延迟会改变多少行为预测？

三模型使用同一地图、Agent、干预和随机种子：

```text
B0：客观可达性，不生成个体活动
B1：全知 Agent，行为始终使用真实地图
M ：动态认知 Agent，行为只使用当前认知地图
```

比较逻辑：

- `B0 - B1`：从客观空间机会到实际活动过程增加了什么；
- `B1 - M`：仅因为信息有限和更新延迟，行为预测改变了多少。

核心指标：

- `activity_success_rate`：计划活动成功的比例；
- `destination_js_divergence`：B1 与 M 的目的地份额差异，0 表示相同；
- `route_difference_ratio`：计划路径和实际路径的集合差异；
- `route_failures`：因真实地图与认知地图不一致造成的失败；
- `detour_distance`：实际路程超过计划路程的部分；
- `intervention_use_count`：实际使用干预道路或干预 AOI 的次数；
- `stability_day`：目的地分布连续稳定的起始日。

主要输出：

- `route_outcome_daily`；
- `activity_distribution_daily`；
- `objective_accessibility_daily`；
- `model_comparison`；
- `output/research_outputs/rq3_model_comparison.csv`。

## 6. 模块分工

| 模块 | 作用 | 关键输出 |
|---|---|---|
| M00–M03 | 管理实验、真实地图、干预和 Agent 状态 | 每日环境与个体状态 |
| M04 | 建立和维护二元认知地图 | 认知快照 |
| M05–M06 | 生成日程并选择目的地 | 活动 Episode、目的地决策 |
| M07–M08 | 在认知图上规划，在真实图上执行 | 计划/实际路径、失败和绕行 |
| M09 | 根据轨迹生成视觉和亲历信息 | 观察事件、视觉诊断 |
| 线下交互 | 根据停留重叠生成交谈和报告 | 共处事件、报告传递 |
| M10 | 信息到达后写入二元认知地图 | 信息账本、认知更新 |
| M11 | 保存数据并计算 RQ1–RQ3 指标 | SQLite、CSV、网页 |

## 7. 最小可复现实验

每轮实验必须同时固定：

- 地图版本与处理参数；
- Agent 档案和地图位置绑定；
- 干预列表；
- 运行天数与随机种子；
- 模型标签和 LLM 版本。

`output/*_inputs.resolved.json` 保存解析后的非敏感输入；SQLite 的 `run_manifest` 保存运行配置。API Key 不进入任何结果文件。
