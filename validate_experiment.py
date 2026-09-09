"""PyCharm 可直接运行的实验输入预检工具；不会调用 LLM，也不会创建数据库。"""
from __future__ import annotations

from run_pycharm import (
    attach_input_provenance,
    load_agents,
    load_interventions,
    load_or_import_map,
    load_settings,
)


def _target_description(event, graph) -> str:
    if event.target_type == "AOI":
        item = graph.aois[event.target_id]
        return f"{item.name or '(无名称)'} / 当前功能={item.function}"
    if event.target_type == "ROAD_EDGE":
        item = graph.edges[event.target_id]
        return f"{item.name or '(无名称道路)'} / 当前状态={item.status}"
    if event.target_type == "ENTRANCE":
        item = graph.entrances[event.target_id]
        return f"所属 AOI={item.aoi_id} / 当前状态={item.status}"
    if event.target_type == "SIGNAGE":
        item = graph.signage[event.target_id]
        return f"标识={getattr(item, 'name', event.target_id)}"
    return event.target_id


def main() -> None:
    settings = load_settings()
    graph = load_or_import_map(settings)
    agents = load_agents(settings, graph)
    interventions = load_interventions(settings, graph)
    attach_input_provenance(settings, graph, agents)

    print("\n========== 实验输入预检通过 ==========")
    print(f"实验目录：{settings.get('_experiment_directory')}")
    print(f"实验 ID：{settings['experiment_id']}")
    print(f"运行天数：{settings['days']}")
    print(f"模型组合：{', '.join(settings.get('research_models', [settings.get('model_label', 'M')]))}")
    print(f"Agent 数：{len(agents)}")
    for agent in agents:
        print(
            f"  - {agent.agent_id}: HOME={agent.home_aoi_id}, "
            f"日常锚点={agent.routine_anchor_aoi_id}, 会面点={agent.regular_meeting_aoi_ids}"
        )
    print(f"干预数：{len(interventions)}")
    for event in interventions:
        print(
            f"  - Day {event.effective_day} | {event.intervention_id} | "
            f"{event.operation_type} | {_target_description(event, graph)} | "
            f"{event.before_value} -> {event.after_value}"
        )
    print("没有调用豆包 API，也没有写入实验数据库。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[预检失败] {error}")
        print("请依次检查 experiments/current 中的 agents.json、run.json、interventions.json 和 map.json。")
        raise
