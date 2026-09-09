from __future__ import annotations

import math
import random
from dataclasses import replace

from .graph import UrbanGraph
from .models import AOI, AgentProfile, Edge, EdgeStatus, InterventionEvent, Node, ScenarioConfig


GROUPS = {
    "active_connected_older": dict(exploration=0.62, search=0.72, social=0.82, safety=0.48, habit=0.42, attention=0.82, age=64, speed=72.0),
    "long_term_routine_older": dict(exploration=0.25, search=0.40, social=0.58, safety=0.72, habit=0.88, attention=0.68, age=72, speed=62.0),
    "mobility_limited_older": dict(exploration=0.15, search=0.32, social=0.30, safety=0.92, habit=0.82, attention=0.58, age=81, speed=48.0),
}


def build_demo_scenario(
    days: int = 30,
    agent_count: int = 3,
    seed: int = 42,
    model_type: str = "cognitive",
    agent_profiles: list[AgentProfile] | None = None,
) -> tuple[ScenarioConfig, UrbanGraph, list[AgentProfile], list[InterventionEvent]]:
    """Build a deterministic synthetic district with a central pedestrian intervention."""
    rng = random.Random(seed)
    nodes = {f"N{x}{y}": Node(f"N{x}{y}", float(x), float(y)) for x in range(5) for y in range(4)}
    edges: dict[str, Edge] = {}
    for x in range(5):
        for y in range(4):
            if x < 4:
                eid = f"H{x}{y}"
                edges[eid] = Edge(eid, f"N{x}{y}", f"N{x+1}{y}", 1.0, 1.0 + rng.random() * 0.15)
            if y < 3:
                eid = f"V{x}{y}"
                edges[eid] = Edge(eid, f"N{x}{y}", f"N{x}{y+1}", 1.0, 1.0 + rng.random() * 0.15)
    # A new diagonal link makes the central district substantially more accessible.
    edges["NEW_LINK"] = Edge(
        "NEW_LINK", "N20", "N40", 2.0, 0.9, EdgeStatus.CLOSED,
        visibility=0.92, safety=0.88, comfort=0.9, is_intervention=True,
    )
    # A familiar central street is closed, forcing belief correction and rerouting.
    edges["H20"].is_intervention = True

    aois = {
        "HOME_W": AOI("HOME_W", "West Housing", "N00", "HOME", 0.45),
        "HOME_E": AOI("HOME_E", "East Housing", "N43", "HOME", 0.45),
        "OFFICE_W": AOI("OFFICE_W", "Civic Offices", "N03", "WORK", 0.6),
        "OFFICE_E": AOI("OFFICE_E", "Innovation Campus", "N40", "WORK", 0.65),
        "CAFE_N": AOI("CAFE_N", "North Cafe", "N13", "CAFE", 0.65, comfort=0.78),
        "CAFE_S": AOI("CAFE_S", "South Cafe", "N30", "CAFE", 0.6),
        "SHOP_W": AOI("SHOP_W", "West Market", "N01", "SHOP", 0.62),
        "SHOP_E": AOI("SHOP_E", "East Market", "N42", "SHOP", 0.66),
        "PARK_OLD": AOI("PARK_OLD", "Riverside Green", "N23", "PARK", 0.63, comfort=0.72),
        "PLAZA_NEW": AOI("PLAZA_NEW", "Foundry Commons", "N22", "INDUSTRIAL", 0.3, safety=0.86, comfort=0.9, capacity=260, is_intervention=True),
    }
    graph = UrbanGraph(nodes, edges, aois)

    group_names = list(GROUPS)
    agents: list[AgentProfile] = []
    for index in range(agent_count):
        group_id = group_names[index % len(group_names)]
        params = GROUPS[group_id]
        west_home = index % 2 == 0
        agent_id = f"A{index + 1:03d}"
        agents.append(AgentProfile(
            agent_id=agent_id, group_id=group_id,
            home_aoi_id="HOME_W" if west_home else "HOME_E",
            routine_anchor_aoi_id="PARK_OLD",
            exploration_tendency=_jitter(rng, params["exploration"]),
            map_search_tendency=_jitter(rng, params["search"]),
            social_participation=_jitter(rng, params["social"]),
            safety_sensitivity=_jitter(rng, params["safety"]),
            route_habit=_jitter(rng, params["habit"]),
            visual_attention=_jitter(rng, params["attention"]),
            social_contact_frequency=_jitter(rng, params["social"]),
            visual_search_radius=1.25,
            visual_detection_skill=_jitter(rng, params["attention"]),
            age=int(params["age"] + index % 3),
            years_residence=float(8 + (index % 4) * 9),
            risk_aversion=_jitter(rng, params["safety"]),
            walking_speed_m_per_min=float(params["speed"]),
            activity_radius_m=float(1100 - 180 * (index % 3)),
            physical_cost_sensitivity=_jitter(rng, params["safety"]),
            employment_status="retired",
            occupation="retired",
            work_days=[],
            chronotype="morning",
            regular_meeting_aoi_ids=["PARK_OLD"],
            meeting_frequency_weekly=max(2.0, 6.0 * params["social"]),
            meeting_start_mean=15.0,
            meeting_start_std=0.25,
            stable_contact_ids=[f"A{((index + 1) % max(1, agent_count)) + 1:03d}"] if agent_count > 1 else [],
            system_prompt=(
                "You represent an independently mobile older neighbourhood resident. "
                "Plan only from known places, protect energy and safety, retain familiar routines, "
                "and add a spatial change to your map when you see, experience, or hear it directly."
            ),
        ))
    if agent_profiles is not None:
        agents = [
            replace(
                profile,
                home_aoi_id="HOME_W" if index % 2 == 0 else "HOME_E",
                routine_anchor_aoi_id="PARK_OLD",
                regular_meeting_aoi_ids=["PARK_OLD"],
            )
            for index, profile in enumerate(agent_profiles)
        ]
        agent_count = len(agents)

    interventions = [
        InterventionEvent("INT_OPEN", 0, "ROAD_OPEN", "ROAD_EDGE", "NEW_LINK", "CLOSED", "OPEN", 0.92, 0.35),
        InterventionEvent("INT_CLOSE", 0, "ROAD_CLOSE", "ROAD_EDGE", "H20", "OPEN", "CLOSED", 0.82, 0.2),
        InterventionEvent("INT_PLAZA", 0, "AOI_FUNCTION_CHANGE", "AOI", "PLAZA_NEW", "INDUSTRIAL", "PARK", 0.88, 0.45),
    ]
    config = ScenarioConfig(end_day=days, random_seed=seed, model_type=model_type, agent_count=agent_count)
    return config, graph, agents, interventions


def _jitter(rng: random.Random, value: float) -> float:
    return min(0.98, max(0.02, value + rng.uniform(-0.08, 0.08)))
