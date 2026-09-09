from __future__ import annotations

import random
import uuid
from collections import defaultdict
from itertools import combinations
from typing import Any, Callable

from .models import (
    ActivityStop,
    AgentProfile,
    CopresenceEvent,
    InformationItem,
    InterpersonalExchange,
    ObservationEvent,
    SourceChannel,
)


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class CoPresenceInteractionEngine:
    """Generate face-to-face information from actual place-time overlap.

    Co-presence, conversation, report selection, and delivery remain separate
    auditable stages. There is no global feed or online public knowledge store;
    a delivered first-hand report is written into the listener's map.
    """

    def __init__(self, minimum_overlap_minutes: float = 8.0, max_reports: int = 2):
        self.minimum_overlap_minutes = max(0.0, float(minimum_overlap_minutes))
        self.max_reports = max(1, int(max_reports))
        self.encounter_counts: dict[tuple[str, str], int] = defaultdict(int)

    def find_copresence(
        self,
        day: int,
        profiles: list[AgentProfile],
        stops: list[ActivityStop],
        rng_for: Callable[[str], random.Random],
    ) -> list[CopresenceEvent]:
        by_place: dict[str, list[ActivityStop]] = defaultdict(list)
        for stop in stops:
            if stop.activity_type != "HOME" and stop.end_time > stop.start_time:
                by_place[stop.place_id].append(stop)
        profile_by_id = {profile.agent_id: profile for profile in profiles}
        best_overlap: dict[tuple[str, str, str], tuple[float, float]] = {}
        for place_id, place_stops in by_place.items():
            for left, right in combinations(place_stops, 2):
                if left.agent_id == right.agent_id:
                    continue
                start = max(left.start_time, right.start_time)
                end = min(left.end_time, right.end_time)
                overlap_minutes = max(0.0, (end - start) * 60.0)
                if overlap_minutes < self.minimum_overlap_minutes:
                    continue
                agent_a, agent_b = sorted((left.agent_id, right.agent_id))
                key = (agent_a, agent_b, place_id)
                current = best_overlap.get(key)
                if current is None or end - start > current[1] - current[0]:
                    best_overlap[key] = (start, end)

        events: list[CopresenceEvent] = []
        for (agent_a, agent_b, place_id), (start, end) in sorted(best_overlap.items()):
            left = profile_by_id[agent_a]
            right = profile_by_id[agent_b]
            pair = (agent_a, agent_b)
            stable = agent_b in left.stable_contact_ids or agent_a in right.stable_contact_ids
            relationship = 0.72 if stable else min(0.68, 0.18 + 0.07 * self.encounter_counts[pair])
            overlap_minutes = (end - start) * 60.0
            overlap_factor = min(1.0, overlap_minutes / 60.0)
            sociality = (left.social_participation + right.social_participation) / 2.0
            contact_frequency = (left.social_contact_frequency + right.social_contact_frequency) / 2.0
            probability = max(
                0.0,
                min(0.98, 0.05 + 0.30 * sociality + 0.25 * contact_frequency + 0.25 * relationship + 0.15 * overlap_factor),
            )
            rng = rng_for(f"{agent_a}|{agent_b}|{place_id}|conversation")
            draw = rng.random()
            event = CopresenceEvent(
                event_id=_uid("COP"), day=day, agent_a=agent_a, agent_b=agent_b,
                place_id=place_id, overlap_start=start, overlap_end=end,
                overlap_minutes=overlap_minutes, relationship_strength=relationship,
                conversation_probability=probability, conversation_draw=draw,
                conversation_flag=draw < probability,
            )
            events.append(event)
            self.encounter_counts[pair] += 1
        return events

    def build_reports(
        self,
        events: list[CopresenceEvent],
        observations_by_agent: dict[str, list[ObservationEvent]],
        information_by_agent: dict[str, list[InformationItem]],
    ) -> dict[str, list[dict[str, Any]]]:
        reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            if not event.conversation_flag:
                continue
            for speaker, listener in ((event.agent_a, event.agent_b), (event.agent_b, event.agent_a)):
                candidates = self._speaker_candidates(
                    speaker,
                    event.overlap_end,
                    observations_by_agent.get(speaker, []),
                    information_by_agent.get(speaker, []),
                )
                for candidate in candidates[: self.max_reports]:
                    report_id = _uid("REPORT")
                    first_hand = bool(candidate["first_hand_source"])
                    prefix = "我亲眼看到/亲自遇到" if first_hand else "我听别人说"
                    reports[listener].append({
                        "report_id": report_id,
                        "copresence_event_id": event.event_id,
                        "speaker_agent_id": speaker,
                        "listener_agent_id": listener,
                        "place_id": event.place_id,
                        "event_time": event.overlap_end,
                        "relationship_strength": event.relationship_strength,
                        "object_id": candidate["object_id"],
                        "object_type": candidate["object_type"],
                        "observed_attribute": candidate["observed_attribute"],
                        "observed_value": candidate["observed_value"],
                        "source_event_id": candidate["source_event_id"],
                        "independence_group": candidate["independence_group"],
                        "first_hand_source": first_hand,
                        "source_reliability": candidate["source_reliability"],
                        "salience": candidate["salience"],
                        "utterance": (
                            f"{prefix}，{candidate['object_id']} 的 "
                            f"{candidate['observed_attribute']} 现在是 {candidate['observed_value']}。"
                        ),
                    })
        return reports

    @staticmethod
    def _speaker_candidates(
        speaker: str,
        encounter_end: float,
        observations: list[ObservationEvent],
        information: list[InformationItem],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for event in observations:
            if event.event_time > encounter_end or event.source_channel == SourceChannel.INTERPERSONAL:
                continue
            source_event_id = event.source_event_id or event.event_id
            candidates.append({
                "object_id": event.object_id,
                "object_type": event.object_type,
                "observed_attribute": event.observed_attribute,
                "observed_value": event.observed_value,
                "source_event_id": source_event_id,
                "independence_group": event.independence_group or (
                    f"FIRSTHAND:{speaker}:{event.object_type}:"
                    f"{event.object_id}:{event.observed_attribute}"
                ),
                "first_hand_source": True,
                "source_reliability": event.source_reliability,
                "salience": event.salience,
                "score": event.source_reliability * event.salience + 0.3,
            })
        for item in information:
            # The core experiment transmits only traceable first-hand knowledge.
            # This keeps "heard = known" from creating unbounded rumour cascades.
            if not item.first_hand_source:
                continue
            candidates.append({
                "object_id": item.object_id,
                "object_type": item.object_type,
                "observed_attribute": item.attribute_name,
                "observed_value": item.observed_value,
                "source_event_id": item.source_event_id,
                "independence_group": item.independence_group or f"INFORMATION:{speaker}:{item.source_event_id}",
                "first_hand_source": bool(item.first_hand_source),
                "source_reliability": item.reliability,
                "salience": item.salience,
                "score": item.reliability * item.salience + 0.2,
            })
        deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for candidate in candidates:
            key = (
                candidate["object_id"], candidate["observed_attribute"],
                candidate["independence_group"],
            )
            if key not in deduped or candidate["score"] > deduped[key]["score"]:
                deduped[key] = candidate
        return sorted(deduped.values(), key=lambda item: item["score"], reverse=True)

    def materialize_exchange(
        self,
        day: int,
        report: dict[str, Any],
    ) -> tuple[InterpersonalExchange, ObservationEvent]:
        relationship = float(report["relationship_strength"])
        first_hand = bool(report["first_hand_source"])
        exchange = InterpersonalExchange(
            exchange_id=_uid("XCH"), day=day,
            copresence_event_id=str(report["copresence_event_id"]),
            speaker_agent_id=str(report["speaker_agent_id"]),
            listener_agent_id=str(report["listener_agent_id"]), place_id=str(report["place_id"]),
            report_id=str(report["report_id"]), object_id=str(report["object_id"]),
            object_type=str(report["object_type"]),
            observed_attribute=str(report["observed_attribute"]),
            observed_value=str(report["observed_value"]),
            source_event_id=str(report["source_event_id"]),
            independence_group=str(report["independence_group"]),
            first_hand_source=first_hand,
            delivered=True,
        )
        reliability = min(
            1.0,
            float(report["source_reliability"]) * (0.55 + 0.45 * relationship)
            * (1.0 if first_hand else 0.78),
        )
        observation = ObservationEvent(
            event_id=_uid("OBS_FACE"), day=day, event_time=float(report["event_time"]),
            agent_id=str(report["listener_agent_id"]),
            episode_id=f"D{day}_{report['listener_agent_id']}_FACE_TO_FACE",
            object_id=str(report["object_id"]), object_type=str(report["object_type"]),
            source_channel=SourceChannel.INTERPERSONAL,
            observed_attribute=str(report["observed_attribute"]),
            observed_value=str(report["observed_value"]),
            source_reliability=reliability, salience=float(report["salience"]) * 0.75,
            source_event_id=str(report["source_event_id"]),
            independence_group=str(report["independence_group"]),
            source_agent_id=str(report["speaker_agent_id"]),
            copresence_event_id=str(report["copresence_event_id"]),
            first_hand_source=first_hand,
        )
        return exchange, observation
