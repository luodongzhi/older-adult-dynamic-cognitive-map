from __future__ import annotations

import uuid
from collections import defaultdict

from .models import InformationItem, ObservationEvent, SourceChannel


class InformationLedger:
    """Source-addressable memory of information that actually reached each Agent.

    The ledger does not score, decay, or accumulate weighted support. A valid visual,
    direct-experience, route-failure, or delivered face-to-face observation is
    information arrival and is therefore eligible for an immediate cognitive
    map write at the current daily simulation step.
    """

    def __init__(self) -> None:
        self._items: dict[str, list[InformationItem]] = defaultdict(list)
        self._dedup: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)

    def ingest(
        self,
        day: int,
        agent_id: str,
        events: list[ObservationEvent],
    ) -> list[InformationItem]:
        created: list[InformationItem] = []
        for event in sorted(events, key=lambda item: item.event_time):
            if event.source_channel not in {
                SourceChannel.VISUAL_SEARCH,
                SourceChannel.DIRECT_EXPERIENCE,
                SourceChannel.ROUTE_FAILURE,
                SourceChannel.INTERPERSONAL,
                SourceChannel.PUBLIC_ANNOUNCEMENT,
            }:
                continue
            source_event_id = event.source_event_id or event.event_id
            independence_group = event.independence_group or (
                f"FIRSTHAND:{agent_id}:{event.object_type}:"
                f"{event.object_id}:{event.observed_attribute}"
            )
            key = (
                independence_group,
                event.object_id,
                event.observed_attribute,
                event.observed_value,
            )
            if key in self._dedup[agent_id]:
                continue
            self._dedup[agent_id].add(key)
            item = InformationItem(
                information_id=f"INF_{uuid.uuid4().hex[:16]}",
                day=day,
                agent_id=agent_id,
                object_id=event.object_id,
                object_type=event.object_type,
                attribute_name=event.observed_attribute,
                observed_value=event.observed_value,
                source_channel=event.source_channel,
                source_event_id=source_event_id,
                reliability=event.source_reliability,
                salience=event.salience,
                independence_group=independence_group,
                source_agent_id=event.source_agent_id,
                copresence_event_id=event.copresence_event_id,
                first_hand_source=(
                    event.source_channel != SourceChannel.INTERPERSONAL
                    if event.first_hand_source is None else event.first_hand_source
                ),
            )
            self._items[agent_id].append(item)
            created.append(item)
        return created

    def items_for(self, agent_id: str, *, first_hand_only: bool = False) -> list[InformationItem]:
        items = list(self._items.get(agent_id, []))
        if first_hand_only:
            return [item for item in items if item.first_hand_source]
        return items
