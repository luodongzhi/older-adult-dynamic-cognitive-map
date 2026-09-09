from __future__ import annotations

import random
import unittest

from cogmap_sim.llm import DoubaoResponsesDriver
from cogmap_sim.models import AgentProfile


class FakeDoubaoDriver(DoubaoResponsesDriver):
    def __init__(self) -> None:
        super().__init__(api_key="test-key")
        self.last_payload = None

    def _http_post(self, payload):
        self.last_payload = payload
        return {
            "object": "response",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": '{"flexible_activity_type":"PARK","destination_preferences":{"CAFE":[],"PARK":["P1"],"SHOP":[]},"reasoning":"test"}',
                }],
            }],
        }


class RetryDoubaoDriver(FakeDoubaoDriver):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def _http_post(self, payload):
        self.calls += 1
        if self.calls == 1:
            return {"object": "response", "output": [{"type": "reasoning"}], "incomplete_details": {"reason": "length"}}
        return super()._http_post(payload)


class MalformedThenValidDriver(FakeDoubaoDriver):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def _http_post(self, payload):
        self.calls += 1
        if self.calls == 1:
            return {"output_text": '{"parsed_reports":[] "reasoning":"bad"}'}
        return {
            "output_text": '{"parsed_reports":[{"report_id":"R1","object_id":"FAKE","observed_attribute":"fake","observed_value":"fake"},{"report_id":"UNKNOWN","object_id":"FAKE","observed_attribute":"fake","observed_value":"fake"}],"reasoning":"ok"}'
        }


class AlwaysMalformedDriver(FakeDoubaoDriver):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def _http_post(self, payload):
        self.calls += 1
        return {"output_text": '{"parsed_reports": [] "reasoning": "bad"}'}


def report() -> dict:
    return {
        "report_id": "R1", "object_id": "P1", "object_type": "AOI",
        "observed_attribute": "function", "observed_value": "PARK",
        "utterance": "我看到 P1 现在是公园。",
    }


def profile() -> AgentProfile:
    return AgentProfile(
        agent_id="A_TEST", group_id="test", home_aoi_id="HOME",
        routine_anchor_aoi_id="WORK", exploration_tendency=0.5,
        map_search_tendency=0.5, social_participation=0.5,
        safety_sensitivity=0.5, route_habit=0.5, visual_attention=0.5,
        llm_model="doubao-seed-2-1-pro-260628",
    )


class DoubaoDriverTests(unittest.TestCase):
    def test_responses_payload_and_output_parsing(self) -> None:
        driver = FakeDoubaoDriver()
        decision = driver.plan_day(
            profile(), 1,
            [{"aoi_id": "P1", "believed_function": "PARK"}], [], random.Random(1),
        )
        self.assertEqual(decision["flexible_activity_type"], "PARK")
        self.assertEqual(driver.last_payload["model"], "doubao-seed-2-1-pro-260628")
        self.assertEqual(driver.last_payload["input"][1]["role"], "user")
        self.assertEqual(driver.last_payload["text"]["format"]["type"], "json_schema")
        self.assertEqual(driver.last_payload["thinking"]["type"], "disabled")
        self.assertEqual(driver.last_payload["max_output_tokens"], 1500)
        self.assertEqual(driver.pop_logs()[0]["provider"], "doubao")

    def test_length_truncation_retries_with_larger_budget(self) -> None:
        driver = RetryDoubaoDriver()
        decision = driver.plan_day(profile(), 1, [], [], random.Random(1))
        self.assertEqual(driver.calls, 2)
        self.assertEqual(driver.last_payload["max_output_tokens"], 4000)
        self.assertEqual(decision["flexible_activity_type"], "PARK")

    def test_invalid_json_is_retried_and_ids_are_validated(self) -> None:
        driver = MalformedThenValidDriver()
        decision = driver.interpret_face_to_face(profile(), 1, [report()], random.Random(1))
        self.assertEqual(driver.calls, 2)
        self.assertEqual([item["report_id"] for item in decision["parsed_reports"]], ["R1"])
        self.assertEqual(decision["parsed_reports"][0]["object_id"], "P1")
        self.assertEqual(decision["parsed_reports"][0]["observed_attribute"], "function")
        self.assertTrue(decision["_adapter"]["format_retry"])
        self.assertEqual(driver.pop_logs()[0]["status"], "OK_AFTER_RETRY")

    def test_repeated_invalid_json_uses_explicit_safe_fallback(self) -> None:
        driver = AlwaysMalformedDriver()
        decision = driver.interpret_face_to_face(profile(), 1, [report()], random.Random(1))
        self.assertEqual(driver.calls, 3)
        self.assertEqual(decision["parsed_reports"], [])
        log = driver.pop_logs()[0]
        self.assertEqual(log["provider"], "doubao-fallback")
        self.assertEqual(log["status"], "FALLBACK_INVALID_JSON")


if __name__ == "__main__":
    unittest.main()
