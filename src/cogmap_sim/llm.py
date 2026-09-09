from __future__ import annotations

import json
import os
import random
import ssl
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from .models import AgentProfile


class AgentLLMDriver(ABC):
    """Behavioral layer. Geometry and physical routing remain deterministic tools."""

    def __init__(self) -> None:
        self._logs: list[dict[str, Any]] = []

    @abstractmethod
    def plan_day(self, profile: AgentProfile, day: int, known_aois: list[dict[str, Any]], recent_memory: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def interpret_face_to_face(
        self, profile: AgentProfile, day: int, reports: list[dict[str, Any]], rng: random.Random,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def pop_logs(self) -> list[dict[str, Any]]:
        logs, self._logs = self._logs, []
        return logs

    def _log(
        self, profile: AgentProfile, day: int, phase: str, provider: str,
        request: dict[str, Any], response: dict[str, Any], model: str,
        latency_ms: float = 0.0, status: str = "OK",
    ) -> None:
        self._logs.append({
            "call_id": f"LLM_{uuid.uuid4().hex[:16]}", "simulation_day": day,
            "agent_id": profile.agent_id, "phase": phase, "provider": provider,
            "model": model, "request_json": request, "response_json": response,
            "latency_ms": latency_ms, "status": status,
        })


class RuleBasedBehaviorDriver(AgentLLMDriver):
    """Reproducible behavioral core with the same contract as optional LLM adapters."""

    def plan_day(self, profile: AgentProfile, day: int, known_aois: list[dict[str, Any]], recent_memory: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
        frequencies = profile.activity_frequency_weekly
        needs = profile.initial_needs
        park_weight = frequencies.get("PARK", 0.0) * (
            0.45 + profile.exploration_tendency + 0.6 * needs.get("leisure", 0.5)
        )
        shop_weight = frequencies.get("SHOP", 0.0) * (
            0.55 + 0.8 * needs.get("errand", 0.35) + 0.25 * profile.route_habit
        )
        none_weight = max(
            0.05,
            7.0 * profile.stay_home_preference * (1.25 - 0.5 * profile.spontaneity),
        )
        flexible = rng.choices(
            ["PARK", "SHOP", "NONE"],
            weights=[max(0.0, park_weight), max(0.0, shop_weight), none_weight],
            k=1,
        )[0]
        preferences: dict[str, list[str]] = {}
        for activity in ("CAFE", "PARK", "SHOP"):
            candidates = [aoi for aoi in known_aois if aoi["believed_function"] == activity]
            candidates.sort(
                key=lambda aoi: (
                    aoi["perceived_attractiveness"] + 0.35 * aoi["perceived_safety"]
                ),
                reverse=True,
            )
            preferences[activity] = [aoi["aoi_id"] for aoi in candidates]
        response = {"flexible_activity_type": flexible, "destination_preferences": preferences, "reasoning": "profile frequencies, needs and routine priors"}
        self._log(profile, day, "PLAN_DAY", "rule_based", {"day_context": _schedule_context(profile, day), "known_aois": known_aois, "memory": recent_memory}, response, "rule_based")
        return response

    def interpret_face_to_face(
        self, profile: AgentProfile, day: int, reports: list[dict[str, Any]], rng: random.Random,
    ) -> dict[str, Any]:
        parsed = [{
            "report_id": report["report_id"],
            "object_id": report["object_id"],
            "observed_attribute": report["observed_attribute"],
            "observed_value": report["observed_value"],
        } for report in reports]
        response = {"parsed_reports": parsed, "reasoning": "deterministic parsing of grounded face-to-face reports"}
        self._log(profile, day, "PARSE_CONVERSATION", "rule_based", {"reports": reports}, response, "rule_based")
        return response


class DeterministicLLMDriver(RuleBasedBehaviorDriver):
    """Backward-compatible name used by older scripts and tests."""


class OpenAIResponsesDriver(AgentLLMDriver):
    """OpenAI Responses API driver using strict JSON-schema outputs."""

    def __init__(self, default_model: str = "gpt-5.4-mini", api_key: str | None = None):
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI behavior requires the SDK. Install with: pip install -e .[llm]") from exc
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set. Set it, or use --llm-provider rule_based for an offline dry run.")
        self.client = OpenAI(api_key=key)
        self.default_model = default_model

    def plan_day(self, profile: AgentProfile, day: int, known_aois: list[dict[str, Any]], recent_memory: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
        request = {"day": day, "day_context": _schedule_context(profile, day), "resident_attributes": _profile_context(profile), "known_aois": known_aois, "recent_memory": recent_memory, "task": "Using the resident's weekly frequencies, needs, routine and current day type, choose one optional activity type and rank at most 3 known destinations per activity. NONE is valid. Never choose an unknown AOI."}
        schema = {"type": "object", "properties": {"flexible_activity_type": {"type": "string", "enum": ["PARK", "SHOP", "NONE"]}, "destination_preferences": {"type": "object", "properties": {key: {"type": "array", "items": {"type": "string"}} for key in ("CAFE", "PARK", "SHOP")}, "required": ["CAFE", "PARK", "SHOP"], "additionalProperties": False}, "reasoning": {"type": "string"}}, "required": ["flexible_activity_type", "destination_preferences", "reasoning"], "additionalProperties": False}
        return self._call(profile, day, "PLAN_DAY", request, schema)

    def interpret_face_to_face(
        self, profile: AgentProfile, day: int, reports: list[dict[str, Any]], rng: random.Random,
    ) -> dict[str, Any]:
        request = _conversation_request(profile, day, reports)
        return self._call(profile, day, "PARSE_CONVERSATION", request, _conversation_schema())

    def _call(self, profile: AgentProfile, day: int, phase: str, request: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        model = profile.llm_model or self.default_model
        started = time.perf_counter()
        response = self.client.responses.create(
            model=model,
            instructions=profile.system_prompt + " Use only supplied cognitive state and conversation reports.",
            input=json.dumps(request, ensure_ascii=False),
            text={"format": {"type": "json_schema", "name": phase.lower(), "strict": True, "schema": schema}},
            max_output_tokens=700,
        )
        parsed = json.loads(response.output_text)
        if phase == "PARSE_CONVERSATION":
            parsed = DoubaoResponsesDriver._normalize_response(phase, request, parsed)
        self._log(profile, day, phase, "openai", request, parsed, model, (time.perf_counter() - started) * 1000)
        return parsed


class DoubaoResponsesDriver(AgentLLMDriver):
    """Volcengine Ark Responses API driver using ARK_API_KEY."""

    def __init__(
        self,
        default_model: str = "doubao-seed-2-1-pro-260628",
        api_key: str | None = None,
        endpoint: str | None = None,
    ):
        super().__init__()
        self.api_key = api_key or os.environ.get("ARK_API_KEY")
        if not self.api_key:
            raise RuntimeError("ARK_API_KEY is not set. Set it, or use --llm-provider rule_based for an offline dry run.")
        self.default_model = default_model
        self.endpoint = endpoint or os.environ.get("ARK_RESPONSES_URL", "https://ark.cn-beijing.volces.com/api/v3/responses")
        self.ssl_context = self._certifi_ssl_context()

    def plan_day(self, profile: AgentProfile, day: int, known_aois: list[dict[str, Any]], recent_memory: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
        request = {"day": day, "day_context": _schedule_context(profile, day), "resident_attributes": _profile_context(profile), "known_aois": known_aois, "recent_memory": recent_memory, "task": "Using the resident's weekly frequencies, needs, routine and current day type, choose one optional activity type and rank at most 3 known destinations per activity. NONE is valid. Never choose an unknown AOI."}
        schema = {"type": "object", "properties": {"flexible_activity_type": {"type": "string", "enum": ["PARK", "SHOP", "NONE"]}, "destination_preferences": {"type": "object", "properties": {key: {"type": "array", "items": {"type": "string"}, "maxItems": 3} for key in ("CAFE", "PARK", "SHOP")}, "required": ["CAFE", "PARK", "SHOP"], "additionalProperties": False}, "reasoning": {"type": "string", "maxLength": 160}}, "required": ["flexible_activity_type", "destination_preferences", "reasoning"], "additionalProperties": False}
        return self._call(profile, day, "PLAN_DAY", request, schema)

    def interpret_face_to_face(
        self, profile: AgentProfile, day: int, reports: list[dict[str, Any]], rng: random.Random,
    ) -> dict[str, Any]:
        request = _conversation_request(profile, day, reports)
        return self._call(profile, day, "PARSE_CONVERSATION", request, _conversation_schema())

    def _call(self, profile: AgentProfile, day: int, phase: str, request: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        model = profile.llm_model or self.default_model
        payload = {
            "model": model,
            "store": False,
            "input": [
                {"type": "message", "role": "system", "content": [{"type": "input_text", "text": profile.system_prompt + " Use only supplied cognitive state and conversation reports."}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": json.dumps(request, ensure_ascii=False)}]},
            ],
            "text": {"format": {"type": "json_schema", "name": phase.lower(), "strict": True, "schema": schema}},
            "thinking": {"type": "disabled"},
            "max_output_tokens": 1500,
        }
        started = time.perf_counter()
        errors: list[str] = []
        for attempt in range(1, 4):
            raw = self._http_post(payload)
            try:
                output_text = self._extract_output_text(raw)
            except RuntimeError as exc:
                incomplete = raw.get("incomplete_details") or {}
                errors.append(f"attempt {attempt}: {incomplete.get('reason') or str(exc)[:120]}")
                if incomplete.get("reason") == "length":
                    payload["max_output_tokens"] = 4000
                if attempt < 3:
                    self._add_json_correction(payload, request)
                    continue
                break
            try:
                parsed = json.loads(_strip_json_fence(output_text))
                parsed = self._normalize_response(phase, request, parsed)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {str(exc)[:160]}")
                if attempt < 3:
                    self._add_json_correction(payload, request)
                    continue
                break
            parsed["_adapter"] = {
                "api_attempts": attempt,
                "format_retry": attempt > 1,
            }
            self._log(
                profile, day, phase, "doubao", request, parsed, model,
                (time.perf_counter() - started) * 1000,
                "OK_AFTER_RETRY" if attempt > 1 else "OK",
            )
            return parsed

        fallback = self._safe_fallback(phase, request)
        fallback["_adapter"] = {
            "api_attempts": 3,
            "format_retry": True,
            "fallback_reason": "INVALID_STRUCTURED_OUTPUT",
            "errors": errors,
        }
        self._log(
            profile, day, phase, "doubao-fallback", request, fallback, model,
            (time.perf_counter() - started) * 1000, "FALLBACK_INVALID_JSON",
        )
        return fallback

    @staticmethod
    def _add_json_correction(payload: dict[str, Any], request: dict[str, Any]) -> None:
        corrected_request = {
            **request,
            "format_correction": (
                "The previous response was invalid. Return one compact JSON object only. "
                "Use double quotes, no markdown, no comments, and keep reasoning under 80 characters."
            ),
        }
        payload["input"][1]["content"][0]["text"] = json.dumps(corrected_request, ensure_ascii=False)

    @staticmethod
    def _normalize_response(phase: str, request: dict[str, Any], parsed: Any) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            raise ValueError("structured response must be a JSON object")
        reasoning = str(parsed.get("reasoning", ""))[:300]
        if phase == "PLAN_DAY":
            activity = str(parsed.get("flexible_activity_type", "NONE")).upper()
            if activity not in {"PARK", "SHOP", "NONE"}:
                raise ValueError(f"invalid flexible activity: {activity}")
            known_ids = {str(item.get("aoi_id")) for item in request.get("known_aois", [])}
            raw_preferences = parsed.get("destination_preferences")
            if not isinstance(raw_preferences, dict):
                raise ValueError("destination_preferences must be an object")
            preferences = {}
            for key in ("CAFE", "PARK", "SHOP"):
                values = raw_preferences.get(key, [])
                if not isinstance(values, list):
                    raise ValueError(f"destination_preferences.{key} must be an array")
                preferences[key] = [str(value) for value in values if str(value) in known_ids][:3]
            return {
                "flexible_activity_type": activity,
                "destination_preferences": preferences,
                "reasoning": reasoning,
            }
        if phase == "PARSE_CONVERSATION":
            supplied = {str(item.get("report_id")): item for item in request.get("reports", [])}
            values = parsed.get("parsed_reports", [])
            if not isinstance(values, list):
                raise ValueError("parsed_reports must be an array")
            normalized = []
            for item in values:
                if not isinstance(item, dict) or str(item.get("report_id")) not in supplied:
                    continue
                source = supplied[str(item["report_id"])]
                normalized.append({
                    "report_id": str(item["report_id"]),
                    "object_id": str(source["object_id"]),
                    "observed_attribute": str(source["observed_attribute"]),
                    "observed_value": str(source["observed_value"]),
                })
            return {"parsed_reports": normalized, "reasoning": reasoning}
        return parsed

    @staticmethod
    def _safe_fallback(phase: str, request: dict[str, Any]) -> dict[str, Any]:
        reason = "豆包结构化输出连续无效，本阶段采用保守安全回退"
        if phase == "PLAN_DAY":
            preferences = {}
            known = request.get("known_aois", [])
            for activity in ("CAFE", "PARK", "SHOP"):
                preferences[activity] = [
                    str(item["aoi_id"]) for item in known
                    if item.get("believed_function") == activity
                ][:3]
            return {
                "flexible_activity_type": "NONE",
                "destination_preferences": preferences,
                "reasoning": reason,
            }
        if phase == "PARSE_CONVERSATION":
            return {"parsed_reports": [], "reasoning": reason}
        return {"reasoning": reason}

    def _http_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        last_error: BaseException | None = None
        for network_attempt in range(1, 4):
            try:
                with urllib.request.urlopen(
                    request, timeout=120, context=self.ssl_context,
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise RuntimeError(
                        f"Doubao Ark API returned HTTP {exc.code}: {detail[:800]}"
                    ) from exc
                last_error = RuntimeError(
                    f"Doubao Ark API temporary HTTP {exc.code}: {detail[:300]}"
                )
            except (urllib.error.URLError, ssl.SSLError, TimeoutError, OSError) as exc:
                last_error = exc
            if network_attempt < 3:
                time.sleep(0.75 * network_attempt)
        detail = str(getattr(last_error, "reason", last_error))
        raise RuntimeError(
            "Cannot connect to Doubao Ark API after 3 attempts. "
            f"TLS/network detail: {detail}. The client is using certifi instead of the Windows certificate store."
        ) from last_error

    @staticmethod
    def _certifi_ssl_context() -> ssl.SSLContext:
        """Use a known CA bundle so damaged Windows certificate-store entries cannot abort TLS setup."""
        try:
            import certifi
        except ImportError as exc:
            raise RuntimeError(
                "Doubao HTTPS requires certifi. Install it in the active interpreter with: "
                "python -m pip install certifi"
            ) from exc
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except (OSError, ssl.SSLError) as exc:
            raise RuntimeError(
                f"Cannot load certifi CA bundle at {certifi.where()}: {exc}"
            ) from exc

    @staticmethod
    def _extract_output_text(response: dict[str, Any]) -> str:
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    return content["text"]
        raise RuntimeError(f"Doubao response contains no output_text: {json.dumps(response, ensure_ascii=False)[:800]}")


def create_llm_driver(provider: str, model: str = "doubao-seed-2-1-pro-260628") -> AgentLLMDriver:
    if provider == "doubao":
        return DoubaoResponsesDriver(model)
    if provider == "openai":
        return OpenAIResponsesDriver(model)
    if provider in {"deterministic", "rule_based"}:
        return RuleBasedBehaviorDriver()
    raise ValueError(f"Unknown LLM provider: {provider}")


def _strip_json_fence(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return value


def _profile_context(profile: AgentProfile) -> dict[str, Any]:
    data = asdict(profile)
    data.pop("system_prompt", None)
    return data


def _schedule_context(profile: AgentProfile, day: int) -> dict[str, Any]:
    day_of_week = ((day - 1) % 7) + 1
    return {
        "day_of_week": day_of_week,
        "is_work_day": day_of_week in profile.work_days,
        "chronotype": profile.chronotype,
        "employment_status": profile.employment_status,
    }


def _conversation_request(
    profile: AgentProfile, day: int, reports: list[dict[str, Any]],
) -> dict[str, Any]:
    safe_reports = [{
        "report_id": item["report_id"],
        "object_id": item["object_id"],
        "observed_attribute": item["observed_attribute"],
        "observed_value": item["observed_value"],
        "utterance": item["utterance"],
    } for item in reports]
    return {
        "day": day,
        "listener_attributes": {"age": profile.age},
        "reports": safe_reports,
        "task": (
            "Parse each supplied face-to-face utterance into the supplied report ID, object, "
            "attribute and state. Parsing a supplied report means the information was delivered; "
            "do not add a belief score and do not invent objects."
        ),
    }


def _conversation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "parsed_reports": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "report_id": {"type": "string"},
                        "object_id": {"type": "string"},
                        "observed_attribute": {"type": "string"},
                        "observed_value": {"type": "string"},
                    },
                    "required": [
                        "report_id", "object_id", "observed_attribute",
                        "observed_value",
                    ],
                    "additionalProperties": False,
                },
            },
            "reasoning": {"type": "string", "maxLength": 160},
        },
        "required": ["parsed_reports", "reasoning"],
        "additionalProperties": False,
    }
