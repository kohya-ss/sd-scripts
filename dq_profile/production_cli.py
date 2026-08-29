from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from dq_profile.production_preset import DiagnosticPreset, get_preset


SENSITIVE_NAME = re.compile(r"(?:token|secret|password|api[_-]?key)", re.IGNORECASE)
CONSUMED_DESTS = {
    "pretrained_model_name_or_path",
    "dataset_config",
    "output_name",
}


@dataclass(frozen=True)
class CompatibilityIssue:
    option: str
    destination: str
    value: Any
    reason: str


class ProfileCompatibilityError(ValueError):
    def __init__(self, issues: Sequence[CompatibilityIssue]) -> None:
        self.issues = tuple(issues)
        lines = ["DQ profile CLI compatibility check failed:"]
        for issue in self.issues:
            rendered = "<redacted>" if SENSITIVE_NAME.search(issue.destination) else repr(issue.value)
            lines.append(f"  {issue.option}={rendered}: {issue.reason}")
        lines.append(
            "Use only canonical-v1 compatible options, or remove the conflicting option. "
            "The profiler never ignores an unsupported training option silently."
        )
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class ResolvedProfileRequest:
    preset: DiagnosticPreset
    model_path: Path
    dataset_config: Path
    output_name: str
    normal_output_dir: Path | None
    parsed_namespace: Any
    dispositions: tuple[dict[str, Any], ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "preset": self.preset.name,
            "pretrained_model_name_or_path": str(self.model_path),
            "dataset_config": str(self.dataset_config),
            "output_name": self.output_name,
            "normal_output_dir": str(self.normal_output_dir) if self.normal_output_dir else None,
            "explicit_option_dispositions": list(self.dispositions),
            "policy": {
                "unknown_options": "reject",
                "unsupported_options": "reject",
                "preset_conflicts": "reject",
                "ignored_options": "record_with_reason",
            },
        }


def _training_parser() -> Any:
    # Import lazily so --help and module discovery stay lightweight.  Build the
    # parser from the diagnostic SDXL wrapper so profile mode never depends on
    # the ordinary sdxl_train_network trainer entry.
    from dq_profile.sdxl_profile_trainer import setup_parser

    return setup_parser()


def _option_map(parser: Any) -> dict[str, Any]:
    return {
        option: action
        for action in parser._actions
        for option in action.option_strings
    }


def _explicit_actions(argv: Sequence[str], parser: Any) -> list[tuple[str, Any]]:
    by_option = _option_map(parser)
    explicit: list[tuple[str, Any]] = []
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        action = by_option.get(option)
        if action is not None:
            explicit.append((option, action))
    return explicit


def _redact(dest: str, value: Any) -> Any:
    if SENSITIVE_NAME.search(dest):
        return "<redacted>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_redact(dest, item) for item in value]
    return value


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str):
        return str(actual).casefold() == expected.casefold()
    return actual == expected


def _network_args_ok(value: Any) -> bool:
    if value is None:
        return True
    values: Iterable[Any] = value if isinstance(value, (list, tuple)) else (value,)
    parsed: dict[str, str] = {}
    for item in values:
        text = str(item)
        if "=" not in text:
            return False
        key, raw = text.split("=", 1)
        parsed[key.strip()] = raw.strip()
    if set(parsed) != {"rank_dropout"}:
        return False
    try:
        return math.isclose(float(parsed["rank_dropout"]), 0.2, abs_tol=1e-12)
    except ValueError:
        return False


def _disposition(option: str, dest: str, value: Any, action: str, reason: str) -> dict[str, Any]:
    return {
        "option": option,
        "destination": dest,
        "value": _redact(dest, value),
        "action": action,
        "reason": reason,
    }


def resolve_training_cli(
    argv: Sequence[str],
    *,
    preset_name: str = "canonical-v1",
) -> ResolvedProfileRequest:
    preset = get_preset(preset_name)
    parser = _training_parser()
    namespace, unknown = parser.parse_known_args(list(argv))
    issues: list[CompatibilityIssue] = []
    if unknown:
        for token in unknown:
            raw_token = str(token)
            option = raw_token.split("=", 1)[0] if raw_token.startswith("-") else "<unknown>"
            issues.append(
                CompatibilityIssue(
                    option=option,
                    destination="unknown",
                    value="<redacted>",
                    reason="unknown to the SDXL training parser or missing a value",
                )
            )

    explicit = _explicit_actions(argv, parser)
    dispositions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for option, action in explicit:
        dest = str(action.dest)
        value = getattr(namespace, dest, None)
        key = (option, dest)
        if key in seen:
            continue
        seen.add(key)
        if dest in CONSUMED_DESTS:
            dispositions.append(
                _disposition(option, dest, value, "consumed", "used to build the diagnostic request")
            )
            continue
        if dest == "network_args":
            if _network_args_ok(value):
                dispositions.append(
                    _disposition(option, dest, value, "matched_preset", "canonical-v1 rank_dropout=0.2")
                )
            else:
                issues.append(
                    CompatibilityIssue(option, dest, value, "canonical-v1 supports only network_args rank_dropout=0.2")
                )
            continue
        if dest == "fp16_safe_norms":
            if bool(value):
                dispositions.append(
                    _disposition(option, dest, value, "matched_preset", "accepted as the strict safe-norms alias")
                )
            else:
                issues.append(CompatibilityIssue(option, dest, value, "canonical-v1 requires fp16 safe norms"))
            continue
        if dest == "fp16_safe_norms_mode":
            if str(value).casefold() == "strict":
                dispositions.append(
                    _disposition(option, dest, value, "matched_preset", "canonical-v1 requires strict mode")
                )
            else:
                issues.append(CompatibilityIssue(option, dest, value, "canonical-v1 requires fp16_safe_norms_mode=strict"))
            continue
        if dest in preset.expected_explicit:
            expected = preset.expected_explicit[dest]
            if _equal(value, expected):
                dispositions.append(
                    _disposition(option, dest, value, "matched_preset", f"matches canonical-v1 value {expected!r}")
                )
            else:
                issues.append(
                    CompatibilityIssue(option, dest, value, f"canonical-v1 requires {dest}={expected!r}")
                )
            continue
        ignored_reason = preset.ignored_explicit.get(dest)
        if ignored_reason is None and dest.startswith("dq_delta_auto_"):
            ignored_reason = "auto range control is disabled during the fixed diagnostic mul scan"
        if ignored_reason is not None:
            dispositions.append(
                _disposition(option, dest, value, "overridden_with_reason", ignored_reason)
            )
            continue
        if dest in preset.unsupported_explicit:
            issues.append(CompatibilityIssue(option, dest, value, preset.unsupported_explicit[dest]))
            continue
        issues.append(
            CompatibilityIssue(
                option,
                dest,
                value,
                f"{dest} is a recognized training option but is not validated by preset {preset.name}",
            )
        )

    for dest, label in (
        ("pretrained_model_name_or_path", "--pretrained_model_name_or_path"),
        ("dataset_config", "--dataset_config"),
    ):
        if not getattr(namespace, dest, None):
            issues.append(CompatibilityIssue(label, dest, None, "required in DQ profile mode"))
    if issues:
        raise ProfileCompatibilityError(issues)

    dataset_config = Path(str(namespace.dataset_config)).expanduser().resolve()
    model_path = Path(str(namespace.pretrained_model_name_or_path)).expanduser().resolve()
    output_name = str(getattr(namespace, "output_name", "") or dataset_config.stem).strip()
    if not output_name or output_name in {".", ".."} or "/" in output_name or "\\" in output_name:
        raise ProfileCompatibilityError(
            (CompatibilityIssue("--output_name", "output_name", output_name, "must be a filename stem without path separators"),)
        )
    raw_output_dir = getattr(namespace, "output_dir", None)
    normal_output_dir = Path(str(raw_output_dir)).expanduser().resolve() if raw_output_dir else None
    return ResolvedProfileRequest(
        preset=preset,
        model_path=model_path,
        dataset_config=dataset_config,
        output_name=output_name,
        normal_output_dir=normal_output_dir,
        parsed_namespace=namespace,
        dispositions=tuple(dispositions),
    )
