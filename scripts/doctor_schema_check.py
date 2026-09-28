"""Just enough JSON Schema (draft 2020-12 subset) to check doctor reports against
docs/doctor.schema.json without a jsonschema dependency. Used by the doctor tests."""
import re


def validate(inst, sch):
    if "const" in sch and inst != sch["const"]:
        return f"const {inst!r}"
    if "enum" in sch and inst not in sch["enum"]:
        return f"enum {inst!r}"
    t = sch.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float)}
    if t and not isinstance(inst, types[t]):
        return f"type {type(inst).__name__} != {t}"
    if t == "object":
        for k in sch.get("required", []):
            if k not in inst:
                return f"missing {k}"
        props = sch.get("properties", {})
        for k, v in inst.items():
            if k not in props:
                if sch.get("additionalProperties") is False:
                    return f"extra {k}"
                continue
            e = validate(v, props[k])
            if e:
                return f"{k}: {e}"
    if t == "array":
        for i, v in enumerate(inst):
            e = validate(v, sch["items"])
            if e:
                return f"[{i}] {e}"
    if t == "string" and "pattern" in sch and not re.match(sch["pattern"], inst):
        return f"pattern {inst!r}"
    return None
