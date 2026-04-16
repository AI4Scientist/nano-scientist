#!/usr/bin/env python3
"""Validate SKILL.md frontmatter for all skills in the repository.

Expected frontmatter format:
---
name: <skill-id>
description: <one-line description>
allowed-tools: Bash          # or omit for plain LLM; "Bash" grants tool-calling loop
required-keys: [KEY1, KEY2]  # optional; skill filtered out if any key missing
metadata:
  author: <author>
  version: '<semver>'
  tags: [tag1, tag2]
---
"""

import glob
import sys
from pathlib import Path

import yaml

REQUIRED_FIELDS = ["name", "description", "metadata"]
REQUIRED_METADATA = ["author", "version", "tags"]

# allowed-tools values that grant bash execution in nano-scientist
BASH_TOOL_VALUES = {"Bash", "bash"}


def extract_frontmatter(filepath: str) -> str | None:
    """Extract YAML frontmatter from a SKILL.md file."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    if not content.startswith("---"):
        return None

    try:
        end = content.index("---", 3)
    except ValueError:
        return None

    return content[3:end].strip()


def validate_skill(filepath: str) -> list[str]:
    """Validate a single SKILL.md file. Returns list of error messages."""
    errors = []
    warnings = []
    skill_name = Path(filepath).parent.name

    try:
        fm_text = extract_frontmatter(filepath)
    except (ValueError, FileNotFoundError) as e:
        return [f"{skill_name}: Failed to read frontmatter — {e}"]

    if not fm_text:
        return [f"{skill_name}: Missing YAML frontmatter (file must start with ---)"]

    try:
        data = yaml.safe_load(fm_text)
    except yaml.YAMLError as e:
        return [f"{skill_name}: Invalid YAML — {e}"]

    if not isinstance(data, dict):
        return [f"{skill_name}: Frontmatter is not a YAML mapping"]

    # --- Required top-level fields ---
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"{skill_name}: Missing required field '{field}'")

    # --- name should match the directory name ---
    if "name" in data and data["name"] != skill_name:
        errors.append(
            f"{skill_name}: 'name' field '{data['name']}' does not match directory name '{skill_name}'"
        )

    # --- description must be non-empty ---
    if "description" in data:
        desc = data["description"]
        if not isinstance(desc, str) or not desc.strip():
            errors.append(f"{skill_name}: 'description' must be a non-empty string")

    # --- allowed-tools: optional but must be valid if present ---
    if "allowed-tools" in data:
        tools_raw = data["allowed-tools"]
        # Normalise: string → list
        if isinstance(tools_raw, str):
            tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
        elif isinstance(tools_raw, list):
            tools = [str(t).strip() for t in tools_raw]
        else:
            errors.append(f"{skill_name}: 'allowed-tools' must be a string or list")
            tools = []

        # Warn if non-standard tool names are present (won't break, but worth noting)
        standard = {"Bash"} | BASH_TOOL_VALUES
        non_standard = [t for t in tools if t not in standard]
        if non_standard:
            warnings.append(
                f"{skill_name}: 'allowed-tools' contains non-standard values {non_standard!r} "
                f"(only 'Bash' is recognised by nano-scientist; others are ignored)"
            )

    # --- required-keys: optional but must be a list of strings if present ---
    if "required-keys" in data:
        keys = data["required-keys"]
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]
        if not isinstance(keys, list):
            errors.append(f"{skill_name}: 'required-keys' must be a list of strings")

    # --- metadata sub-fields ---
    meta = data.get("metadata", {})
    if isinstance(meta, dict):
        for field in REQUIRED_METADATA:
            if field not in meta:
                errors.append(f"{skill_name}: Missing metadata field '{field}'")
        # version should be a quoted string
        if "version" in meta and not isinstance(meta["version"], str):
            errors.append(
                f"{skill_name}: metadata.version should be a quoted string (e.g. '1.0.0')"
            )
        # tags should be a non-empty list
        if "tags" in meta:
            if not isinstance(meta["tags"], list) or not meta["tags"]:
                errors.append(f"{skill_name}: metadata.tags must be a non-empty list")
    elif "metadata" in data:
        errors.append(f"{skill_name}: 'metadata' must be a mapping")

    return errors + [f"[WARN] {w}" for w in warnings]


def main():
    skill_files = sorted(glob.glob("skills/*/SKILL.md"))

    if not skill_files:
        print("No SKILL.md files found in skills/*/")
        sys.exit(1)

    all_errors = []
    all_warnings = []

    for filepath in skill_files:
        skill_name = Path(filepath).parent.name
        messages = validate_skill(filepath)
        errors = [m for m in messages if not m.startswith("[WARN]")]
        warnings = [m[7:] for m in messages if m.startswith("[WARN]")]

        if errors:
            for e in errors:
                print(f"  ❌ {e}")
            all_errors.extend(errors)
        elif warnings:
            for w in warnings:
                print(f"  ⚠️  {w}")
            all_warnings.extend(warnings)
            print(f"  ✅ {skill_name} (with warnings)")
        else:
            print(f"  ✅ {skill_name}")

    print()
    if all_errors:
        error_skills = len(set(e.split(":")[0] for e in all_errors))
        print(f"❌ {len(all_errors)} error(s) in {error_skills} skill(s)")
        if all_warnings:
            print(f"⚠️  {len(all_warnings)} warning(s)")
        sys.exit(1)
    else:
        if all_warnings:
            print(f"⚠️  All {len(skill_files)} skills passed validation ({len(all_warnings)} warning(s))")
        else:
            print(f"✅ All {len(skill_files)} skills passed validation")


if __name__ == "__main__":
    main()
