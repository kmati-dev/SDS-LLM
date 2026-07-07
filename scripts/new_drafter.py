#!/usr/bin/env python3
"""
Scaffold a new drafter, verifier, or playback class.
Usage: python scripts/new_drafter.py <ClassName> [--type drafter|verifier|playback]
"""

import argparse
import os
import sys

TEMPLATES = {
    "drafter": {
        "src_dir": os.path.join("src", "specdecode", "simulator", "drafter"),
        "test_dir": os.path.join("tests", "drafter"),
        "base_import": "from specdecode.interface.abstractDrafter import AbstractDrafter",
        "base_class": "AbstractDrafter",
        "method": "    def generate_draft(self, prompt: List[int]) -> List[int]:\n        return []",
        "extra_imports": "from typing import List",
    },
    "verifier": {
        "src_dir": os.path.join("src", "specdecode", "simulator", "verifier"),
        "test_dir": os.path.join("tests", "verifier"),
        "base_import": "from specdecode.interface.abstractVerifier import AbstractVerifier",
        "base_class": "AbstractVerifier",
        "method": '    def verify(self, draft_tokens: List[int], current_prefix: List[int], complete_tokens: List[int]) -> dict:\n        return {"accepted_tokens": [], "accepted_count": 0, "rejected_count": 0}',
        "extra_imports": "from typing import List",
    },
    "playback": {
        "src_dir": os.path.join("src", "specdecode", "simulator", "playback"),
        "test_dir": os.path.join("tests", "playback"),
        "base_import": "from specdecode.interface.abstractPlayback import AbstractPlayback",
        "base_class": "AbstractPlayback",
        "method": '    def run_playback(self, input_data: str, use_drafter: bool = True) -> str:\n        return ""',
        "extra_imports": "",
        "test_body": (
            "from unittest.mock import MagicMock\n"
            "from specdecode.simulator.playback.{module} import {cls}\n\n\n"
            "def test_{module}_instantiation() -> None:\n"
            "    obj = {cls}(tokenizer=MagicMock(), drafter=MagicMock(), verifier=MagicMock())\n"
            "    assert obj is not None\n"
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new simulator component.")
    parser.add_argument("class_name", help="CamelCase class name, e.g. MyDrafter")
    parser.add_argument(
        "--type",
        choices=["drafter", "verifier", "playback"],
        default="drafter",
        help="Component type (default: drafter)",
    )
    args = parser.parse_args()

    class_name = args.class_name
    component_type = args.type

    if not class_name.isidentifier():
        print(f"Error: '{class_name}' is not a valid Python identifier.")
        sys.exit(1)

    module_name = class_name[0].lower() + class_name[1:]
    tmpl = TEMPLATES[component_type]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(project_root, tmpl["src_dir"], f"{module_name}.py")
    test_path = os.path.join(project_root, tmpl["test_dir"], f"test_{module_name}.py")

    for path in (src_path, test_path):
        if os.path.exists(path):
            print(f"Error: {path} already exists.")
            sys.exit(1)

    extra = f"\n{tmpl['extra_imports']}\n" if tmpl["extra_imports"] else ""
    src_content = f'"""{class_name} implementation."""\n{extra}\n{tmpl["base_import"]}\n\n\nclass {class_name}({tmpl["base_class"]}):\n{tmpl["method"]}\n'

    if "test_body" in tmpl:
        test_content = tmpl["test_body"].format(module=module_name, cls=class_name)
    else:
        test_content = f'"""Tests for {class_name}."""\nfrom specdecode.simulator.{component_type}.{module_name} import {class_name}\n\n\ndef test_{module_name}_instantiation() -> None:\n    obj = {class_name}()\n    assert obj is not None\n'

    with open(src_path, "w") as f:
        f.write(src_content)
    with open(test_path, "w") as f:
        f.write(test_content)

    print(f"Created {src_path}")
    print(f"Created {test_path}")


if __name__ == "__main__":
    main()
