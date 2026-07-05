#!/usr/bin/env python3
"""
Scaffolding script to create a new drafter component.
Usage: python scripts/new_drafter.py MyDrafter
"""
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/new_drafter.py <DrafterClassName>")
        sys.exit(1)
        
    class_name = sys.argv[1]
    if not class_name.isidentifier():
        print(f"Error: '{class_name}' is not a valid Python identifier.")
        sys.exit(1)
        
    # Lowercase first letter for module name
    module_name = class_name[0].lower() + class_name[1:]
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drafter_dir = os.path.join(project_root, "src", "specdecode", "simulator", "drafter")
    test_dir = os.path.join(project_root, "tests")
    
    drafter_path = os.path.join(drafter_dir, f"{module_name}.py")
    test_path = os.path.join(test_dir, f"test_{module_name}.py")
    
    if os.path.exists(drafter_path):
        print(f"Error: {drafter_path} already exists.")
        sys.exit(1)
        
    if os.path.exists(test_path):
        print(f"Error: {test_path} already exists.")
        sys.exit(1)
        
    # Create the drafter file
    drafter_content = f'''"""Implementation of {class_name}."""
from typing import List

from specdecode.interface.abstractDrafter import AbstractDrafter


class {class_name}(AbstractDrafter):
    """
    Concrete implementation of {class_name}.
    """

    def generate_draft(self, prompt: List[int]) -> List[int]:
        """
        Generate speculative token guesses.

        Args:
            prompt: List of token IDs representing the current prefix context.

        Returns:
            A list of guessed token IDs (draft tokens).
        """
        # TODO: Implement draft generation logic
        return []
'''
    with open(drafter_path, "w") as f:
        f.write(drafter_content)
        
    # Create the test file
    test_content = f'''"""Tests for {class_name}."""
from specdecode.simulator.drafter.{module_name} import {class_name}


def test_{module_name}_instantiation() -> None:
    """Test that {class_name} can be instantiated."""
    drafter = {class_name}()
    assert drafter is not None


def test_{module_name}_generate_draft_returns_list() -> None:
    """Test that {class_name} returns a list."""
    drafter = {class_name}()
    result = drafter.generate_draft([1, 2, 3])
    assert isinstance(result, list)
'''
    with open(test_path, "w") as f:
        f.write(test_content)
        
    print(f"Created {drafter_path}")
    print(f"Created {test_path}")

if __name__ == "__main__":
    main()
