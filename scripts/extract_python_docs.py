"""Extract Python stdlib documentation into structured format with relationships."""

import ast
import builtins
import importlib
import inspect
import json
import pkgutil
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from typing_extensions import TypeIs

OUTPUT_PATH = Path("data/python_docs.json")


def get_function_calls(obj: Any, module_name: str) -> list[str]:
    """
    Get function calls made within a function using AST.

    Args:
        obj: A function or method object

    Returns:
        List of function names called (e.g., ["loads", "open"])

    Note:
        - May fail for built-in functions (no source code)
        - Returns empty list on failure rather than crashing
    """

    #  Get source and parse AST
    try:
        function_source_code = inspect.getsource(obj)
        function_source_code = textwrap.dedent(function_source_code)
        ast_tree = ast.parse(function_source_code)
    except Exception as e:
        print(f"Error getting source code for {obj}: {e}")
        return []

    # Collect called functions from AST
    function_calls = []
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Call):
            function_name = None
            if hasattr(node.func, "id"):
                function_name = node.func.id

            # Filter out built-in functions
            if function_name and not hasattr(builtins, function_name):
                function_calls.append(function_name)

    # Validate function calls - only keep functions that exist in the module
    try:
        module = importlib.import_module(module_name)
        module_functions = {
            name
            for name, obj in inspect.getmembers(module)
            if inspect.isfunction(obj) and obj.__module__ == module_name
        }
        function_calls = [f for f in function_calls if f in module_functions]
    except Exception as e:
        print(f"  Warning: Could not validate function calls for {module_name}: {e}")
        pass

    return function_calls


def get_base_classes(obj: Any) -> list[str]:
    """
    Extract base classes using inspect.

    Args:
        cls: A class object

    Returns:
        List of base class names (excluding 'object')
    """

    class_mro = inspect.getmro(obj)
    return [base_class.__name__ for base_class in class_mro[1:] if base_class.__name__ != "object"]


def get_base_class_module(obj: type, base_class_name: str) -> str:
    """
    Get the module name for a base class.

    Args:
        obj: The class object being inspected
        base_class_name: Name of the base class to find

    Returns:
        Module name (defaults to "builtins" if not found)
    """
    try:
        if hasattr(obj, "__bases__"):
            for base in obj.__bases__:
                if base.__name__ == base_class_name:
                    return base.__module__
    except Exception as e:
        print(f"  Warning: Could not get base class module for {obj}: {e}")
        pass

    return "builtins"  # Default assumption


def inspect_callable(obj: Any, module_name: str, obj_name: str) -> Optional[dict]:
    """
    Extract structured info from a function or class.

    Args:
        obj: Function or class object
        module_name: Name of parent module (e.g., "json")
        obj_name: Name of the object (e.g., "load")

    Returns:
        Dict with structure:
        {
            "id": "stdlib.json.load",
            "name": "load",
            "module": "json",
            "type": "function" or "class",
            "signature": "load(fp, *, cls=None, ...)",
            "description": "Docstring text...",
            "relationships": [
                {"target": "stdlib.json.loads", "type": "calls"},
                {"target": "stdlib.io.TextIOWrapper", "type": "parameter_type"}
            ]
        }

        Returns None if object should be skipped (private, no docs, etc.)
    """
    if obj_name.startswith("_"):
        return None

    is_function = inspect.isfunction(obj)
    is_class = inspect.isclass(obj)

    if not is_function and not is_class:
        print(f"Object {obj_name} is not a function or class")
        return None

    try:
        signature = inspect.signature(obj)
    except Exception as e:
        print(f"Error getting signature for {obj_name}: {e}")
        return None

    try:
        docstring = inspect.getdoc(obj) or ""
    except Exception as e:
        print(f"Error getting docstring for {obj_name}: {e}")
        docstring = ""

    relationships = []
    if is_function:
        function_calls = get_function_calls(obj, module_name)
        for function_call in function_calls:
            relationships.append(
                {"target": f"stdlib.{module_name}.{function_call}", "type": "calls"}
            )

    if is_class:
        base_classes = get_base_classes(obj)
        for base_class in base_classes:
            base_class_module = get_base_class_module(obj, base_class)
            relationships.append(
                {"target": f"stdlib.{base_class_module}.{base_class}", "type": "base_class"}
            )

    callable_data = {
        "id": f"stdlib.{module_name}.{obj_name}",
        "name": obj_name,
        "module": module_name,
        "type": "function" if is_function else "class",
        "signature": str(signature),
        "description": docstring,
        "relationships": relationships,
    }
    return callable_data


def inspect_module(module_name: str) -> list[dict]:
    """
    Import a module and extract all functions/classes from it.

    Args:
        module_name: Name of module to extract (e.g., "json")

    Returns:
        List of info dicts for all callables in the module
    """

    def is_function_or_class(object: object) -> TypeIs[type[Any]]:
        return inspect.isfunction(object) or inspect.isclass(object)

    try:
        module = importlib.import_module(module_name)
    except (ImportError, Exception) as e:
        print(f"Error importing module {module_name}: {e}")
        return []

    callable_members = inspect.getmembers(module, is_function_or_class)
    callable_members = [
        (name, obj)
        for name, obj in callable_members
        if getattr(obj, "__module__", None) == module_name
    ]

    callable_data = [
        inspect_callable(member[1], module_name, member[0]) for member in callable_members
    ]
    callable_data = [data for data in callable_data if data is not None]

    return callable_data


def main() -> None:
    """
    Extract data from all configured modules and save to JSON.
    """

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\nExtracting Python stdlib documentation...")
    now = datetime.now()

    stdlib_modules = sys.stdlib_module_names  # Python 3.10+

    all_modules = sorted(
        [
            m.name
            for m in pkgutil.iter_modules()
            if m.name in stdlib_modules and not m.name.startswith("_")
        ]
    )

    all_data = []
    for idx, module_name in enumerate(all_modules, 1):
        print(f"\n[{idx}/{len(all_modules)}] Processing: {module_name}")
        module_data = inspect_module(module_name)
        all_data.extend(module_data)
        print(f"  ✓ Extracted {len(module_data)} items")

    elapsed = datetime.now() - now
    metadata = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "module_count": len(all_modules),
        "function_count": len(all_data),
        "extracted_at": now.isoformat(),
    }

    data = {"metadata": metadata, "data": all_data}

    with OUTPUT_PATH.open("w") as f:
        json.dump(data, f, indent=2)

    print("\nExtraction Complete!")

    print(f" - Modules processed: {metadata['module_count']}")
    print(f" - Functions/classes extracted: {metadata['function_count']}")
    print(f" - Elapsed time: {elapsed.total_seconds()}s")
    print(f" - Saved to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
