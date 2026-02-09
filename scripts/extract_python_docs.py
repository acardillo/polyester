"""Extract Python stdlib documentation into structured format with relationships."""

import sys
import ast
import inspect
import importlib
import json
import pkgutil
import textwrap
import builtins
from pathlib import Path
from typing import Any, Optional
from typing_extensions import TypeIs
from datetime import datetime

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
            functionSourceCode = inspect.getsource(obj)
            functionSourceCode = textwrap.dedent(functionSourceCode)
            astTree = ast.parse(functionSourceCode)
        except Exception as e:
            print(f"Error getting source code for {obj}: {e}")
            return []

        # Collect called functions from AST
        functionCalls = []
        for node in ast.walk(astTree):
            if isinstance(node, ast.Call):
                functionName = None
                if hasattr(node.func, 'id'):
                    functionName = node.func.id

                # Filter out built-in functions
                if functionName and not hasattr(builtins, functionName):
                    functionCalls.append(functionName)

        # Validate function calls - only keep functions that exist in the module
        try:
            module = importlib.import_module(module_name)
            module_functions = {
                name for name, obj in inspect.getmembers(module)
                if inspect.isfunction(obj) and obj.__module__ == module_name
            }
            functionCalls = [f for f in functionCalls if f in module_functions]
        except Exception as e:
            print(f"  Warning: Could not validate function calls for {module_name}: {e}")
            pass

        return functionCalls

def get_base_classes(obj: Any) -> list[str]:
    """
    Extract base classes using inspect.
    
    Args:
        cls: A class object
        
    Returns:
        List of base class names (excluding 'object')
    """

    classMRO = inspect.getmro(obj)
    return [baseClass.__name__ for baseClass in classMRO[1:] if baseClass.__name__ != "object"]

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
        if hasattr(obj, '__bases__'):
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

    isFunction = inspect.isfunction(obj)
    isClass = inspect.isclass(obj)
    
    if not isFunction and not isClass:
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
    if isFunction:
        functionCalls = get_function_calls(obj, module_name)
        for functionCall in functionCalls:
            relationships.append({
                "target": f"stdlib.{module_name}.{functionCall}",
                "type": "calls"
            })

    if isClass:
        baseClasses = get_base_classes(obj)
        for baseClass in baseClasses:
            baseClassModule = get_base_class_module(obj, baseClass)
            relationships.append({
                "target": f"stdlib.{baseClassModule}.{baseClass}",
                "type": "base_class"
            })

    callableData = {
        "id": f"stdlib.{module_name}.{obj_name}",
        "name": obj_name,
        "module": module_name,
        "type": "function" if isFunction else "class",
        "signature": str(signature),
        "description": docstring,
        "relationships": relationships
    }
    return callableData

def inspect_module(module_name: str) -> list[dict]:
    """
    Import a module and extract all functions/classes from it.
    
    Args:
        module_name: Name of module to extract (e.g., "json")
        
    Returns:
        List of info dicts for all callables in the module
    """

    def isFunctionOrClass(object: object) -> TypeIs[type[Any]]:
        return inspect.isfunction(object) or inspect.isclass(object)

    try:
        module = importlib.import_module(module_name)
    except (ImportError, Exception) as e:
        print(f"Error importing module {module_name}: {e}")
        return []

    callable_members = inspect.getmembers(module, isFunctionOrClass)
    callable_members = [
        (name, obj) for name, obj in callable_members
        if getattr(obj, '__module__', None) == module_name
    ]

    callable_data = [inspect_callable(member[1], module_name, member[0]) for member in callable_members]
    callable_data = [data for data in callable_data if data is not None]

    return callable_data

def main() -> None:
    """
    Extract data from all configured modules and save to JSON.
    """

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting Python stdlib documentation...")
    now = datetime.now()

    stdlib_modules = sys.stdlib_module_names  # Python 3.10+

    all_modules = sorted([
        m.name 
        for m in pkgutil.iter_modules()
        if m.name in stdlib_modules
        and not m.name.startswith("_")
    ])

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

    data = {
        "metadata": metadata,
        "data": all_data
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"\nExtraction Complete!")
    
    print(f" - Modules processed: {metadata['module_count']}")
    print(f" - Functions/classes extracted: {metadata['function_count']}")
    print(f" - Elapsed time: {elapsed.total_seconds()}s")
    print(f" - Saved to {OUTPUT_PATH}\n")

if __name__ == "__main__":
    main()