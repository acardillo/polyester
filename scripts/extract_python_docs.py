"""Extract Python stdlib documentation into structured format with relationships."""

import sys
import sysconfig
import ast
import inspect
import importlib
import json
import pkgutil
import textwrap
from pathlib import Path
from typing import Any, Optional
from typing_extensions import TypeIs
from datetime import datetime

OUTPUT_PATH = Path("data/python_docs.json")

def get_function_calls(obj: Any) -> list[str]:
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

        try:
            functionSourceCode = inspect.getsource(obj)
        except Exception as e:
            print(f"Error getting source code for {obj}: {e}")
            return []

        # Remove leading indentation before parsing
        functionSourceCode = textwrap.dedent(functionSourceCode)

        functionCalls = []
        try:
            astTree = ast.parse(functionSourceCode)
            for node in ast.walk(astTree):
                if isinstance(node, ast.Call):
                    if hasattr(node.func, 'id'):
                        functionCalls.append(node.func.id)
                    elif hasattr(node.func, 'attr'):
                        functionCalls.append(node.func.attr)
        except Exception as e:
            # Parsing failed, return empty list
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
        functionCalls = get_function_calls(obj)
        for functionCall in functionCalls:
            relationships.append({
                "target": f"stdlib.{module_name}.{functionCall}",
                "type": "calls"
            })

    if isClass:
        baseClasses = get_base_classes(obj)
        for baseClass in baseClasses:
            relationships.append({
                "target": f"{baseClass}",
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

    all_modules = sorted([
        m.name 
        for m in pkgutil.iter_modules()
        if m.module_finder.path == sysconfig.get_path('stdlib') 
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