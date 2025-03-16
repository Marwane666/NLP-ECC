"""
Compatibility layer for different Pydantic versions.
This allows the code to work with both Pydantic v1 and v2.
"""
import importlib.util
import sys
import warnings

# Check if pydantic is installed and which version
pydantic_spec = importlib.util.find_spec("pydantic")
pydantic_installed = pydantic_spec is not None

if pydantic_installed:
    import pydantic
    
    # Get the version
    PYDANTIC_VERSION = getattr(pydantic, "__version__", "1.0.0")
    IS_PYDANTIC_V2 = PYDANTIC_VERSION.startswith("2.")
    
    if IS_PYDANTIC_V2:
        # Pydantic v2 imports
        from pydantic import field_validator, model_validator, Field, ConfigDict, PrivateAttr
    else:
        # Pydantic v1 compatibility
        from pydantic import validator, root_validator, Field
        # Create aliases for v2 validators in v1
        field_validator = validator
        model_validator = root_validator
        # v1 doesn't have ConfigDict, but we can create a simple placeholder
        ConfigDict = dict
        # v1 doesn't have PrivateAttr, use a simple function as placeholder
        def PrivateAttr(default=None, **kwargs):
            return default
        
        warnings.warn(
            "Using Pydantic v1 compatibility mode. Some features may not work correctly. "
            "Consider upgrading to Pydantic v2 for full compatibility with LangChain."
        )
else:
    # Pydantic not installed, create placeholder functions that raise errors
    def field_validator(*args, **kwargs):
        raise ImportError("Pydantic is not installed. Please install with 'pip install pydantic>=2.4.0'")
    
    def model_validator(*args, **kwargs):
        raise ImportError("Pydantic is not installed. Please install with 'pip install pydantic>=2.4.0'")
    
    def Field(*args, **kwargs):
        raise ImportError("Pydantic is not installed. Please install with 'pip install pydantic>=2.4.0'")
    
    def ConfigDict(*args, **kwargs):
        raise ImportError("Pydantic is not installed. Please install with 'pip install pydantic>=2.4.0'")
    
    def PrivateAttr(*args, **kwargs):
        raise ImportError("Pydantic is not installed. Please install with 'pip install pydantic>=2.4.0'")
    
    warnings.warn(
        "Pydantic is not installed. You need to install it with 'pip install pydantic>=2.4.0'"
    )

# Export all the compatible names
__all__ = [
    "field_validator", 
    "model_validator", 
    "Field", 
    "ConfigDict", 
    "PrivateAttr",
    "IS_PYDANTIC_V2"
]
