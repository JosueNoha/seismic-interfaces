"""
Módulo de utilidades centralizadas para interfaces sísmicas
==========================================================

Este módulo centraliza las funciones de utilidad compartidas
entre las diferentes normativas del proyecto.

Componentes disponibles:
- validators: Validadores comunes ya implementados
- converters: Convertidores ya implementados  
- file_handlers: Manejadores de archivos ya implementados
"""

import logging
from typing import Dict, List, Any, Optional

# Configurar logger
logger = logging.getLogger(__name__)

# Información del módulo
__version__ = "1.0.0"
__author__ = "Equipo Seismic Interfaces"

# Control de importaciones
_IMPORT_ERRORS = []
_AVAILABLE_MODULES = {}

# Importar validators si está disponible
try:
    from .validators import *
    from .validators import __all__ as validators_all
    _AVAILABLE_MODULES['validators'] = True
except ImportError as e:
    _IMPORT_ERRORS.append(f"validators: {str(e)}")
    _AVAILABLE_MODULES['validators'] = False
    validators_all = []

# Importar converters si está disponible  
try:
    from .converters import *
    from .converters import __all__ as converters_all
    _AVAILABLE_MODULES['converters'] = True
except ImportError as e:
    _IMPORT_ERRORS.append(f"converters: {str(e)}")
    _AVAILABLE_MODULES['converters'] = False
    converters_all = []

# Importar file_handlers si está disponible
try:
    from .file_handlers import *
    from .file_handlers import __all__ as file_handlers_all
    _AVAILABLE_MODULES['file_handlers'] = True
except ImportError as e:
    _IMPORT_ERRORS.append(f"file_handlers: {str(e)}")
    _AVAILABLE_MODULES['file_handlers'] = False
    file_handlers_all = []

# Combinar todas las exportaciones disponibles
__all__ = []
__all__.extend(validators_all)
__all__.extend(converters_all) 
__all__.extend(file_handlers_all)

# Eliminar duplicados
__all__ = list(set(__all__))


def get_available_modules() -> Dict[str, bool]:
    """
    Obtiene información sobre la disponibilidad de módulos
    
    Returns
    -------
    Dict[str, bool]
        Diccionario con disponibilidad de cada módulo
    """
    return _AVAILABLE_MODULES.copy()


def get_import_errors() -> List[str]:
    """
    Obtiene la lista de errores de importación
    
    Returns
    -------
    List[str]
        Lista de errores de importación
    """
    return _IMPORT_ERRORS.copy()


# Mostrar advertencias si hay errores de importación
if _IMPORT_ERRORS:
    import warnings
    warnings.warn(
        f"Algunos módulos de utilidades no se pudieron importar: {_IMPORT_ERRORS}. "
        f"Funcionalidad limitada disponible."
    )

# Log inicial
logger.info(f"Módulo utils inicializado - Módulos disponibles: {sum(_AVAILABLE_MODULES.values())}")