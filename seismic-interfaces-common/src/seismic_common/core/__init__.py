"""
Seismic Common Core Module

Módulo central del sistema de interfaces sísmicas centralizado.
Proporciona las funcionalidades básicas compartidas entre todos los proyectos
de análisis sísmico (Perú, Bolivia, etc.).

Modules included:
- etabs_utils: Utilidades para conexión y manejo de ETABS
- sismo_utils: Clases y funciones para análisis sísmico
- unit_tool: Sistema de unidades y conversiones
- latex_utils: Utilidades para procesamiento de LaTeX

Author: Generated for Seismic Interfaces Project
Version: 1.0.0
"""

from typing import Dict, Any, Optional
import warnings

# Configuración de logging para el módulo
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Información del módulo
__version__ = "1.0.0"
__author__ = "Seismic Interfaces Project"
__email__ = "support@seismicinterfaces.com"

# Importaciones principales del core
try:
    from . import etabs_utils
    from . import sismo_utils
    from . import unit_tool
    from . import latex_utils
    
    # Importar clases principales para acceso directo
    from .etabs_utils import (
        connect_to_etabs,
        connect_to_safe,
        get_modal_data,
        get_table,
        set_units,
        validate_connection
    )
    
    from .sismo_utils import (
        BaseSeismicAnalysis,
        Sismo_e30,
        SeismicLoads,
        SeismicTables,
        SeismicData,
        UNIT_DICT
    )
    
    from .unit_tool import (
        Units,
        UnitSystem,
        RegionalUnits,
        EngineeringConstants,
        create_unit_dict
    )
    
    from .latex_utils import (
        escape_for_latex,
        dataframe_to_latex,
        highlight_cell,
        compile_latex,
        process_latex_variables
    )
    
    # Marcar como importaciones exitosas
    _IMPORTS_OK = True
    
except ImportError as e:
    warnings.warn(f"Algunas importaciones del core fallaron: {e}")
    _IMPORTS_OK = False


# Configuración global por defecto del sistema
DEFAULT_CONFIG = {
    'units': {
        'system': 'SI',
        'length': 'm',
        'force': 'kN',
        'pressure': 'MPa'
    },
    'analysis': {
        'modal_mass_threshold': 0.90,
        'max_drift_default': 0.007,
        'base_story': 'Base'
    },
    'latex': {
        'encoding': 'utf-8',
        'compile_runs': 2,
        'clean_aux': True
    },
    'etabs': {
        'api_version': 'v1',
        'timeout': 30,
        'auto_run_analysis': True
    }
}


def get_version() -> str:
    """
    Obtiene la versión del módulo core
    
    Returns
    -------
    str
        Versión del módulo
    """
    return __version__


def get_config() -> Dict[str, Any]:
    """
    Obtiene la configuración por defecto del sistema
    
    Returns
    -------
    Dict[str, Any]
        Diccionario con configuración por defecto
    """
    return DEFAULT_CONFIG.copy()


def check_dependencies() -> Dict[str, bool]:
    """
    Verifica las dependencias necesarias del sistema
    
    Returns
    -------
    Dict[str, bool]
        Diccionario con estado de dependencias
    """
    dependencies = {}
    
    # Verificar bibliotecas Python esenciales
    try:
        import pandas
        dependencies['pandas'] = True
    except ImportError:
        dependencies['pandas'] = False
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False
    
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        dependencies['matplotlib'] = False
    
    try:
        from PyQt5 import QtWidgets
        dependencies['PyQt5'] = True
    except ImportError:
        dependencies['PyQt5'] = False
    
    try:
        import comtypes
        dependencies['comtypes'] = True
    except ImportError:
        dependencies['comtypes'] = False
    
    try:
        from PIL import Image
        dependencies['PIL'] = True
    except ImportError:
        dependencies['PIL'] = False
    
    # Verificar módulos propios
    dependencies['core_modules'] = _IMPORTS_OK
    
    return dependencies


def validate_environment() -> Dict[str, Any]:
    """
    Valida el entorno de ejecución
    
    Returns
    -------
    Dict[str, Any]
        Diccionario con información del entorno
    """
    import sys
    import platform
    
    environment = {
        'python_version': sys.version,
        'platform': platform.system(),
        'architecture': platform.architecture()[0],
        'dependencies': check_dependencies(),
        'core_version': __version__
    }
    
    return environment


def create_default_units() -> Units:
    """
    Crea una instancia de Units con configuración por defecto
    
    Returns
    -------
    Units
        Instancia de Units configurada
    """
    if not _IMPORTS_OK:
        raise ImportError("No se pudieron importar los módulos necesarios")
    
    return Units(DEFAULT_CONFIG['units']['system'])


def create_default_unit_dict() -> Dict[str, float]:
    """
    Crea un diccionario de unidades por defecto
    
    Returns
    -------
    Dict[str, float]
        Diccionario con factores de conversión
    """
    units = create_default_units()
    return create_unit_dict(units)


def setup_seismic_analysis(country_code: str = 'PE') -> BaseSeismicAnalysis:
    """
    Configura un análisis sísmico según el país
    
    Parameters
    ----------
    country_code : str
        Código del país ('PE', 'BO', 'US', etc.)
        
    Returns
    -------
    BaseSeismicAnalysis
        Instancia configurada para el país
    """
    if not _IMPORTS_OK:
        raise ImportError("No se pudieron importar los módulos necesarios")
    
    # Mapeo de países a clases específicas
    country_classes = {
        'PE': Sismo_e30,  # Perú - E030
        'BO': BaseSeismicAnalysis,  # Bolivia - Base class por ahora
        'US': BaseSeismicAnalysis,  # USA - Base class por ahora
    }
    
    analysis_class = country_classes.get(country_code.upper(), BaseSeismicAnalysis)
    
    # Crear instancia
    analysis = analysis_class()
    
    # Configurar unidades según región
    regional_units = RegionalUnits.get_preferred_units(country_code)
    analysis.set_units(
        u_h=DEFAULT_CONFIG['units']['length'],
        u_f=DEFAULT_CONFIG['units']['force']
    )
    
    return analysis


def get_sample_data() -> Dict[str, Any]:
    """
    Proporciona datos de muestra para testing
    
    Returns
    -------
    Dict[str, Any]
        Datos de muestra
    """
    import pandas as pd
    
    sample_modal = pd.DataFrame({
        'Mode': [1, 2, 3, 4, 5],
        'Period': [1.2, 0.8, 0.6, 0.4, 0.3],
        'Ux': [0.65, 0.15, 0.05, 0.10, 0.05],
        'Uy': [0.10, 0.60, 0.20, 0.05, 0.05],
        'SumUx': [0.65, 0.80, 0.85, 0.95, 1.00],
        'SumUy': [0.10, 0.70, 0.90, 0.95, 1.00]
    })
    
    sample_drifts = pd.DataFrame({
        'Story': ['Piso 4', 'Piso 3', 'Piso 2', 'Piso 1'],
        'DriftX': [0.004, 0.005, 0.006, 0.003],
        'DriftY': [0.003, 0.004, 0.005, 0.002]
    })
    
    sample_variables = {
        'Z': 0.35,
        'U': 1.0,
        'S': 1.2,
        'Tx': 1.2,
        'Ty': 0.8,
        'MP_x': 0.95,
        'MP_y': 0.90
    }
    
    return {
        'modal': sample_modal,
        'drifts': sample_drifts,
        'variables': sample_variables
    }


# Configurar logging por defecto
def setup_logging(level: str = 'INFO') -> None:
    """
    Configura el sistema de logging
    
    Parameters
    ----------
    level : str
        Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    import logging
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Información exportada del módulo
__all__ = [
    # Módulos
    'etabs_utils',
    'sismo_utils', 
    'unit_tool',
    'latex_utils',
    
    # Clases principales
    'BaseSeismicAnalysis',
    'Sismo_e30',
    'SeismicLoads',
    'SeismicTables', 
    'SeismicData',
    'Units',
    'UnitSystem',
    'RegionalUnits',
    'EngineeringConstants',
    
    # Funciones principales
    'connect_to_etabs',
    'get_modal_data',
    'dataframe_to_latex',
    'compile_latex',
    'create_unit_dict',
    
    # Utilidades del módulo
    'get_version',
    'get_config',
    'check_dependencies',
    'validate_environment',
    'create_default_units',
    'create_default_unit_dict',
    'setup_seismic_analysis',
    'get_sample_data',
    'setup_logging',
    
    # Constantes
    'DEFAULT_CONFIG',
    'UNIT_DICT'
]


# Mensaje de bienvenida al importar
if _IMPORTS_OK:
    print(f"Seismic Common Core v{__version__} - Módulos cargados correctamente")
else:
    warnings.warn("Seismic Common Core - Algunos módulos no se pudieron cargar")


# Auto-configuración si se ejecuta como script principal
if __name__ == '__main__':
    print("=== Seismic Common Core - Información del Sistema ===\n")
    
    print(f"Versión: {get_version()}")
    
    print("\n=== Estado de Dependencias ===")
    deps = check_dependencies()
    for dep, status in deps.items():
        status_str = "✓ OK" if status else "✗ FALTA"
        print(f"{dep}: {status_str}")
    
    print(f"\n=== Configuración por Defecto ===")
    config = get_config()
    for category, settings in config.items():
        print(f"\n{category.title()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    print(f"\n=== Entorno de Ejecución ===")
    env = validate_environment()
    print(f"Python: {env['python_version'].split()[0]}")
    print(f"Plataforma: {env['platform']} ({env['architecture']})")
    
    if _IMPORTS_OK:
        print(f"\n=== Ejemplo de Uso ===")
        # Crear instancia de análisis sísmico
        try:
            sismo = setup_seismic_analysis('PE')
            print(f"Análisis sísmico configurado para Perú")
            print(f"Unidades: {sismo.u_h} (longitud), {sismo.u_f} (fuerza)")
            
            # Crear unidades
            units = create_default_units()
            print(f"Sistema de unidades: {units.get_system().value}")
            
            print("\n¡Core inicializado correctamente!")
            
        except Exception as e:
            print(f"Error en ejemplo: {e}")
    else:
        print("\n⚠️ Algunos módulos no están disponibles")