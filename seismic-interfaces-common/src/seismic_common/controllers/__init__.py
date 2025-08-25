"""
Paquete de controladores centralizados para interfaces sísmicas

Este paquete contiene todos los controladores base que proporcionan funcionalidades
comunes para los proyectos específicos por país (Perú, Bolivia, etc.).

Modules:
--------
base_controller: Controlador base con funcionalidades comunes
seismic_controller: Controlador especializado para análisis sísmicos avanzados  
table_controller: Controlador para manejo de tablas de datos
graph_controller: Controlador para generación y visualización de gráficos

Classes:
--------
BaseController: Clase base abstracta para todos los controladores
SeismicController: Controlador avanzado para análisis sísmicos
TableController: Controlador para operaciones con tablas
GraphController: Controlador para gráficos y visualizaciones

Usage:
------
# Importar controladores individuales
from seismic_common.controllers import BaseController, SeismicController

# Importar factory function para crear controladores específicos por país
from seismic_common.controllers import create_country_controller

# Crear controlador para Perú
controller = create_country_controller('PE', main_window)

Author: Seismic Interfaces Project
Version: 1.0.0
"""

import warnings
from typing import Dict, Any, Optional, Type, Union
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Información del paquete
__version__ = "1.0.0"
__author__ = "Seismic Interfaces Project"

# Intentar importar controladores base
try:
    from .base_controller import (
        BaseController,
        PandasTableModel,
        BaseGraphDialog,
        BaseTableDialog
    )
    _BASE_AVAILABLE = True
    logger.info("BaseController cargado correctamente")
except ImportError as e:
    _BASE_AVAILABLE = False
    logger.warning(f"No se pudo cargar BaseController: {e}")
    warnings.warn(f"BaseController no disponible: {e}")

# Intentar importar controlador sísmico
try:
    from .seismic_controller import (
        SeismicController,
        AnalysisWorker,
        SeismicProgressDialog,
        BatchAnalysisManager
    )
    _SEISMIC_AVAILABLE = True
    logger.info("SeismicController cargado correctamente")
except ImportError as e:
    _SEISMIC_AVAILABLE = False
    logger.warning(f"No se pudo cargar SeismicController: {e}")

# Intentar importar controlador de tablas
try:
    from .table_controller import (
        TableController,
        AdvancedTableModel,
        AdvancedTableDialog,
        TableExporter,
        TableFilterManager
    )
    _TABLE_AVAILABLE = True
    logger.info("TableController cargado correctamente")
except ImportError as e:
    _TABLE_AVAILABLE = False
    logger.warning(f"No se pudo cargar TableController: {e}")

# Intentar importar controlador de gráficos
try:
    from .graph_controller import (
        GraphController,
        BaseGraphDialog as AdvancedGraphDialog,  # Alias para distinguir
        GraphType
    )
    _GRAPH_AVAILABLE = True
    logger.info("GraphController cargado correctamente")
except ImportError as e:
    _GRAPH_AVAILABLE = False
    logger.warning(f"No se pudo cargar GraphController: {e}")


# Lista de todos los controladores disponibles
_AVAILABLE_CONTROLLERS = []
if _BASE_AVAILABLE:
    _AVAILABLE_CONTROLLERS.append('BaseController')
if _SEISMIC_AVAILABLE:
    _AVAILABLE_CONTROLLERS.append('SeismicController')
if _TABLE_AVAILABLE:
    _AVAILABLE_CONTROLLERS.append('TableController')
if _GRAPH_AVAILABLE:
    _AVAILABLE_CONTROLLERS.append('GraphController')


def get_available_controllers() -> list:
    """
    Retorna lista de controladores disponibles
    
    Returns
    -------
    list
        Lista de nombres de controladores disponibles
    """
    return _AVAILABLE_CONTROLLERS.copy()


def is_controller_available(controller_name: str) -> bool:
    """
    Verifica si un controlador específico está disponible
    
    Parameters
    ----------
    controller_name : str
        Nombre del controlador a verificar
        
    Returns
    -------
    bool
        True si el controlador está disponible
    """
    return controller_name in _AVAILABLE_CONTROLLERS


def get_controller_status() -> Dict[str, bool]:
    """
    Retorna el estado de disponibilidad de todos los controladores
    
    Returns
    -------
    Dict[str, bool]
        Diccionario con el estado de cada controlador
    """
    return {
        'BaseController': _BASE_AVAILABLE,
        'SeismicController': _SEISMIC_AVAILABLE,
        'TableController': _TABLE_AVAILABLE,
        'GraphController': _GRAPH_AVAILABLE
    }


def create_controller_suite(main_window, country_code: str = None) -> Dict[str, Any]:
    """
    Crea una suite completa de controladores para una aplicación
    
    Parameters
    ----------
    main_window : QtWidgets.QMainWindow
        Ventana principal de la aplicación
    country_code : str, optional
        Código del país para configuraciones específicas
        
    Returns
    -------
    Dict[str, Any]
        Diccionario con todos los controladores disponibles
    """
    suite = {}
    
    # Crear controlador base si está disponible
    if _BASE_AVAILABLE:
        # Crear una implementación genérica del controlador base
        class GenericController(BaseController):
            def initialize_seismic_analysis(self):
                """Implementación por defecto"""
                if hasattr(self, 'seismic_analysis') and self.seismic_analysis:
                    logger.info("Análisis sísmico ya inicializado")
                else:
                    logger.info("Inicializando análisis sísmico genérico")
            
            def set_seismic_parameters(self):
                """Implementación por defecto"""
                logger.info("Configurando parámetros sísmicos genéricos")
        
        suite['base'] = GenericController(main_window)
    
    # Crear controlador sísmico si está disponible
    if _SEISMIC_AVAILABLE:
        suite['seismic'] = SeismicController(main_window)
        
        # Conectar con controlador base si existe
        if 'base' in suite:
            suite['seismic'].base_controller = suite['base']
    
    # Crear controlador de tablas si está disponible
    if _TABLE_AVAILABLE:
        suite['table'] = TableController(main_window)
        
        # Conectar con otros controladores
        if 'base' in suite:
            suite['table'].base_controller = suite['base']
    
    # Crear controlador de gráficos si está disponible  
    if _GRAPH_AVAILABLE:
        suite['graph'] = GraphController(main_window)
        
        # Conectar con otros controladores
        if 'base' in suite:
            suite['graph'].base_controller = suite['base']
    
    # Configurar para país específico si se proporciona
    if country_code:
        _configure_for_country(suite, country_code)
    
    logger.info(f"Suite de controladores creada: {list(suite.keys())}")
    return suite


def _configure_for_country(suite: Dict[str, Any], country_code: str) -> None:
    """
    Configura la suite de controladores para un país específico
    
    Parameters
    ----------
    suite : Dict[str, Any]
        Suite de controladores
    country_code : str
        Código del país
    """
    # Configuraciones específicas por país
    country_configs = {
        'PE': {  # Perú
            'units': {'length': 'm', 'force': 'tonf'},
            'seismic_code': 'E030',
            'default_soil': 'S1',
            'default_zone': 'Z4'
        },
        'BO': {  # Bolivia
            'units': {'length': 'm', 'force': 'kN'},
            'seismic_code': 'CNBDS',
            'default_soil': 'I',
            'default_zone': 'A'
        },
        'CO': {  # Colombia
            'units': {'length': 'm', 'force': 'kN'},
            'seismic_code': 'NSR10',
            'default_soil': 'C',
            'default_zone': 'ALTA'
        }
    }
    
    config = country_configs.get(country_code, {})
    
    # Aplicar configuración a cada controlador
    for controller_name, controller in suite.items():
        if hasattr(controller, 'set_country_config'):
            controller.set_country_config(country_code, config)
            logger.info(f"Configurado {controller_name} para {country_code}")


def create_base_controller(main_window, **kwargs) -> Optional['BaseController']:
    """
    Factory function para crear controlador base
    
    Parameters
    ----------
    main_window : QtWidgets.QMainWindow
        Ventana principal
    **kwargs
        Argumentos adicionales para configuración
        
    Returns
    -------
    BaseController or None
        Controlador base o None si no está disponible
    """
    if not _BASE_AVAILABLE:
        logger.error("BaseController no está disponible")
        return None
    
    class ConfigurableController(BaseController):
        def __init__(self, main_window):
            super().__init__(main_window)
            # Aplicar configuración adicional
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        def initialize_seismic_analysis(self):
            logger.info("Inicializando análisis sísmico configurable")
        
        def set_seismic_parameters(self):
            logger.info("Configurando parámetros sísmicos configurables")
    
    return ConfigurableController(main_window)


def create_seismic_controller(main_window, **kwargs) -> Optional['SeismicController']:
    """
    Factory function para crear controlador sísmico
    
    Parameters
    ----------
    main_window : QtWidgets.QMainWindow
        Ventana principal
    **kwargs
        Argumentos adicionales
        
    Returns
    -------
    SeismicController or None
        Controlador sísmico o None si no está disponible
    """
    if not _SEISMIC_AVAILABLE:
        logger.error("SeismicController no está disponible")
        return None
    
    return SeismicController(main_window, **kwargs)


def create_table_controller(main_window, **kwargs) -> Optional['TableController']:
    """
    Factory function para crear controlador de tablas
    
    Parameters
    ----------
    main_window : QtWidgets.QMainWindow
        Ventana principal
    **kwargs
        Argumentos adicionales
        
    Returns
    -------
    TableController or None
        Controlador de tablas o None si no está disponible
    """
    if not _TABLE_AVAILABLE:
        logger.error("TableController no está disponible")
        return None
    
    return TableController(main_window, **kwargs)


def create_graph_controller(main_window, **kwargs) -> Optional['GraphController']:
    """
    Factory function para crear controlador de gráficos
    
    Parameters
    ----------
    main_window : QtWidgets.QMainWindow
        Ventana principal
    **kwargs
        Argumentos adicionales
        
    Returns
    -------
    GraphController or None
        Controlador de gráficos o None si no está disponible
    """
    if not _GRAPH_AVAILABLE:
        logger.error("GraphController no está disponible")
        return None
    
    return GraphController(main_window, **kwargs)


# Exportar funciones y clases principales
__all__ = [
    # Clases principales (si están disponibles)
    'BaseController' if _BASE_AVAILABLE else None,
    'SeismicController' if _SEISMIC_AVAILABLE else None,
    'TableController' if _TABLE_AVAILABLE else None,
    'GraphController' if _GRAPH_AVAILABLE else None,
    
    # Clases auxiliares
    'PandasTableModel' if _BASE_AVAILABLE else None,
    'BaseGraphDialog' if _BASE_AVAILABLE else None,
    'BaseTableDialog' if _BASE_AVAILABLE else None,
    'AnalysisWorker' if _SEISMIC_AVAILABLE else None,
    'GraphType' if _GRAPH_AVAILABLE else None,
    
    # Factory functions
    'create_controller_suite',
    'create_base_controller',
    'create_seismic_controller', 
    'create_table_controller',
    'create_graph_controller',
    
    # Utility functions
    'get_available_controllers',
    'is_controller_available',
    'get_controller_status'
]

# Filtrar None values de __all__
__all__ = [item for item in __all__ if item is not None]


def get_version() -> str:
    """Retorna la versión del paquete de controladores"""
    return __version__


def check_all_dependencies() -> Dict[str, bool]:
    """
    Verifica todas las dependencias de los controladores
    
    Returns
    -------
    Dict[str, bool]
        Estado de las dependencias
    """
    dependencies = {
        'PyQt5': False,
        'pandas': False,
        'numpy': False,
        'matplotlib': False,
        'seismic_common.core': False
    }
    
    # Verificar PyQt5
    try:
        from PyQt5 import QtWidgets, QtCore
        dependencies['PyQt5'] = True
    except ImportError:
        pass
    
    # Verificar pandas
    try:
        import pandas
        dependencies['pandas'] = True
    except ImportError:
        pass
    
    # Verificar numpy
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    # Verificar matplotlib
    try:
        import matplotlib
        dependencies['matplotlib'] = True
    except ImportError:
        pass
    
    # Verificar core
    try:
        from seismic_common import core
        dependencies['seismic_common.core'] = True
    except ImportError:
        pass
    
    return dependencies


# Mensaje de inicialización
if __name__ == '__main__':
    print("=== Seismic Common Controllers ===")
    print(f"Versión: {get_version()}")
    print()
    
    # Mostrar estado de controladores
    print("Estado de controladores:")
    status = get_controller_status()
    for controller, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {controller}: {status_icon}")
    
    print()
    
    # Mostrar dependencias
    print("Estado de dependencias:")
    deps = check_all_dependencies()
    for dep, available in deps.items():
        status_icon = "✅" if available else "❌"
        print(f"  {dep}: {status_icon}")
    
    print()
    print("Controladores disponibles:", get_available_controllers())
    
    print("\n=== Uso Recomendado ===")
    print("# Crear suite completa")
    print("from seismic_common.controllers import create_controller_suite")
    print("controllers = create_controller_suite(main_window, 'PE')")
    print()
    print("# Usar controladores individuales")
    print("seismic = controllers['seismic']")
    print("graph = controllers['graph']")
    print("table = controllers['table']")
else:
    # Log de inicialización silenciosa
    available_count = sum(get_controller_status().values())
    total_count = len(get_controller_status())
    logger.info(f"Seismic Controllers inicializado: {available_count}/{total_count} disponibles")