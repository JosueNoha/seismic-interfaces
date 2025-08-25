"""
Módulo de modelos de datos centralizados para interfaces sísmicas
===============================================================

Este módulo proporciona modelos de datos reutilizables para análisis sísmico
según diferentes normativas internacionales.

Modelos disponibles:
- PandasTableModel: Modelo de tabla pandas para mostrar datos tabulares
- SeismicTableModel: Modelo especializado para datos sísmicos con validación
- SeismicData: Modelo base de datos sísmicos común a todas las normativas
- NormativeSeismicData: Clase base abstracta para normativas específicas
- LocationModel: Modelo de ubicaciones geográficas jerárquicas
- SeismicZoneDatabase: Base de datos de zonificación sísmica

Características principales:
- Compatibilidad total con código existente
- Extensibilidad para normativas específicas
- Validación automática de datos
- Serialización/deserialización
- Funciones de migración desde código heredado

Ejemplo de uso:
    ```python
    from seismic_common.models import (
        SeismicData, 
        PandasTableModel, 
        LocationModel,
        SeismicZoneDatabase
    )
    
    # Crear datos sísmicos
    seismic_data = SeismicData()
    seismic_data.set_reduction_factors(8.0, 7.0)
    
    # Crear modelo de tabla
    table_model = PandasTableModel(dataframe)
    
    # Crear base de ubicaciones
    locations = LocationModel('PE')
    locations.load_from_csv('peru_locations.csv')
    ```
"""

# Metadatos del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Modelos de datos centralizados para interfaces sísmicas"
__license__ = "MIT"
__status__ = "Production"

# Dependencias requeridas
__requires__ = [
    "pandas>=1.3.0",
    "numpy>=1.21.0", 
    "PyQt5>=5.15.0"
]

import sys
import logging
from typing import List, Dict, Any

# Configurar logging para el módulo
logger = logging.getLogger(__name__)

# Verificar dependencias críticas
def _check_dependencies() -> Dict[str, bool]:
    """Verifica la disponibilidad de dependencias críticas"""
    deps = {
        'pandas': False,
        'numpy': False,
        'PyQt5': False
    }
    
    try:
        import pandas
        deps['pandas'] = True
    except ImportError:
        logger.warning("pandas no disponible - funcionalidad de modelos limitada")
    
    try:
        import numpy
        deps['numpy'] = True
    except ImportError:
        logger.warning("numpy no disponible - funcionalidad numérica limitada")
        
    try:
        from PyQt5 import QtCore
        deps['PyQt5'] = True
    except ImportError:
        logger.warning("PyQt5 no disponible - modelos de tabla limitados")
    
    return deps

# Verificar dependencias al importar
_DEPENDENCIES = _check_dependencies()

#============================================================================
# IMPORTACIONES PRINCIPALES
#============================================================================

# Importaciones de modelos de tabla pandas
try:
    from .pandas_table_model import (
        # Clases principales
        PandasTableModel,
        SeismicTableModel,
        
        # Funciones de conveniencia  
        create_modal_table_model,
        create_drift_table_model,
        create_irregularity_table_model,
        create_static_table_model,
        
        # Alias de compatibilidad
        pandasModel,
        
        # Constantes
        DECIMAL_PLACES,
        COLORS,
        DEFAULT_LIMITS,
        DEFAULT_R_FACTORS
    )
    _PANDAS_MODELS_AVAILABLE = True
    logger.debug("Modelos de tabla pandas cargados exitosamente")
except ImportError as e:
    logger.error(f"Error cargando modelos de tabla pandas: {e}")
    _PANDAS_MODELS_AVAILABLE = False
    
    # Crear stubs para evitar errores
    class PandasTableModel: pass
    class SeismicTableModel: pass
    def create_modal_table_model(*args, **kwargs): raise NotImplementedError("Modelos de tabla no disponibles")
    def create_drift_table_model(*args, **kwargs): raise NotImplementedError("Modelos de tabla no disponibles")
    def create_irregularity_table_model(*args, **kwargs): raise NotImplementedError("Modelos de tabla no disponibles")
    def create_static_table_model(*args, **kwargs): raise NotImplementedError("Modelos de tabla no disponibles")
    pandasModel = PandasTableModel
    DECIMAL_PLACES = {}
    COLORS = {}
    DEFAULT_LIMITS = {}
    DEFAULT_R_FACTORS = {}

# Importaciones de modelos de datos sísmicos
try:
    from .seismic_data_model import (
        # Clases principales
        SeismicData,
        NormativeSeismicData,
        
        # Estructuras de datos
        ProjectInfo,
        LoadPatterns,
        ModalData,
        DriftData,
        
        # Funciones de utilidad
        create_seismic_data_for_normative,
        get_available_normatives,
        create_seismic_data_from_legacy,
        migrate_tables_to_dataframes,
        
        # Clases de compatibilidad
        SeismicLoads,
        SeismicTables,
        
        # Constantes
        DEFAULT_LIMITS,
        DEFAULT_R_FACTORS,
        DEFAULT_IRREGULARITY_FACTORS
    )
    _SEISMIC_DATA_AVAILABLE = True
    logger.debug("Modelos de datos sísmicos cargados exitosamente")
except ImportError as e:
    logger.error(f"Error cargando modelos de datos sísmicos: {e}")
    _SEISMIC_DATA_AVAILABLE = False
    
    # Crear stubs para evitar errores
    class SeismicData: pass
    class NormativeSeismicData: pass
    class ProjectInfo: pass
    class LoadPatterns: pass
    class ModalData: pass
    class DriftData: pass
    class SeismicLoads: pass
    class SeismicTables: pass
    def create_seismic_data_for_normative(*args, **kwargs): raise NotImplementedError("Modelos sísmicos no disponibles")
    def get_available_normatives(): return []
    def create_seismic_data_from_legacy(*args, **kwargs): raise NotImplementedError("Modelos sísmicos no disponibles")
    def migrate_tables_to_dataframes(*args, **kwargs): raise NotImplementedError("Modelos sísmicos no disponibles")

# Importaciones de modelos de ubicación
try:
    from .location_model import (
        # Clases principales
        LocationModel,
        SeismicZoneDatabase,
        
        # Estructuras de datos
        LocationInfo,
        SeismicZoneInfo,
        
        # Funciones de utilidad
        get_country_hierarchy,
        create_location_from_address,
        validate_country_data_structure,
        merge_location_databases,
        export_to_geojson,
        migrate_basedatos_zonificacion,
        
        # Funciones de análisis
        find_duplicates_in_hierarchy,
        get_location_tree_structure,
        validate_location_consistency,
        create_location_lookup_table,
        
        # Clases de compatibilidad
        BaseDatos_Zonas_Sismicas,
        
        # Constantes
        LOCATION_HIERARCHIES,
        LEGACY_COLUMN_MAPPING
    )
    _LOCATION_MODELS_AVAILABLE = True
    logger.debug("Modelos de ubicación cargados exitosamente")
except ImportError as e:
    logger.error(f"Error cargando modelos de ubicación: {e}")
    _LOCATION_MODELS_AVAILABLE = False
    
    # Crear stubs para evitar errores
    class LocationModel: pass
    class SeismicZoneDatabase: pass
    class LocationInfo: pass
    class SeismicZoneInfo: pass
    class BaseDatos_Zonas_Sismicas: pass
    def get_country_hierarchy(*args, **kwargs): return []
    def create_location_from_address(*args, **kwargs): return {}
    def validate_country_data_structure(*args, **kwargs): return False, []
    def merge_location_databases(*args, **kwargs): raise NotImplementedError("Modelos de ubicación no disponibles")
    def export_to_geojson(*args, **kwargs): raise NotImplementedError("Modelos de ubicación no disponibles")
    def migrate_basedatos_zonificacion(*args, **kwargs): raise NotImplementedError("Modelos de ubicación no disponibles")
    def find_duplicates_in_hierarchy(*args, **kwargs): raise NotImplementedError("Modelos de ubicación no disponibles")
    def get_location_tree_structure(*args, **kwargs): return {}
    def validate_location_consistency(*args, **kwargs): return {}
    def create_location_lookup_table(*args, **kwargs): return {}
    LOCATION_HIERARCHIES = {}
    LEGACY_COLUMN_MAPPING = {}

#============================================================================
# LISTA DE EXPORTACIONES PÚBLICAS
#============================================================================

__all__ = [
    # Modelos de tabla pandas
    'PandasTableModel',
    'SeismicTableModel', 
    'create_modal_table_model',
    'create_drift_table_model',
    'create_irregularity_table_model',
    'create_static_table_model',
    'pandasModel',  # Compatibilidad
    
    # Modelos de datos sísmicos
    'SeismicData',
    'NormativeSeismicData',
    'ProjectInfo',
    'LoadPatterns',
    'ModalData', 
    'DriftData',
    'SeismicLoads',
    'SeismicTables',
    'create_seismic_data_for_normative',
    'get_available_normatives',
    'create_seismic_data_from_legacy',
    'migrate_tables_to_dataframes',
    
    # Modelos de ubicación
    'LocationModel',
    'SeismicZoneDatabase',
    'LocationInfo',
    'SeismicZoneInfo',
    'BaseDatos_Zonas_Sismicas',  # Compatibilidad
    'get_country_hierarchy',
    'create_location_from_address',
    'validate_country_data_structure',
    'merge_location_databases',
    'export_to_geojson',
    'migrate_basedatos_zonificacion',
    'find_duplicates_in_hierarchy',
    'get_location_tree_structure',
    'validate_location_consistency',
    'create_location_lookup_table',
    
    # Constantes importantes
    'LOCATION_HIERARCHIES',
    'LEGACY_COLUMN_MAPPING',
    'DEFAULT_LIMITS',
    'DEFAULT_R_FACTORS',
    'DEFAULT_IRREGULARITY_FACTORS'
]

#============================================================================
# FUNCIONES DE UTILIDAD DEL MÓDULO
#============================================================================

def get_available_models() -> Dict[str, bool]:
    """
    Obtiene información sobre qué modelos están disponibles
    
    Returns
    -------
    Dict[str, bool]
        Diccionario con disponibilidad de cada tipo de modelo
    """
    return {
        'pandas_models': _PANDAS_MODELS_AVAILABLE,
        'seismic_data': _SEISMIC_DATA_AVAILABLE,
        'location_models': _LOCATION_MODELS_AVAILABLE,
        'dependencies': _DEPENDENCIES
    }


def validate_all_models() -> Dict[str, bool]:
    """
    Ejecuta validaciones de todos los modelos disponibles
    
    Returns
    -------
    Dict[str, bool]
        Resultados de validación por modelo
    """
    results = {
        'pandas_models': False,
        'seismic_data': False,
        'location_models': False
    }
    
    # Validar modelos de tabla pandas
    if _PANDAS_MODELS_AVAILABLE:
        try:
            from .pandas_table_model import validate_model_compatibility
            results['pandas_models'] = validate_model_compatibility()
        except Exception as e:
            logger.error(f"Error validando modelos pandas: {e}")
    
    # Validar modelos de datos sísmicos
    if _SEISMIC_DATA_AVAILABLE:
        try:
            from .seismic_data_model import validate_seismic_data_model
            results['seismic_data'] = validate_seismic_data_model()
        except Exception as e:
            logger.error(f"Error validando modelos sísmicos: {e}")
    
    # Validar modelos de ubicación
    if _LOCATION_MODELS_AVAILABLE:
        try:
            from .location_model import validate_location_model
            results['location_models'] = validate_location_model()
        except Exception as e:
            logger.error(f"Error validando modelos de ubicación: {e}")
    
    return results


def create_complete_seismic_project(country_code: str = 'PE') -> Dict[str, Any]:
    """
    Crea un proyecto sísmico completo con todos los modelos integrados
    
    Parameters
    ----------
    country_code : str
        Código del país para el proyecto
        
    Returns
    -------
    Dict[str, Any]
        Diccionario con todos los modelos configurados
    """
    project = {}
    
    try:
        # Crear datos sísmicos
        if _SEISMIC_DATA_AVAILABLE:
            project['seismic_data'] = SeismicData()
            project['seismic_data'].project.name = f"Proyecto {country_code}"
        
        # Crear base de ubicaciones
        if _LOCATION_MODELS_AVAILABLE:
            project['locations'] = LocationModel(country_code)
            project['seismic_zones'] = SeismicZoneDatabase(country_code)
        
        # Configurar información del proyecto
        project['country_code'] = country_code
        project['hierarchy'] = get_country_hierarchy(country_code) if _LOCATION_MODELS_AVAILABLE else []
        project['available_models'] = get_available_models()
        
        logger.info(f"Proyecto sísmico completo creado para país: {country_code}")
        
    except Exception as e:
        logger.error(f"Error creando proyecto completo: {e}")
        project['error'] = str(e)
    
    return project


def get_compatibility_aliases() -> Dict[str, Any]:
    """
    Obtiene aliases de compatibilidad con código existente
    
    Returns
    -------
    Dict[str, Any]
        Diccionario con aliases de compatibilidad
    """
    aliases = {}
    
    # Aliases para modelos de tabla
    if _PANDAS_MODELS_AVAILABLE:
        aliases.update({
            'pandasModel': PandasTableModel,
            'FormTabla': SeismicTableModel,
            'TableDialog': SeismicTableModel,
            'diagTable': create_modal_table_model
        })
    
    # Aliases para datos sísmicos
    if _SEISMIC_DATA_AVAILABLE:
        aliases.update({
            'SeismicLoads': SeismicLoads,
            'SeismicTables': SeismicTables,
            'SismoData': SeismicData
        })
    
    # Aliases para ubicaciones
    if _LOCATION_MODELS_AVAILABLE:
        aliases.update({
            'BaseDatos_Zonas_Sismicas': BaseDatos_Zonas_Sismicas,
            'BD_Zonas_Sismicas': BaseDatos_Zonas_Sismicas
        })
    
    return aliases


def get_version_info() -> Dict[str, Any]:
    """
    Obtiene información de versión del módulo de modelos
    
    Returns
    -------
    Dict[str, Any]
        Información de versión y estado
    """
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'status': __status__,
        'available_models': get_available_models(),
        'total_exports': len(__all__),
        'dependencies_ok': all(_DEPENDENCIES.values())
    }


#============================================================================
# CONFIGURACIÓN DE COMPATIBILIDAD
#============================================================================

# Crear aliases globales para compatibilidad con código existente
_compatibility_aliases = get_compatibility_aliases()
globals().update(_compatibility_aliases)

# Agregar aliases a las exportaciones públicas
__all__.extend(_compatibility_aliases.keys())

#============================================================================
# INICIALIZACIÓN DEL MÓDULO
#============================================================================

def _initialize_module():
    """Inicializa el módulo de modelos"""
    try:
        # Verificar que al menos un tipo de modelo esté disponible
        if not any([_PANDAS_MODELS_AVAILABLE, _SEISMIC_DATA_AVAILABLE, _LOCATION_MODELS_AVAILABLE]):
            logger.warning("⚠️ Ningún modelo está completamente disponible")
            return False
        
        # Log de estado
        available_count = sum([_PANDAS_MODELS_AVAILABLE, _SEISMIC_DATA_AVAILABLE, _LOCATION_MODELS_AVAILABLE])
        logger.info(f"✓ Módulo de modelos inicializado ({available_count}/3 tipos disponibles)")
        
        # Log de detalles
        if _PANDAS_MODELS_AVAILABLE:
            logger.debug("  - Modelos de tabla pandas: ✓")
        if _SEISMIC_DATA_AVAILABLE:
            logger.debug("  - Modelos de datos sísmicos: ✓")
        if _LOCATION_MODELS_AVAILABLE:
            logger.debug("  - Modelos de ubicación: ✓")
        
        return True
        
    except Exception as e:
        logger.error(f"Error inicializando módulo de modelos: {e}")
        return False

# Ejecutar inicialización
_module_initialized = _initialize_module()

if __name__ == "__main__":
    # Ejecutar validaciones si el módulo se ejecuta directamente
    print("🔍 Validando módulo de modelos...")
    print(f"📊 Información de versión: {get_version_info()}")
    print(f"🔧 Modelos disponibles: {get_available_models()}")
    
    if _module_initialized:
        print("🧪 Ejecutando validaciones...")
        validation_results = validate_all_models()
        
        for model_type, result in validation_results.items():
            status = "✓" if result else "✗"
            print(f"   {model_type}: {status}")
        
        if all(validation_results.values()):
            print("🎉 Todas las validaciones pasaron exitosamente")
        else:
            print("⚠️ Algunas validaciones fallaron")
    else:
        print("❌ Error en la inicialización del módulo")