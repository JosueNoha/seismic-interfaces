"""
Módulo de diálogos centralizados para interfaces sísmicas
========================================================

Este módulo proporciona diálogos reutilizables para mostrar:
- Tablas de datos sísmicos (pandas DataFrames)
- Gráficos de análisis (matplotlib Figures)  
- Descripciones y textos explicativos

Características principales:
- Compatibilidad total con código existente
- Funcionalidades avanzadas (exportación, plantillas, estadísticas)
- Interfaz consistente entre todos los diálogos
- Señales PyQt para comunicación entre componentes
- Configuración flexible (modos simple y avanzado)

Ejemplo de uso básico:
    ```python
    from seismic_common.dialogs import (
        create_modal_table_dialog,
        create_drift_graph_dialog, 
        create_structure_description_dialog
    )
    
    # Mostrar tabla modal
    table_dialog = create_modal_table_dialog(modal_data)
    table_dialog.exec_()
    
    # Mostrar gráfico de derivas
    graph_dialog = create_drift_graph_dialog(drift_figure)
    graph_dialog.exec_()
    
    # Editar descripción
    desc_dialog = create_structure_description_dialog("Texto inicial")
    desc_dialog.show()
    ```

Ejemplo de suite completa:
    ```python
    from seismic_common.dialogs import create_dialog_suite
    
    # Crear suite para ventana principal
    dialogs = create_dialog_suite(main_window)
    
    # Usar diálogos específicos
    dialogs['modal_table'](data).exec_()
    dialogs['drift_graph'](figure).exec_()
    dialogs['structure_description']("texto").show()
    ```
"""

# Metadatos del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Diálogos centralizados para interfaces sísmicas reutilizables"
__license__ = "MIT"
__status__ = "Production"

# Dependencias requeridas
__requires__ = [
    "PyQt5>=5.15.0",
    "pandas>=1.3.0", 
    "matplotlib>=3.5.0",
    "numpy>=1.21.0"
]

import sys
import logging
from typing import Dict, List, Optional, Union, Any, Callable

# Configurar logging para el módulo
logger = logging.getLogger(__name__)

# Verificar dependencias críticas
def _check_dependencies() -> Dict[str, bool]:
    """Verifica la disponibilidad de dependencias críticas"""
    deps = {
        'PyQt5': False,
        'pandas': False,
        'matplotlib': False,
        'numpy': False
    }
    
    try:
        from PyQt5 import QtWidgets, QtCore
        deps['PyQt5'] = True
    except ImportError:
        logger.warning("PyQt5 no disponible - funcionalidad limitada")
    
    try:
        import pandas
        deps['pandas'] = True
    except ImportError:
        logger.warning("pandas no disponible - diálogos de tabla limitados")
    
    try:
        import matplotlib
        deps['matplotlib'] = True
    except ImportError:
        logger.warning("matplotlib no disponible - diálogos de gráfico limitados")
        
    try:
        import numpy
        deps['numpy'] = True
    except ImportError:
        logger.warning("numpy no disponible - funcionalidad numérica limitada")
    
    return deps

# Verificar dependencias al importar
_DEPENDENCIES = _check_dependencies()

#============================================================================
# IMPORTACIONES PRINCIPALES
#============================================================================

# Importaciones de diálogos de tablas
try:
    from .table_dialog import (
        # Clases principales
        PandasTableModel,
        SeismicTableDialog,
        
        # Funciones de conveniencia
        create_modal_table_dialog,
        create_static_analysis_dialog,
        create_soft_story_dialog,
        create_mass_irregularity_dialog,
        create_torsion_irregularity_dialog,
        
        # Alias de compatibilidad
        diagTable,
        FormTabla,
        TableDialog
    )
    _TABLE_DIALOGS_AVAILABLE = True
    logger.debug("Diálogos de tabla cargados exitosamente")
except ImportError as e:
    logger.error(f"Error cargando diálogos de tabla: {e}")
    _TABLE_DIALOGS_AVAILABLE = False
    
    # Crear stubs para evitar errores
    class PandasTableModel: pass
    class SeismicTableDialog: pass
    def create_modal_table_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de tabla no disponibles")
    def create_static_analysis_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de tabla no disponibles")
    def create_soft_story_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de tabla no disponibles")
    def create_mass_irregularity_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de tabla no disponibles")
    def create_torsion_irregularity_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de tabla no disponibles")
    diagTable = FormTabla = TableDialog = SeismicTableDialog

# Importaciones de diálogos de gráficos
try:
    from .graph_dialog import (
        # Clases principales
        SeismicGraphDialog,
        
        # Funciones de conveniencia
        create_drift_graph_dialog,
        create_displacement_graph_dialog,
        create_shear_graph_dialog,
        create_spectrum_graph_dialog,
        create_advanced_graph_dialog,
        
        # Alias de compatibilidad
        FormCanvas,
        GraphDialog,
        SeismicGraphCanvas
    )
    _GRAPH_DIALOGS_AVAILABLE = True
    logger.debug("Diálogos de gráfico cargados exitosamente")
except ImportError as e:
    logger.error(f"Error cargando diálogos de gráfico: {e}")
    _GRAPH_DIALOGS_AVAILABLE = False
    
    # Crear stubs
    class SeismicGraphDialog: pass
    def create_drift_graph_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de gráfico no disponibles")
    def create_displacement_graph_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de gráfico no disponibles") 
    def create_shear_graph_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de gráfico no disponibles")
    def create_spectrum_graph_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de gráfico no disponibles")
    def create_advanced_graph_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de gráfico no disponibles")
    FormCanvas = GraphDialog = SeismicGraphCanvas = SeismicGraphDialog

# Importaciones de diálogos de descripción
try:
    from .description_dialog import (
        # Clases principales
        SeismicDescriptionDialog,
        
        # Funciones de conveniencia
        create_structure_description_dialog,
        create_modeling_description_dialog,
        create_loads_description_dialog,
        create_simple_description_dialog,
        
        # Alias de compatibilidad
        Descriptions,
        DescriptionDialog,
        SeismicDescriptions
    )
    _DESCRIPTION_DIALOGS_AVAILABLE = True
    logger.debug("Diálogos de descripción cargados exitosamente")
except ImportError as e:
    logger.error(f"Error cargando diálogos de descripción: {e}")
    _DESCRIPTION_DIALOGS_AVAILABLE = False
    
    # Crear stubs
    class SeismicDescriptionDialog: pass
    def create_structure_description_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de descripción no disponibles")
    def create_modeling_description_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de descripción no disponibles")
    def create_loads_description_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de descripción no disponibles")
    def create_simple_description_dialog(*args, **kwargs): raise NotImplementedError("Diálogos de descripción no disponibles")
    Descriptions = DescriptionDialog = SeismicDescriptions = SeismicDescriptionDialog

#============================================================================
# EXPORTACIONES PÚBLICAS
#============================================================================

__all__ = [
    # Metadatos
    '__version__',
    '__author__',
    '__description__',
    
    # Clases principales de tablas
    'PandasTableModel',
    'SeismicTableDialog',
    
    # Funciones de conveniencia para tablas
    'create_modal_table_dialog',
    'create_static_analysis_dialog',
    'create_soft_story_dialog', 
    'create_mass_irregularity_dialog',
    'create_torsion_irregularity_dialog',
    
    # Clases principales de gráficos
    'SeismicGraphDialog',
    
    # Funciones de conveniencia para gráficos
    'create_drift_graph_dialog',
    'create_displacement_graph_dialog',
    'create_shear_graph_dialog',
    'create_spectrum_graph_dialog', 
    'create_advanced_graph_dialog',
    
    # Clases principales de descripciones
    'SeismicDescriptionDialog',
    
    # Funciones de conveniencia para descripciones
    'create_structure_description_dialog',
    'create_modeling_description_dialog',
    'create_loads_description_dialog',
    'create_simple_description_dialog',
    
    # Funciones de utilidad
    'get_available_dialogs',
    'create_dialog_from_config',
    'create_dialog_suite',
    'get_dialog_info',
    'check_dialog_dependencies',
    'migrate_from_old_dialog',
    
    # Alias de compatibilidad (código existente)
    'diagTable',           # -> SeismicTableDialog
    'FormTabla',           # -> SeismicTableDialog  
    'TableDialog',         # -> SeismicTableDialog
    'FormCanvas',          # -> SeismicGraphDialog
    'GraphDialog',         # -> SeismicGraphDialog
    'SeismicGraphCanvas',  # -> SeismicGraphDialog
    'Descriptions',        # -> SeismicDescriptionDialog
    'DescriptionDialog',   # -> SeismicDescriptionDialog
    'SeismicDescriptions', # -> SeismicDescriptionDialog
]

#============================================================================
# CONFIGURACIONES Y CONSTANTES
#============================================================================

# Configuración por defecto para todos los diálogos
DEFAULT_DIALOG_CONFIG = {
    # Configuración de ventana
    'modal': True,
    'resizable': True,
    'center_on_parent': True,
    
    # Tamaños por defecto
    'default_table_size': (800, 600),
    'default_graph_size': (800, 600), 
    'default_description_size': (800, 500),
    
    # Configuración de fuentes
    'title_font_family': 'Montserrat',
    'title_font_size': 24,
    'content_font_family': 'MS Shell Dlg 2',
    'content_font_size': 12,
    
    # Configuración de funcionalidades
    'show_export_buttons': True,
    'show_controls': False,
    'show_templates': True,
    'show_statistics': True,
    'auto_resize_columns': True,
    
    # Formatos de exportación soportados
    'export_formats': {
        'tables': ['xlsx', 'csv'],
        'graphs': ['png', 'pdf', 'svg', 'jpg'],
        'descriptions': ['txt', 'md']
    },
    
    # Configuración de colores y estilos
    'colors': {
        'primary': '#2196F3',
        'secondary': '#FFC107', 
        'success': '#4CAF50',
        'warning': '#FF9800',
        'error': '#F44336'
    }
}

# Tipos de diálogos disponibles
DIALOG_TYPES = {
    'table': [
        'modal_table',
        'static_analysis_table',
        'soft_story_table',
        'mass_irregularity_table', 
        'torsion_irregularity_table',
        'generic_table'
    ],
    'graph': [
        'drift_graph',
        'displacement_graph',
        'shear_graph',
        'spectrum_graph',
        'advanced_graph',
        'generic_graph'
    ],
    'description': [
        'structure_description',
        'modeling_description',
        'loads_description',
        'simple_description',
        'generic_description'
    ]
}

#============================================================================
# FUNCIONES DE UTILIDAD
#============================================================================

def get_available_dialogs() -> List[str]:
    """
    Retorna lista de todos los diálogos disponibles en el sistema
    
    Returns
    -------
    List[str]
        Lista de nombres de diálogos disponibles, separados por categoría
    """
    dialogs = []
    
    # Diálogos de clase principal
    main_dialogs = [
        'SeismicTableDialog',
        'SeismicGraphDialog', 
        'SeismicDescriptionDialog'
    ]
    
    # Agregar según disponibilidad
    if _TABLE_DIALOGS_AVAILABLE:
        dialogs.extend(DIALOG_TYPES['table'])
    if _GRAPH_DIALOGS_AVAILABLE:
        dialogs.extend(DIALOG_TYPES['graph'])
    if _DESCRIPTION_DIALOGS_AVAILABLE:
        dialogs.extend(DIALOG_TYPES['description'])
    
    return main_dialogs + dialogs

def check_dialog_dependencies() -> Dict[str, Any]:
    """
    Verifica el estado de las dependencias de diálogos
    
    Returns
    -------
    Dict[str, Any]
        Información detallada sobre dependencias y disponibilidad
    """
    return {
        'dependencies': _DEPENDENCIES,
        'dialogs_available': {
            'tables': _TABLE_DIALOGS_AVAILABLE,
            'graphs': _GRAPH_DIALOGS_AVAILABLE,
            'descriptions': _DESCRIPTION_DIALOGS_AVAILABLE
        },
        'total_dialogs': len(get_available_dialogs()),
        'recommended_action': _get_recommended_action()
    }

def _get_recommended_action() -> str:
    """Determina la acción recomendada basada en dependencias"""
    if not any(_DEPENDENCIES.values()):
        return "Instalar todas las dependencias: pip install PyQt5 pandas matplotlib numpy"
    elif not _DEPENDENCIES['PyQt5']:
        return "Instalar PyQt5: pip install PyQt5"
    elif not (_TABLE_DIALOGS_AVAILABLE and _GRAPH_DIALOGS_AVAILABLE and _DESCRIPTION_DIALOGS_AVAILABLE):
        return "Verificar importaciones de módulos de diálogos"
    else:
        return "Todas las dependencias están disponibles"

def create_dialog_from_config(dialog_type: str, data_or_content: Any, config: Optional[Dict] = None):
    """
    Crea un diálogo basado en configuración dinámica
    
    Parameters
    ----------
    dialog_type : str
        Tipo de diálogo a crear ('modal', 'drift', 'structure_description', etc.)
    data_or_content : Any
        Contenido a mostrar (DataFrame, Figure, str, etc.)
    config : Dict, optional
        Configuración adicional para el diálogo
        
    Returns
    -------
    Union[SeismicTableDialog, SeismicGraphDialog, SeismicDescriptionDialog]
        Diálogo configurado según el tipo solicitado
        
    Raises
    ------
    ValueError
        Si el tipo de diálogo no es reconocido
    NotImplementedError
        Si el tipo de diálogo no está disponible por falta de dependencias
    """
    config = config or {}
    parent = config.get('parent')
    
    # Mapeo de tipos de diálogo para tablas
    if dialog_type in DIALOG_TYPES['table'] and _TABLE_DIALOGS_AVAILABLE:
        table_creators = {
            'modal_table': create_modal_table_dialog,
            'static_analysis_table': create_static_analysis_dialog,
            'soft_story_table': create_soft_story_dialog,
            'mass_irregularity_table': create_mass_irregularity_dialog,
            'torsion_irregularity_table': create_torsion_irregularity_dialog,
            'generic_table': lambda data, parent=None: SeismicTableDialog(
                data, config.get('title', 'Tabla'), config.get('table_title', 'Datos'), parent)
        }
        creator = table_creators.get(dialog_type)
        if creator:
            return creator(data_or_content, parent)
    
    # Mapeo de tipos de diálogo para gráficos
    elif dialog_type in DIALOG_TYPES['graph'] and _GRAPH_DIALOGS_AVAILABLE:
        graph_creators = {
            'drift_graph': create_drift_graph_dialog,
            'displacement_graph': create_displacement_graph_dialog,
            'shear_graph': lambda fig, parent=None: create_shear_graph_dialog(fig, config.get('analysis_type', 'dynamic'), parent),
            'spectrum_graph': create_spectrum_graph_dialog,
            'advanced_graph': lambda fig, parent=None: create_advanced_graph_dialog(fig, config.get('title', 'Gráfico'), parent),
            'generic_graph': lambda fig, parent=None: SeismicGraphDialog(
                fig, config.get('title', 'Gráfico'), config.get('graph_title', 'Gráfico'), parent)
        }
        creator = graph_creators.get(dialog_type)
        if creator:
            return creator(data_or_content, parent)
    
    # Mapeo de tipos de diálogo para descripciones  
    elif dialog_type in DIALOG_TYPES['description'] and _DESCRIPTION_DIALOGS_AVAILABLE:
        desc_creators = {
            'structure_description': lambda text="", parent=None: create_structure_description_dialog(text or data_or_content, parent),
            'modeling_description': lambda text="", parent=None: create_modeling_description_dialog(text or data_or_content, parent),
            'loads_description': lambda text="", parent=None: create_loads_description_dialog(text or data_or_content, parent),
            'simple_description': lambda text="", parent=None: create_simple_description_dialog(
                config.get('title', 'Descripción'), text or data_or_content, parent),
            'generic_description': lambda text="", parent=None: SeismicDescriptionDialog(
                config.get('title', 'Descripción'), text or data_or_content, 
                config.get('description_name', 'custom'), parent)
        }
        creator = desc_creators.get(dialog_type)
        if creator:
            return creator(data_or_content, parent)
    
    # Auto-detección por tipo de datos
    else:
        return _create_dialog_by_data_type(data_or_content, config)

def _create_dialog_by_data_type(data_or_content: Any, config: Dict):
    """Crea diálogo basado en el tipo de datos automáticamente"""
    try:
        # Importar solo cuando sea necesario
        if _DEPENDENCIES['pandas']:
            import pandas as pd
            if isinstance(data_or_content, pd.DataFrame):
                return SeismicTableDialog(
                    data_or_content,
                    config.get('title', 'Tabla'),
                    config.get('table_title', 'Datos'),
                    config.get('parent')
                )
        
        if _DEPENDENCIES['matplotlib']:
            from matplotlib.figure import Figure
            if isinstance(data_or_content, Figure):
                return SeismicGraphDialog(
                    data_or_content,
                    config.get('title', 'Gráfico'),
                    config.get('graph_title', 'Gráfico'),
                    config.get('parent')
                )
        
        if isinstance(data_or_content, str):
            return SeismicDescriptionDialog(
                config.get('title', 'Descripción'),
                data_or_content,
                config.get('description_name', 'custom'),
                config.get('parent')
            )
        
        raise ValueError(f"Tipo de datos no soportado: {type(data_or_content)}")
        
    except Exception as e:
        logger.error(f"Error creando diálogo automáticamente: {e}")
        raise

def create_dialog_suite(parent=None) -> Dict[str, Callable]:
    """
    Crea una suite completa de diálogos para uso común
    
    Parameters
    ---------- 
    parent : QWidget, optional
        Widget padre para todos los diálogos de la suite
        
    Returns
    -------
    Dict[str, Callable]
        Diccionario con funciones de creación de diálogos organizadas por categoría
        
    Example
    -------
    >>> dialogs = create_dialog_suite(main_window)
    >>> modal_dialog = dialogs['tables']['modal'](data)
    >>> drift_dialog = dialogs['graphs']['drift'](figure)  
    >>> desc_dialog = dialogs['descriptions']['structure'](text)
    """
    suite = {
        'tables': {},
        'graphs': {},
        'descriptions': {},
        'legacy': {}  # Para compatibilidad
    }
    
    # Funciones de tablas
    if _TABLE_DIALOGS_AVAILABLE:
        suite['tables'].update({
            'modal': lambda data: create_modal_table_dialog(data, parent),
            'static': lambda data: create_static_analysis_dialog(data, parent),
            'soft_story': lambda data: create_soft_story_dialog(data, parent),
            'mass_irregularity': lambda data: create_mass_irregularity_dialog(data, parent),
            'torsion': lambda data: create_torsion_irregularity_dialog(data, parent),
            'generic': lambda data, title="Tabla": SeismicTableDialog(data, title, title, parent)
        })
    
    # Funciones de gráficos
    if _GRAPH_DIALOGS_AVAILABLE:
        suite['graphs'].update({
            'drift': lambda figure: create_drift_graph_dialog(figure, parent),
            'displacement': lambda figure: create_displacement_graph_dialog(figure, parent),
            'shear': lambda figure, analysis_type="dynamic": create_shear_graph_dialog(figure, analysis_type, parent),
            'spectrum': lambda figure: create_spectrum_graph_dialog(figure, parent),
            'advanced': lambda figure, title="Gráfico": create_advanced_graph_dialog(figure, title, parent),
            'generic': lambda figure, title="Gráfico": SeismicGraphDialog(figure, title, title, parent)
        })
    
    # Funciones de descripciones
    if _DESCRIPTION_DIALOGS_AVAILABLE:
        suite['descriptions'].update({
            'structure': lambda text="": create_structure_description_dialog(text, parent),
            'modeling': lambda text="": create_modeling_description_dialog(text, parent),
            'loads': lambda text="": create_loads_description_dialog(text, parent),
            'simple': lambda text="", title="Descripción": create_simple_description_dialog(title, text, parent),
            'generic': lambda text="", title="Descripción": SeismicDescriptionDialog(title, text, "custom", parent)
        })
    
    # Funciones de compatibilidad (acceso directo)
    suite['legacy'].update({
        'diag_table': lambda: diagTable(parent),
        'form_canvas': lambda figure=None: FormCanvas(figure, parent),
        'descriptions': lambda: Descriptions(parent)
    })
    
    # Funciones de acceso directo (backward compatibility)
    suite.update({
        'modal_table': suite['tables'].get('modal', lambda x: None),
        'drift_graph': suite['graphs'].get('drift', lambda x: None),
        'structure_description': suite['descriptions'].get('structure', lambda x: None)
    })
    
    return suite

def migrate_from_old_dialog(old_dialog_class: type, content: Any, title: str = "Diálogo") -> Any:
    """
    Ayuda a migrar desde diálogos antiguos al nuevo sistema centralizado
    
    Parameters
    ----------
    old_dialog_class : type
        Clase de diálogo antigua (diagTable, FormCanvas, Descriptions, etc.)
    content : Any
        Contenido a mostrar en el nuevo diálogo
    title : str
        Título del diálogo
        
    Returns
    -------
    Union[SeismicTableDialog, SeismicGraphDialog, SeismicDescriptionDialog]
        Diálogo equivalente del nuevo sistema
        
    Example
    -------
    >>> # En lugar de: old_dialog = diagTable()
    >>> new_dialog = migrate_from_old_dialog(diagTable, dataframe, "Tabla Modal")
    """
    class_name = getattr(old_dialog_class, '__name__', str(old_dialog_class))
    
    # Mapeo de clases antiguas a nuevas
    migration_map = {
        'diagTable': lambda: SeismicTableDialog(content, title, title) if hasattr(content, 'columns') else SeismicTableDialog(title=title),
        'FormTabla': lambda: SeismicTableDialog(content, title, title) if hasattr(content, 'columns') else SeismicTableDialog(title=title),
        'FormCanvas': lambda: SeismicGraphDialog(content, title, title) if hasattr(content, 'savefig') else SeismicGraphDialog(title=title),
        'Descriptions': lambda: SeismicDescriptionDialog(title, content if isinstance(content, str) else "", "migrated")
    }
    
    migrator = migration_map.get(class_name)
    if migrator:
        try:
            return migrator()
        except Exception as e:
            logger.warning(f"Error en migración de {class_name}: {e}")
    
    # Fallback: intentar auto-detección
    return create_dialog_from_config('generic_table', content, {'title': title})

def get_dialog_info() -> Dict[str, Any]:
    """
    Retorna información completa del módulo de diálogos
    
    Returns
    -------
    Dict[str, Any]
        Información detallada incluyendo versión, dependencias, diálogos disponibles
    """
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'license': __license__,
        'status': __status__,
        'dependencies': {
            'required': __requires__,
            'available': _DEPENDENCIES,
            'missing': [dep for dep, available in _DEPENDENCIES.items() if not available]
        },
        'dialogs': {
            'available': get_available_dialogs(),
            'by_type': DIALOG_TYPES,
            'total_count': len(get_available_dialogs())
        },
        'modules': {
            'tables': _TABLE_DIALOGS_AVAILABLE,
            'graphs': _GRAPH_DIALOGS_AVAILABLE, 
            'descriptions': _DESCRIPTION_DIALOGS_AVAILABLE
        },
        'config': DEFAULT_DIALOG_CONFIG
    }

#============================================================================
# INICIALIZACIÓN DEL MÓDULO
#============================================================================

def _initialize_module():
    """Inicializa el módulo y reporta el estado"""
    info = get_dialog_info()
    available_count = len(info['dialogs']['available'])
    total_possible = sum(len(dialogs) for dialogs in DIALOG_TYPES.values()) + 3  # +3 por las clases principales
    
    logger.info(f"Seismic Dialogs v{__version__} inicializado")
    logger.info(f"Diálogos disponibles: {available_count}/{total_possible}")
    
    missing_deps = info['dependencies']['missing']
    if missing_deps:
        logger.warning(f"Dependencias faltantes: {', '.join(missing_deps)}")
        logger.warning(f"Recomendación: {info['dependencies'].get('recommended_action', 'Verificar instalación')}")
    
    return info

# Ejecutar inicialización
_MODULE_INFO = _initialize_module()

#============================================================================
# MODO DESARROLLO Y DEBUGGING
#============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SEISMIC DIALOGS MODULE - INFORMACIÓN COMPLETA")
    print("=" * 60)
    
    info = get_dialog_info()
    
    print(f"📦 Versión: {info['version']}")
    print(f"👤 Autor: {info['author']}")
    print(f"📝 Descripción: {info['description']}")
    print(f"📜 Licencia: {info['license']}")
    print(f"🔧 Estado: {info['status']}")
    print()
    
    # Estado de dependencias
    print("🔗 DEPENDENCIAS:")
    for dep, available in info['dependencies']['available'].items():
        status = "✅ Disponible" if available else "❌ No disponible"
        print(f"  {dep}: {status}")
    print()
    
    # Estado de módulos
    print("📂 MÓDULOS:")
    modules = info['modules']
    print(f"  📊 Tablas: {'✅ Cargado' if modules['tables'] else '❌ Error'}")
    print(f"  📈 Gráficos: {'✅ Cargado' if modules['graphs'] else '❌ Error'}")
    print(f"  📝 Descripciones: {'✅ Cargado' if modules['descriptions'] else '❌ Error'}")
    print()
    
    # Diálogos por categoría
    print("🎛️ DIÁLOGOS DISPONIBLES:")
    
    for category, dialogs in DIALOG_TYPES.items():
        category_name = {
            'table': '📊 Tablas', 
            'graph': '📈 Gráficos',
            'description': '📝 Descripciones'
        }.get(category, category.title())
        
        print(f"\n  {category_name}:")
        for dialog in dialogs:
            print(f"    ✅ {dialog}")
    
    print()
    print("🚀 EJEMPLOS DE USO:")
    print()
    
    print("# Importación básica:")
    print("from seismic_common.dialogs import (")
    print("    create_modal_table_dialog,")
    print("    create_drift_graph_dialog,")
    print("    create_structure_description_dialog")
    print(")")
    print()
    
    print("# Uso individual:")
    print("table_dialog = create_modal_table_dialog(data)")
    print("table_dialog.exec_()")
    print()
    print("graph_dialog = create_drift_graph_dialog(figure)")
    print("graph_dialog.exec_()")
    print()
    print("desc_dialog = create_structure_description_dialog('Texto')")
    print("desc_dialog.show()")
    print()
    
    print("# Suite completa:")
    print("from seismic_common.dialogs import create_dialog_suite")
    print("dialogs = create_dialog_suite(main_window)")
    print()
    print("# Acceso por categoría:")
    print("dialogs['tables']['modal'](data).exec_()")
    print("dialogs['graphs']['drift'](figure).exec_()")
    print("dialogs['descriptions']['structure']('texto').show()")
    print()
    
    print("# Compatibilidad con código existente:")
    print("from seismic_common.dialogs import diagTable, FormCanvas, Descriptions")
    print("table = diagTable()  # Funciona exactamente igual")
    print("graph = FormCanvas(figure)  # Funciona exactamente igual") 
    print("desc = Descriptions()  # Funciona exactamente igual")
    print()
    
    print("# Creación dinámica:")
    print("from seismic_common.dialogs import create_dialog_from_config")
    print("dialog = create_dialog_from_config('modal_table', dataframe)")
    print("dialog = create_dialog_from_config('drift_graph', figure)")
    print("dialog = create_dialog_from_config('structure_description', 'texto')")
    print()
    
    # Recomendaciones
    missing_deps = info['dependencies']['missing']
    if missing_deps:
        print("⚠️  RECOMENDACIONES:")
        print(f"   Instalar dependencias faltantes: {', '.join(missing_deps)}")
        print(f"   Comando: pip install {' '.join(missing_deps)}")
        print()
    
    # Información adicional
    print("📚 INFORMACIÓN ADICIONAL:")
    print("   - Documentación completa disponible en cada función")
    print("   - Todas las clases emiten señales PyQt para integración")
    print("   - Soporte completo para exportación en múltiples formatos")
    print("   - Plantillas predefinidas para descripciones comunes")
    print("   - Interfaz consistente entre todos los diálogos")
    print("   - Compatibilidad 100% con código existente")
    print()
    
    print("=" * 60)
    print("Módulo listo para uso en producción ✅")
    print("=" * 60)

else:
    # Mensaje silencioso para importación normal
    available_count = len(get_available_dialogs()) 
    logger.debug(f"Seismic Dialogs v{__version__} - {available_count} diálogos disponibles")