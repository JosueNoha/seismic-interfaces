"""
Módulo de widgets centralizados para interfaces sísmicas
Proporciona widgets reutilizables para análisis sísmico según diferentes normativas
"""

# Información del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Widgets centralizados para interfaces sísmicas reutilizables"

# Importaciones principales de parameter_widgets
from .parameter_widgets import (
    # Widgets base
    ParameterLineEdit,
    ParameterSpinBox,
    ParameterDoubleSpinBox,
    ParameterComboBox,
    ParameterCheckBox,
    
    # Widgets compuestos
    LabeledParameter,
    ParameterGroup,
    GridParameterGroup,
    
    # Widgets dinámicos
    LoadSelectionWidget,
    DynamicParameterWidget,
    
    # Funciones de utilidad para parámetros
    create_dropdown,
    create_input_box,
    create_spinbox,
    create_double_spinbox,
    create_check_box,
    create_button,
    create_parameter_from_config
)

# Importaciones principales de location_selectors
from .location_selectors import (
    # Clases principales
    LocationDatabase,
    HierarchicalLocationSelector,
    LocationSelectorWithToggle,
    CountryLocationSelector,
    
    # Funciones de utilidad para ubicaciones
    create_simple_location_selector,
    create_csv_location_selector
)

# Importaciones principales de system_selectors
from .system_selectors import (
    # Clases principales
    StructuralSystemDatabase,
    StructuralSystemSelector,
    DualDirectionSystemSelector,
    SystemPropertiesWidget,
    CategoryBasedSystemSelector,
    
    # Funciones de utilidad para sistemas
    create_basic_structural_systems,
    create_detailed_structural_systems,
    create_systems_from_dataframe
)

# Importaciones principales de irregularity_widgets
from .irregularity_widgets import (
    # Clases principales
    IrregularityDatabase,
    IrregularityCheckBox,
    IrregularityCategoryWidget,
    IrregularitySelector,
    IrregularitySummaryWidget,
    IrregularityAnalysisWidget,
    
    # Funciones de utilidad para irregularidades
    create_basic_irregularities,
    create_detailed_irregularities,
    create_irregularities_from_dataframe
)

# Exportar todas las clases y funciones principales
__all__ = [
    # Parameter widgets
    'ParameterLineEdit',
    'ParameterSpinBox', 
    'ParameterDoubleSpinBox',
    'ParameterComboBox',
    'ParameterCheckBox',
    'LabeledParameter',
    'ParameterGroup',
    'GridParameterGroup',
    'LoadSelectionWidget',
    'DynamicParameterWidget',
    'create_dropdown',
    'create_input_box',
    'create_spinbox',
    'create_double_spinbox',
    'create_check_box',
    'create_button',
    'create_parameter_from_config',
    
    # Location selectors
    'LocationDatabase',
    'HierarchicalLocationSelector',
    'LocationSelectorWithToggle',
    'CountryLocationSelector',
    'create_simple_location_selector',
    'create_csv_location_selector',
    
    # System selectors
    'StructuralSystemDatabase',
    'StructuralSystemSelector',
    'DualDirectionSystemSelector',
    'SystemPropertiesWidget',
    'CategoryBasedSystemSelector',
    'create_basic_structural_systems',
    'create_detailed_structural_systems',
    'create_systems_from_dataframe',
    
    # Irregularity widgets
    'IrregularityDatabase',
    'IrregularityCheckBox',
    'IrregularityCategoryWidget',
    'IrregularitySelector',
    'IrregularitySummaryWidget',
    'IrregularityAnalysisWidget',
    'create_basic_irregularities',
    'create_detailed_irregularities',
    'create_irregularities_from_dataframe'
]

# Diccionario de categorías de widgets para facilitar la importación selectiva
WIDGET_CATEGORIES = {
    'parameters': [
        'ParameterLineEdit', 'ParameterSpinBox', 'ParameterDoubleSpinBox',
        'ParameterComboBox', 'ParameterCheckBox', 'LabeledParameter',
        'ParameterGroup', 'GridParameterGroup', 'LoadSelectionWidget',
        'DynamicParameterWidget'
    ],
    'locations': [
        'LocationDatabase', 'HierarchicalLocationSelector',
        'LocationSelectorWithToggle', 'CountryLocationSelector'
    ],
    'systems': [
        'StructuralSystemDatabase', 'StructuralSystemSelector',
        'DualDirectionSystemSelector', 'SystemPropertiesWidget',
        'CategoryBasedSystemSelector'
    ],
    'irregularities': [
        'IrregularityDatabase', 'IrregularityCheckBox',
        'IrregularityCategoryWidget', 'IrregularitySelector',
        'IrregularitySummaryWidget', 'IrregularityAnalysisWidget'
    ]
}

# Funciones de utilidad por categoría
UTILITY_FUNCTIONS = {
    'parameters': [
        'create_dropdown', 'create_input_box', 'create_spinbox',
        'create_double_spinbox', 'create_check_box', 'create_button',
        'create_parameter_from_config'
    ],
    'locations': [
        'create_simple_location_selector', 'create_csv_location_selector'
    ],
    'systems': [
        'create_basic_structural_systems', 'create_detailed_structural_systems',
        'create_systems_from_dataframe'
    ],
    'irregularities': [
        'create_basic_irregularities', 'create_detailed_irregularities',
        'create_irregularities_from_dataframe'
    ]
}


def get_widgets_by_category(category: str):
    """
    Obtiene las clases de widgets de una categoría específica
    
    Parameters
    ----------
    category : str
        Categoría de widgets ('parameters', 'locations', 'systems', 'irregularities')
    
    Returns
    -------
    dict
        Diccionario con las clases de la categoría especificada
    """
    if category not in WIDGET_CATEGORIES:
        available_categories = list(WIDGET_CATEGORIES.keys())
        raise ValueError(f"Categoría '{category}' no válida. Disponibles: {available_categories}")
    
    # Obtener el módulo actual
    current_module = __import__(__name__, fromlist=[''])
    
    # Crear diccionario con las clases
    widgets = {}
    for widget_name in WIDGET_CATEGORIES[category]:
        if hasattr(current_module, widget_name):
            widgets[widget_name] = getattr(current_module, widget_name)
    
    return widgets


def get_utility_functions_by_category(category: str):
    """
    Obtiene las funciones de utilidad de una categoría específica
    
    Parameters
    ----------
    category : str
        Categoría de funciones ('parameters', 'locations', 'systems', 'irregularities')
    
    Returns
    -------
    dict
        Diccionario con las funciones de la categoría especificada
    """
    if category not in UTILITY_FUNCTIONS:
        available_categories = list(UTILITY_FUNCTIONS.keys())
        raise ValueError(f"Categoría '{category}' no válida. Disponibles: {available_categories}")
    
    # Obtener el módulo actual
    current_module = __import__(__name__, fromlist=[''])
    
    # Crear diccionario con las funciones
    functions = {}
    for function_name in UTILITY_FUNCTIONS[category]:
        if hasattr(current_module, function_name):
            functions[function_name] = getattr(current_module, function_name)
    
    return functions


def create_complete_seismic_interface():
    """
    Crea una interfaz sísmica completa con todos los widgets principales
    
    Returns
    -------
    dict
        Diccionario con instancias de los widgets principales configurados
    """
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
    
    # Crear widget principal
    main_widget = QWidget()
    layout = QVBoxLayout(main_widget)
    
    # Crear tab widget
    tab_widget = QTabWidget()
    
    # Tab de parámetros sísmicos básicos
    parameters_widget = DynamicParameterWidget("Parámetros Sísmicos")
    tab_widget.addTab(parameters_widget, "Parámetros")
    
    # Tab de ubicación
    # Nota: Requiere datos específicos del país, se crea vacío como ejemplo
    location_widget = QWidget()
    tab_widget.addTab(location_widget, "Ubicación")
    
    # Tab de sistemas estructurales
    systems_db = create_basic_structural_systems()
    systems_widget = DualDirectionSystemSelector(systems_db)
    tab_widget.addTab(systems_widget, "Sistemas")
    
    # Tab de irregularidades
    irregularities_db = create_basic_irregularities()
    irregularities_widget = IrregularitySelector(irregularities_db, 'tabs', 2)
    tab_widget.addTab(irregularities_widget, "Irregularidades")
    
    layout.addWidget(tab_widget)
    
    return {
        'main_widget': main_widget,
        'parameters': parameters_widget,
        'location': location_widget,
        'systems': systems_widget,
        'irregularities': irregularities_widget,
        'tabs': tab_widget
    }


def get_version_info():
    """
    Obtiene información de versión del módulo
    
    Returns
    -------
    dict
        Diccionario con información de versión
    """
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'widgets_count': len(__all__),
        'categories': list(WIDGET_CATEGORIES.keys())
    }


# Verificación de dependencias al importar
def _check_dependencies():
    """Verifica que las dependencias estén disponibles"""
    try:
        import PyQt5
        return True
    except ImportError:
        print("Advertencia: PyQt5 no está instalado. Los widgets no funcionarán correctamente.")
        return False


# Verificar dependencias al importar el módulo
_DEPENDENCIES_OK = _check_dependencies()


def are_dependencies_satisfied():
    """
    Verifica si todas las dependencias están satisfechas
    
    Returns
    -------
    bool
        True si todas las dependencias están disponibles
    """
    return _DEPENDENCIES_OK


# Configuración de logging para el módulo
import logging

# Crear logger específico para widgets
widgets_logger = logging.getLogger('seismic_common.widgets')
widgets_logger.setLevel(logging.INFO)

# Configurar formato si no hay handlers
if not widgets_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    widgets_logger.addHandler(handler)


def enable_debug_logging():
    """Habilita logging de debug para los widgets"""
    widgets_logger.setLevel(logging.DEBUG)
    widgets_logger.debug("Debug logging habilitado para seismic_common.widgets")


def disable_debug_logging():
    """Deshabilita logging de debug para los widgets"""
    widgets_logger.setLevel(logging.INFO)


# Mensaje de inicialización
widgets_logger.info(f"Módulo seismic_common.widgets v{__version__} inicializado correctamente")
widgets_logger.info(f"Widgets disponibles: {len(__all__)} en {len(WIDGET_CATEGORIES)} categorías")

if not _DEPENDENCIES_OK:
    widgets_logger.warning("Algunas dependencias no están disponibles. Funcionalidad limitada.")