"""
Sistema de procesadores para memorias de cálculo sísmico
Proporciona procesadores especializados para tablas, gráficos y variables
"""

# Información del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Sistema de procesadores especializados para memorias sísmicas"

# Verificación de importaciones
_IMPORTS_OK = True
_IMPORT_ERRORS = []

try:
    from .table_processor import (
        TableProcessor,
        create_modal_table_legacy,
        create_torsion_table_legacy,
        create_mass_table_legacy,
        create_stiffness_table_legacy
    )
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"table_processor: {e}")

try:
    from .graph_processor import (
        GraphProcessor,
        create_spectrum_figure_legacy,
        create_displacement_figure_legacy,
        create_drift_figure_legacy,
        create_shear_figures_legacy,
        actualize_images_legacy
    )
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"graph_processor: {e}")

try:
    from .variable_processor import (
        VariableProcessor,
        TemplateVariableExtractor,
        save_variables_legacy,
        create_bolivia_unit_dict,
        create_peru_unit_dict,
        create_variable_processor_for_country
    )
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"variable_processor: {e}")

# Configuración por defecto de procesadores
DEFAULT_PROCESSOR_CONFIG = {
    'tables': {
        'default_decimals': 3,
        'default_position': 'H',
        'textwidth_tables': ['stiffness', 'irregularity']
    },
    'graphs': {
        'default_dpi': 300,
        'default_format': 'pdf',
        'default_figsize': (10, 8),
        'bbox_inches': 'tight'
    },
    'variables': {
        'default_unit_system': 'SI',
        'default_decimals': 3,
        'validation_enabled': True
    }
}

# Exportaciones principales si las importaciones están OK
if _IMPORTS_OK:
    __all__ = [
        # Clases principales
        'TableProcessor',
        'GraphProcessor', 
        'VariableProcessor',
        'TemplateVariableExtractor',
        
        # Funciones de configuración
        'create_processor_suite',
        'get_default_config',
        'validate_processors_environment',
        
        # Funciones de compatibilidad (legacy)
        'create_modal_table_legacy',
        'create_torsion_table_legacy',
        'create_mass_table_legacy',
        'create_stiffness_table_legacy',
        'create_spectrum_figure_legacy',
        'create_displacement_figure_legacy', 
        'create_drift_figure_legacy',
        'create_shear_figures_legacy',
        'actualize_images_legacy',
        'save_variables_legacy',
        
        # Funciones de utilidad por país
        'create_bolivia_unit_dict',
        'create_peru_unit_dict',
        'create_variable_processor_for_country',
        'create_processors_for_country',
        
        # Información del módulo
        '__version__',
        '__author__', 
        '__description__'
    ]
else:
    # Si hay errores de importación, exportar solo funciones de diagnóstico
    __all__ = [
        'get_import_errors',
        'validate_processors_environment',
        '__version__',
        '__author__',
        '__description__'
    ]


class ProcessorSuite:
    """
    Suite completa de procesadores para memorias sísmicas
    Agrupa todos los procesadores para uso conjunto
    """
    
    def __init__(self, unit_system: str = 'SI', config: dict = None):
        """
        Inicializa la suite de procesadores
        
        Parameters
        ----------
        unit_system : str
            Sistema de unidades ('SI', 'FPS', 'MKS')
        config : dict, optional
            Configuración personalizada
        """
        if not _IMPORTS_OK:
            raise ImportError(f"No se pudieron importar todos los procesadores: {_IMPORT_ERRORS}")
        
        self.config = config or DEFAULT_PROCESSOR_CONFIG.copy()
        self.unit_system = unit_system
        
        # Inicializar procesadores
        self.table_processor = TableProcessor()
        self.graph_processor = GraphProcessor()
        self.variable_processor = VariableProcessor(unit_system)
        
        # Configurar procesadores según config
        self._apply_configuration()
    
    def _apply_configuration(self):
        """Aplica configuración a todos los procesadores"""
        # Configurar procesador de tablas
        if 'tables' in self.config:
            table_config = self.config['tables']
            if 'default_decimals' in table_config:
                self.table_processor.default_decimals = table_config['default_decimals']
            if 'default_position' in table_config:
                self.table_processor.default_position = table_config['default_position']
        
        # Configurar procesador de gráficos
        if 'graphs' in self.config:
            graph_config = self.config['graphs']
            if 'default_dpi' in graph_config:
                self.graph_processor.default_dpi = graph_config['default_dpi']
            if 'default_format' in graph_config:
                self.graph_processor.default_format = graph_config['default_format']
            if 'default_figsize' in graph_config:
                self.graph_processor.default_figsize = graph_config['default_figsize']
        
        # Configurar procesador de variables
        if 'variables' in self.config:
            var_config = self.config['variables']
            if 'default_decimals' in var_config:
                self.variable_processor.default_decimals = var_config['default_decimals']
    
    def process_complete_memory(self, seismic_data, template_content: str, 
                              output_dir: str) -> str:
        """
        Procesa una memoria completa usando todos los procesadores
        
        Parameters
        ----------
        seismic_data : Any
            Datos sísmicos del análisis
        template_content : str
            Contenido del template LaTeX
        output_dir : str
            Directorio de salida
            
        Returns
        -------
        str
            Contenido completamente procesado
        """
        # 1. Procesar variables
        if hasattr(seismic_data, 'variables'):
            template_content = self.variable_processor.process_variables(
                seismic_data.variables, template_content
            )
        
        # 2. Procesar tablas
        if hasattr(seismic_data, 'tables'):
            tables = seismic_data.tables
            
            if hasattr(tables, 'modal'):
                template_content = self.table_processor.create_modal_table(
                    tables.modal, template_content
                )
            
            if hasattr(tables, 'torsion_x') and hasattr(tables, 'torsion_y'):
                template_content = self.table_processor.create_torsion_table(
                    tables.torsion_x, tables.torsion_y, template_content
                )
            
            if hasattr(tables, 'mass_table'):
                template_content = self.table_processor.create_mass_table(
                    tables.mass_table, template_content
                )
            
            if hasattr(tables, 'stiffness_x') and hasattr(tables, 'stiffness_y'):
                template_content = self.table_processor.create_stiffness_table(
                    tables.stiffness_x, tables.stiffness_y, template_content
                )
        
        # 3. Generar gráficos
        self.graph_processor.create_response_spectrum(seismic_data, output_dir)
        self.graph_processor.create_displacement_figure(seismic_data, output_dir)
        self.graph_processor.create_drift_figure(seismic_data, output_dir)
        self.graph_processor.create_shear_figures(seismic_data, output_dir)
        
        return template_content
    
    def validate_data(self, seismic_data, template_content: str) -> dict:
        """
        Valida datos antes del procesamiento
        
        Parameters
        ----------
        seismic_data : Any
            Datos sísmicos
        template_content : str
            Contenido del template
            
        Returns
        -------
        dict
            Reporte de validación
        """
        validation_report = {
            'variables': {},
            'tables': {'available': [], 'missing': []},
            'graphs': {'available': [], 'missing': []},
            'overall_status': 'unknown'
        }
        
        # Validar variables
        if hasattr(seismic_data, 'variables'):
            validation_report['variables'] = self.variable_processor.validate_variables(
                seismic_data.variables, template_content
            )
        
        # Validar tablas disponibles
        if hasattr(seismic_data, 'tables'):
            tables = seismic_data.tables
            available_tables = []
            
            if hasattr(tables, 'modal'):
                available_tables.append('modal')
            if hasattr(tables, 'torsion_table'):
                available_tables.append('torsion')
            if hasattr(tables, 'mass_table'):
                available_tables.append('mass')
            if hasattr(tables, 'stiffness_x'):
                available_tables.append('stiffness')
            
            validation_report['tables']['available'] = available_tables
        
        # Validar gráficos disponibles
        graph_indicators = ['fig_spectrum', 'fig_displacements', 'fig_drifts', 
                          'dynamic_shear_fig', 'static_shear_fig']
        available_graphs = []
        
        for indicator in graph_indicators:
            if hasattr(seismic_data, indicator):
                available_graphs.append(indicator)
        
        validation_report['graphs']['available'] = available_graphs
        
        # Determinar estado general
        has_vars = bool(validation_report['variables'])
        has_tables = bool(validation_report['tables']['available'])
        has_graphs = bool(validation_report['graphs']['available'])
        
        if has_vars and has_tables and has_graphs:
            validation_report['overall_status'] = 'complete'
        elif has_vars or has_tables or has_graphs:
            validation_report['overall_status'] = 'partial'
        else:
            validation_report['overall_status'] = 'insufficient'
        
        return validation_report


def create_processor_suite(unit_system: str = 'SI', config: dict = None) -> ProcessorSuite:
    """
    Crea una suite completa de procesadores
    
    Parameters
    ----------
    unit_system : str
        Sistema de unidades
    config : dict, optional
        Configuración personalizada
        
    Returns
    -------
    ProcessorSuite
        Suite de procesadores configurada
    """
    if not _IMPORTS_OK:
        raise ImportError(f"No se pudieron importar todos los procesadores: {_IMPORT_ERRORS}")
    
    return ProcessorSuite(unit_system, config)


def create_processors_for_country(country_code: str) -> ProcessorSuite:
    """
    Crea procesadores configurados para un país específico
    
    Parameters
    ----------
    country_code : str
        Código del país ('PE', 'BO', 'US', etc.)
        
    Returns
    -------
    ProcessorSuite
        Suite configurada para el país
    """
    if not _IMPORTS_OK:
        raise ImportError(f"No se pudieron importar todos los procesadores: {_IMPORT_ERRORS}")
    
    # Configuraciones específicas por país
    country_configs = {
        'PE': {  # Perú
            'unit_system': 'SI',
            'config': {
                'variables': {'default_decimals': 2},
                'tables': {'textwidth_tables': ['stiffness']},
                'graphs': {'default_format': 'pdf'}
            }
        },
        'BO': {  # Bolivia
            'unit_system': 'SI', 
            'config': {
                'variables': {'default_decimals': 3},
                'tables': {'textwidth_tables': ['stiffness', 'torsion']},
                'graphs': {'default_format': 'pdf'}
            }
        },
        'US': {  # Estados Unidos
            'unit_system': 'FPS',
            'config': {
                'variables': {'default_decimals': 2},
                'graphs': {'default_format': 'png'}
            }
        }
    }
    
    country_setup = country_configs.get(country_code, {
        'unit_system': 'SI',
        'config': DEFAULT_PROCESSOR_CONFIG.copy()
    })
    
    return ProcessorSuite(
        unit_system=country_setup['unit_system'],
        config=country_setup.get('config', DEFAULT_PROCESSOR_CONFIG.copy())
    )


def get_default_config() -> dict:
    """
    Obtiene la configuración por defecto de procesadores
    
    Returns
    -------
    dict
        Configuración por defecto
    """
    return DEFAULT_PROCESSOR_CONFIG.copy()


def validate_processors_environment() -> dict:
    """
    Valida el entorno de procesadores
    
    Returns
    -------
    dict
        Información sobre el estado del entorno
    """
    import sys
    import platform
    from pathlib import Path
    
    validation = {
        'imports_ok': _IMPORTS_OK,
        'import_errors': _IMPORT_ERRORS.copy() if _IMPORT_ERRORS else [],
        'python_version': sys.version,
        'platform': platform.system(),
        'processors_version': __version__,
        'modules_available': {},
        'paths': {}
    }
    
    # Verificar dependencias externas
    try:
        import matplotlib
        validation['modules_available']['matplotlib'] = matplotlib.__version__
    except ImportError:
        validation['modules_available']['matplotlib'] = 'No disponible'
    
    try:
        import pandas
        validation['modules_available']['pandas'] = pandas.__version__
    except ImportError:
        validation['modules_available']['pandas'] = 'No disponible'
    
    try:
        import numpy
        validation['modules_available']['numpy'] = numpy.__version__
    except ImportError:
        validation['modules_available']['numpy'] = 'No disponible'
    
    # Verificar rutas del sistema
    processors_path = Path(__file__).parent
    validation['paths']['processors_dir'] = str(processors_path)
    validation['paths']['table_processor'] = str(processors_path / 'table_processor.py')
    validation['paths']['graph_processor'] = str(processors_path / 'graph_processor.py')
    validation['paths']['variable_processor'] = str(processors_path / 'variable_processor.py')
    
    # Verificar existencia de archivos
    validation['paths']['files_exist'] = {
        'table_processor': (processors_path / 'table_processor.py').exists(),
        'graph_processor': (processors_path / 'graph_processor.py').exists(),
        'variable_processor': (processors_path / 'variable_processor.py').exists()
    }
    
    return validation


def get_import_errors() -> list:
    """
    Obtiene la lista de errores de importación
    
    Returns
    -------
    list
        Lista de errores de importación
    """
    return _IMPORT_ERRORS.copy()


def get_processor_capabilities() -> dict:
    """
    Obtiene información sobre las capacidades de los procesadores
    
    Returns
    -------
    dict
        Diccionario con capacidades disponibles
    """
    capabilities = {
        'tables': [],
        'graphs': [],
        'variables': [],
        'legacy_functions': []
    }
    
    if _IMPORTS_OK:
        # Capacidades de tablas
        capabilities['tables'] = [
            'modal_table',
            'torsion_table', 
            'mass_table',
            'stiffness_table',
            'shear_table',
            'drift_table',
            'displacement_table',
            'static_analysis_table'
        ]
        
        # Capacidades de gráficos
        capabilities['graphs'] = [
            'response_spectrum',
            'displacement_figure',
            'drift_figure',
            'shear_figures',
            'static_image_copy'
        ]
        
        # Capacidades de variables
        capabilities['variables'] = [
            'latex_variable_processing',
            'unit_conversion',
            'validation',
            'csv_import_export',
            'variable_registration',
            'conversion_history'
        ]
        
        # Funciones legacy disponibles
        capabilities['legacy_functions'] = [
            'save_variables_legacy',
            'create_modal_table_legacy',
            'actualize_images_legacy'
        ]
    
    return capabilities


# Mostrar advertencias de importación al importar el módulo
if not _IMPORTS_OK:
    import warnings
    warnings.warn(
        f"Algunos procesadores no se pudieron importar: {_IMPORT_ERRORS}. "
        f"Funcionalidad limitada disponible.",
        ImportWarning,
        stacklevel=2
    )


# Configuración de logging para procesadores
def setup_processors_logging(level: str = 'INFO'):
    """
    Configura logging para el sistema de procesadores
    
    Parameters
    ----------
    level : str
        Nivel de logging ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    import logging
    
    # Configurar logger específico para procesadores
    logger = logging.getLogger('seismic_common.memory.processors')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Crear handler si no existe
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.info(f"Sistema de procesadores v{__version__} inicializado")
    
    if not _IMPORTS_OK:
        logger.warning(f"Errores de importación detectados: {_IMPORT_ERRORS}")


if __name__ == '__main__':
    # Modo de diagnóstico cuando se ejecuta directamente
    print(f"=== Sistema de Procesadores v{__version__} ===\n")
    
    # Mostrar información del sistema
    info = validate_processors_environment()
    
    print("Estado del entorno:")
    print(f"  Importaciones OK: {info['imports_ok']}")
    if not info['imports_ok']:
        print(f"  Errores: {info['import_errors']}")
    
    print(f"\nSistema: {info['platform']}")
    print(f"Python: {info['python_version'].split()[0]}")
    
    print("\nMódulos disponibles:")
    for module, version in info['modules_available'].items():
        print(f"  {module}: {version}")
    
    print("\nArchivos de procesadores:")
    for name, exists in info['paths']['files_exist'].items():
        print(f"  {name}: {'✓' if exists else '✗'}")
    
    # Mostrar capacidades
    capabilities = get_processor_capabilities()
    print(f"\nCapacidades disponibles:")
    for category, items in capabilities.items():
        if items:
            print(f"  {category.title()}: {len(items)} funciones")
        else:
            print(f"  {category.title()}: No disponible")
    
    print(f"\nDirectorio: {info['paths']['processors_dir']}")