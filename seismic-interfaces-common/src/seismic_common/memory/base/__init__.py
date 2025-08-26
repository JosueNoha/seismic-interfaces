"""
Sistema base de generación de memorias de cálculo sísmico
Proporciona clases y funciones base para la generación de memorias LaTeX
"""

# Información del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Sistema base de generación de memorias sísmicas"

# Verificación de importaciones
_IMPORTS_OK = True
_IMPORT_ERRORS = []

try:
    from .memory_generator import BaseMemoryGenerator
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"memory_generator: {e}")

try:
    from .template_manager import (
        TemplateManager,
        load_template_content,
        process_zone_table,
        process_soil_table
    )
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERRORS.append(f"template_manager: {e}")

# Configuración por defecto
DEFAULT_CONFIG = {
    'template': {
        'base_name': 'base_template.ltx',
        'common_sections': 'common_sections.ltx',
        'encoding': 'utf-8'
    },
    'output': {
        'latex_filename': 'memoria.tex',
        'pdf_filename': 'memoria.pdf',
        'images_dir': 'images',
        'clean_aux': True
    },
    'compilation': {
        'runs': 1,
        'interaction_mode': 'nonstopmode'
    }
}

# Exportaciones principales si las importaciones están OK
if _IMPORTS_OK:
    # Clases principales
    __all__ = [
        # Clases base
        'BaseMemoryGenerator',
        'TemplateManager',
        
        # Funciones de utilidad
        'load_template_content',
        'process_zone_table',
        'process_soil_table',
        
        # Funciones de configuración
        'get_default_config',
        'create_memory_generator',
        'validate_base_environment',
        
        # Información del módulo
        '__version__',
        '__author__',
        '__description__'
    ]
else:
    # Si hay errores de importación, exportar solo funciones de diagnóstico
    __all__ = [
        'get_import_errors',
        'validate_base_environment',
        '__version__',
        '__author__',
        '__description__'
    ]


def get_default_config() -> dict:
    """
    Obtiene la configuración por defecto
    
    Returns
    -------
    dict
        Configuración por defecto del sistema
    """
    return DEFAULT_CONFIG.copy()


def create_memory_generator(country_template: str = None, 
                          config: dict = None) -> 'BaseMemoryGenerator':
    """
    Crea una instancia del generador de memorias
    
    Parameters
    ----------
    country_template : str, optional
        Ruta a la plantilla específica del país
    config : dict, optional
        Configuración personalizada
        
    Returns
    -------
    BaseMemoryGenerator
        Instancia del generador de memorias
        
    Raises
    ------
    ImportError
        Si no se pudieron importar los módulos necesarios
    NotImplementedError
        Si se intenta usar BaseMemoryGenerator directamente
    """
    if not _IMPORTS_OK:
        raise ImportError(f"No se pudieron importar los módulos base: {_IMPORT_ERRORS}")
    
    # Nota: BaseMemoryGenerator es abstracta, por lo que esta función
    # es más conceptual. En la práctica, se usarán las implementaciones específicas
    raise NotImplementedError(
        "BaseMemoryGenerator es una clase abstracta. "
        "Use las implementaciones específicas por país (PeruMemoryGenerator, BoliviaMemoryGenerator, etc.)"
    )


def create_template_manager(base_templates_path: str = None) -> 'TemplateManager':
    """
    Crea una instancia del gestor de plantillas
    
    Parameters
    ----------
    base_templates_path : str, optional
        Ruta personalizada a las plantillas base
        
    Returns
    -------
    TemplateManager
        Instancia del gestor de plantillas
        
    Raises
    ------
    ImportError
        Si no se pudieron importar los módulos necesarios
    """
    if not _IMPORTS_OK:
        raise ImportError(f"No se pudieron importar los módulos base: {_IMPORT_ERRORS}")
    
    manager = TemplateManager()
    
    # Configurar ruta personalizada si se proporciona
    if base_templates_path:
        from pathlib import Path
        manager._base_templates_path = Path(base_templates_path)
    
    return manager


def validate_base_environment() -> dict:
    """
    Valida el entorno base del sistema de memorias
    
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
        'base_version': __version__,
        'modules_available': {},
        'paths': {}
    }
    
    # Verificar módulos principales
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
    
    # Verificar rutas importantes
    base_path = Path(__file__).parent
    validation['paths']['base_dir'] = str(base_path)
    validation['paths']['templates_dir'] = str(base_path.parent / "templates")
    validation['paths']['processors_dir'] = str(base_path.parent / "processors")
    
    # Verificar existencia de directorios clave
    templates_path = base_path.parent / "templates"
    processors_path = base_path.parent / "processors"
    
    validation['paths']['templates_exists'] = templates_path.exists()
    validation['paths']['processors_exists'] = processors_path.exists()
    
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


def check_latex_installation() -> dict:
    """
    Verifica si LaTeX está instalado y disponible
    
    Returns
    -------
    dict
        Información sobre la instalación de LaTeX
    """
    import subprocess
    import shutil
    
    latex_info = {
        'pdflatex_available': False,
        'pdflatex_path': None,
        'version': None,
        'error': None
    }
    
    # Buscar pdflatex
    pdflatex_path = shutil.which('pdflatex')
    if pdflatex_path:
        latex_info['pdflatex_available'] = True
        latex_info['pdflatex_path'] = pdflatex_path
        
        # Obtener versión
        try:
            result = subprocess.run(
                ['pdflatex', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Extraer versión de la primera línea
                version_line = result.stdout.split('\n')[0]
                latex_info['version'] = version_line
            else:
                latex_info['error'] = "Error ejecutando pdflatex --version"
                
        except subprocess.TimeoutExpired:
            latex_info['error'] = "Timeout ejecutando pdflatex --version"
        except Exception as e:
            latex_info['error'] = f"Error verificando versión: {e}"
    else:
        latex_info['error'] = "pdflatex no encontrado en PATH"
    
    return latex_info


def get_system_info() -> dict:
    """
    Obtiene información completa del sistema
    
    Returns
    -------
    dict
        Información completa del sistema y entorno
    """
    return {
        'base_environment': validate_base_environment(),
        'latex_installation': check_latex_installation(),
        'config': get_default_config()
    }


# Funciones de compatibilidad con versiones anteriores
def load_base_template(template_name: str = None) -> str:
    """
    Función de compatibilidad para cargar plantillas base
    
    Parameters
    ----------
    template_name : str, optional
        Nombre de la plantilla
        
    Returns
    -------
    str
        Contenido de la plantilla
        
    Raises
    ------
    ImportError
        Si no se pudieron importar los módulos necesarios
    """
    if not _IMPORTS_OK:
        raise ImportError(f"No se pudieron importar los módulos base: {_IMPORT_ERRORS}")
    
    manager = TemplateManager()
    template_name = template_name or DEFAULT_CONFIG['template']['base_name']
    return manager.get_base_template(template_name)


def create_base_memory_processor():
    """
    Crea un procesador básico de memorias
    Función de compatibilidad para transición
    
    Returns
    -------
    dict
        Diccionario con funciones de procesamiento básico
    """
    if not _IMPORTS_OK:
        raise ImportError(f"No se pudieron importar los módulos base: {_IMPORT_ERRORS}")
    
    manager = TemplateManager()
    
    return {
        'load_template': manager.load_template,
        'process_zone_table': manager.process_zone_factor_table,
        'process_soil_table': manager.process_soil_factor_table,
        'process_usage_table': manager.process_usage_factor_table,
        'validate_template': manager.validate_template
    }


# Mostrar advertencias de importación al importar el módulo
if not _IMPORTS_OK:
    import warnings
    warnings.warn(
        f"Algunos módulos del sistema base no se pudieron importar: {_IMPORT_ERRORS}. "
        f"Funcionalidad limitada disponible.",
        ImportWarning,
        stacklevel=2
    )


# Configuración de logging básico
def setup_logging(level: str = 'INFO'):
    """
    Configura logging básico para el sistema de memorias
    
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
    
    # Configurar logger específico
    logger = logging.getLogger('seismic_common.memory.base')
    logger.info(f"Sistema base de memorias sísmicas v{__version__} inicializado")
    
    if not _IMPORTS_OK:
        logger.warning(f"Errores de importación detectados: {_IMPORT_ERRORS}")


if __name__ == '__main__':
    # Modo de diagnóstico cuando se ejecuta directamente
    print(f"=== Sistema Base de Memorias Sísmicas v{__version__} ===\n")
    
    # Mostrar información del sistema
    info = get_system_info()
    
    print("Estado del entorno:")
    print(f"  Importaciones OK: {info['base_environment']['imports_ok']}")
    if not info['base_environment']['imports_ok']:
        print(f"  Errores: {info['base_environment']['import_errors']}")
    
    print(f"\nSistema: {info['base_environment']['platform']}")
    print(f"Python: {info['base_environment']['python_version'].split()[0]}")
    
    print("\nMódulos disponibles:")
    for module, version in info['base_environment']['modules_available'].items():
        print(f"  {module}: {version}")
    
    print("\nLatex:")
    latex = info['latex_installation']
    if latex['pdflatex_available']:
        print(f"  pdflatex: Disponible ({latex['pdflatex_path']})")
        if latex['version']:
            print(f"  Versión: {latex['version']}")
    else:
        print(f"  pdflatex: No disponible - {latex['error']}")
    
    print(f"\nDirectorios:")
    paths = info['base_environment']['paths']
    print(f"  Base: {paths['base_dir']}")
    print(f"  Templates: {paths['templates_dir']} {'✓' if paths['templates_exists'] else '✗'}")
    print(f"  Processors: {paths['processors_dir']} {'✓' if paths['processors_exists'] else '✗'}")