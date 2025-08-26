"""
Parámetros sísmicos comunes a todas las normativas
=================================================

Este módulo define parámetros, constantes y configuraciones que son comunes
entre diferentes normativas sísmicas, proporcionando una base unificada para
el desarrollo de interfaces específicas por país.

Ejemplo de uso:
    ```python
    from seismic_common.standards.common_parameters import (
        SEISMIC_CONSTANTS,
        DRIFT_LIMITS,
        REDUCTION_FACTORS,
        get_default_parameters
    )
    
    # Usar constantes comunes
    gravity = SEISMIC_CONSTANTS['GRAVITY']
    drift_limit = DRIFT_LIMITS['concrete']
    r_factor = REDUCTION_FACTORS['concrete_moment_frame']
    
    # Configuración por defecto
    defaults = get_default_parameters('PE')
    ```
"""

# Metadatos del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Parámetros sísmicos comunes a todas las normativas"
__license__ = "MIT"
__status__ = "Production"

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Configurar logging
logger = logging.getLogger(__name__)


# Enumeraciones básicas

class AnalysisType(Enum):
    """Tipos de análisis sísmico"""
    STATIC = "static"
    MODAL = "modal"
    TIME_HISTORY = "time_history"


class StructuralMaterial(Enum):
    """Materiales estructurales principales"""
    CONCRETE = "concrete"
    STEEL = "steel"
    MASONRY = "masonry"
    WOOD = "wood"


class BuildingCategory(Enum):
    """Categorías de edificación"""
    ESSENTIAL = "essential"    # A - Hospitales, bomberos
    IMPORTANT = "important"    # B - Escuelas, centros comerciales
    STANDARD = "standard"      # C - Viviendas, oficinas
    MINOR = "minor"           # D - Construcciones menores


# Constantes sísmicas fundamentales

SEISMIC_CONSTANTS = {
    # Constantes físicas
    'GRAVITY': 9.80665,                     # m/s²
    'DEFAULT_DAMPING': 0.05,                # 5%
    
    # Participación de masa
    'MIN_MASS_PARTICIPATION': 0.90,        # 90%
    
    # Factores de escala dinámico
    'MIN_DYNAMIC_SCALE_REGULAR': 0.80,     # 80%
    'MIN_DYNAMIC_SCALE_IRREGULAR': 0.90,   # 90%
    
    # Límites de periodo
    'MIN_PERIOD': 0.01,                    # s
    'MAX_PERIOD_FACTOR': 1.4,              # Factor sobre T aproximado
    
    # Cortante basal mínimo
    'MIN_BASE_SHEAR': 0.044,               # 4.4%
}


# Límites de deriva por material

DRIFT_LIMITS = {
    # Por material básico
    StructuralMaterial.CONCRETE: 0.007,    # 0.7%
    StructuralMaterial.STEEL: 0.010,       # 1.0%
    StructuralMaterial.MASONRY: 0.005,     # 0.5%
    StructuralMaterial.WOOD: 0.010,        # 1.0%
    
    # Por sistema específico
    'concrete_moment_frame': 0.007,
    'concrete_dual_system': 0.007,
    'concrete_shear_walls': 0.005,
    'steel_moment_frame_special': 0.010,
    'steel_moment_frame_ordinary': 0.015,
    'steel_braced_frame': 0.015,
    'masonry_reinforced': 0.005,
    'masonry_confined': 0.005,
    'wood_frame': 0.010,
}


# Factores de reducción sísmica

REDUCTION_FACTORS = {
    # Básicos por material
    StructuralMaterial.CONCRETE: {'R': 8.0, 'Omega': 3.0, 'Cd': 5.5},
    StructuralMaterial.STEEL: {'R': 8.0, 'Omega': 3.0, 'Cd': 5.5},
    StructuralMaterial.MASONRY: {'R': 3.0, 'Omega': 2.5, 'Cd': 2.5},
    StructuralMaterial.WOOD: {'R': 7.0, 'Omega': 3.0, 'Cd': 4.0},
    
    # Específicos por sistema
    'concrete_moment_frame_special': {'R': 8.0, 'Omega': 3.0, 'Cd': 5.5},
    'concrete_moment_frame_ordinary': {'R': 3.0, 'Omega': 3.0, 'Cd': 2.5},
    'concrete_dual_system': {'R': 7.0, 'Omega': 2.5, 'Cd': 5.5},
    'concrete_shear_walls': {'R': 6.0, 'Omega': 2.5, 'Cd': 5.0},
    'steel_moment_frame_special': {'R': 8.0, 'Omega': 3.0, 'Cd': 5.5},
    'steel_moment_frame_ordinary': {'R': 3.5, 'Omega': 3.0, 'Cd': 3.0},
    'steel_braced_frame_special': {'R': 6.0, 'Omega': 2.0, 'Cd': 5.0},
    'masonry_reinforced': {'R': 3.5, 'Omega': 2.5, 'Cd': 3.5},
    'wood_light_frame': {'R': 6.5, 'Omega': 3.0, 'Cd': 4.0},
}


# Factores de importancia

IMPORTANCE_FACTORS = {
    BuildingCategory.ESSENTIAL: 1.5,
    BuildingCategory.IMPORTANT: 1.25,
    BuildingCategory.STANDARD: 1.0,
    BuildingCategory.MINOR: 0.8,
}


# Límites de irregularidades

IRREGULARITY_LIMITS = {
    # Irregularidades en planta
    'torsional_displacement_ratio': 1.2,        # δmax > 1.2 * δavg
    'reentrant_corner_ratio': 0.20,             # > 20% dimensión
    'diaphragm_opening_ratio': 0.50,            # > 50% área
    
    # Irregularidades verticales
    'soft_story_stiffness_ratio': 0.70,         # < 70% piso superior
    'weak_story_strength_ratio': 0.80,          # < 80% piso superior  
    'mass_irregularity_ratio': 1.50,            # > 150% piso adyacente
}


# Combinaciones de carga básicas

LOAD_COMBINATIONS = {
    'gravity': {'D': 1.4, 'L': 1.7},
    'seismic_1': {'D': 1.2, 'L': 1.0, 'E': 1.0},
    'seismic_2': {'D': 0.9, 'E': 1.0},
    'service': {'D': 1.0, 'L': 1.0},
    'service_seismic': {'D': 1.0, 'L': 0.5, 'E': 0.7},
}


# Configuraciones por país/normativa

COUNTRY_DEFAULTS = {
    'PE': {  # Perú - E.030
        'normative': 'E.030',
        'zones': [1, 2, 3, 4],
        'zone_factors': {1: 0.10, 2: 0.25, 3: 0.35, 4: 0.45},
        'default_zone': 4,
        'soil_types': ['S0', 'S1', 'S2', 'S3'],
        'default_soil': 'S1',
        'drift_concrete': 0.007,
        'drift_steel': 0.010,
        'drift_masonry': 0.005,
    },
    
    'BO': {  # Bolivia - NBC
        'normative': 'NBC',
        'zones': [1, 2, 3],
        'zone_factors': {1: 0.15, 2: 0.20, 3: 0.30},
        'default_zone': 3,
        'soil_types': ['A', 'B', 'C', 'D'],
        'default_soil': 'B',
        'drift_concrete': 0.007,
        'drift_steel': 0.010,
        'drift_masonry': 0.005,
    },
    
    'CL': {  # Chile - NCh433
        'normative': 'NCh433',
        'zones': ['A0', 'A1', 'A2', 'A3'],
        'zone_factors': {'A0': 0.20, 'A1': 0.25, 'A2': 0.30, 'A3': 0.40},
        'default_zone': 'A2',
        'soil_types': ['A', 'B', 'C', 'D', 'E'],
        'default_soil': 'B',
        'drift_concrete': 0.002,  # Más restrictivo
        'drift_steel': 0.005,
        'drift_masonry': 0.002,
    },
    
    'CO': {  # Colombia - NSR-10
        'normative': 'NSR-10',
        'zones': ['Low', 'Intermediate', 'High'],
        'zone_factors': {'Low': 0.10, 'Intermediate': 0.20, 'High': 0.35},
        'default_zone': 'High',
        'soil_types': ['A', 'B', 'C', 'D', 'E'],
        'default_soil': 'C',
        'drift_concrete': 0.010,
        'drift_steel': 0.010,
        'drift_masonry': 0.005,
    },
    
    'EC': {  # Ecuador - NEC
        'normative': 'NEC',
        'zones': ['I', 'II', 'III', 'IV', 'V', 'VI'],
        'zone_factors': {'I': 0.15, 'II': 0.25, 'III': 0.30, 'IV': 0.35, 'V': 0.40, 'VI': 0.50},
        'default_zone': 'V',
        'soil_types': ['A', 'B', 'C', 'D', 'E', 'F'],
        'default_soil': 'C',
        'drift_concrete': 0.020,
        'drift_steel': 0.020,
        'drift_masonry': 0.010,
    }
}


# Funciones de utilidad principales

def get_drift_limit(system: str) -> float:
    """
    Obtiene límite de deriva para un sistema específico
    
    Parameters
    ----------
    system : str
        Sistema estructural o material
        
    Returns
    -------
    float
        Límite de deriva
    """
    # Buscar en sistemas específicos
    if system in DRIFT_LIMITS:
        return DRIFT_LIMITS[system]
    
    # Buscar por material
    for material in StructuralMaterial:
        if material.value in system.lower():
            return DRIFT_LIMITS.get(material, 0.007)
    
    # Default conservador
    return 0.007


def get_reduction_factor(system: str, factor_type: str = 'R') -> float:
    """
    Obtiene factor de reducción para un sistema
    
    Parameters
    ----------
    system : str
        Sistema estructural
    factor_type : str
        Tipo de factor ('R', 'Omega', 'Cd')
        
    Returns
    -------
    float
        Factor solicitado
    """
    # Buscar en sistemas específicos
    if system in REDUCTION_FACTORS:
        factors = REDUCTION_FACTORS[system]
        if isinstance(factors, dict):
            return factors.get(factor_type, factors.get('R', 8.0))
    
    # Buscar por material
    for material in StructuralMaterial:
        if material.value in system.lower():
            factors = REDUCTION_FACTORS.get(material, {'R': 8.0, 'Omega': 3.0, 'Cd': 5.5})
            return factors.get(factor_type, 8.0)
    
    # Defaults
    defaults = {'R': 8.0, 'Omega': 3.0, 'Cd': 5.5}
    return defaults.get(factor_type, 8.0)


def get_importance_factor(category: str) -> float:
    """
    Obtiene factor de importancia
    
    Parameters
    ----------
    category : str
        Categoría de edificación
        
    Returns
    -------
    float
        Factor de importancia
    """
    # Buscar por enum
    for cat in BuildingCategory:
        if cat.value in category.lower() or cat.name.lower() in category.lower():
            return IMPORTANCE_FACTORS.get(cat, 1.0)
    
    # Mapeo directo por strings comunes
    category_map = {
        'a': BuildingCategory.ESSENTIAL,
        'b': BuildingCategory.IMPORTANT, 
        'c': BuildingCategory.STANDARD,
        'd': BuildingCategory.MINOR,
        'hospital': BuildingCategory.ESSENTIAL,
        'school': BuildingCategory.IMPORTANT,
        'residential': BuildingCategory.STANDARD,
    }
    
    cat = category_map.get(category.lower())
    if cat:
        return IMPORTANCE_FACTORS.get(cat, 1.0)
    
    return 1.0  # Default


def get_default_parameters(country_code: str) -> Dict[str, Any]:
    """
    Obtiene parámetros por defecto para un país
    
    Parameters
    ----------
    country_code : str
        Código del país ('PE', 'BO', 'CL', etc.)
        
    Returns
    -------
    Dict[str, Any]
        Parámetros por defecto del país
    """
    return COUNTRY_DEFAULTS.get(country_code.upper(), COUNTRY_DEFAULTS['PE'])


def validate_parameters(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valida parámetros sísmicos básicos
    
    Parameters
    ----------
    params : Dict[str, Any]
        Parámetros a validar
        
    Returns
    -------
    Tuple[bool, List[str]]
        (es_válido, lista_errores)
    """
    errors = []
    
    # Validar factores de reducción
    for r_param in ['Rx', 'Ry', 'R_x', 'R_y', 'reduction_factor_x', 'reduction_factor_y']:
        if r_param in params:
            r_value = params[r_param]
            if not isinstance(r_value, (int, float)) or r_value <= 0:
                errors.append(f"Factor de reducción {r_param} debe ser positivo")
            elif r_value > 12:
                errors.append(f"Factor de reducción {r_param} parece excesivo (>{r_value})")
    
    # Validar irregularidades
    for irr_param in ['Ia', 'Ip', 'irregularity_height', 'irregularity_plan']:
        if irr_param in params:
            irr_value = params[irr_param]
            if not isinstance(irr_value, (int, float)) or not (0.5 <= irr_value <= 1.0):
                errors.append(f"Factor de irregularidad {irr_param} debe estar entre 0.5 y 1.0")
    
    # Validar factor de importancia
    for i_param in ['I', 'importance_factor']:
        if i_param in params:
            i_value = params[i_param]
            if not isinstance(i_value, (int, float)) or i_value <= 0:
                errors.append(f"Factor de importancia {i_param} debe ser positivo")
            elif i_value > 2.0:
                errors.append(f"Factor de importancia {i_param} parece excesivo (>{i_value})")
    
    # Validar límites de deriva
    for drift_param in ['drift_limit', 'max_drift_x', 'max_drift_y']:
        if drift_param in params:
            drift_value = params[drift_param]
            if not isinstance(drift_value, (int, float)) or drift_value <= 0:
                errors.append(f"Límite de deriva {drift_param} debe ser positivo")
            elif drift_value > 0.05:
                errors.append(f"Límite de deriva {drift_param} parece excesivo (>{drift_value})")
    
    return len(errors) == 0, errors


def create_configuration(building_height: float, structural_system: str, 
                        importance: str, country: str = 'PE') -> Dict[str, Any]:
    """
    Crea configuración completa basada en parámetros del edificio
    
    Parameters
    ----------
    building_height : float
        Altura del edificio en metros
    structural_system : str
        Sistema estructural
    importance : str
        Categoría de importancia
    country : str
        Código del país
        
    Returns
    -------
    Dict[str, Any]
        Configuración completa
    """
    # Obtener defaults del país
    country_defaults = get_default_parameters(country)
    
    # Tipo de análisis según altura
    if building_height <= 30:
        analysis_type = AnalysisType.MODAL
        can_use_static = True
    elif building_height <= 60:
        analysis_type = AnalysisType.MODAL
        can_use_static = False
    else:
        analysis_type = AnalysisType.TIME_HISTORY
        can_use_static = False
    
    # Factores según sistema estructural
    r_factor = get_reduction_factor(structural_system, 'R')
    omega_factor = get_reduction_factor(structural_system, 'Omega') 
    cd_factor = get_reduction_factor(structural_system, 'Cd')
    
    # Factor de importancia
    importance_factor = get_importance_factor(importance)
    
    # Límite de deriva
    drift_limit = get_drift_limit(structural_system)
    
    # Número de modos según altura
    if building_height <= 20:
        min_modes = 9
    elif building_height <= 50:
        min_modes = 12
    else:
        min_modes = 15
    
    # Configuración completa
    config = {
        'country': country,
        'normative': country_defaults['normative'],
        'building_height': building_height,
        'structural_system': structural_system,
        'importance_category': importance,
        'analysis_type': analysis_type.value,
        'can_use_static': can_use_static,
        'reduction_factor': r_factor,
        'omega_factor': omega_factor,
        'cd_factor': cd_factor,
        'importance_factor': importance_factor,
        'drift_limit': drift_limit,
        'min_modes': min_modes,
        'damping': SEISMIC_CONSTANTS['DEFAULT_DAMPING'],
        'min_mass_participation': SEISMIC_CONSTANTS['MIN_MASS_PARTICIPATION'],
    }
    
    # Agregar defaults del país
    config.update(country_defaults)
    
    return config


# Función de validación del módulo

def validate_module() -> bool:
    """Valida el funcionamiento básico del módulo"""
    try:
        # Probar constantes
        assert SEISMIC_CONSTANTS['GRAVITY'] == 9.80665
        assert SEISMIC_CONSTANTS['DEFAULT_DAMPING'] == 0.05
        
        # Probar límites de deriva
        assert get_drift_limit('concrete') == 0.007
        assert get_drift_limit('steel_moment_frame_special') == 0.010
        
        # Probar factores de reducción
        assert get_reduction_factor('concrete_moment_frame_special', 'R') == 8.0
        assert get_reduction_factor('masonry_reinforced', 'R') == 3.5
        
        # Probar factores de importancia
        assert get_importance_factor('essential') == 1.5
        assert get_importance_factor('hospital') == 1.5
        assert get_importance_factor('residential') == 1.0
        
        # Probar defaults por país
        pe_defaults = get_default_parameters('PE')
        assert pe_defaults['normative'] == 'E.030'
        assert 4 in pe_defaults['zones']
        
        # Probar validación
        valid_params = {'Rx': 8.0, 'Ia': 1.0, 'I': 1.25, 'drift_limit': 0.007}
        is_valid, errors = validate_parameters(valid_params)
        assert is_valid and len(errors) == 0
        
        # Probar configuración completa
        config = create_configuration(25.0, 'concrete_moment_frame', 'standard', 'PE')
        assert config['country'] == 'PE'
        assert config['normative'] == 'E.030'
        assert config['reduction_factor'] == 8.0
        
        return True
        
    except Exception as e:
        logger.error(f"Error en validación: {e}")
        return False


# Punto de entrada para pruebas
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    if validate_module():
        print("✓ Módulo common_parameters validado correctamente")
        
        # Ejemplos básicos
        print(f"\nConstantes principales:")
        print(f"- Gravedad: {SEISMIC_CONSTANTS['GRAVITY']} m/s²")
        print(f"- Amortiguamiento: {SEISMIC_CONSTANTS['DEFAULT_DAMPING']:.1%}")
        
        print(f"\nLímites de deriva:")
        print(f"- Concreto: {get_drift_limit('concrete'):.3%}")
        print(f"- Acero: {get_drift_limit('steel'):.3%}")
        print(f"- Mampostería: {get_drift_limit('masonry'):.3%}")
        
        print(f"\nFactores de reducción:")
        print(f"- Pórticos concreto: R={get_reduction_factor('concrete_moment_frame_special', 'R')}")
        print(f"- Pórticos acero: R={get_reduction_factor('steel_moment_frame_special', 'R')}")
        print(f"- Mampostería: R={get_reduction_factor('masonry_reinforced', 'R')}")
        
        print(f"\nPaíses soportados:")
        for country, data in COUNTRY_DEFAULTS.items():
            print(f"- {country}: {data['normative']} (Zonas: {data['zones']})")
        
        print(f"\nConfiguración automática para edificio de 8 pisos:")
        config = create_configuration(24.0, 'concrete_moment_frame', 'standard', 'PE')
        print(f"- Normativa: {config['normative']}")
        print(f"- Factor R: {config['reduction_factor']}")
        print(f"- Factor I: {config['importance_factor']}")
        print(f"- Deriva límite: {config['drift_limit']:.3%}")
        print(f"- Modos mínimos: {config['min_modes']}")
        
    else:
        print("✗ Error en validación del módulo")