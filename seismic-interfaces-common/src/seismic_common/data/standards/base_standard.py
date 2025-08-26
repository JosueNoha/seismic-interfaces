"""
Clase base abstracta para estándares sísmicos centralizados
=========================================================

Este módulo proporciona la clase base abstracta para implementar diferentes
normativas sísmicas de manera consistente y extensible.

Características principales:
- Clase base abstracta para normativas específicas
- Definición de parámetros sísmicos comunes
- Métodos abstractos para cálculo de espectros
- Validación de parámetros según normativas
- Compatibilidad con código existente
- Extensible para cualquier normativa nacional

Ejemplo de uso:
    ```python
    from seismic_common.standards import BaseSeismicStandard
    
    class E030Standard(BaseSeismicStandard):
        def calculate_design_spectrum(self, periods):
            # Implementación específica para E.030
            pass
        
        def validate_parameters(self):
            # Validación específica para E.030
            pass
    
    # Usar estándar específico
    standard = E030Standard()
    standard.set_zone_factor(0.45)  # Zona 4 Perú
    spectrum = standard.get_design_spectrum([0.1, 0.2, 0.5, 1.0])
    ```
"""

# Metadatos del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Clase base abstracta para estándares sísmicos centralizados"
__license__ = "MIT"
__status__ = "Production"

import sys
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Configurar logging
logger = logging.getLogger(__name__)

# Constantes globales para estándares sísmicos
SEISMIC_CONSTANTS = {
    'GRAVITY': 9.81,                    # m/s²
    'DEFAULT_DAMPING': 0.05,            # 5% amortiguamiento crítico
    'MIN_BASE_SHEAR': 0.044,           # Cortante basal mínimo típico
    'MAX_PERIOD_FACTOR': 1.4,          # Factor de limitación de periodo
    'DEFAULT_IMPORTANCE_FACTOR': 1.0,   # Factor de importancia por defecto
}

# Sistemas estructurales comunes y sus factores R básicos
STRUCTURAL_SYSTEMS = {
    # Sistemas de concreto armado
    'concrete_moment_frame': {
        'name_es': 'Pórticos de Concreto Armado',
        'name_en': 'Reinforced Concrete Moment Frames',
        'R_basic': 8.0,
        'ductility': 'special',
        'material': 'concrete'
    },
    'concrete_dual': {
        'name_es': 'Sistema Dual de Concreto Armado',
        'name_en': 'Reinforced Concrete Dual System',
        'R_basic': 7.0,
        'ductility': 'special',
        'material': 'concrete'
    },
    'concrete_shear_wall': {
        'name_es': 'Muros de Concreto Armado',
        'name_en': 'Reinforced Concrete Shear Walls',
        'R_basic': 6.0,
        'ductility': 'special',
        'material': 'concrete'
    },
    'concrete_limited_ductility': {
        'name_es': 'Muros de Ductilidad Limitada',
        'name_en': 'Limited Ductility Concrete Walls',
        'R_basic': 4.0,
        'ductility': 'limited',
        'material': 'concrete'
    },
    
    # Sistemas de acero
    'steel_smf': {
        'name_es': 'Pórticos Especiales de Acero Resistentes a Momento',
        'name_en': 'Special Steel Moment Frames (SMF)',
        'R_basic': 8.0,
        'ductility': 'special',
        'material': 'steel'
    },
    'steel_imf': {
        'name_es': 'Pórticos Intermedios de Acero Resistentes a Momento',
        'name_en': 'Intermediate Steel Moment Frames (IMF)',
        'R_basic': 5.0,
        'ductility': 'intermediate',
        'material': 'steel'
    },
    'steel_omf': {
        'name_es': 'Pórticos Ordinarios de Acero Resistentes a Momento',
        'name_en': 'Ordinary Steel Moment Frames (OMF)',
        'R_basic': 4.0,
        'ductility': 'ordinary',
        'material': 'steel'
    },
    'steel_scbf': {
        'name_es': 'Pórticos de Acero Concéntricamente Arriostrados Especiales',
        'name_en': 'Special Concentrically Braced Frames (SCBF)',
        'R_basic': 7.0,
        'ductility': 'special',
        'material': 'steel'
    },
    'steel_ocbf': {
        'name_es': 'Pórticos de Acero Concéntricamente Arriostrados Ordinarios',
        'name_en': 'Ordinary Concentrically Braced Frames (OCBF)',
        'R_basic': 4.0,
        'ductility': 'ordinary',
        'material': 'steel'
    },
    'steel_ebf': {
        'name_es': 'Pórticos de Acero Excéntricamente Arriostrados',
        'name_en': 'Eccentrically Braced Frames (EBF)',
        'R_basic': 8.0,
        'ductility': 'special',
        'material': 'steel'
    },
    
    # Otros sistemas
    'masonry': {
        'name_es': 'Albañilería Armada o Confinada',
        'name_en': 'Reinforced or Confined Masonry',
        'R_basic': 3.0,
        'ductility': 'limited',
        'material': 'masonry'
    },
    'wood': {
        'name_es': 'Madera',
        'name_en': 'Wood',
        'R_basic': 7.0,
        'ductility': 'special',
        'material': 'wood'
    }
}

# Categorías de edificaciones típicas
BUILDING_CATEGORIES = {
    'essential': {
        'name_es': 'Esenciales (Categoría A)',
        'name_en': 'Essential Facilities (Category A)',
        'importance_factor': 1.5,
        'description_es': 'Hospitales, estaciones de bomberos, centrales eléctricas',
        'description_en': 'Hospitals, fire stations, power plants'
    },
    'important': {
        'name_es': 'Importantes (Categoría B)',
        'name_en': 'Important Facilities (Category B)',
        'importance_factor': 1.3,
        'description_es': 'Escuelas, centros comerciales, edificios públicos',
        'description_en': 'Schools, shopping centers, public buildings'
    },
    'common': {
        'name_es': 'Comunes (Categoría C)',
        'name_en': 'Common Facilities (Category C)',
        'importance_factor': 1.0,
        'description_es': 'Viviendas, oficinas, hoteles',
        'description_en': 'Residential, offices, hotels'
    },
    'minor': {
        'name_es': 'Menores (Categoría D)',
        'name_en': 'Minor Facilities (Category D)',
        'importance_factor': 0.8,
        'description_es': 'Construcciones temporales, almacenes menores',
        'description_en': 'Temporary structures, minor storage'
    }
}

# Tipos de suelo comunes
SOIL_TYPES = {
    'S0': {
        'name_es': 'Roca Dura',
        'name_en': 'Hard Rock',
        'description_es': 'Roca sana con velocidades de onda de corte > 1500 m/s',
        'vs30_range': (1500, 3000)
    },
    'S1': {
        'name_es': 'Roca o Suelos Muy Rígidos',
        'name_en': 'Rock or Very Dense Soil',
        'description_es': 'Roca con algunas fracturas, suelos granulares densos',
        'vs30_range': (760, 1500)
    },
    'S2': {
        'name_es': 'Suelos Intermedios',
        'name_en': 'Dense or Stiff Soil',
        'description_es': 'Suelos granulares medianamente densos, arcillas duras',
        'vs30_range': (360, 760)
    },
    'S3': {
        'name_es': 'Suelos Blandos',
        'name_en': 'Soft Soil',
        'description_es': 'Suelos granulares sueltos, arcillas blandas',
        'vs30_range': (180, 360)
    },
    'S4': {
        'name_es': 'Condiciones Especiales de Sitio',
        'name_en': 'Special Site Conditions',
        'description_es': 'Suelos susceptibles a licuefacción, arcillas orgánicas',
        'vs30_range': (0, 180)
    }
}


@dataclass
class SeismicParameters:
    """Parámetros sísmicos básicos comunes a todas las normativas"""
    # Parámetros de zonificación
    zone: Union[int, str] = 0
    zone_factor: float = 0.0
    
    # Parámetros de sitio
    soil_type: str = "S1"
    soil_factor_short: float = 1.0  # Fa
    soil_factor_long: float = 1.0   # Fv
    
    # Parámetros de la estructura
    structural_system_x: str = "concrete_moment_frame"
    structural_system_y: str = "concrete_moment_frame"
    building_category: str = "common"
    importance_factor: float = 1.0
    
    # Factores de reducción
    R_basic_x: float = 8.0
    R_basic_y: float = 8.0
    R_final_x: float = 8.0
    R_final_y: float = 8.0
    
    # Factores de irregularidad
    irregularity_height: float = 1.0    # Ia
    irregularity_plan: float = 1.0      # Ip
    
    # Parámetros del espectro
    short_period_region: float = 0.2    # To o Ts
    transition_period: float = 0.6      # Ts
    long_period_transition: float = 2.0  # TL
    
    # Amortiguamiento
    damping_ratio: float = SEISMIC_CONSTANTS['DEFAULT_DAMPING']
    
    # Parámetros calculados
    fundamental_period_x: float = 0.0
    fundamental_period_y: float = 0.0
    design_base_shear_x: float = 0.0
    design_base_shear_y: float = 0.0
    
    def validate_basic_parameters(self) -> Tuple[bool, List[str]]:
        """
        Valida parámetros básicos comunes
        
        Returns
        -------
        Tuple[bool, List[str]]
            (es_válido, lista_errores)
        """
        errors = []
        
        # Validar factor de zona
        if self.zone_factor <= 0:
            errors.append("Factor de zona debe ser mayor a 0")
        
        # Validar factores de reducción
        if self.R_basic_x <= 0:
            errors.append("Factor R básico en X debe ser mayor a 0")
        if self.R_basic_y <= 0:
            errors.append("Factor R básico en Y debe ser mayor a 0")
        
        # Validar factores de irregularidad
        if not (0.5 <= self.irregularity_height <= 1.0):
            errors.append("Factor de irregularidad en altura debe estar entre 0.5 y 1.0")
        if not (0.5 <= self.irregularity_plan <= 1.0):
            errors.append("Factor de irregularidad en planta debe estar entre 0.5 y 1.0")
        
        # Validar factor de importancia
        if self.importance_factor <= 0:
            errors.append("Factor de importancia debe ser mayor a 0")
        
        # Validar amortiguamiento
        if not (0.01 <= self.damping_ratio <= 0.20):
            errors.append("Razón de amortiguamiento debe estar entre 1% y 20%")
        
        return len(errors) == 0, errors


@dataclass
class SpectrumPoint:
    """Punto individual del espectro de respuesta"""
    period: float
    acceleration: float
    velocity: float = 0.0
    displacement: float = 0.0


class BaseSeismicStandard(ABC):
    """
    Clase base abstracta para estándares sísmicos
    
    Define la interfaz común que deben implementar todas las normativas
    sísmicas específicas (E.030, NBC, NCh433, etc.)
    """
    
    def __init__(self, country_code: str, normative_name: str):
        """
        Inicializa el estándar sísmico base
        
        Parameters
        ----------
        country_code : str
            Código del país (PE, BO, CL, etc.)
        normative_name : str
            Nombre de la normativa (E.030, NBC, NCh433, etc.)
        """
        self.country_code = country_code.upper()
        self.normative_name = normative_name
        
        # Parámetros sísmicos
        self.parameters = SeismicParameters()
        
        # Información de la normativa
        self.normative_info = {
            'name': normative_name,
            'country': country_code,
            'version': '',
            'year': '',
            'description': '',
            'units': 'metric'
        }
        
        # Cache para espectros calculados
        self._spectrum_cache = {}
        
        # Configuración específica de la normativa
        self.config = self._initialize_normative_config()
        
        logger.info(f"BaseSeismicStandard inicializado: {normative_name} ({country_code})")
    
    @abstractmethod
    def _initialize_normative_config(self) -> Dict[str, Any]:
        """
        Inicializa configuración específica de la normativa
        
        Returns
        -------
        Dict[str, Any]
            Configuración específica de la normativa
        """
        pass
    
    @abstractmethod
    def calculate_design_spectrum(self, periods: List[float]) -> List[SpectrumPoint]:
        """
        Calcula el espectro de diseño para periodos dados
        
        Parameters
        ----------
        periods : List[float]
            Lista de periodos en segundos
            
        Returns
        -------
        List[SpectrumPoint]
            Puntos del espectro de diseño
        """
        pass
    
    @abstractmethod
    def calculate_base_shear(self, weight: float, period: float, direction: str = 'X') -> float:
        """
        Calcula el cortante basal de diseño
        
        Parameters
        ----------
        weight : float
            Peso sísmico de la estructura
        period : float
            Periodo fundamental de la estructura
        direction : str
            Dirección de análisis ('X' o 'Y')
            
        Returns
        -------
        float
            Cortante basal de diseño
        """
        pass
    
    @abstractmethod
    def calculate_design_accelerations(self, periods: List[float]) -> List[float]:
        """
        Calcula aceleraciones espectrales de diseño
        
        Parameters
        ----------
        periods : List[float]
            Lista de periodos en segundos
            
        Returns
        -------
        List[float]
            Aceleraciones espectrales de diseño
        """
        pass
    
    @abstractmethod
    def validate_drift_limits(self, drift_values: List[float], 
                            structural_system: str) -> Tuple[bool, List[str]]:
        """
        Valida límites de deriva según la normativa
        
        Parameters
        ----------
        drift_values : List[float]
            Valores de deriva de entrepiso
        structural_system : str
            Sistema estructural
            
        Returns
        -------
        Tuple[bool, List[str]]
            (cumple_límites, lista_violaciones)
        """
        pass
    
    # Métodos comunes implementados en la clase base
    
    def set_seismic_zone(self, zone: Union[int, str], zone_factor: float = None):
        """
        Establece la zona sísmica y su factor
        
        Parameters
        ----------
        zone : int or str
            Número o identificador de zona sísmica
        zone_factor : float, optional
            Factor de zona (se calcula automáticamente si no se proporciona)
        """
        self.parameters.zone = zone
        
        if zone_factor is not None:
            self.parameters.zone_factor = zone_factor
        else:
            # Calcular factor de zona según normativa específica
            self.parameters.zone_factor = self._get_zone_factor(zone)
        
        # Limpiar cache
        self._clear_spectrum_cache()
        
        logger.debug(f"Zona sísmica establecida: {zone}, Factor: {self.parameters.zone_factor}")
    
    def set_soil_parameters(self, soil_type: str, fa: float = None, fv: float = None):
        """
        Establece parámetros de sitio
        
        Parameters
        ----------
        soil_type : str
            Tipo de suelo (S0, S1, S2, S3, S4)
        fa : float, optional
            Factor de sitio para periodos cortos
        fv : float, optional
            Factor de sitio para periodos largos
        """
        self.parameters.soil_type = soil_type
        
        if fa is not None:
            self.parameters.soil_factor_short = fa
        else:
            self.parameters.soil_factor_short = self._get_soil_factor_short(soil_type)
        
        if fv is not None:
            self.parameters.soil_factor_long = fv
        else:
            self.parameters.soil_factor_long = self._get_soil_factor_long(soil_type)
        
        # Limpiar cache
        self._clear_spectrum_cache()
        
        logger.debug(f"Suelo establecido: {soil_type}, Fa: {self.parameters.soil_factor_short}, Fv: {self.parameters.soil_factor_long}")
    
    def set_structural_system(self, system_x: str, system_y: str = None):
        """
        Establece el sistema estructural
        
        Parameters
        ----------
        system_x : str
            Sistema estructural en dirección X
        system_y : str, optional
            Sistema estructural en dirección Y (igual a X si no se especifica)
        """
        if system_y is None:
            system_y = system_x
        
        self.parameters.structural_system_x = system_x
        self.parameters.structural_system_y = system_y
        
        # Actualizar factores R básicos
        if system_x in STRUCTURAL_SYSTEMS:
            self.parameters.R_basic_x = STRUCTURAL_SYSTEMS[system_x]['R_basic']
        
        if system_y in STRUCTURAL_SYSTEMS:
            self.parameters.R_basic_y = STRUCTURAL_SYSTEMS[system_y]['R_basic']
        
        # Calcular factores R finales
        self._update_reduction_factors()
        
        logger.debug(f"Sistema estructural: X={system_x} (R={self.parameters.R_basic_x}), Y={system_y} (R={self.parameters.R_basic_y})")
    
    def set_building_category(self, category: str):
        """
        Establece la categoría de edificación
        
        Parameters
        ----------
        category : str
            Categoría de edificación ('essential', 'important', 'common', 'minor')
        """
        self.parameters.building_category = category
        
        if category in BUILDING_CATEGORIES:
            self.parameters.importance_factor = BUILDING_CATEGORIES[category]['importance_factor']
        else:
            logger.warning(f"Categoría de edificación desconocida: {category}")
            self.parameters.importance_factor = 1.0
        
        logger.debug(f"Categoría de edificación: {category}, Factor I: {self.parameters.importance_factor}")
    
    def set_irregularity_factors(self, height: float = 1.0, plan: float = 1.0):
        """
        Establece factores de irregularidad
        
        Parameters
        ----------
        height : float
            Factor de irregularidad en altura (Ia)
        plan : float
            Factor de irregularidad en planta (Ip)
        """
        self.parameters.irregularity_height = height
        self.parameters.irregularity_plan = plan
        
        # Actualizar factores R finales
        self._update_reduction_factors()
        
        logger.debug(f"Factores de irregularidad: Ia={height}, Ip={plan}")
    
    def get_design_spectrum(self, periods: List[float], use_cache: bool = True) -> List[SpectrumPoint]:
        """
        Obtiene el espectro de diseño con cache opcional
        
        Parameters
        ----------
        periods : List[float]
            Lista de periodos
        use_cache : bool
            Usar cache si está disponible
            
        Returns
        -------
        List[SpectrumPoint]
            Espectro de diseño
        """
        if use_cache:
            cache_key = tuple(periods)
            if cache_key in self._spectrum_cache:
                return self._spectrum_cache[cache_key]
        
        spectrum = self.calculate_design_spectrum(periods)
        
        if use_cache:
            self._spectrum_cache[cache_key] = spectrum
        
        return spectrum
    
    def get_spectrum_dataframe(self, periods: List[float]) -> pd.DataFrame:
        """
        Obtiene espectro como DataFrame de pandas
        
        Parameters
        ----------
        periods : List[float]
            Lista de periodos
            
        Returns
        -------
        pd.DataFrame
            DataFrame con el espectro
        """
        spectrum = self.get_design_spectrum(periods)
        
        data = {
            'Period': [point.period for point in spectrum],
            'Acceleration': [point.acceleration for point in spectrum],
            'Velocity': [point.velocity for point in spectrum],
            'Displacement': [point.displacement for point in spectrum]
        }
        
        return pd.DataFrame(data)
    
    def validate_all_parameters(self) -> Tuple[bool, List[str]]:
        """
        Valida todos los parámetros del estándar
        
        Returns
        -------
        Tuple[bool, List[str]]
            (es_válido, lista_errores)
        """
        # Validar parámetros básicos
        basic_valid, basic_errors = self.parameters.validate_basic_parameters()
        
        # Validar parámetros específicos de la normativa
        specific_valid, specific_errors = self._validate_normative_specific()
        
        all_errors = basic_errors + specific_errors
        is_valid = len(all_errors) == 0
        
        return is_valid, all_errors
    
    def export_parameters(self) -> Dict[str, Any]:
        """
        Exporta todos los parámetros como diccionario
        
        Returns
        -------
        Dict[str, Any]
            Diccionario con todos los parámetros
        """
        return {
            'normative_info': self.normative_info,
            'parameters': asdict(self.parameters),
            'config': self.config
        }
    
    def import_parameters(self, data: Dict[str, Any]):
        """
        Importa parámetros desde diccionario
        
        Parameters
        ----------
        data : Dict[str, Any]
            Diccionario con parámetros
        """
        if 'normative_info' in data:
            self.normative_info.update(data['normative_info'])
        
        if 'parameters' in data:
            param_data = data['parameters']
            for key, value in param_data.items():
                if hasattr(self.parameters, key):
                    setattr(self.parameters, key, value)
        
        if 'config' in data:
            self.config.update(data['config'])
        
        # Limpiar cache después de importar
        self._clear_spectrum_cache()
        
        logger.debug("Parámetros importados exitosamente")
    
    # Métodos privados y de utilidad
    
    def _update_reduction_factors(self):
        """Actualiza factores de reducción finales aplicando irregularidades"""
        self.parameters.R_final_x = (
            self.parameters.R_basic_x * 
            self.parameters.irregularity_height * 
            self.parameters.irregularity_plan
        )
        
        self.parameters.R_final_y = (
            self.parameters.R_basic_y * 
            self.parameters.irregularity_height * 
            self.parameters.irregularity_plan
        )
    
    def _clear_spectrum_cache(self):
        """Limpia el cache de espectros"""
        self._spectrum_cache.clear()
    
    # Métodos abstractos para implementación específica
    
    @abstractmethod
    def _get_zone_factor(self, zone: Union[int, str]) -> float:
        """Obtiene factor de zona según normativa específica"""
        pass
    
    @abstractmethod
    def _get_soil_factor_short(self, soil_type: str) -> float:
        """Obtiene factor de sitio para periodos cortos"""
        pass
    
    @abstractmethod
    def _get_soil_factor_long(self, soil_type: str) -> float:
        """Obtiene factor de sitio para periodos largos"""
        pass
    
    @abstractmethod
    def _validate_normative_specific(self) -> Tuple[bool, List[str]]:
        """Valida parámetros específicos de la normativa"""
        pass
    
    # Métodos de utilidad estáticos
    
    @staticmethod
    def get_available_structural_systems() -> Dict[str, Dict[str, Any]]:
        """
        Obtiene sistemas estructurales disponibles
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Diccionario con sistemas estructurales
        """
        return STRUCTURAL_SYSTEMS.copy()
    
    @staticmethod
    def get_available_building_categories() -> Dict[str, Dict[str, Any]]:
        """
        Obtiene categorías de edificación disponibles
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Diccionario con categorías de edificación
        """
        return BUILDING_CATEGORIES.copy()
    
    @staticmethod
    def get_available_soil_types() -> Dict[str, Dict[str, Any]]:
        """
        Obtiene tipos de suelo disponibles
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Diccionario con tipos de suelo
        """
        return SOIL_TYPES.copy()
    
    def __str__(self) -> str:
        """Representación en cadena del estándar"""
        return f"{self.normative_name} ({self.country_code}) - Zona {self.parameters.zone}"
    
    def __repr__(self) -> str:
        """Representación detallada del estándar"""
        return f"BaseSeismicStandard(country='{self.country_code}', normative='{self.normative_name}')"


# Funciones de utilidad para estándares sísmicos

def create_standard_factory(country_code: str) -> Dict[str, type]:
    """
    Factory function para crear estándares según país
    
    Nota: Las implementaciones específicas estarán en cada proyecto
    
    Parameters
    ----------
    country_code : str
        Código del país
        
    Returns
    -------
    Dict[str, type]
        Diccionario con clases de estándares disponibles
    """
    # Esta función retorna la clase base
    # Las implementaciones específicas deben sobrescribir esta función
    logger.info(f"Factory genérico para {country_code}")
    return {'base': BaseSeismicStandard}


def get_normative_by_country(country_code: str) -> List[str]:
    """
    Obtiene normativas disponibles por país
    
    Parameters
    ----------
    country_code : str
        Código del país
        
    Returns
    -------
    List[str]
        Lista de normativas disponibles
    """
    normatives_map = {
        'PE': ['E.030'],
        'BO': ['NBC'],
        'CL': ['NCh433'],
        'CO': ['NSR-10'],
        'EC': ['NEC'],
        'MX': ['CFE', 'RCDF'],
        'US': ['ASCE7', 'IBC'],
        'EU': ['EC8']
    }
    
    return normatives_map.get(country_code.upper(), [])


def validate_standard_compatibility(standard1: BaseSeismicStandard, 
                                   standard2: BaseSeismicStandard) -> Tuple[bool, List[str]]:
    """
    Valida compatibilidad entre dos estándares sísmicos
    
    Parameters
    ----------
    standard1 : BaseSeismicStandard
        Primer estándar a comparar
    standard2 : BaseSeismicStandard
        Segundo estándar a comparar
        
    Returns
    -------
    Tuple[bool, List[str]]
        (son_compatibles, lista_incompatibilidades)
    """
    incompatibilities = []
    
    # Verificar mismo país
    if standard1.country_code != standard2.country_code:
        incompatibilities.append(f"Países diferentes: {standard1.country_code} vs {standard2.country_code}")
    
    # Verificar misma normativa
    if standard1.normative_name != standard2.normative_name:
        incompatibilities.append(f"Normativas diferentes: {standard1.normative_name} vs {standard2.normative_name}")
    
    # Verificar unidades compatibles
    units1 = standard1.normative_info.get('units', 'metric')
    units2 = standard2.normative_info.get('units', 'metric')
    if units1 != units2:
        incompatibilities.append(f"Sistemas de unidades diferentes: {units1} vs {units2}")
    
    return len(incompatibilities) == 0, incompatibilities


class StandardRegistry:
    """
    Registro centralizado de estándares sísmicos disponibles
    
    Mantiene un catálogo de normativas implementadas y sus capacidades
    """
    
    def __init__(self):
        self._standards = {}
        self._metadata = {}
        
    def register_standard(self, country_code: str, normative_name: str, 
                         standard_class: type, metadata: Dict[str, Any] = None):
        """
        Registra un estándar sísmico
        
        Parameters
        ----------
        country_code : str
            Código del país
        normative_name : str
            Nombre de la normativa
        standard_class : type
            Clase que implementa el estándar
        metadata : Dict[str, Any], optional
            Metadatos adicionales del estándar
        """
        key = f"{country_code.upper()}_{normative_name}"
        self._standards[key] = standard_class
        
        if metadata is None:
            metadata = {}
        
        self._metadata[key] = {
            'country_code': country_code.upper(),
            'normative_name': normative_name,
            'class_name': standard_class.__name__,
            'registered_at': pd.Timestamp.now().isoformat(),
            **metadata
        }
        
        logger.info(f"Estándar registrado: {key}")
    
    def get_standard(self, country_code: str, normative_name: str) -> Optional[type]:
        """
        Obtiene una clase de estándar registrada
        
        Parameters
        ----------
        country_code : str
            Código del país
        normative_name : str
            Nombre de la normativa
            
        Returns
        -------
        type or None
            Clase del estándar o None si no existe
        """
        key = f"{country_code.upper()}_{normative_name}"
        return self._standards.get(key)
    
    def list_standards(self, country_code: str = None) -> List[Dict[str, Any]]:
        """
        Lista estándares disponibles
        
        Parameters
        ----------
        country_code : str, optional
            Filtrar por código de país
            
        Returns
        -------
        List[Dict[str, Any]]
            Lista de metadatos de estándares
        """
        if country_code is None:
            return list(self._metadata.values())
        
        country_upper = country_code.upper()
        return [meta for meta in self._metadata.values() 
                if meta['country_code'] == country_upper]
    
    def create_standard_instance(self, country_code: str, normative_name: str) -> Optional[BaseSeismicStandard]:
        """
        Crea una instancia de estándar
        
        Parameters
        ----------
        country_code : str
            Código del país
        normative_name : str
            Nombre de la normativa
            
        Returns
        -------
        BaseSeismicStandard or None
            Instancia del estándar o None si no existe
        """
        standard_class = self.get_standard(country_code, normative_name)
        if standard_class:
            return standard_class(country_code, normative_name)
        return None


# Instancia global del registro
STANDARD_REGISTRY = StandardRegistry()


class SeismicCalculator:
    """
    Calculadora de parámetros sísmicos comunes
    
    Proporciona métodos de cálculo que son independientes de normativas específicas
    """
    
    @staticmethod
    def calculate_fundamental_period_approx(height: float, building_type: str = 'concrete') -> float:
        """
        Calcula periodo fundamental aproximado
        
        Parameters
        ----------
        height : float
            Altura del edificio en metros
        building_type : str
            Tipo de edificio ('concrete', 'steel', 'masonry')
            
        Returns
        -------
        float
            Periodo fundamental aproximado en segundos
        """
        # Coeficientes típicos según tipo de estructura
        coefficients = {
            'concrete': {'Ct': 0.0488, 'x': 0.75},      # Pórticos de concreto
            'steel': {'Ct': 0.0724, 'x': 0.80},         # Pórticos de acero
            'masonry': {'Ct': 0.0331, 'x': 0.75},       # Muros de corte
            'wood': {'Ct': 0.0488, 'x': 0.75}           # Madera
        }
        
        coeff = coefficients.get(building_type, coefficients['concrete'])
        period = coeff['Ct'] * (height ** coeff['x'])
        
        return period
    
    @staticmethod
    def calculate_period_limits(approx_period: float, upper_limit_factor: float = 1.4) -> Tuple[float, float]:
        """
        Calcula límites de periodo para análisis dinámico
        
        Parameters
        ----------
        approx_period : float
            Periodo aproximado
        upper_limit_factor : float
            Factor de límite superior (típicamente 1.4)
            
        Returns
        -------
        Tuple[float, float]
            (periodo_mínimo, periodo_máximo)
        """
        # Periodo mínimo típicamente es 0.85 veces el aproximado
        min_period = 0.85 * approx_period
        # Periodo máximo es el factor por el aproximado
        max_period = upper_limit_factor * approx_period
        
        return min_period, max_period
    
    @staticmethod
    def calculate_mass_participation_adequacy(participation_ratios: List[float], 
                                            min_required: float = 0.9) -> Tuple[bool, float]:
        """
        Verifica adecuación de participación de masa
        
        Parameters
        ----------
        participation_ratios : List[float]
            Ratios de participación modal de masa
        min_required : float
            Participación mínima requerida (típicamente 90%)
            
        Returns
        -------
        Tuple[bool, float]
            (es_adecuado, participación_total)
        """
        total_participation = sum(participation_ratios)
        is_adequate = total_participation >= min_required
        
        return is_adequate, total_participation
    
    @staticmethod
    def calculate_torsional_irregularity(max_displacement: float, 
                                       avg_displacement: float,
                                       limit_factor: float = 1.2) -> Tuple[bool, float]:
        """
        Verifica irregularidad torsional
        
        Parameters
        ----------
        max_displacement : float
            Desplazamiento máximo del diafragma
        avg_displacement : float
            Desplazamiento promedio de los extremos
        limit_factor : float
            Factor límite (típicamente 1.2)
            
        Returns
        -------
        Tuple[bool, float]
            (hay_irregularidad, ratio_calculado)
        """
        if avg_displacement == 0:
            return False, 0.0
        
        ratio = max_displacement / avg_displacement
        has_irregularity = ratio > limit_factor
        
        return has_irregularity, ratio
    
    @staticmethod
    def calculate_soft_story_irregularity(story_rigidities: List[float],
                                        limit_factor: float = 0.7) -> Tuple[bool, List[int]]:
        """
        Verifica irregularidad de piso blando
        
        Parameters
        ----------
        story_rigidities : List[float]
            Rigideces de cada piso
        limit_factor : float
            Factor límite (típicamente 0.7)
            
        Returns
        -------
        Tuple[bool, List[int]]
            (hay_irregularidad, pisos_irregulares)
        """
        if len(story_rigidities) < 2:
            return False, []
        
        irregular_stories = []
        
        for i in range(len(story_rigidities) - 1):
            current_rigidity = story_rigidities[i]
            upper_rigidity = story_rigidities[i + 1]
            
            if upper_rigidity > 0:
                ratio = current_rigidity / upper_rigidity
                if ratio < limit_factor:
                    irregular_stories.append(i)
        
        has_irregularity = len(irregular_stories) > 0
        
        return has_irregularity, irregular_stories
    
    @staticmethod
    def interpolate_spectrum_value(periods: List[float], accelerations: List[float], 
                                 target_period: float) -> float:
        """
        Interpola valor del espectro para un periodo específico
        
        Parameters
        ----------
        periods : List[float]
            Periodos del espectro
        accelerations : List[float]
            Aceleraciones espectrales
        target_period : float
            Periodo objetivo para interpolación
            
        Returns
        -------
        float
            Aceleración espectral interpolada
        """
        if not periods or not accelerations or len(periods) != len(accelerations):
            return 0.0
        
        # Si el periodo objetivo está fuera del rango, usar valores extremos
        if target_period <= periods[0]:
            return accelerations[0]
        if target_period >= periods[-1]:
            return accelerations[-1]
        
        # Interpolación lineal
        for i in range(len(periods) - 1):
            if periods[i] <= target_period <= periods[i + 1]:
                t1, t2 = periods[i], periods[i + 1]
                a1, a2 = accelerations[i], accelerations[i + 1]
                
                # Interpolación lineal
                factor = (target_period - t1) / (t2 - t1)
                interpolated = a1 + factor * (a2 - a1)
                
                return interpolated
        
        return 0.0


# Funciones de validación del módulo

def validate_base_standard_module() -> bool:
    """
    Valida el funcionamiento del módulo base_standard
    
    Returns
    -------
    bool
        True si todas las validaciones pasan
    """
    try:
        # Probar creación de parámetros sísmicos
        params = SeismicParameters()
        params.zone_factor = 0.45
        params.R_basic_x = 8.0
        params.R_basic_y = 8.0
        
        is_valid, errors = params.validate_basic_parameters()
        assert is_valid, f"Parámetros básicos inválidos: {errors}"
        
        # Probar registro de estándares
        registry = StandardRegistry()
        
        # Crear una clase de prueba
        class TestStandard(BaseSeismicStandard):
            def _initialize_normative_config(self):
                return {'test': True}
            
            def calculate_design_spectrum(self, periods):
                return [SpectrumPoint(p, 0.5) for p in periods]
            
            def calculate_base_shear(self, weight, period, direction):
                return weight * 0.1
            
            def calculate_design_accelerations(self, periods):
                return [0.5] * len(periods)
            
            def validate_drift_limits(self, drift_values, structural_system):
                return True, []
            
            def _get_zone_factor(self, zone):
                return 0.45
            
            def _get_soil_factor_short(self, soil_type):
                return 1.0
            
            def _get_soil_factor_long(self, soil_type):
                return 1.0
            
            def _validate_normative_specific(self):
                return True, []
        
        # Registrar estándar de prueba
        registry.register_standard('TEST', 'TEST_NORM', TestStandard)
        
        # Crear instancia
        test_instance = registry.create_standard_instance('TEST', 'TEST_NORM')
        assert test_instance is not None
        assert isinstance(test_instance, BaseSeismicStandard)
        
        # Probar configuración de parámetros
        test_instance.set_seismic_zone(4, 0.45)
        test_instance.set_soil_parameters('S2')
        test_instance.set_structural_system('concrete_moment_frame')
        test_instance.set_building_category('common')
        test_instance.set_irregularity_factors(1.0, 1.0)
        
        # Probar cálculo de espectro
        periods = [0.1, 0.5, 1.0, 2.0]
        spectrum = test_instance.get_design_spectrum(periods)
        assert len(spectrum) == len(periods)
        assert all(isinstance(point, SpectrumPoint) for point in spectrum)
        
        # Probar DataFrame del espectro
        df = test_instance.get_spectrum_dataframe(periods)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(periods)
        assert 'Period' in df.columns
        assert 'Acceleration' in df.columns
        
        # Probar calculadora sísmica
        calc = SeismicCalculator()
        
        # Calcular periodo aproximado
        period_approx = calc.calculate_fundamental_period_approx(30.0, 'concrete')
        assert period_approx > 0
        
        # Calcular límites de periodo
        min_p, max_p = calc.calculate_period_limits(period_approx)
        assert min_p < period_approx < max_p
        
        # Probar participación de masa
        participation = [0.6, 0.2, 0.1, 0.05]
        is_adequate, total = calc.calculate_mass_participation_adequacy(participation)
        assert abs(total - 0.95) < 1e-6
        assert is_adequate
        
        # Probar irregularidad torsional
        has_torsional, ratio = calc.calculate_torsional_irregularity(12.0, 10.0)
        assert not has_torsional
        assert abs(ratio - 1.2) < 1e-6
        
        # Probar interpolación espectral
        spectrum_periods = [0.0, 0.5, 1.0, 2.0]
        spectrum_accels = [0.4, 1.0, 0.8, 0.3]
        interp_value = calc.interpolate_spectrum_value(spectrum_periods, spectrum_accels, 0.75)
        assert 0.8 < interp_value < 1.0  # Debe estar entre 1.0 y 0.8
        
        # Probar exportación/importación de parámetros
        exported = test_instance.export_parameters()
        assert isinstance(exported, dict)
        assert 'normative_info' in exported
        assert 'parameters' in exported
        
        # Probar sistemas estructurales estáticos
        systems = BaseSeismicStandard.get_available_structural_systems()
        assert 'concrete_moment_frame' in systems
        assert systems['concrete_moment_frame']['R_basic'] == 8.0
        
        categories = BaseSeismicStandard.get_available_building_categories()
        assert 'common' in categories
        assert categories['common']['importance_factor'] == 1.0
        
        soil_types = BaseSeismicStandard.get_available_soil_types()
        assert 'S1' in soil_types
        
        # Probar compatibilidad entre estándares
        test_instance2 = TestStandard('TEST', 'TEST_NORM')
        compatible, issues = validate_standard_compatibility(test_instance, test_instance2)
        assert compatible
        assert len(issues) == 0
        
        logger.info("✓ Validación del módulo base_standard exitosa")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en validación del módulo base_standard: {e}")
        return False


# Funciones de utilidad adicionales

def create_default_parameters_by_country(country_code: str) -> SeismicParameters:
    """
    Crea parámetros sísmicos por defecto según país
    
    Parameters
    ----------
    country_code : str
        Código del país
        
    Returns
    -------
    SeismicParameters
        Parámetros con valores por defecto del país
    """
    params = SeismicParameters()
    
    # Configuraciones por defecto por país
    defaults = {
        'PE': {
            'zone': 4,
            'zone_factor': 0.45,
            'soil_type': 'S1',
            'structural_system_x': 'concrete_moment_frame',
            'structural_system_y': 'concrete_moment_frame',
            'building_category': 'common'
        },
        'BO': {
            'zone': 3,
            'zone_factor': 0.30,
            'soil_type': 'S2',
            'structural_system_x': 'concrete_moment_frame',
            'structural_system_y': 'concrete_moment_frame',
            'building_category': 'common'
        },
        'CL': {
            'zone': 'A2',
            'zone_factor': 0.30,
            'soil_type': 'S2',
            'structural_system_x': 'concrete_moment_frame',
            'structural_system_y': 'concrete_moment_frame',
            'building_category': 'common'
        }
    }
    
    country_defaults = defaults.get(country_code.upper(), defaults['PE'])
    
    for key, value in country_defaults.items():
        if hasattr(params, key):
            setattr(params, key, value)
    
    return params


def generate_standard_periods(min_period: float = 0.01, max_period: float = 4.0, 
                            num_points: int = 100, distribution: str = 'log') -> List[float]:
    """
    Genera lista de periodos para cálculo de espectros
    
    Parameters
    ----------
    min_period : float
        Periodo mínimo
    max_period : float
        Periodo máximo
    num_points : int
        Número de puntos
    distribution : str
        Distribución ('linear' o 'log')
        
    Returns
    -------
    List[float]
        Lista de periodos
    """
    if distribution == 'log':
        periods = np.logspace(np.log10(min_period), np.log10(max_period), num_points)
    else:
        periods = np.linspace(min_period, max_period, num_points)
    
    return periods.tolist()


def compare_standards_parameters(standard1: BaseSeismicStandard, 
                               standard2: BaseSeismicStandard) -> pd.DataFrame:
    """
    Compara parámetros entre dos estándares
    
    Parameters
    ----------
    standard1 : BaseSeismicStandard
        Primer estándar
    standard2 : BaseSeismicStandard
        Segundo estándar
        
    Returns
    -------
    pd.DataFrame
        DataFrame con comparación de parámetros
    """
    params1 = asdict(standard1.parameters)
    params2 = asdict(standard2.parameters)
    
    comparison_data = []
    
    for param_name in params1.keys():
        value1 = params1[param_name]
        value2 = params2[param_name]
        
        comparison_data.append({
            'Parameter': param_name,
            'Standard_1': value1,
            'Standard_2': value2,
            'Equal': value1 == value2,
            'Difference': value1 - value2 if isinstance(value1, (int, float)) and isinstance(value2, (int, float)) else None
        })
    
    return pd.DataFrame(comparison_data)


# Punto de entrada para pruebas del módulo
if __name__ == "__main__":
    # Configurar logging para pruebas
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Ejecutar validación
    if validate_base_standard_module():
        print("✓ Módulo base_standard validado correctamente")
        
        # Ejemplo de uso básico
        print("\nEjemplo de uso:")
        
        # Crear parámetros por defecto para Perú
        params = create_default_parameters_by_country('PE')
        print(f"Parámetros por defecto para PE:")
        print(f"- Zona: {params.zone}, Factor Z: {params.zone_factor}")
        print(f"- Sistema estructural: {params.structural_system_x}")
        print(f"- Categoría: {params.building_category}")
        
        # Generar periodos estándar
        periods = generate_standard_periods(0.01, 3.0, 50)
        print(f"Generados {len(periods)} periodos de 0.01 a 3.0 segundos")
        
        # Usar calculadora sísmica
        calc = SeismicCalculator()
        T_approx = calc.calculate_fundamental_period_approx(25.0, 'concrete')
        T_min, T_max = calc.calculate_period_limits(T_approx)
        
        print(f"\nCálculos para edificio de 25m de altura:")
        print(f"- Periodo aproximado: {T_approx:.3f} s")
        print(f"- Rango permitido: {T_min:.3f} - {T_max:.3f} s")
        
        # Mostrar sistemas disponibles
        systems = BaseSeismicStandard.get_available_structural_systems()
        print(f"\nSistemas estructurales disponibles: {len(systems)}")
        for key, system in list(systems.items())[:3]:  # Mostrar solo los primeros 3
            print(f"- {key}: {system['name_es']} (R = {system['R_basic']})")
        
    else:
        print("✗ Error en validación del módulo base_standard")
        sys.exit(1)