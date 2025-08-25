"""
Modelos centralizados de datos sísmicos para interfaces de análisis
=================================================================

Este módulo proporciona modelos de datos centralizados para almacenar y manejar
información sísmica común a todas las normativas, así como extensiones específicas
para cada país/normativa.

Características principales:
- Modelo base común a todas las normativas sísmicas
- Extensiones específicas para cada normativa (Perú E.030, Bolivia NCH, etc.)
- Validación automática de parámetros según normativas
- Serialización/deserialización de datos
- Compatibilidad total con código existente
- Gestión de cargas sísmicas y combinaciones

Ejemplo de uso:
    ```python
    from seismic_common.models import SeismicData, PeruSeismicData, BoliviaSeismicData
    
    # Uso básico (genérico)
    seismic_data = SeismicData()
    seismic_data.set_reduction_factors(8.0, 8.0)
    seismic_data.set_irregularity_factors(0.75, 1.0)
    
    # Uso específico para normativa peruana
    peru_data = PeruSeismicData()
    peru_data.set_zone_factor(0.45)  # Zona 4
    peru_data.set_usage_factor(1.0)   # Categoría A
    ```
"""

# Metadatos del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Modelos de datos sísmicos centralizados para diferentes normativas"
__license__ = "MIT"
__status__ = "Production"

import sys
import json
import pickle
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

import pandas as pd
import numpy as np

# Configurar logging
logger = logging.getLogger(__name__)

# Constantes comunes a todas las normativas
DEFAULT_LIMITS = {
    'drift_limit': 0.007,        # Deriva límite típica (0.7%)
    'torsional_limit': 1.2,      # Irregularidad torsional
    'soft_story_limit': 0.7,     # Piso blando
    'mass_irregularity': 1.5,    # Irregularidad de masa
    'min_participation': 0.9     # Participación de masa mínima
}

# Factores de reducción típicos por sistema estructural
DEFAULT_R_FACTORS = {
    'porticos_concreto': 8.0,
    'sistema_dual': 7.0,
    'muros_concreto': 6.0,
    'porticos_acero_especial': 8.0,
    'porticos_acero_intermedio': 5.0,
    'porticos_acero_ordinario': 4.0,
    'albañileria': 3.0,
    'madera': 7.0
}

# Valores típicos de irregularidades
DEFAULT_IRREGULARITY_FACTORS = {
    'regular_structure': {'Ia': 1.0, 'Ip': 1.0},
    'height_irregularity': {'Ia': 0.75, 'Ip': 1.0},
    'plan_irregularity': {'Ia': 1.0, 'Ip': 0.75},
    'both_irregularities': {'Ia': 0.75, 'Ip': 0.75}
}


@dataclass
class ProjectInfo:
    """Información básica del proyecto estructural"""
    name: str = ""
    location: str = ""
    author: str = ""
    date: str = ""
    description: str = ""
    structural_system: str = ""
    analysis_type: str = "dynamic"
    notes: str = ""
    
    def __post_init__(self):
        """Inicialización posterior al constructor"""
        if not self.date:
            self.date = datetime.now().strftime("%Y-%m-%d")


@dataclass
class LoadPatterns:
    """Patrones de carga sísmica"""
    static_x: str = "SX"
    static_y: str = "SY" 
    dynamic_x: str = "SDX"
    dynamic_y: str = "SDY"
    spectrum_x: str = "SPECX"
    spectrum_y: str = "SPECY"
    
    # Mapeo para compatibilidad con código existente
    seism_loads: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Actualizar mapeo de cargas sísmicas"""
        self.seism_loads = {
            'SX': self.static_x,
            'SY': self.static_y,
            'SDX': self.dynamic_x,
            'SDY': self.dynamic_y,
            'SPECX': self.spectrum_x,
            'SPECY': self.spectrum_y
        }
    
    def get_x_direction_loads(self) -> List[str]:
        """Obtiene cargas en dirección X"""
        return [self.static_x, self.dynamic_x, self.spectrum_x]
    
    def get_y_direction_loads(self) -> List[str]:
        """Obtiene cargas en dirección Y"""
        return [self.static_y, self.dynamic_y, self.spectrum_y]


@dataclass
class ModalData:
    """Datos del análisis modal"""
    periods_x: List[float] = field(default_factory=list)
    periods_y: List[float] = field(default_factory=list)
    frequencies: List[float] = field(default_factory=list)
    mass_participation_x: List[float] = field(default_factory=list)
    mass_participation_y: List[float] = field(default_factory=list)
    
    # Valores calculados principales
    fundamental_period_x: float = 0.0
    fundamental_period_y: float = 0.0
    total_participation_x: float = 0.0
    total_participation_y: float = 0.0
    
    # DataFrame para compatibilidad
    modal_table: Optional[pd.DataFrame] = None
    
    def calculate_fundamental_periods(self):
        """Calcula periodos fundamentales"""
        if self.periods_x:
            self.fundamental_period_x = max(self.periods_x)
        if self.periods_y:
            self.fundamental_period_y = max(self.periods_y)
    
    def calculate_total_participation(self):
        """Calcula participación total de masa"""
        self.total_participation_x = sum(self.mass_participation_x)
        self.total_participation_y = sum(self.mass_participation_y)
    
    def is_participation_adequate(self, min_participation: float = 0.9) -> Tuple[bool, bool]:
        """Verifica si la participación de masa es adecuada"""
        x_adequate = self.total_participation_x >= min_participation
        y_adequate = self.total_participation_y >= min_participation
        return x_adequate, y_adequate


@dataclass
class DriftData:
    """Datos de derivas de piso"""
    story_names: List[str] = field(default_factory=list)
    story_heights: List[float] = field(default_factory=list)
    drifts_x: List[float] = field(default_factory=list)
    drifts_y: List[float] = field(default_factory=list)
    
    # Límites de deriva
    max_drift_x: float = DEFAULT_LIMITS['drift_limit']
    max_drift_y: float = DEFAULT_LIMITS['drift_limit']
    
    # DataFrame para compatibilidad
    drift_table: Optional[pd.DataFrame] = None
    
    def get_max_drifts(self) -> Tuple[float, float]:
        """Obtiene las derivas máximas"""
        max_drift_x = max(self.drifts_x) if self.drifts_x else 0.0
        max_drift_y = max(self.drifts_y) if self.drifts_y else 0.0
        return max_drift_x, max_drift_y
    
    def check_drift_compliance(self) -> Tuple[bool, bool]:
        """Verifica cumplimiento de derivas"""
        max_x, max_y = self.get_max_drifts()
        x_compliant = max_x <= self.max_drift_x
        y_compliant = max_y <= self.max_drift_y
        return x_compliant, y_compliant
    
    def get_critical_stories(self) -> Tuple[List[str], List[str]]:
        """Obtiene pisos con derivas críticas"""
        critical_x = []
        critical_y = []
        
        for i, story in enumerate(self.story_names):
            if i < len(self.drifts_x) and self.drifts_x[i] > self.max_drift_x:
                critical_x.append(story)
            if i < len(self.drifts_y) and self.drifts_y[i] > self.max_drift_y:
                critical_y.append(story)
        
        return critical_x, critical_y


class SeismicData:
    """
    Clase base para almacenamiento de datos sísmicos comunes
    
    Compatible con código existente y extensible para normativas específicas
    """
    
    def __init__(self):
        """Inicializa con valores por defecto comunes"""
        # Información del proyecto
        self.project = ProjectInfo()
        
        # Patrones de carga
        self.loads = LoadPatterns()
        
        # Factores de reducción sísmica (común a todas las normativas)
        self.Rx: float = 8.0  # Factor de reducción X
        self.Ry: float = 8.0  # Factor de reducción Y
        self.Rox: float = 8.0  # Factor básico de reducción X
        self.Roy: float = 8.0  # Factor básico de reducción Y
        
        # Factores de irregularidad (común a todas las normativas)
        self.Ia: float = 1.0  # Factor de irregularidad en altura
        self.Ip: float = 1.0  # Factor de irregularidad en planta
        
        # Datos de análisis modal
        self.modal_data = ModalData()
        
        # Datos de derivas
        self.drift_data = DriftData()
        
        # Periodos fundamentales (calculados)
        self.Tx: float = 0.0  # Periodo fundamental X
        self.Ty: float = 0.0  # Periodo fundamental Y
        
        # Masas participativas (calculadas)
        self.MP_x: float = 0.0  # Masa participativa X
        self.MP_y: float = 0.0  # Masa participativa Y
        
        # Parámetros de deriva (compatibilidad)
        self.max_drift_x: float = DEFAULT_LIMITS['drift_limit']
        self.max_drift_y: float = DEFAULT_LIMITS['drift_limit']
        
        # Tablas de resultados (compatibilidad con código existente)
        self.modal_table: Optional[pd.DataFrame] = None
        self.static_seism_table: Optional[pd.DataFrame] = None
        self.drift_table: Optional[pd.DataFrame] = None
        self.irregularity_tables: Dict[str, pd.DataFrame] = {}
        
        # Parámetros específicos de normativas (extensibles)
        self.normative_params: Dict[str, Any] = {}
        
        # Información adicional del proyecto (compatibilidad)
        self.proyecto: str = ""
        self.ubicacion: str = ""
        self.autor: str = ""
        self.fecha: str = ""
        self.descripcion: str = ""
        self.modelamiento: str = ""
        self.cargas: str = ""
        
        logger.debug("SeismicData inicializado con valores por defecto")
    
    # Métodos de configuración de factores básicos
    def set_reduction_factors(self, rx: float, ry: float, 
                             rox: Optional[float] = None, roy: Optional[float] = None):
        """
        Establece factores de reducción sísmica
        
        Parameters
        ----------
        rx, ry : float
            Factores de reducción finales
        rox, roy : float, optional
            Factores básicos de reducción (si no se proporcionan, usan rx, ry)
        """
        self.Rx = float(rx)
        self.Ry = float(ry)
        self.Rox = float(rox) if rox is not None else self.Rx
        self.Roy = float(roy) if roy is not None else self.Ry
        
        logger.debug(f"Factores de reducción establecidos: Rx={self.Rx}, Ry={self.Ry}")
    
    def set_irregularity_factors(self, ia: float, ip: float):
        """
        Establece factores de irregularidad
        
        Parameters
        ----------
        ia : float
            Factor de irregularidad en altura
        ip : float
            Factor de irregularidad en planta
        """
        self.Ia = float(ia)
        self.Ip = float(ip)
        
        # Recalcular factores de reducción finales
        self.Rx = self.Rox * self.Ia * self.Ip
        self.Ry = self.Roy * self.Ia * self.Ip
        
        logger.debug(f"Factores de irregularidad establecidos: Ia={self.Ia}, Ip={self.Ip}")
    
    def set_fundamental_periods(self, tx: float, ty: float):
        """
        Establece periodos fundamentales
        
        Parameters
        ----------
        tx, ty : float
            Periodos fundamentales en X e Y
        """
        self.Tx = float(tx)
        self.Ty = float(ty)
        self.modal_data.fundamental_period_x = self.Tx
        self.modal_data.fundamental_period_y = self.Ty
        
        logger.debug(f"Periodos fundamentales establecidos: Tx={self.Tx}, Ty={self.Ty}")
    
    def set_mass_participation(self, mp_x: float, mp_y: float):
        """
        Establece masas participativas totales
        
        Parameters
        ----------
        mp_x, mp_y : float
            Participación de masa en X e Y (0-1 o 0-100)
        """
        # Convertir a fracción si está en porcentaje
        if mp_x > 1.0:
            mp_x /= 100.0
        if mp_y > 1.0:
            mp_y /= 100.0
            
        self.MP_x = float(mp_x)
        self.MP_y = float(mp_y)
        self.modal_data.total_participation_x = self.MP_x
        self.modal_data.total_participation_y = self.MP_y
        
        logger.debug(f"Masas participativas establecidas: MP_x={self.MP_x}, MP_y={self.MP_y}")
    
    def set_drift_limits(self, max_drift_x: float, max_drift_y: float):
        """
        Establece límites de deriva
        
        Parameters
        ----------
        max_drift_x, max_drift_y : float
            Límites máximos de deriva
        """
        self.max_drift_x = float(max_drift_x)
        self.max_drift_y = float(max_drift_y)
        self.drift_data.max_drift_x = self.max_drift_x
        self.drift_data.max_drift_y = self.max_drift_y
        
        logger.debug(f"Límites de deriva establecidos: {self.max_drift_x}, {self.max_drift_y}")
    
    # Métodos de configuración de información del proyecto
    def set_project_info(self, proyecto: str = "", ubicacion: str = "", 
                        autor: str = "", fecha: str = "", descripcion: str = ""):
        """
        Establece información básica del proyecto (compatibilidad)
        
        Parameters
        ----------
        proyecto, ubicacion, autor, fecha, descripcion : str
            Información del proyecto
        """
        self.proyecto = proyecto
        self.ubicacion = ubicacion
        self.autor = autor
        self.fecha = fecha if fecha else datetime.now().strftime("%Y-%m-%d")
        self.descripcion = descripcion
        
        # Actualizar estructura moderna
        self.project.name = proyecto
        self.project.location = ubicacion
        self.project.author = autor
        self.project.date = self.fecha
        self.project.description = descripcion
        
        logger.debug(f"Información del proyecto establecida: {proyecto}")
    
    def set_load_patterns(self, seism_loads: Dict[str, str]):
        """
        Establece patrones de carga sísmica (compatibilidad)
        
        Parameters
        ----------
        seism_loads : dict
            Diccionario con patrones de carga
        """
        self.loads.seism_loads = seism_loads.copy()
        
        # Actualizar campos individuales si están presentes
        self.loads.static_x = seism_loads.get('SX', 'SX')
        self.loads.static_y = seism_loads.get('SY', 'SY')
        self.loads.dynamic_x = seism_loads.get('SDX', 'SDX')
        self.loads.dynamic_y = seism_loads.get('SDY', 'SDY')
        
        logger.debug(f"Patrones de carga establecidos: {seism_loads}")
    
    # Métodos de validación
    def validate_basic_parameters(self) -> Tuple[bool, List[str]]:
        """
        Valida parámetros básicos comunes a todas las normativas
        
        Returns
        -------
        Tuple[bool, List[str]]
            (es_válido, lista_errores)
        """
        errors = []
        
        # Validar factores de reducción
        if self.Rx <= 0:
            errors.append("Factor de reducción Rx debe ser mayor a 0")
        if self.Ry <= 0:
            errors.append("Factor de reducción Ry debe ser mayor a 0")
        
        # Validar irregularidades
        if not (0.5 <= self.Ia <= 1.0):
            errors.append("Factor de irregularidad Ia debe estar entre 0.5 y 1.0")
        if not (0.5 <= self.Ip <= 1.0):
            errors.append("Factor de irregularidad Ip debe estar entre 0.5 y 1.0")
        
        # Validar periodos
        if self.Tx <= 0:
            errors.append("Periodo fundamental Tx debe ser mayor a 0")
        if self.Ty <= 0:
            errors.append("Periodo fundamental Ty debe ser mayor a 0")
        
        # Validar participación de masa
        if self.MP_x < 0.9:
            errors.append("Participación de masa en X menor al 90%")
        if self.MP_y < 0.9:
            errors.append("Participación de masa en Y menor al 90%")
        
        # Validar límites de deriva
        if self.max_drift_x <= 0:
            errors.append("Límite de deriva en X debe ser mayor a 0")
        if self.max_drift_y <= 0:
            errors.append("Límite de deriva en Y debe ser mayor a 0")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    # Métodos de análisis
    def get_effective_reduction_factors(self) -> Tuple[float, float]:
        """Obtiene factores de reducción efectivos"""
        return self.Rx, self.Ry
    
    def get_fundamental_periods(self) -> Tuple[float, float]:
        """Obtiene periodos fundamentales"""
        return self.Tx, self.Ty
    
    def get_mass_participation(self) -> Tuple[float, float]:
        """Obtiene participación de masa"""
        return self.MP_x, self.MP_y
    
    def is_structure_regular(self) -> bool:
        """Verifica si la estructura es regular"""
        return self.Ia == 1.0 and self.Ip == 1.0
    
    def get_irregularity_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de irregularidades"""
        return {
            'is_regular': self.is_structure_regular(),
            'height_irregular': self.Ia < 1.0,
            'plan_irregular': self.Ip < 1.0,
            'Ia': self.Ia,
            'Ip': self.Ip,
            'total_factor': self.Ia * self.Ip
        }
    
    # Métodos de exportación/importación
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            'project': asdict(self.project),
            'loads': asdict(self.loads),
            'reduction_factors': {
                'Rx': self.Rx, 'Ry': self.Ry, 'Rox': self.Rox, 'Roy': self.Roy
            },
            'irregularity_factors': {'Ia': self.Ia, 'Ip': self.Ip},
            'periods': {'Tx': self.Tx, 'Ty': self.Ty},
            'mass_participation': {'MP_x': self.MP_x, 'MP_y': self.MP_y},
            'drift_limits': {'max_drift_x': self.max_drift_x, 'max_drift_y': self.max_drift_y},
            'modal_data': asdict(self.modal_data),
            'drift_data': asdict(self.drift_data),
            'normative_params': self.normative_params,
            'compatibility_fields': {
                'proyecto': self.proyecto,
                'ubicacion': self.ubicacion,
                'autor': self.autor,
                'fecha': self.fecha,
                'descripcion': self.descripcion,
                'modelamiento': self.modelamiento,
                'cargas': self.cargas
            }
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Carga desde diccionario"""
        try:
            # Cargar información del proyecto
            if 'project' in data:
                self.project = ProjectInfo(**data['project'])
            
            # Cargar patrones de carga
            if 'loads' in data:
                self.loads = LoadPatterns(**data['loads'])
            
            # Cargar factores de reducción
            if 'reduction_factors' in data:
                rf = data['reduction_factors']
                self.Rx = rf.get('Rx', self.Rx)
                self.Ry = rf.get('Ry', self.Ry)
                self.Rox = rf.get('Rox', self.Rox)
                self.Roy = rf.get('Roy', self.Roy)
            
            # Cargar factores de irregularidad
            if 'irregularity_factors' in data:
                if_data = data['irregularity_factors']
                self.Ia = if_data.get('Ia', self.Ia)
                self.Ip = if_data.get('Ip', self.Ip)
            
            # Cargar otros parámetros
            if 'periods' in data:
                periods = data['periods']
                self.Tx = periods.get('Tx', self.Tx)
                self.Ty = periods.get('Ty', self.Ty)
            
            if 'mass_participation' in data:
                mp = data['mass_participation']
                self.MP_x = mp.get('MP_x', self.MP_x)
                self.MP_y = mp.get('MP_y', self.MP_y)
            
            # Cargar campos de compatibilidad
            if 'compatibility_fields' in data:
                cf = data['compatibility_fields']
                self.proyecto = cf.get('proyecto', '')
                self.ubicacion = cf.get('ubicacion', '')
                self.autor = cf.get('autor', '')
                self.fecha = cf.get('fecha', '')
                self.descripcion = cf.get('descripcion', '')
                self.modelamiento = cf.get('modelamiento', '')
                self.cargas = cf.get('cargas', '')
            
            # Cargar parámetros específicos de normativas
            if 'normative_params' in data:
                self.normative_params = data['normative_params'].copy()
            
            logger.debug("Datos sísmicos cargados desde diccionario")
            
        except Exception as e:
            logger.error(f"Error cargando datos desde diccionario: {e}")
            raise
    
    def save_to_file(self, filepath: Union[str, Path], format: str = 'json'):
        """
        Guarda datos a archivo
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo
        format : str
            Formato de archivo ('json' o 'pickle')
        """
        filepath = Path(filepath)
        
        try:
            data = self.to_dict()
            
            if format.lower() == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"Formato no soportado: {format}")
            
            logger.info(f"Datos sísmicos guardados en: {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando datos a {filepath}: {e}")
            raise
    
    def load_from_file(self, filepath: Union[str, Path], format: str = 'json'):
        """
        Carga datos desde archivo
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo
        format : str
            Formato de archivo ('json' o 'pickle')
        """
        filepath = Path(filepath)
        
        try:
            if format.lower() == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif format.lower() == 'pickle':
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Formato no soportado: {format}")
            
            self.from_dict(data)
            logger.info(f"Datos sísmicos cargados desde: {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando datos desde {filepath}: {e}")
            raise
    
    # Métodos de compatibilidad con código existente
    def get_seismic_loads(self) -> Dict[str, str]:
        """Obtiene patrones de carga (compatibilidad)"""
        return self.loads.seism_loads.copy()


    # Métodos de compatibilidad con código existente
    def get_seismic_loads(self) -> Dict[str, str]:
        """Obtiene patrones de carga (compatibilidad)"""
        return self.loads.seism_loads.copy()


# NOTA: Las clases especializadas para normativas específicas se encuentran
# en sus respectivas carpetas/módulos:
# - PeruSeismicData -> seismic_common/normatives/peru/
# - BoliviaSeismicData -> seismic_common/normatives/bolivia/
# - ChileSeismicData -> seismic_common/normatives/chile/
# etc.

# Clase base abstracta para normativas específicas
class NormativeSeismicData(SeismicData, ABC):
    """
    Clase base abstracta para datos sísmicos específicos de normativas
    
    Las implementaciones específicas deben estar en sus módulos correspondientes:
    - seismic_common.normatives.peru.PeruSeismicData
    - seismic_common.normatives.bolivia.BoliviaSeismicData
    - etc.
    """
    
    def __init__(self, normative_name: str):
        super().__init__()
        self.normative_name = normative_name
        self.normative_version = ""
        self.normative_year = ""
        
        # Cada normativa debe definir sus parámetros específicos
        self._initialize_normative_parameters()
        
        logger.debug(f"NormativeSeismicData inicializado para normativa: {normative_name}")
    
    @abstractmethod
    def _initialize_normative_parameters(self):
        """
        Inicializa parámetros específicos de la normativa
        DEBE SER IMPLEMENTADO por cada normativa específica
        """
        pass
    
    @abstractmethod
    def validate_normative_parameters(self) -> Tuple[bool, List[str]]:
        """
        Valida parámetros específicos de la normativa
        DEBE SER IMPLEMENTADO por cada normativa específica
        
        Returns
        -------
        Tuple[bool, List[str]]
            (es_válido, lista_errores)
        """
        pass
    
    @abstractmethod
    def calculate_design_spectrum(self) -> Tuple[List[float], List[float]]:
        """
        Calcula espectro de diseño según la normativa
        DEBE SER IMPLEMENTADO por cada normativa específica
        
        Returns
        -------
        Tuple[List[float], List[float]]
            (periodos, aceleraciones_espectrales)
        """
        pass
    
    @abstractmethod
    def calculate_base_shear(self, total_weight: float) -> Tuple[float, float]:
        """
        Calcula cortante basal según la normativa
        DEBE SER IMPLEMENTADO por cada normativa específica
        
        Parameters
        ----------
        total_weight : float
            Peso sísmico total de la estructura
            
        Returns
        -------
        Tuple[float, float]
            (cortante_x, cortante_y)
        """
        pass
    
    def get_normative_info(self) -> Dict[str, str]:
        """Obtiene información de la normativa"""
        return {
            'name': self.normative_name,
            'version': self.normative_version,
            'year': self.normative_year
        }
    


# Funciones de utilidad para crear instancias específicas de normativas
def create_seismic_data_for_normative(normative: str) -> SeismicData:
    """
    Crea instancia de datos sísmicos para una normativa específica
    
    NOTA: Las implementaciones específicas de normativas están en sus respectivos
    proyectos (appPeru, appBolivia, etc.) y no en el código centralizado.
    
    Esta función sirve como interfaz común pero retorna la clase base.
    Cada proyecto específico debe implementar su propia factory function.
    
    Parameters
    ----------
    normative : str
        Nombre de la normativa ('peru', 'bolivia', 'chile', etc.)
        
    Returns
    -------
    SeismicData
        Instancia genérica de datos sísmicos
    """
    logger.info(f"Creando datos sísmicos genéricos para normativa: {normative}")
    
    # Crear instancia genérica con configuración básica
    seismic_data = SeismicData()
    
    # Configurar algunos valores por defecto según normativa conocida
    normative = normative.lower().strip()
    
    if normative == 'peru':
        seismic_data.project.name = "Proyecto Perú E.030"
        seismic_data.normative_params = {
            'normative_name': 'E.030',
            'country': 'PE',
            'default_drift_limit': 0.007
        }
    elif normative == 'bolivia':
        seismic_data.project.name = "Proyecto Bolivia NBC"
        seismic_data.normative_params = {
            'normative_name': 'NBC',
            'country': 'BO',
            'default_drift_limit': 0.007
        }
    elif normative == 'chile':
        seismic_data.project.name = "Proyecto Chile NCh"
        seismic_data.normative_params = {
            'normative_name': 'NCh433',
            'country': 'CL',
            'default_drift_limit': 0.002
        }
    else:
        logger.warning(f"Normativa '{normative}' no reconocida, usando configuración genérica")
        seismic_data.normative_params = {
            'normative_name': normative,
            'country': 'UNKNOWN'
        }
    
    return seismic_data


def get_available_normatives() -> List[str]:
    """
    Obtiene lista de normativas conocidas (no necesariamente implementadas)
    
    Las implementaciones específicas están en cada proyecto individual.
    
    Returns
    -------
    List[str]
        Lista de normativas conocidas
    """
    return ['generic', 'peru', 'bolivia', 'chile', 'colombia', 'ecuador']


# Funciones de utilidad para migración de código existente
def create_seismic_data_from_legacy(legacy_data: Dict[str, Any]) -> SeismicData:
    """
    Crea SeismicData desde estructura de datos heredada
    
    Parameters
    ----------
    legacy_data : dict
        Diccionario con datos en formato antiguo
        
    Returns
    -------
    SeismicData
        Instancia con datos migrados
    """
    seismic_data = SeismicData()
    
    try:
        # Migrar campos básicos
        if 'Rx' in legacy_data:
            seismic_data.Rx = float(legacy_data['Rx'])
        if 'Ry' in legacy_data:
            seismic_data.Ry = float(legacy_data['Ry'])
        if 'Rox' in legacy_data:
            seismic_data.Rox = float(legacy_data['Rox'])
        if 'Roy' in legacy_data:
            seismic_data.Roy = float(legacy_data['Roy'])
        
        # Migrar irregularidades
        if 'Ia' in legacy_data:
            seismic_data.Ia = float(legacy_data['Ia'])
        if 'Ip' in legacy_data:
            seismic_data.Ip = float(legacy_data['Ip'])
        
        # Migrar periodos
        if 'Tx' in legacy_data:
            seismic_data.Tx = float(legacy_data['Tx'])
        if 'Ty' in legacy_data:
            seismic_data.Ty = float(legacy_data['Ty'])
        
        # Migrar información del proyecto
        project_fields = ['proyecto', 'ubicacion', 'autor', 'fecha', 'descripcion']
        for field in project_fields:
            if field in legacy_data:
                setattr(seismic_data, field, str(legacy_data[field]))
        
        # Migrar cargas sísmicas
        if 'seism_loads' in legacy_data:
            seismic_data.set_load_patterns(legacy_data['seism_loads'])
        
        logger.debug("Datos heredados migrados exitosamente")
        
    except Exception as e:
        logger.error(f"Error migrando datos heredados: {e}")
        raise
    
    return seismic_data


def migrate_tables_to_dataframes(tables_dict: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Migra tablas heredadas a DataFrames estándar
    
    Parameters
    ----------
    tables_dict : dict
        Diccionario con tablas en formato heredado
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Diccionario con DataFrames estandarizados
    """
    migrated_tables = {}
    
    try:
        for table_name, table_data in tables_dict.items():
            if isinstance(table_data, pd.DataFrame):
                migrated_tables[table_name] = table_data.copy()
            elif isinstance(table_data, (list, tuple)) and len(table_data) > 0:
                # Convertir lista/tupla a DataFrame
                migrated_tables[table_name] = pd.DataFrame(table_data)
            elif isinstance(table_data, dict):
                # Convertir diccionario a DataFrame
                migrated_tables[table_name] = pd.DataFrame([table_data])
            else:
                logger.warning(f"No se pudo migrar tabla '{table_name}' de tipo {type(table_data)}")
        
        logger.debug(f"Migradas {len(migrated_tables)} tablas a DataFrames")
        
    except Exception as e:
        logger.error(f"Error migrando tablas: {e}")
        raise
    
    return migrated_tables


# Clases de compatibilidad con código existente
class SeismicLoads:
    """Clase para manejo de cargas sísmicas (compatibilidad)"""
    
    def __init__(self):
        self.seism_loads = {}
        self.load_patterns = {}
        self.combinations = {}
    
    def set_seism_loads(self, seism_loads: Dict[str, str]) -> None:
        """
        Establece las cargas sísmicas del modelo
        
        Parameters
        ----------
        seism_loads : Dict[str, str]
            Diccionario con nombres de cargas sísmicas
        """
        self.seism_loads = seism_loads.copy()
        logger.debug(f"Cargas sísmicas establecidas: {seism_loads}")


class SeismicTables:
    """Clase para almacenamiento de tablas de resultados sísmicos (compatibilidad)"""
    
    def __init__(self):
        # Tablas de análisis
        self.modal = pd.DataFrame()
        self.static_seism = pd.DataFrame()
        
        # Tablas de irregularidades
        self.rigidez_table = pd.DataFrame()
        self.torsion_table = pd.DataFrame()
        self.masa_table = pd.DataFrame()
        
        # Tablas de resultados
        self.story_drifts = pd.DataFrame()
        self.joint_displacements = pd.DataFrame()
        self.base_reactions = pd.DataFrame()


# Funciones de validación del módulo
def validate_seismic_data_model():
    """
    Valida el funcionamiento correcto del modelo de datos sísmicos
    
    Returns
    -------
    bool
        True si todas las validaciones pasan
    """
    try:
        # Probar creación de instancia básica
        basic_data = SeismicData()
        assert basic_data.Rx == 8.0
        assert basic_data.Ry == 8.0
        assert basic_data.Ia == 1.0
        assert basic_data.Ip == 1.0
        
        # Probar establecimiento de parámetros
        basic_data.set_reduction_factors(7.0, 6.0)
        assert basic_data.Rx == 7.0
        assert basic_data.Ry == 6.0
        
        basic_data.set_irregularity_factors(0.75, 0.8)
        assert basic_data.Ia == 0.75
        assert basic_data.Ip == 0.8
        
        # Probar validación
        is_valid, errors = basic_data.validate_basic_parameters()
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        # Probar serialización
        data_dict = basic_data.to_dict()
        assert isinstance(data_dict, dict)
        
        new_data = SeismicData()
        new_data.from_dict(data_dict)
        assert new_data.Rx == basic_data.Rx
        assert new_data.Ry == basic_data.Ry
        
        # Probar compatibilidad con datos heredados
        legacy_data = {
            'Rx': 8.0, 'Ry': 7.0, 'Ia': 0.75, 'Ip': 1.0,
            'proyecto': 'Test', 'autor': 'Test Author'
        }
        migrated_data = create_seismic_data_from_legacy(legacy_data)
        assert migrated_data.Rx == 8.0
        assert migrated_data.proyecto == 'Test'
        
        logger.info("✓ Validación del modelo de datos sísmicos exitosa")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en validación del modelo de datos sísmicos: {e}")
        return False


if __name__ == "__main__":
    # Ejecutar validaciones si el módulo se ejecuta directamente
    validate_seismic_data_model()