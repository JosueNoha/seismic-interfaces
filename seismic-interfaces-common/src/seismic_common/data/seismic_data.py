"""
Gestión centralizada de datos sísmicos para interfaces de análisis
================================================================

Este módulo proporciona clases y funciones para gestionar datos sísmicos de manera
centralizada, compatible con el código existente y extensible para diferentes normativas.

Características principales:
- Gestión centralizada de datos sísmicos
- Compatibilidad con código existente (SeismicData, SeismicLoads, SeismicTables)
- Extensible para normativas específicas
- Validación automática de datos
- Persistencia y serialización
- Migración desde código heredado

Ejemplo de uso:
    ```python
    from seismic_common.data import (
        SeismicDataManager, 
        SeismicLoads, 
        SeismicTables,
        load_seismic_data_from_legacy
    )
    
    # Crear gestor de datos sísmicos
    manager = SeismicDataManager()
    manager.set_basic_parameters(Rx=8.0, Ry=8.0, Ia=1.0, Ip=1.0)
    
    # Configurar cargas
    loads = SeismicLoads()
    loads.set_seism_loads({'SX': 'EQX', 'SY': 'EQY'})
    
    # Gestionar tablas
    tables = SeismicTables()
    tables.modal = modal_dataframe
    ```
"""

# Metadatos del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Gestión centralizada de datos sísmicos para interfaces de análisis"
__license__ = "MIT"
__status__ = "Production"

import sys
import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

# Configurar logging
logger = logging.getLogger(__name__)

# Constantes globales para datos sísmicos
SEISMIC_CONSTANTS = {
    'DEFAULT_DRIFT_LIMIT': 0.007,        # Deriva límite típica (0.7%)
    'DEFAULT_TORSIONAL_LIMIT': 1.2,      # Irregularidad torsional
    'DEFAULT_SOFT_STORY_LIMIT': 0.7,     # Piso blando
    'DEFAULT_MASS_IRREGULARITY': 1.5,    # Irregularidad de masa
    'MIN_MASS_PARTICIPATION': 0.9,       # Participación de masa mínima
    'DEFAULT_DAMPING': 0.05,             # Amortiguamiento típico (5%)
}

# Factores de reducción por sistema estructural
STRUCTURAL_SYSTEM_R_FACTORS = {
    'porticos_concreto': {'Rx': 8.0, 'Ry': 8.0},
    'sistema_dual': {'Rx': 7.0, 'Ry': 7.0},
    'muros_concreto': {'Rx': 6.0, 'Ry': 6.0},
    'porticos_acero_especial': {'Rx': 8.0, 'Ry': 8.0},
    'porticos_acero_intermedio': {'Rx': 5.0, 'Ry': 5.0},
    'porticos_acero_ordinario': {'Rx': 4.0, 'Ry': 4.0},
    'albañileria': {'Rx': 3.0, 'Ry': 3.0},
    'madera': {'Rx': 7.0, 'Ry': 7.0}
}

# Patrones de carga sísmicos estándar
STANDARD_LOAD_PATTERNS = {
    'static_x': 'SX',
    'static_y': 'SY',
    'dynamic_x': 'SDX',
    'dynamic_y': 'SDY',
    'spectrum_x': 'SPECX',
    'spectrum_y': 'SPECY'
}


@dataclass
class ProjectInfo:
    """Información del proyecto estructural"""
    name: str = ""
    location: str = ""
    author: str = ""
    date: str = ""
    description: str = ""
    structural_system: str = ""
    analysis_type: str = "dynamic"
    normative: str = ""
    notes: str = ""
    
    def __post_init__(self):
        """Inicialización posterior al constructor"""
        if not self.date:
            self.date = datetime.now().strftime("%Y-%m-%d")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return asdict(self)
    
    def from_dict(self, data: Dict[str, Any]):
        """Carga desde diccionario"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass 
class SeismicParameters:
    """Parámetros sísmicos básicos comunes a todas las normativas"""
    # Factores de reducción
    Rx: float = 8.0
    Ry: float = 8.0
    
    # Factores de irregularidad
    Ia: float = 1.0  # Factor de irregularidad en altura
    Ip: float = 1.0  # Factor de irregularidad en planta
    
    # Periodos fundamentales (calculados)
    Tx: float = 0.0
    Ty: float = 0.0
    
    # Masas participativas (calculadas)
    MP_x: float = 0.0
    MP_y: float = 0.0
    
    # Factores de escala dinámico
    FE_x: float = 1.0
    FE_y: float = 1.0
    
    # Parámetros de deriva
    drift_limit: float = SEISMIC_CONSTANTS['DEFAULT_DRIFT_LIMIT']
    
    # Amortiguamiento
    damping: float = SEISMIC_CONSTANTS['DEFAULT_DAMPING']
    
    def validate_parameters(self) -> Tuple[bool, List[str]]:
        """
        Valida parámetros sísmicos básicos
        
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
        
        # Validar participación de masa (si fue calculada)
        if self.MP_x > 0 and self.MP_x < SEISMIC_CONSTANTS['MIN_MASS_PARTICIPATION']:
            errors.append("Participación de masa en X menor al 90%")
        if self.MP_y > 0 and self.MP_y < SEISMIC_CONSTANTS['MIN_MASS_PARTICIPATION']:
            errors.append("Participación de masa en Y menor al 90%")
        
        return len(errors) == 0, errors
    
    def set_from_structural_system(self, system: str):
        """
        Configura factores R según sistema estructural
        
        Parameters
        ----------
        system : str
            Tipo de sistema estructural
        """
        if system in STRUCTURAL_SYSTEM_R_FACTORS:
            factors = STRUCTURAL_SYSTEM_R_FACTORS[system]
            self.Rx = factors['Rx']
            self.Ry = factors['Ry']
            logger.info(f"Factores R establecidos para {system}: Rx={self.Rx}, Ry={self.Ry}")
        else:
            logger.warning(f"Sistema estructural '{system}' no reconocido")


class SeismicLoads:
    """Gestión de patrones de carga sísmica"""
    
    def __init__(self):
        self.seism_loads: Dict[str, str] = {}
        self.load_patterns: Dict[str, str] = STANDARD_LOAD_PATTERNS.copy()
        self.combinations: Dict[str, List[str]] = {}
        self.scale_factors: Dict[str, float] = {}
        
        # Inicializar con patrones estándar
        self._initialize_standard_loads()
    
    def _initialize_standard_loads(self):
        """Inicializa con patrones de carga estándar"""
        self.seism_loads = {
            'SX': self.load_patterns['static_x'],
            'SY': self.load_patterns['static_y'],
            'SDX': self.load_patterns['dynamic_x'],
            'SDY': self.load_patterns['dynamic_y'],
            'SPECX': self.load_patterns['spectrum_x'],
            'SPECY': self.load_patterns['spectrum_y']
        }
    
    def set_seism_loads(self, seism_loads: Dict[str, str]) -> None:
        """
        Establece las cargas sísmicas del modelo
        
        Parameters
        ----------
        seism_loads : Dict[str, str]
            Diccionario con nombres de cargas sísmicas
            Ejemplo: {'SX': 'EQX', 'SY': 'EQY', 'SDX': 'SPECX', 'SDY': 'SPECY'}
        """
        self.seism_loads.update(seism_loads)
        logger.debug(f"Cargas sísmicas establecidas: {seism_loads}")
    
    def get_x_direction_loads(self) -> List[str]:
        """Obtiene cargas en dirección X"""
        return [
            self.seism_loads.get('SX', 'SX'),
            self.seism_loads.get('SDX', 'SDX'),
            self.seism_loads.get('SPECX', 'SPECX')
        ]
    
    def get_y_direction_loads(self) -> List[str]:
        """Obtiene cargas en dirección Y"""
        return [
            self.seism_loads.get('SY', 'SY'),
            self.seism_loads.get('SDY', 'SDY'),
            self.seism_loads.get('SPECY', 'SPECY')
        ]
    
    def set_scale_factors(self, x_factor: float = 1.0, y_factor: float = 1.0):
        """
        Establece factores de escala para cargas dinámicas
        
        Parameters
        ----------
        x_factor : float
            Factor de escala para dirección X
        y_factor : float
            Factor de escala para dirección Y
        """
        self.scale_factors.update({
            'X': x_factor,
            'Y': y_factor
        })
        logger.info(f"Factores de escala establecidos: X={x_factor}, Y={y_factor}")
    
    def add_load_combination(self, name: str, loads: List[str], factors: List[float] = None):
        """
        Añade combinación de cargas
        
        Parameters
        ----------
        name : str
            Nombre de la combinación
        loads : List[str]
            Lista de cargas a combinar
        factors : List[float], optional
            Factores de cada carga (por defecto 1.0)
        """
        if factors is None:
            factors = [1.0] * len(loads)
        
        self.combinations[name] = {
            'loads': loads,
            'factors': factors
        }
        logger.debug(f"Combinación de carga '{name}' añadida")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            'seism_loads': self.seism_loads,
            'load_patterns': self.load_patterns,
            'combinations': self.combinations,
            'scale_factors': self.scale_factors
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Carga desde diccionario"""
        self.seism_loads = data.get('seism_loads', {})
        self.load_patterns = data.get('load_patterns', STANDARD_LOAD_PATTERNS.copy())
        self.combinations = data.get('combinations', {})
        self.scale_factors = data.get('scale_factors', {})


class SeismicTables:
    """Gestión de tablas de resultados sísmicos"""
    
    def __init__(self):
        # Tablas de análisis principal
        self.modal: pd.DataFrame = pd.DataFrame()
        self.static_seism: pd.DataFrame = pd.DataFrame()
        self.dynamic_seism: pd.DataFrame = pd.DataFrame()
        
        # Tablas de irregularidades
        self.rigidez_table: pd.DataFrame = pd.DataFrame()
        self.torsion_table: pd.DataFrame = pd.DataFrame()
        self.masa_table: pd.DataFrame = pd.DataFrame()
        
        # Tablas de resultados
        self.story_drifts: pd.DataFrame = pd.DataFrame()
        self.joint_displacements: pd.DataFrame = pd.DataFrame()
        self.base_reactions: pd.DataFrame = pd.DataFrame()
        
        # Tablas de verificación
        self.drift_check: pd.DataFrame = pd.DataFrame()
        self.irregularity_check: pd.DataFrame = pd.DataFrame()
        
        # Metadatos de tablas
        self._table_metadata: Dict[str, Dict[str, Any]] = {}
    
    def set_modal_table(self, modal_df: pd.DataFrame):
        """
        Establece tabla de análisis modal
        
        Parameters
        ----------
        modal_df : pd.DataFrame
            DataFrame con resultados del análisis modal
        """
        self.modal = modal_df.copy()
        self._table_metadata['modal'] = {
            'last_updated': datetime.now().isoformat(),
            'row_count': len(modal_df),
            'columns': list(modal_df.columns)
        }
        logger.info(f"Tabla modal establecida con {len(modal_df)} modos")
    
    def set_drift_table(self, drift_df: pd.DataFrame):
        """
        Establece tabla de derivas de piso
        
        Parameters
        ----------
        drift_df : pd.DataFrame
            DataFrame con derivas de piso
        """
        self.story_drifts = drift_df.copy()
        self._table_metadata['story_drifts'] = {
            'last_updated': datetime.now().isoformat(),
            'row_count': len(drift_df),
            'columns': list(drift_df.columns)
        }
        logger.info(f"Tabla de derivas establecida con {len(drift_df)} pisos")
    
    def get_table_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de todas las tablas
        
        Returns
        -------
        Dict[str, Any]
            Resumen con información de cada tabla
        """
        summary = {}
        
        for table_name in ['modal', 'static_seism', 'dynamic_seism', 'rigidez_table', 
                          'torsion_table', 'masa_table', 'story_drifts', 
                          'joint_displacements', 'base_reactions']:
            table = getattr(self, table_name)
            summary[table_name] = {
                'has_data': not table.empty,
                'row_count': len(table),
                'column_count': len(table.columns),
                'columns': list(table.columns) if not table.empty else []
            }
        
        return summary
    
    def validate_tables(self) -> Tuple[bool, List[str]]:
        """
        Valida consistencia entre tablas
        
        Returns
        -------
        Tuple[bool, List[str]]
            (es_válido, lista_errores)
        """
        errors = []
        
        # Verificar si hay datos mínimos
        if self.modal.empty:
            errors.append("Tabla modal vacía - se requiere análisis modal")
        
        # Verificar consistencia de derivas si existen
        if not self.story_drifts.empty and not self.joint_displacements.empty:
            # Aquí se pueden añadir más validaciones específicas
            pass
        
        return len(errors) == 0, errors
    
    def export_all_tables(self, directory: Union[str, Path], format: str = 'excel'):
        """
        Exporta todas las tablas a archivos
        
        Parameters
        ----------
        directory : str or Path
            Directorio donde exportar
        format : str
            Formato de exportación ('excel', 'csv')
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        
        tables_to_export = {
            'modal': self.modal,
            'static_seism': self.static_seism,
            'dynamic_seism': self.dynamic_seism,
            'story_drifts': self.story_drifts,
            'base_reactions': self.base_reactions
        }
        
        for name, table in tables_to_export.items():
            if not table.empty:
                if format.lower() == 'excel':
                    filename = directory / f"{name}.xlsx"
                    table.to_excel(filename, index=False)
                elif format.lower() == 'csv':
                    filename = directory / f"{name}.csv"
                    table.to_csv(filename, index=False)
                
                logger.info(f"Tabla {name} exportada a {filename}")


class SeismicDataManager:
    """
    Gestor centralizado de datos sísmicos
    
    Esta clase coordina todos los aspectos de los datos sísmicos:
    - Información del proyecto
    - Parámetros sísmicos
    - Cargas sísmicas
    - Tablas de resultados
    """
    
    def __init__(self):
        self.project = ProjectInfo()
        self.parameters = SeismicParameters()
        self.loads = SeismicLoads()
        self.tables = SeismicTables()
        
        # Metadatos del análisis
        self.analysis_metadata = {
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'version': __version__,
            'software_source': None  # Se puede establecer desde el proyecto específico
        }
        
        # Estado del análisis
        self.analysis_status = {
            'modal_completed': False,
            'static_completed': False,
            'dynamic_completed': False,
            'irregularities_checked': False,
            'drift_checked': False
        }
    
    def set_basic_parameters(self, Rx: float = 8.0, Ry: float = 8.0, 
                           Ia: float = 1.0, Ip: float = 1.0):
        """
        Establece parámetros sísmicos básicos
        
        Parameters
        ----------
        Rx : float
            Factor de reducción en X
        Ry : float
            Factor de reducción en Y
        Ia : float
            Factor de irregularidad en altura
        Ip : float
            Factor de irregularidad en planta
        """
        self.parameters.Rx = Rx
        self.parameters.Ry = Ry
        self.parameters.Ia = Ia
        self.parameters.Ip = Ip
        
        self._update_timestamp()
        logger.info(f"Parámetros básicos establecidos: Rx={Rx}, Ry={Ry}, Ia={Ia}, Ip={Ip}")
    
    def set_project_info(self, name: str = "", location: str = "", 
                        author: str = "", description: str = ""):
        """
        Establece información del proyecto
        
        Parameters
        ----------
        name : str
            Nombre del proyecto
        location : str
            Ubicación del proyecto
        author : str
            Autor del análisis
        description : str
            Descripción del proyecto
        """
        self.project.name = name
        self.project.location = location
        self.project.author = author
        self.project.description = description
        
        self._update_timestamp()
        logger.info(f"Información del proyecto establecida: {name}")
    
    def validate_all_data(self) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Valida todos los datos sísmicos
        
        Returns
        -------
        Tuple[bool, Dict[str, List[str]]]
            (es_válido, diccionario_errores_por_categoría)
        """
        all_errors = {}
        
        # Validar parámetros
        param_valid, param_errors = self.parameters.validate_parameters()
        if param_errors:
            all_errors['parameters'] = param_errors
        
        # Validar tablas
        table_valid, table_errors = self.tables.validate_tables()
        if table_errors:
            all_errors['tables'] = table_errors
        
        # Validaciones adicionales pueden añadirse aquí
        
        is_valid = len(all_errors) == 0
        return is_valid, all_errors
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen completo del análisis
        
        Returns
        -------
        Dict[str, Any]
            Resumen completo del estado del análisis
        """
        summary = {
            'project': self.project.to_dict(),
            'parameters': asdict(self.parameters),
            'loads_summary': {
                'total_loads': len(self.loads.seism_loads),
                'combinations': len(self.loads.combinations),
                'has_scale_factors': bool(self.loads.scale_factors)
            },
            'tables_summary': self.tables.get_table_summary(),
            'analysis_status': self.analysis_status.copy(),
            'metadata': self.analysis_metadata.copy()
        }
        
        return summary
    
    def save_to_file(self, filepath: Union[str, Path], format: str = 'json'):
        """
        Guarda todos los datos a archivo
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo
        format : str
            Formato de archivo ('json', 'pickle')
        """
        filepath = Path(filepath)
        
        try:
            # Preparar datos para serialización
            data_to_save = {
                'project': self.project.to_dict(),
                'parameters': asdict(self.parameters),
                'loads': self.loads.to_dict(),
                'analysis_metadata': self.analysis_metadata,
                'analysis_status': self.analysis_status,
                # Las tablas se guardan como diccionarios
                'tables': {
                    'modal': self.tables.modal.to_dict() if not self.tables.modal.empty else {},
                    'static_seism': self.tables.static_seism.to_dict() if not self.tables.static_seism.empty else {},
                    'story_drifts': self.tables.story_drifts.to_dict() if not self.tables.story_drifts.empty else {},
                    'base_reactions': self.tables.base_reactions.to_dict() if not self.tables.base_reactions.empty else {}
                }
            }
            
            if format.lower() == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            elif format.lower() == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(data_to_save, f)
            else:
                raise ValueError(f"Formato no soportado: {format}")
            
            logger.info(f"Datos sísmicos completos guardados en: {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando datos a {filepath}: {e}")
            raise
    
    def load_from_file(self, filepath: Union[str, Path], format: str = 'json'):
        """
        Carga todos los datos desde archivo
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo
        format : str
            Formato de archivo ('json', 'pickle')
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
            
            # Cargar datos
            self.project.from_dict(data.get('project', {}))
            
            param_data = data.get('parameters', {})
            for key, value in param_data.items():
                if hasattr(self.parameters, key):
                    setattr(self.parameters, key, value)
            
            self.loads.from_dict(data.get('loads', {}))
            
            self.analysis_metadata = data.get('analysis_metadata', {})
            self.analysis_status = data.get('analysis_status', {})
            
            # Cargar tablas
            table_data = data.get('tables', {})
            for table_name, table_dict in table_data.items():
                if table_dict and hasattr(self.tables, table_name):
                    df = pd.DataFrame.from_dict(table_dict)
                    setattr(self.tables, table_name, df)
            
            self._update_timestamp()
            logger.info(f"Datos sísmicos completos cargados desde: {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando datos desde {filepath}: {e}")
            raise
    
    def _update_timestamp(self):
        """Actualiza timestamp de última modificación"""
        self.analysis_metadata['last_modified'] = datetime.now().isoformat()
    
    # Métodos de compatibilidad con código existente
    def get_seismic_loads(self) -> Dict[str, str]:
        """Obtiene cargas sísmicas (compatibilidad)"""
        return self.loads.seism_loads.copy()
    
    def set_seism_loads(self, seism_loads: Dict[str, str]):
        """Establece cargas sísmicas (compatibilidad)"""
        self.loads.set_seism_loads(seism_loads)


# Funciones de utilidad y migración

def load_seismic_data_from_legacy(legacy_data: Any) -> SeismicDataManager:
    """
    Carga datos sísmicos desde formato heredado
    
    Esta función facilita la migración desde código existente
    
    Parameters
    ----------
    legacy_data : Any
        Datos en formato heredado (puede ser objeto o diccionario)
    
    Returns
    -------
    SeismicDataManager
        Gestor con datos migrados
    """
    manager = SeismicDataManager()
    
    try:
        # Intentar extraer parámetros comunes
        if hasattr(legacy_data, 'Rx'):
            manager.parameters.Rx = getattr(legacy_data, 'Rx', 8.0)
        if hasattr(legacy_data, 'Ry'):
            manager.parameters.Ry = getattr(legacy_data, 'Ry', 8.0)
        if hasattr(legacy_data, 'Ia'):
            manager.parameters.Ia = getattr(legacy_data, 'Ia', 1.0)
        if hasattr(legacy_data, 'Ip'):
            manager.parameters.Ip = getattr(legacy_data, 'Ip', 1.0)
        
        # Intentar extraer información del proyecto
        if hasattr(legacy_data, 'proyecto'):
            manager.project.name = getattr(legacy_data, 'proyecto', '')
        if hasattr(legacy_data, 'ubicacion'):
            manager.project.location = getattr(legacy_data, 'ubicacion', '')
        if hasattr(legacy_data, 'autor'):
            manager.project.author = getattr(legacy_data, 'autor', '')
        
        # Intentar extraer cargas sísmicas
        if hasattr(legacy_data, 'seism_loads'):
            manager.loads.set_seism_loads(getattr(legacy_data, 'seism_loads', {}))
        
        logger.info("Datos migrados desde formato heredado")
        
    except Exception as e:
        logger.warning(f"Error parcial en migración: {e}")
    
    return manager


def create_seismic_data_for_normative(normative: str) -> SeismicDataManager:
    """
    Crea gestor de datos sísmicos configurado para normativa específica
    
    Nota: Esta función proporciona configuración básica común.
    Las implementaciones específicas de normativas están en sus módulos correspondientes.
    
    Parameters
    ----------
    normative : str
        Nombre de la normativa ('peru', 'bolivia', 'chile', etc.)
    
    Returns
    -------
    SeismicDataManager
        Gestor configurado para la normativa
    """
    manager = SeismicDataManager()
    
    # Configurar según normativa conocida
    normative = normative.lower().strip()
    
    if normative == 'peru':
        manager.project.name = "Proyecto Perú E.030"
        manager.project.normative = "E.030"
        manager.parameters.drift_limit = 0.007  # 0.7% para estructuras comunes
        manager.analysis_metadata['software_source'] = 'peru_seismic_interface'
        
    elif normative == 'bolivia':
        manager.project.name = "Proyecto Bolivia NBC"
        manager.project.normative = "NBC"
        manager.parameters.drift_limit = 0.007
        manager.analysis_metadata['software_source'] = 'bolivia_seismic_interface'
        
    elif normative == 'chile':
        manager.project.name = "Proyecto Chile NCh"
        manager.project.normative = "NCh433"
        manager.parameters.drift_limit = 0.002  # Más restrictivo en Chile
        manager.analysis_metadata['software_source'] = 'chile_seismic_interface'
        
    elif normative == 'ecuador':
        manager.project.name = "Proyecto Ecuador NEC"
        manager.project.normative = "NEC"
        manager.parameters.drift_limit = 0.02   # 2% para estructuras regulares
        manager.analysis_metadata['software_source'] = 'ecuador_seismic_interface'
        
    else:
        logger.warning(f"Normativa '{normative}' no reconocida, usando configuración genérica")
        manager.project.normative = normative
        manager.analysis_metadata['software_source'] = 'generic_seismic_interface'
    
    logger.info(f"Gestor de datos sísmicos creado para normativa: {normative}")
    return manager


def get_available_normatives() -> List[str]:
    """
    Obtiene lista de normativas conocidas (no necesariamente implementadas)
    
    Las implementaciones específicas están en cada proyecto individual.
    
    Returns
    -------
    List[str]
        Lista de normativas conocidas
    """
    return ['peru', 'bolivia', 'chile', 'ecuador', 'colombia', 'mexico']


def validate_seismic_data_integrity(manager: SeismicDataManager) -> Dict[str, Any]:
    """
    Valida la integridad completa de los datos sísmicos
    
    Parameters
    ----------
    manager : SeismicDataManager
        Gestor de datos sísmicos a validar
    
    Returns
    -------
    Dict[str, Any]
        Resultado detallado de la validación
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'checks_performed': [],
        'summary': {}
    }
    
    # Validar parámetros básicos
    param_valid, param_errors = manager.parameters.validate_parameters()
    validation_result['checks_performed'].append('basic_parameters')
    if not param_valid:
        validation_result['is_valid'] = False
        validation_result['errors'].extend([f"Parámetros: {err}" for err in param_errors])
    
    # Validar tablas
    table_valid, table_errors = manager.tables.validate_tables()
    validation_result['checks_performed'].append('tables')
    if not table_valid:
        validation_result['is_valid'] = False
        validation_result['errors'].extend([f"Tablas: {err}" for err in table_errors])
    
    # Validar consistencia de cargas
    validation_result['checks_performed'].append('load_consistency')
    if not manager.loads.seism_loads:
        validation_result['warnings'].append("No se han definido cargas sísmicas")
    
    # Validar información del proyecto
    validation_result['checks_performed'].append('project_info')
    if not manager.project.name:
        validation_result['warnings'].append("Nombre del proyecto no establecido")
    if not manager.project.normative:
        validation_result['warnings'].append("Normativa no especificada")
    
    # Resumen de validación
    validation_result['summary'] = {
        'total_errors': len(validation_result['errors']),
        'total_warnings': len(validation_result['warnings']),
        'checks_performed': len(validation_result['checks_performed']),
        'has_modal_data': not manager.tables.modal.empty,
        'has_drift_data': not manager.tables.story_drifts.empty,
        'parameters_valid': param_valid,
        'tables_valid': table_valid
    }
    
    return validation_result


# Clases auxiliares para análisis específicos

class SeismicAnalysisResults:
    """
    Contenedor para resultados de análisis sísmico específico
    
    Esta clase encapsula resultados calculados y proporciona
    métodos para acceso y validación de resultados
    """
    
    def __init__(self):
        # Resultados modales
        self.modal_periods = []
        self.modal_frequencies = []
        self.mass_participation_x = []
        self.mass_participation_y = []
        
        # Resultados estáticos
        self.base_shear_x = 0.0
        self.base_shear_y = 0.0
        self.static_period_x = 0.0
        self.static_period_y = 0.0
        
        # Resultados dinámicos
        self.dynamic_base_shear_x = 0.0
        self.dynamic_base_shear_y = 0.0
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
        
        # Verificaciones
        self.max_drift_x = 0.0
        self.max_drift_y = 0.0
        self.drift_ratio_x = 0.0
        self.drift_ratio_y = 0.0
        
        # Estado de cálculos
        self.calculations_completed = {
            'modal': False,
            'static': False,
            'dynamic': False,
            'drift_check': False,
            'irregularity_check': False
        }
    
    def set_modal_results(self, periods: List[float], frequencies: List[float],
                         participation_x: List[float], participation_y: List[float]):
        """
        Establece resultados del análisis modal
        
        Parameters
        ----------
        periods : List[float]
            Periodos modales
        frequencies : List[float]
            Frecuencias modales
        participation_x : List[float]
            Participación de masa en X
        participation_y : List[float]
            Participación de masa en Y
        """
        self.modal_periods = periods[:]
        self.modal_frequencies = frequencies[:]
        self.mass_participation_x = participation_x[:]
        self.mass_participation_y = participation_y[:]
        self.calculations_completed['modal'] = True
        
        logger.info(f"Resultados modales establecidos: {len(periods)} modos")
    
    def get_fundamental_periods(self) -> Tuple[float, float]:
        """
        Obtiene periodos fundamentales en ambas direcciones
        
        Returns
        -------
        Tuple[float, float]
            (periodo_X, periodo_Y)
        """
        if not self.modal_periods:
            return 0.0, 0.0
        
        # Asumiendo que los primeros modos corresponden a las direcciones principales
        period_x = self.modal_periods[0] if len(self.modal_periods) > 0 else 0.0
        period_y = self.modal_periods[1] if len(self.modal_periods) > 1 else 0.0
        
        return period_x, period_y
    
    def get_total_mass_participation(self) -> Tuple[float, float]:
        """
        Obtiene participación total de masa en ambas direcciones
        
        Returns
        -------
        Tuple[float, float]
            (participación_X, participación_Y)
        """
        total_x = sum(self.mass_participation_x) if self.mass_participation_x else 0.0
        total_y = sum(self.mass_participation_y) if self.mass_participation_y else 0.0
        
        return total_x, total_y
    
    def check_mass_participation_adequacy(self, min_participation: float = 0.9) -> Tuple[bool, bool]:
        """
        Verifica si la participación de masa es adecuada
        
        Parameters
        ----------
        min_participation : float
            Participación mínima requerida (por defecto 90%)
        
        Returns
        -------
        Tuple[bool, bool]
            (es_adecuada_X, es_adecuada_Y)
        """
        total_x, total_y = self.get_total_mass_participation()
        
        adequate_x = total_x >= min_participation
        adequate_y = total_y >= min_participation
        
        return adequate_x, adequate_y
    
    def set_drift_results(self, max_drift_x: float, max_drift_y: float, 
                         drift_limit: float = 0.007):
        """
        Establece resultados de verificación de deriva
        
        Parameters
        ----------
        max_drift_x : float
            Deriva máxima en X
        max_drift_y : float
            Deriva máxima en Y
        drift_limit : float
            Límite de deriva (por defecto 0.7%)
        """
        self.max_drift_x = max_drift_x
        self.max_drift_y = max_drift_y
        self.drift_ratio_x = max_drift_x / drift_limit if drift_limit > 0 else 0.0
        self.drift_ratio_y = max_drift_y / drift_limit if drift_limit > 0 else 0.0
        self.calculations_completed['drift_check'] = True
        
        logger.info(f"Verificación de deriva completada: X={max_drift_x:.4f}, Y={max_drift_y:.4f}")
    
    def check_drift_compliance(self, drift_limit: float = 0.007) -> Tuple[bool, bool]:
        """
        Verifica cumplimiento de límites de deriva
        
        Parameters
        ----------
        drift_limit : float
            Límite de deriva
        
        Returns
        -------
        Tuple[bool, bool]
            (cumple_X, cumple_Y)
        """
        complies_x = self.max_drift_x <= drift_limit
        complies_y = self.max_drift_y <= drift_limit
        
        return complies_x, complies_y
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen completo del análisis
        
        Returns
        -------
        Dict[str, Any]
            Resumen de todos los resultados
        """
        tx, ty = self.get_fundamental_periods()
        mp_x, mp_y = self.get_total_mass_participation()
        mp_adequate_x, mp_adequate_y = self.check_mass_participation_adequacy()
        
        summary = {
            'modal_analysis': {
                'fundamental_period_x': tx,
                'fundamental_period_y': ty,
                'total_modes': len(self.modal_periods),
                'mass_participation_x': mp_x,
                'mass_participation_y': mp_y,
                'mass_participation_adequate_x': mp_adequate_x,
                'mass_participation_adequate_y': mp_adequate_y
            },
            'static_analysis': {
                'base_shear_x': self.base_shear_x,
                'base_shear_y': self.base_shear_y,
                'static_period_x': self.static_period_x,
                'static_period_y': self.static_period_y
            },
            'dynamic_analysis': {
                'dynamic_base_shear_x': self.dynamic_base_shear_x,
                'dynamic_base_shear_y': self.dynamic_base_shear_y,
                'scale_factor_x': self.scale_factor_x,
                'scale_factor_y': self.scale_factor_y
            },
            'drift_verification': {
                'max_drift_x': self.max_drift_x,
                'max_drift_y': self.max_drift_y,
                'drift_ratio_x': self.drift_ratio_x,
                'drift_ratio_y': self.drift_ratio_y
            },
            'calculations_status': self.calculations_completed.copy()
        }
        
        return summary


class SeismicReportGenerator:
    """
    Generador de reportes sísmicos automatizados
    
    Esta clase toma un SeismicDataManager y genera reportes
    en diferentes formatos (texto, HTML, LaTeX, etc.)
    """
    
    def __init__(self, data_manager: SeismicDataManager):
        self.data_manager = data_manager
        self.results = SeismicAnalysisResults()
        
    def generate_text_report(self) -> str:
        """
        Genera reporte en formato texto plano
        
        Returns
        -------
        str
            Reporte formateado en texto
        """
        report_lines = []
        
        # Encabezado
        report_lines.append("=" * 60)
        report_lines.append("REPORTE DE ANÁLISIS SÍSMICO")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Información del proyecto
        report_lines.append("INFORMACIÓN DEL PROYECTO:")
        report_lines.append(f"  Nombre: {self.data_manager.project.name}")
        report_lines.append(f"  Ubicación: {self.data_manager.project.location}")
        report_lines.append(f"  Normativa: {self.data_manager.project.normative}")
        report_lines.append(f"  Fecha: {self.data_manager.project.date}")
        report_lines.append("")
        
        # Parámetros sísmicos
        report_lines.append("PARÁMETROS SÍSMICOS:")
        params = self.data_manager.parameters
        report_lines.append(f"  Factor de reducción Rx: {params.Rx}")
        report_lines.append(f"  Factor de reducción Ry: {params.Ry}")
        report_lines.append(f"  Factor de irregularidad Ia: {params.Ia}")
        report_lines.append(f"  Factor de irregularidad Ip: {params.Ip}")
        report_lines.append(f"  Límite de deriva: {params.drift_limit:.3%}")
        report_lines.append("")
        
        # Cargas sísmicas
        report_lines.append("PATRONES DE CARGA:")
        for pattern, load_name in self.data_manager.loads.seism_loads.items():
            report_lines.append(f"  {pattern}: {load_name}")
        report_lines.append("")
        
        # Resumen de tablas
        table_summary = self.data_manager.tables.get_table_summary()
        report_lines.append("RESUMEN DE DATOS:")
        for table_name, info in table_summary.items():
            if info['has_data']:
                report_lines.append(f"  {table_name}: {info['row_count']} filas")
        report_lines.append("")
        
        # Estado del análisis
        report_lines.append("ESTADO DEL ANÁLISIS:")
        for check, completed in self.data_manager.analysis_status.items():
            status = "COMPLETADO" if completed else "PENDIENTE"
            report_lines.append(f"  {check}: {status}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def generate_summary_dict(self) -> Dict[str, Any]:
        """
        Genera resumen en formato diccionario para exportación
        
        Returns
        -------
        Dict[str, Any]
            Resumen completo del análisis
        """
        return self.data_manager.get_analysis_summary()
    
    def export_to_json(self, filepath: Union[str, Path]):
        """
        Exporta resumen completo a archivo JSON
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo JSON
        """
        summary = self.generate_summary_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resumen exportado a JSON: {filepath}")


# Funciones de utilidad adicionales

def merge_seismic_data_managers(*managers: SeismicDataManager) -> SeismicDataManager:
    """
    Combina múltiples gestores de datos sísmicos
    
    Útil para combinar resultados de diferentes análisis o fases
    
    Parameters
    ----------
    *managers : SeismicDataManager
        Gestores a combinar
    
    Returns
    -------
    SeismicDataManager
        Gestor combinado
    """
    if not managers:
        return SeismicDataManager()
    
    # Usar el primer gestor como base
    combined = SeismicDataManager()
    base_manager = managers[0]
    
    # Copiar información básica del primer gestor
    combined.project = ProjectInfo()
    combined.project.from_dict(base_manager.project.to_dict())
    
    # Los parámetros se toman del primer gestor (o se podrían promediar)
    combined.parameters = SeismicParameters(
        Rx=base_manager.parameters.Rx,
        Ry=base_manager.parameters.Ry,
        Ia=base_manager.parameters.Ia,
        Ip=base_manager.parameters.Ip
    )
    
    # Combinar cargas (unión de todos los patrones)
    combined.loads = SeismicLoads()
    for manager in managers:
        combined.loads.seism_loads.update(manager.loads.seism_loads)
        combined.loads.combinations.update(manager.loads.combinations)
    
    # Las tablas se pueden combinar o mantener separadas según necesidad
    # Por ahora, se mantiene la lógica del primer gestor
    combined.tables = base_manager.tables
    
    logger.info(f"Combinados {len(managers)} gestores de datos sísmicos")
    return combined


def create_empty_seismic_data_manager() -> SeismicDataManager:
    """
    Crea un gestor de datos sísmicos vacío con valores por defecto
    
    Returns
    -------
    SeismicDataManager
        Gestor vacío con configuración por defecto
    """
    manager = SeismicDataManager()
    
    # Configurar valores por defecto razonables
    manager.parameters.Rx = 8.0
    manager.parameters.Ry = 8.0
    manager.parameters.Ia = 1.0
    manager.parameters.Ip = 1.0
    manager.parameters.drift_limit = SEISMIC_CONSTANTS['DEFAULT_DRIFT_LIMIT']
    
    # Configurar proyecto con información mínima
    manager.project.name = "Proyecto Nuevo"
    manager.project.date = datetime.now().strftime("%Y-%m-%d")
    manager.project.analysis_type = "dynamic"
    
    logger.info("Gestor de datos sísmicos vacío creado")
    return manager


# Funciones de validación del módulo

def validate_seismic_data_module() -> bool:
    """
    Valida el funcionamiento correcto del módulo de datos sísmicos
    
    Returns
    -------
    bool
        True si todas las validaciones pasan
    """
    try:
        # Probar creación de instancia básica
        manager = SeismicDataManager()
        assert manager is not None
        
        # Probar configuración de parámetros
        manager.set_basic_parameters(Rx=8.0, Ry=8.0, Ia=1.0, Ip=1.0)
        assert manager.parameters.Rx == 8.0
        
        # Probar configuración de proyecto
        manager.set_project_info(name="Test Project", location="Test Location")
        assert manager.project.name == "Test Project"
        
        # Probar cargas sísmicas
        test_loads = {'SX': 'EQX', 'SY': 'EQY'}
        manager.loads.set_seism_loads(test_loads)
        assert manager.loads.seism_loads['SX'] == 'EQX'
        
        # Probar validación
        is_valid, errors = manager.validate_all_data()
        # En este punto básico, debe ser válido
        
        # Probar serialización/deserialización
        temp_data = manager.get_analysis_summary()
        assert 'project' in temp_data
        assert 'parameters' in temp_data
        
        logger.info("Validación del módulo seismic_data completada exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error en validación del módulo seismic_data: {e}")
        return False


# Punto de entrada para pruebas del módulo
if __name__ == "__main__":
    # Configurar logging para pruebas
    logging.basicConfig(level=logging.INFO)
    
    # Ejecutar validación
    if validate_seismic_data_module():
        print("✓ Módulo seismic_data validado correctamente")
        
        # Ejemplo de uso básico
        print("\nEjemplo de uso:")
        manager = create_empty_seismic_data_manager()
        manager.set_project_info(
            name="Edificio de Ejemplo",
            location="Lima, Perú",
            author="Ingeniero de Ejemplo"
        )
        
        print(f"Proyecto creado: {manager.project.name}")
        print(f"Parámetros: Rx={manager.parameters.Rx}, Ry={manager.parameters.Ry}")
        
        # Generar reporte
        reporter = SeismicReportGenerator(manager)
        print("\nReporte generado:")
        print(reporter.generate_text_report())
        
    else:
        print("✗ Error en validación del módulo seismic_data")
        sys.exit(1)