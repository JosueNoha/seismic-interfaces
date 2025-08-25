"""
Modelos centralizados de ubicaciones y zonificación sísmica
=========================================================

Este módulo proporciona modelos de datos centralizados para manejar ubicaciones
geográficas y su correspondiente zonificación sísmica según diferentes normativas.

Características principales:
- Modelo base genérico para ubicaciones jerárquicas
- Soporte para diferentes países y normativas
- Integración con bases de datos de zonificación sísmica
- Validación automática de datos de ubicación
- Compatibilidad con código existente
- Funciones de búsqueda y filtrado avanzado

Ejemplo de uso:
    ```python
    from seismic_common.models import LocationModel, SeismicZoneDatabase
    
    # Crear base de datos de ubicaciones
    locations = LocationModel()
    locations.load_from_csv('peru_locations.csv')
    locations.set_hierarchy(['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO'])
    
    # Buscar ubicación específica
    cusco_data = locations.find_location({
        'DEPARTAMENTO': 'CUSCO',
        'PROVINCIA': 'CUSCO', 
        'DISTRITO': 'CUSCO'
    })
    ```
"""

# Metadatos del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Modelos centralizados de ubicaciones y zonificación sísmica"
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

# Constantes para jerarquías de ubicación por país
LOCATION_HIERARCHIES = {
    'PE': ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO'],
    'BO': ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO'],
    'CL': ['REGION', 'PROVINCIA', 'COMUNA'],
    'CO': ['DEPARTAMENTO', 'MUNICIPIO'],
    'EC': ['PROVINCIA', 'CANTON', 'PARROQUIA'],
    'AR': ['PROVINCIA', 'DEPARTAMENTO', 'LOCALIDAD'],
    'MX': ['ESTADO', 'MUNICIPIO'],
    'US': ['STATE', 'COUNTY', 'CITY'],
    'BR': ['ESTADO', 'MUNICIPIO']
}

# Alias de compatibilidad para código existente
LEGACY_COLUMN_MAPPING = {
    'ZONA': 'ZONA_SISMICA',
    'Z': 'FACTOR_ZONA',
    'PGA': 'ACELERACION_PICO',
    'SOIL': 'TIPO_SUELO',
    'SUELO': 'TIPO_SUELO'
}


@dataclass
class LocationInfo:
    """Información básica de una ubicación geográfica"""
    name: str = ""
    level: str = ""  # 'departamento', 'provincia', 'distrito', etc.
    parent: str = ""  # Ubicación padre en la jerarquía
    country_code: str = ""
    coordinates: Tuple[float, float] = (0.0, 0.0)  # (latitud, longitud)
    altitude: float = 0.0
    population: int = 0
    area_km2: float = 0.0
    
    def __post_init__(self):
        """Validaciones después de la inicialización"""
        if self.coordinates != (0.0, 0.0):
            lat, lon = self.coordinates
            if not (-90 <= lat <= 90):
                logger.warning(f"Latitud fuera de rango: {lat}")
            if not (-180 <= lon <= 180):
                logger.warning(f"Longitud fuera de rango: {lon}")


@dataclass
class SeismicZoneInfo:
    """Información de zonificación sísmica para una ubicación"""
    zone_number: Union[int, str] = 0
    zone_factor: float = 0.0
    pga: float = 0.0  # Peak Ground Acceleration
    soil_type: str = ""
    normative: str = ""
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_zone_description(self) -> str:
        """Obtiene descripción textual de la zona sísmica"""
        zone_descriptions = {
            1: "Zona de sismicidad baja",
            2: "Zona de sismicidad media",
            3: "Zona de sismicidad alta",
            4: "Zona de sismicidad muy alta"
        }
        
        if isinstance(self.zone_number, int):
            return zone_descriptions.get(self.zone_number, f"Zona {self.zone_number}")
        else:
            return f"Zona {self.zone_number}"


class LocationModel:
    """
    Modelo centralizado para manejo de ubicaciones geográficas
    
    Proporciona funcionalidad para almacenar, buscar y validar ubicaciones
    con soporte para diferentes países y jerarquías administrativas.
    """
    
    def __init__(self, country_code: str = 'PE'):
        """
        Inicializa el modelo de ubicaciones
        
        Parameters
        ----------
        country_code : str
            Código del país (PE, BO, CL, etc.)
        """
        self.country_code = country_code.upper()
        self.data = pd.DataFrame()
        self.hierarchy = LOCATION_HIERARCHIES.get(self.country_code, ['LEVEL1', 'LEVEL2', 'LEVEL3'])
        self.metadata = {}
        
        # Índices para búsqueda rápida
        self._location_index = {}
        self._parent_child_index = {}
        
        logger.debug(f"LocationModel inicializado para país: {self.country_code}")
    
    def load_from_dataframe(self, df: pd.DataFrame, validate: bool = True):
        """
        Carga datos desde un DataFrame
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de ubicaciones
        validate : bool
            Si validar los datos después de cargar
        """
        self.data = df.copy()
        
        # Normalizar nombres de columnas
        self._normalize_column_names()
        
        # Validar jerarquía
        if validate:
            self._validate_hierarchy()
        
        # Construir índices
        self._build_indexes()
        
        logger.info(f"Cargadas {len(self.data)} ubicaciones desde DataFrame")
    
    def load_from_csv(self, filepath: Union[str, Path], encoding: str = 'utf-8', **kwargs):
        """
        Carga datos desde archivo CSV
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo CSV
        encoding : str
            Codificación del archivo
        **kwargs
            Argumentos adicionales para pandas.read_csv
        """
        try:
            filepath = Path(filepath)
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            self.load_from_dataframe(df)
            
            logger.info(f"Datos cargados desde CSV: {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando CSV {filepath}: {e}")
            raise
    
    def load_from_json(self, filepath: Union[str, Path]):
        """
        Carga datos desde archivo JSON
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo JSON
        """
        try:
            filepath = Path(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convertir a DataFrame según el formato
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'locations' in data:
                    df = pd.DataFrame(data['locations'])
                    self.metadata = data.get('metadata', {})
                else:
                    # Asumir estructura jerárquica
                    rows = self._flatten_hierarchical_dict(data)
                    df = pd.DataFrame(rows)
            else:
                raise ValueError("Formato JSON no soportado")
            
            self.load_from_dataframe(df)
            logger.info(f"Datos cargados desde JSON: {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando JSON {filepath}: {e}")
            raise
    
    def set_hierarchy(self, hierarchy: List[str]):
        """
        Establece la jerarquía administrativa
        
        Parameters
        ----------
        hierarchy : List[str]
            Lista de nombres de columnas en orden jerárquico
        """
        missing_cols = [col for col in hierarchy if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Columnas no encontradas en los datos: {missing_cols}")
        
        self.hierarchy = hierarchy.copy()
        self._build_indexes()
        
        logger.debug(f"Jerarquía establecida: {hierarchy}")
    
    def find_location(self, location: Dict[str, str], exact_match: bool = True) -> pd.DataFrame:
        """
        Busca ubicaciones que coincidan con los criterios
        
        Parameters
        ----------
        location : Dict[str, str]
            Criterios de búsqueda (ej: {'DEPARTAMENTO': 'CUSCO'})
        exact_match : bool
            Si requiere coincidencia exacta o permite parcial
            
        Returns
        -------
        pd.DataFrame
            Ubicaciones que coinciden con los criterios
        """
        filtered_data = self.data.copy()
        
        for column, value in location.items():
            if column not in self.data.columns:
                logger.warning(f"Columna '{column}' no existe en los datos")
                continue
            
            if exact_match:
                filtered_data = filtered_data[filtered_data[column] == value]
            else:
                # Búsqueda parcial (case-insensitive)
                mask = filtered_data[column].str.contains(value, case=False, na=False)
                filtered_data = filtered_data[mask]
        
        return filtered_data
    
    def get_children(self, parent_location: Dict[str, str], child_level: str) -> List[str]:
        """
        Obtiene ubicaciones hijas de un nivel específico
        
        Parameters
        ----------
        parent_location : Dict[str, str]
            Ubicación padre
        child_level : str
            Nivel de las ubicaciones hijas deseadas
            
        Returns
        -------
        List[str]
            Lista de ubicaciones hijas
        """
        if child_level not in self.hierarchy:
            return []
        
        filtered_data = self.find_location(parent_location)
        
        if filtered_data.empty:
            return []
        
        children = filtered_data[child_level].dropna().unique().tolist()
        return sorted(children)
    
    def get_full_hierarchy_path(self, location: Dict[str, str]) -> List[str]:
        """
        Obtiene la ruta completa en la jerarquía para una ubicación
        
        Parameters
        ----------
        location : Dict[str, str]
            Ubicación específica
            
        Returns
        -------
        List[str]
            Ruta jerárquica completa
        """
        path = []
        found_data = self.find_location(location)
        
        if not found_data.empty:
            row = found_data.iloc[0]
            for level in self.hierarchy:
                if level in row and pd.notna(row[level]):
                    path.append(str(row[level]))
        
        return path
    
    def validate_location(self, location: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Valida si una ubicación existe en la base de datos
        
        Parameters
        ----------
        location : Dict[str, str]
            Ubicación a validar
            
        Returns
        -------
        Tuple[bool, List[str]]
            (es_válida, lista_errores)
        """
        errors = []
        
        # Verificar que las columnas existen
        for column in location.keys():
            if column not in self.data.columns:
                errors.append(f"Columna '{column}' no existe")
        
        if errors:
            return False, errors
        
        # Verificar que la ubicación existe
        found_data = self.find_location(location)
        if found_data.empty:
            errors.append(f"Ubicación no encontrada: {location}")
            return False, errors
        
        # Verificar jerarquía válida
        row = found_data.iloc[0]
        for i, level in enumerate(self.hierarchy):
            if level in location:
                expected_value = location[level]
                actual_value = row.get(level)
                if str(actual_value) != str(expected_value):
                    errors.append(f"Valor inconsistente en {level}: esperado '{expected_value}', encontrado '{actual_value}'")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_location_options(self, level: str, filters: Dict[str, str] = None) -> List[str]:
        """
        Obtiene opciones disponibles para un nivel específico
        
        Parameters
        ----------
        level : str
            Nivel de la jerarquía
        filters : Dict[str, str], optional
            Filtros basados en niveles superiores
            
        Returns
        -------
        List[str]
            Lista de opciones disponibles
        """
        if level not in self.data.columns:
            return []
        
        filtered_data = self.data.copy()
        
        # Aplicar filtros
        if filters:
            for filter_col, filter_val in filters.items():
                if filter_col in self.data.columns:
                    filtered_data = filtered_data[filtered_data[filter_col] == filter_val]
        
        # Obtener opciones únicas y ordenadas
        options = filtered_data[level].dropna().unique().tolist()
        return sorted(options)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la base de datos de ubicaciones
        
        Returns
        -------
        Dict[str, Any]
            Estadísticas de la base de datos
        """
        stats = {
            'total_locations': len(self.data),
            'country_code': self.country_code,
            'hierarchy': self.hierarchy.copy(),
            'columns': list(self.data.columns),
            'levels_count': {}
        }
        
        # Contar elementos por nivel
        for level in self.hierarchy:
            if level in self.data.columns:
                stats['levels_count'][level] = self.data[level].nunique()
        
        return stats
    
    def _normalize_column_names(self):
        """Normaliza nombres de columnas para compatibilidad"""
        # Convertir a mayúsculas
        self.data.columns = self.data.columns.str.upper()
        
        # Aplicar mapeo de compatibilidad
        rename_mapping = {}
        for old_name, new_name in LEGACY_COLUMN_MAPPING.items():
            if old_name in self.data.columns and new_name not in self.data.columns:
                rename_mapping[old_name] = new_name
        
        if rename_mapping:
            self.data.rename(columns=rename_mapping, inplace=True)
            logger.debug(f"Columnas renombradas: {rename_mapping}")
    
    def _validate_hierarchy(self):
        """Valida que la jerarquía sea consistente"""
        missing_columns = [col for col in self.hierarchy if col not in self.data.columns]
        if missing_columns:
            logger.warning(f"Columnas de jerarquía faltantes: {missing_columns}")
            # Mantener solo las columnas que existen
            self.hierarchy = [col for col in self.hierarchy if col in self.data.columns]
    
    def _build_indexes(self):
        """Construye índices para búsqueda rápida"""
        self._location_index = {}
        self._parent_child_index = {}
        
        for idx, row in self.data.iterrows():
            # Índice de ubicación completa
            location_key = tuple(str(row.get(col, '')) for col in self.hierarchy)
            self._location_index[location_key] = idx
            
            # Índice padre-hijo
            for i in range(len(self.hierarchy) - 1):
                parent_col = self.hierarchy[i]
                child_col = self.hierarchy[i + 1]
                
                if parent_col in row and child_col in row:
                    parent_val = str(row[parent_col])
                    child_val = str(row[child_col])
                    
                    if parent_val not in self._parent_child_index:
                        self._parent_child_index[parent_val] = set()
                    self._parent_child_index[parent_val].add(child_val)
    
    def _flatten_hierarchical_dict(self, data: Dict, parent_path: List = None) -> List[Dict]:
        """Aplana un diccionario jerárquico a lista de registros"""
        if parent_path is None:
            parent_path = []
        
        rows = []
        for key, value in data.items():
            current_path = parent_path + [key]
            
            if isinstance(value, dict) and not any(isinstance(v, (str, int, float)) for v in value.values()):
                # Continúa la jerarquía
                rows.extend(self._flatten_hierarchical_dict(value, current_path))
            else:
                # Crear registro
                row = {}
                for i, level in enumerate(self.hierarchy):
                    if i < len(current_path):
                        row[level] = current_path[i]
                
                # Agregar datos adicionales
                if isinstance(value, dict):
                    row.update(value)
                
                rows.append(row)
        
        return rows
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            'country_code': self.country_code,
            'hierarchy': self.hierarchy,
            'data': self.data.to_dict('records'),
            'metadata': self.metadata
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Carga desde diccionario"""
        self.country_code = data.get('country_code', 'PE')
        self.hierarchy = data.get('hierarchy', LOCATION_HIERARCHIES.get(self.country_code, ['LEVEL1']))
        self.metadata = data.get('metadata', {})
        
        df = pd.DataFrame(data.get('data', []))
        if not df.empty:
            self.load_from_dataframe(df, validate=False)
    
    def save_to_file(self, filepath: Union[str, Path], format: str = 'json'):
        """
        Guarda datos a archivo
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo
        format : str
            Formato ('json', 'csv', 'pickle')
        """
        filepath = Path(filepath)
        
        try:
            if format.lower() == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            elif format.lower() == 'csv':
                self.data.to_csv(filepath, index=False, encoding='utf-8')
            elif format.lower() == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(self.to_dict(), f)
            else:
                raise ValueError(f"Formato no soportado: {format}")
            
            logger.info(f"Datos guardados en: {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando a {filepath}: {e}")
            raise


class SeismicZoneDatabase:
    """
    Base de datos especializada para zonificación sísmica
    
    Extiende LocationModel con funcionalidad específica para datos sísmicos
    """
    
    def __init__(self, country_code: str = 'PE'):
        """
        Inicializa la base de datos de zonificación sísmica
        
        Parameters
        ----------
        country_code : str
            Código del país
        """
        self.location_model = LocationModel(country_code)
        self.seismic_data_columns = ['ZONA_SISMICA', 'FACTOR_ZONA', 'ACELERACION_PICO', 'TIPO_SUELO']
        self.normative = ""
        
        logger.debug(f"SeismicZoneDatabase inicializado para país: {country_code}")
    
    def load_seismic_data(self, df: pd.DataFrame, normative: str = ""):
        """
        Carga datos de zonificación sísmica
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de ubicaciones y zonificación sísmica
        normative : str
            Normativa sísmica aplicable
        """
        self.location_model.load_from_dataframe(df)
        self.normative = normative
        
        # Validar que existen columnas sísmicas
        available_seismic_cols = [col for col in self.seismic_data_columns 
                                 if col in df.columns]
        
        if not available_seismic_cols:
            logger.warning("No se encontraron columnas de datos sísmicos")
        
        logger.info(f"Datos sísmicos cargados para normativa: {normative}")
    
    def get_seismic_zone(self, location: Dict[str, str]) -> Optional[SeismicZoneInfo]:
        """
        Obtiene información de zona sísmica para una ubicación
        
        Parameters
        ----------
        location : Dict[str, str]
            Ubicación específica
            
        Returns
        -------
        SeismicZoneInfo or None
            Información de zona sísmica
        """
        location_data = self.location_model.find_location(location)
        
        if location_data.empty:
            return None
        
        row = location_data.iloc[0]
        
        zone_info = SeismicZoneInfo(
            normative=self.normative
        )
        
        # Mapear datos disponibles
        if 'ZONA_SISMICA' in row:
            zone_info.zone_number = row['ZONA_SISMICA']
        elif 'ZONA' in row:
            zone_info.zone_number = row['ZONA']
        
        if 'FACTOR_ZONA' in row:
            zone_info.zone_factor = float(row['FACTOR_ZONA'])
        elif 'Z' in row:
            zone_info.zone_factor = float(row['Z'])
        
        if 'ACELERACION_PICO' in row:
            zone_info.pga = float(row['ACELERACION_PICO'])
        elif 'PGA' in row:
            zone_info.pga = float(row['PGA'])
        
        if 'TIPO_SUELO' in row:
            zone_info.soil_type = str(row['TIPO_SUELO'])
        elif 'SUELO' in row:
            zone_info.soil_type = str(row['SUELO'])
        
        # Datos adicionales
        additional_params = {}
        for col in row.index:
            if col not in self.location_model.hierarchy and col not in self.seismic_data_columns:
                additional_params[col] = row[col]
        
        zone_info.additional_params = additional_params
        
        return zone_info
    
    def find_locations_by_zone(self, zone_number: Union[int, str]) -> pd.DataFrame:
        """
        Encuentra ubicaciones en una zona sísmica específica
        
        Parameters
        ----------
        zone_number : int or str
            Número o identificador de zona sísmica
            
        Returns
        -------
        pd.DataFrame
            Ubicaciones en la zona especificada
        """
        data = self.location_model.data
        
        # Buscar en diferentes columnas posibles
        zone_columns = ['ZONA_SISMICA', 'ZONA', 'ZONE']
        
        for col in zone_columns:
            if col in data.columns:
                return data[data[col] == zone_number]
        
        return pd.DataFrame()
    
    def get_zone_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de zonificación sísmica
        
        Returns
        -------
        Dict[str, Any]
            Estadísticas de zonificación
        """
        stats = self.location_model.get_statistics()
        data = self.location_model.data
        
        # Estadísticas por zona sísmica
        zone_stats = {}
        zone_columns = ['ZONA_SISMICA', 'ZONA', 'ZONE']
        
        for col in zone_columns:
            if col in data.columns:
                zone_counts = data[col].value_counts().to_dict()
                zone_stats['zones'] = zone_counts
                break
        
        stats.update({
            'normative': self.normative,
            'seismic_stats': zone_stats
        })
        
        return stats


# Funciones de migración de código existente
def migrate_basedatos_zonificacion(bd_df: pd.DataFrame) -> SeismicZoneDatabase:
    """
    Migra datos de BaseDatos_Zonas_Sismicas existente
    
    Parameters
    ----------
    bd_df : pd.DataFrame
        DataFrame del sistema heredado
        
    Returns
    -------
    SeismicZoneDatabase
        Base de datos migrada
    """
    # Detectar país por columnas
    if 'DEPARTAMENTO' in bd_df.columns:
        country_code = 'PE' if 'DISTRITO' in bd_df.columns else 'BO'
    else:
        country_code = 'PE'  # Default
    
    db = SeismicZoneDatabase(country_code)
    
    # Mapear columnas heredadas
    column_mapping = {
        'ZONA(Z)': 'ZONA_SISMICA',
        'ZONA': 'ZONA_SISMICA'
    }
    
    migrated_df = bd_df.rename(columns=column_mapping)
    
    # Determinar normativa
    normative = 'E.030' if country_code == 'PE' else 'NBC'
    
    db.load_seismic_data(migrated_df, normative)
    
    logger.info(f"Migrados datos heredados para país: {country_code}")
    return db


# Clases de compatibilidad
class BaseDatos_Zonas_Sismicas:
    """Clase de compatibilidad con código existente"""
    
    def __init__(self, csv_path: Optional[str] = None):
        """Inicializa con datos heredados"""
        if csv_path:
            self.BD_Zonas_Sismicas = pd.read_csv(csv_path)
        else:
            # Crear DataFrame vacío con estructura esperada
            self.BD_Zonas_Sismicas = pd.DataFrame(columns=[
                'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'ZONA'
            ])


# Funciones de utilidad adicionales
def get_country_hierarchy(country_code: str) -> List[str]:
    """
    Obtiene la jerarquía administrativa estándar para un país
    
    Parameters
    ----------
    country_code : str
        Código del país
        
    Returns
    -------
    List[str]
        Jerarquía administrativa
    """
    return LOCATION_HIERARCHIES.get(country_code.upper(), ['LEVEL1', 'LEVEL2', 'LEVEL3'])


def create_location_from_address(address_parts: List[str], country_code: str) -> Dict[str, str]:
    """
    Crea diccionario de ubicación desde partes de dirección
    
    Parameters
    ----------
    address_parts : List[str]
        Partes de la dirección en orden jerárquico
    country_code : str
        Código del país
        
    Returns
    -------
    Dict[str, str]
        Diccionario de ubicación
    """
    hierarchy = get_country_hierarchy(country_code)
    location = {}
    
    for i, part in enumerate(address_parts):
        if i < len(hierarchy):
            location[hierarchy[i]] = part.strip().upper()
    
    return location


def validate_country_data_structure(df: pd.DataFrame, country_code: str) -> Tuple[bool, List[str]]:
    """
    Valida que un DataFrame tenga la estructura correcta para un país
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar
    country_code : str
        Código del país
        
    Returns
    -------
    Tuple[bool, List[str]]
        (es_válido, lista_errores)
    """
    errors = []
    expected_hierarchy = get_country_hierarchy(country_code)
    
    # Verificar columnas requeridas
    missing_columns = [col for col in expected_hierarchy if col not in df.columns]
    if missing_columns:
        errors.append(f"Columnas faltantes para {country_code}: {missing_columns}")
    
    # Verificar que no hay valores nulos en columnas principales
    for col in expected_hierarchy:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Columna '{col}' tiene {null_count} valores nulos")
    
    # Verificar estructura jerárquica
    if len(expected_hierarchy) > 1:
        for i in range(len(expected_hierarchy) - 1):
            parent_col = expected_hierarchy[i]
            child_col = expected_hierarchy[i + 1]
            
            if parent_col in df.columns and child_col in df.columns:
                # Verificar que cada hijo tiene exactamente un padre
                child_parent_counts = df.groupby(child_col)[parent_col].nunique()
                invalid_children = child_parent_counts[child_parent_counts > 1]
                
                if not invalid_children.empty:
                    errors.append(f"Elementos en '{child_col}' con múltiples padres en '{parent_col}': {list(invalid_children.index)}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def merge_location_databases(*databases: LocationModel) -> LocationModel:
    """
    Combina múltiples bases de datos de ubicaciones
    
    Parameters
    ----------
    *databases : LocationModel
        Bases de datos a combinar
        
    Returns
    -------
    LocationModel
        Base de datos combinada
    """
    if not databases:
        return LocationModel()
    
    # Usar el primer país como base
    combined = LocationModel(databases[0].country_code)
    combined_data = []
    
    for db in databases:
        if not db.data.empty:
            combined_data.append(db.data)
    
    if combined_data:
        merged_df = pd.concat(combined_data, ignore_index=True)
        # Eliminar duplicados
        hierarchy_cols = combined.hierarchy
        available_cols = [col for col in hierarchy_cols if col in merged_df.columns]
        
        if available_cols:
            merged_df = merged_df.drop_duplicates(subset=available_cols)
        
        combined.load_from_dataframe(merged_df)
    
    logger.info(f"Combinadas {len(databases)} bases de datos de ubicaciones")
    return combined


def export_to_geojson(location_model: LocationModel, output_path: Union[str, Path]):
    """
    Exporta ubicaciones a formato GeoJSON (requiere coordenadas)
    
    Parameters
    ----------
    location_model : LocationModel
        Modelo de ubicaciones
    output_path : str or Path
        Ruta del archivo de salida
    """
    try:
        import geojson
        from geojson import Feature, FeatureCollection, Point
    except ImportError:
        logger.error("Librería 'geojson' no disponible para exportación")
        return
    
    features = []
    
    for _, row in location_model.data.iterrows():
        # Buscar coordenadas en diferentes formatos
        lat, lon = 0.0, 0.0
        
        coord_columns = ['LAT', 'LATITUDE', 'LATITUD', 'LNG', 'LONGITUDE', 'LONGITUD']
        for col in coord_columns:
            if col in row:
                if 'LAT' in col.upper():
                    lat = float(row[col]) if pd.notna(row[col]) else 0.0
                else:
                    lon = float(row[col]) if pd.notna(row[col]) else 0.0
        
        if lat != 0.0 or lon != 0.0:
            # Crear propiedades
            properties = {}
            for col in location_model.hierarchy:
                if col in row and pd.notna(row[col]):
                    properties[col] = str(row[col])
            
            # Agregar datos adicionales
            for col in row.index:
                if col not in location_model.hierarchy and col not in coord_columns:
                    if pd.notna(row[col]):
                        properties[col] = str(row[col])
            
            # Crear feature
            point = Point((lon, lat))
            feature = Feature(geometry=point, properties=properties)
            features.append(feature)
    
    if features:
        feature_collection = FeatureCollection(features)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            geojson.dump(feature_collection, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exportadas {len(features)} ubicaciones a GeoJSON: {output_path}")
    else:
        logger.warning("No se encontraron coordenadas válidas para exportar")


def create_empty_location_model(country_code: str) -> LocationModel:
    """
    Crea un modelo de ubicaciones vacío para un país específico
    
    Parameters
    ----------
    country_code : str
        Código del país
        
    Returns
    -------
    LocationModel
        Modelo vacío con jerarquía configurada
    """
    model = LocationModel(country_code)
    
    # Crear DataFrame vacío con columnas de la jerarquía
    hierarchy = get_country_hierarchy(country_code)
    empty_df = pd.DataFrame(columns=hierarchy)
    model.load_from_dataframe(empty_df, validate=False)
    
    return model


def load_location_model_from_dict(data_dict: Dict[str, Any]) -> LocationModel:
    """
    Carga un modelo de ubicaciones desde diccionario
    
    Parameters
    ----------
    data_dict : Dict[str, Any]
        Diccionario con datos de ubicaciones
        
    Returns
    -------
    LocationModel
        Modelo cargado
    """
    country_code = data_dict.get('country_code', 'PE')
    model = LocationModel(country_code)
    model.from_dict(data_dict)
    return model


def create_location_lookup_table(location_model: LocationModel) -> Dict[str, Dict[str, Any]]:
    """
    Crea tabla de búsqueda rápida para ubicaciones
    
    Parameters
    ----------
    location_model : LocationModel
        Modelo de ubicaciones
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Tabla de búsqueda con claves concatenadas
    """
    lookup = {}
    
    for _, row in location_model.data.iterrows():
        # Crear clave única concatenando la jerarquía
        key_parts = []
        for level in location_model.hierarchy:
            if level in row and pd.notna(row[level]):
                key_parts.append(str(row[level]).upper())
        
        if key_parts:
            key = "|".join(key_parts)
            lookup[key] = row.to_dict()
    
    return lookup


def find_duplicates_in_hierarchy(location_model: LocationModel) -> pd.DataFrame:
    """
    Encuentra duplicados en la jerarquía de ubicaciones
    
    Parameters
    ----------
    location_model : LocationModel
        Modelo de ubicaciones
        
    Returns
    -------
    pd.DataFrame
        Ubicaciones duplicadas
    """
    hierarchy_cols = [col for col in location_model.hierarchy if col in location_model.data.columns]
    
    if not hierarchy_cols:
        return pd.DataFrame()
    
    # Encontrar duplicados basados en la jerarquía completa
    duplicates = location_model.data[location_model.data.duplicated(subset=hierarchy_cols, keep=False)]
    
    return duplicates.sort_values(hierarchy_cols)


def get_location_tree_structure(location_model: LocationModel) -> Dict[str, Any]:
    """
    Obtiene estructura de árbol de ubicaciones
    
    Parameters
    ----------
    location_model : LocationModel
        Modelo de ubicaciones
        
    Returns
    -------
    Dict[str, Any]
        Estructura de árbol jerárquica
    """
    if not location_model.hierarchy or location_model.data.empty:
        return {}
    
    tree = {}
    
    for _, row in location_model.data.iterrows():
        current_level = tree
        
        for level in location_model.hierarchy:
            if level in row and pd.notna(row[level]):
                value = str(row[level])
                if value not in current_level:
                    current_level[value] = {}
                current_level = current_level[value]
    
    return tree


def validate_location_consistency(location_model: LocationModel) -> Dict[str, Any]:
    """
    Valida la consistencia de datos de ubicación
    
    Parameters
    ----------
    location_model : LocationModel
        Modelo de ubicaciones
        
    Returns
    -------
    Dict[str, Any]
        Reporte de validación
    """
    report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    if location_model.data.empty:
        report['errors'].append("No hay datos para validar")
        report['is_valid'] = False
        return report
    
    # Validar jerarquía
    hierarchy_cols = [col for col in location_model.hierarchy if col in location_model.data.columns]
    
    if not hierarchy_cols:
        report['errors'].append("No se encontraron columnas de jerarquía válidas")
        report['is_valid'] = False
        return report
    
    # Verificar valores nulos
    for col in hierarchy_cols:
        null_count = location_model.data[col].isnull().sum()
        if null_count > 0:
            report['warnings'].append(f"Columna '{col}' tiene {null_count} valores nulos")
    
    # Verificar duplicados
    duplicates = find_duplicates_in_hierarchy(location_model)
    if not duplicates.empty:
        report['errors'].append(f"Se encontraron {len(duplicates)} ubicaciones duplicadas")
        report['is_valid'] = False
    
    # Verificar consistencia jerárquica
    if len(hierarchy_cols) > 1:
        for i in range(len(hierarchy_cols) - 1):
            parent_col = hierarchy_cols[i]
            child_col = hierarchy_cols[i + 1]
            
            # Verificar que cada hijo pertenece a un solo padre
            child_parent_map = location_model.data.groupby(child_col)[parent_col].nunique()
            inconsistent = child_parent_map[child_parent_map > 1]
            
            if not inconsistent.empty:
                report['errors'].append(f"Inconsistencia jerárquica entre '{parent_col}' y '{child_col}': {len(inconsistent)} elementos")
                report['is_valid'] = False
    
    # Estadísticas
    report['statistics'] = location_model.get_statistics()
    
    if report['errors']:
        report['is_valid'] = False
    
    return report


# Validación del módulo
def validate_location_model():
    """
    Valida el funcionamiento correcto del modelo de ubicaciones
    
    Returns
    -------
    bool
        True si todas las validaciones pasan
    """
    try:
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'DEPARTAMENTO': ['CUSCO', 'CUSCO', 'LIMA'],
            'PROVINCIA': ['CUSCO', 'ANTA', 'LIMA'], 
            'DISTRITO': ['CUSCO', 'ANTA', 'MIRAFLORES'],
            'ZONA_SISMICA': [4, 3, 4],
            'FACTOR_ZONA': [0.45, 0.35, 0.45]
        })
        
        # Probar LocationModel básico
        location_model = LocationModel('PE')
        location_model.load_from_dataframe(test_data)
        
        assert len(location_model.data) == 3
        assert location_model.hierarchy == ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']
        
        # Probar búsqueda
        cusco_data = location_model.find_location({'DEPARTAMENTO': 'CUSCO'})
        assert len(cusco_data) == 2
        
        # Probar obtener hijos
        provincias = location_model.get_children({'DEPARTAMENTO': 'CUSCO'}, 'PROVINCIA')
        assert len(provincias) == 2
        assert 'CUSCO' in provincias and 'ANTA' in provincias
        
        # Probar validación
        is_valid, errors = location_model.validate_location({
            'DEPARTAMENTO': 'CUSCO', 
            'PROVINCIA': 'CUSCO', 
            'DISTRITO': 'CUSCO'
        })
        assert is_valid
        
        # Probar SeismicZoneDatabase
        seismic_db = SeismicZoneDatabase('PE')
        seismic_db.load_seismic_data(test_data, 'E.030')
        
        zone_info = seismic_db.get_seismic_zone({
            'DEPARTAMENTO': 'CUSCO',
            'PROVINCIA': 'CUSCO',
            'DISTRITO': 'CUSCO'
        })
        
        assert zone_info is not None
        assert zone_info.zone_number == 4
        assert zone_info.zone_factor == 0.45
        assert zone_info.normative == 'E.030'
        
        # Probar serialización
        data_dict = location_model.to_dict()
        assert isinstance(data_dict, dict)
        
        new_model = LocationModel('PE')
        new_model.from_dict(data_dict)
        assert len(new_model.data) == 3
        
        # Probar migración de datos heredados
        legacy_df = pd.DataFrame({
            'DEPARTAMENTO': ['CUSCO'],
            'PROVINCIA': ['CUSCO'],
            'DISTRITO': ['CUSCO'],
            'ZONA(Z)': [4]
        })
        
        migrated_db = migrate_basedatos_zonificacion(legacy_df)
        assert isinstance(migrated_db, SeismicZoneDatabase)
        assert 'ZONA_SISMICA' in migrated_db.location_model.data.columns
        
        # Probar funciones de utilidad
        hierarchy = get_country_hierarchy('PE')
        assert hierarchy == ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']
        
        address = create_location_from_address(['Cusco', 'Cusco', 'Cusco'], 'PE')
        assert address['DEPARTAMENTO'] == 'CUSCO'
        
        is_valid, errors = validate_country_data_structure(test_data, 'PE')
        assert is_valid
        
        # Probar estructura de árbol
        tree = get_location_tree_structure(location_model)
        assert 'CUSCO' in tree
        assert 'LIMA' in tree
        
        # Probar tabla de búsqueda
        lookup = create_location_lookup_table(location_model)
        assert len(lookup) == 3
        
        # Probar validación de consistencia
        validation_report = validate_location_consistency(location_model)
        assert validation_report['is_valid']
        
        logger.info("✓ Validación del modelo de ubicaciones exitosa")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en validación del modelo de ubicaciones: {e}")
        return False


if __name__ == "__main__":
    # Ejecutar validaciones si el módulo se ejecuta directamente
    validate_location_model()