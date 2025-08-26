"""
Base de datos centralizada de ubicaciones y zonificación sísmica
==============================================================

Este módulo proporciona gestión centralizada de bases de datos de ubicaciones
geográficas con información de zonificación sísmica para diferentes países.

Características principales:
- Gestión jerárquica de ubicaciones (país/departamento/provincia/distrito)
- Integración de zonificación sísmica por normativas
- Compatibilidad con código existente (BaseDatos_Zonas_Sismicas)
- Búsqueda y filtrado avanzado
- Validación de datos geográficos
- Exportación e importación múltiple formato
- Cache inteligente para optimización

Ejemplo de uso:
    ```python
    from seismic_common.data import (
        LocationDatabase,
        SeismicZoneDatabase,
        load_peru_locations,
        migrate_legacy_database
    )
    
    # Crear base de datos de ubicaciones
    db = LocationDatabase(country_code='PE')
    db.load_from_csv('peru_locations.csv')
    
    # Buscar ubicación específica
    location = db.find_location(
        departamento='CUSCO',
        provincia='CUSCO',
        distrito='CUSCO'
    )
    
    # Obtener información sísmica
    seismic_info = location.get_seismic_zone_info()
    ```
"""

# Metadatos del módulo
__version__ = "1.0.0"
__author__ = "Proyecto Interfaces Sísmicas"
__description__ = "Base de datos centralizada de ubicaciones y zonificación sísmica"
__license__ = "MIT"
__status__ = "Production"

import sys
import os
import json
import pickle
import sqlite3
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

# Constantes para jerarquías administrativas por país
COUNTRY_HIERARCHIES = {
    'PE': ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO'],          # Perú
    'BO': ['DEPARTAMENTO', 'PROVINCIA', 'MUNICIPIO'],         # Bolivia
    'CL': ['REGION', 'PROVINCIA', 'COMUNA'],                  # Chile
    'CO': ['DEPARTAMENTO', 'MUNICIPIO'],                      # Colombia
    'EC': ['PROVINCIA', 'CANTON', 'PARROQUIA'],              # Ecuador
    'AR': ['PROVINCIA', 'DEPARTAMENTO', 'LOCALIDAD'],        # Argentina
    'MX': ['ESTADO', 'MUNICIPIO'],                           # México
    'US': ['STATE', 'COUNTY', 'CITY'],                       # Estados Unidos
    'BR': ['ESTADO', 'MUNICIPIO'],                           # Brasil
    'CR': ['PROVINCIA', 'CANTON', 'DISTRITO'],               # Costa Rica
    'GT': ['DEPARTAMENTO', 'MUNICIPIO']                      # Guatemala
}

# Mapeo de aliases de columnas para compatibilidad
COLUMN_ALIASES = {
    'ZONA': 'ZONA_SISMICA',
    'ZONA(Z)': 'ZONA_SISMICA',
    'Z': 'FACTOR_ZONA',
    'FACTOR_Z': 'FACTOR_ZONA',
    'PGA': 'ACELERACION_PICO',
    'ACELERACION': 'ACELERACION_PICO',
    'SUELO': 'TIPO_SUELO',
    'SOIL': 'TIPO_SUELO',
    'TIPO_S': 'TIPO_SUELO',
    'LAT': 'LATITUD',
    'LON': 'LONGITUD',
    'LONGITUDE': 'LONGITUD',
    'LATITUDE': 'LATITUD'
}

# Información de normativas sísmicas por país
SEISMIC_NORMATIVES = {
    'PE': {'name': 'E.030', 'zones': [1, 2, 3, 4], 'factor_range': (0.1, 0.45)},
    'BO': {'name': 'NBC', 'zones': [1, 2, 3], 'factor_range': (0.15, 0.30)},
    'CL': {'name': 'NCh433', 'zones': ['A0', 'A1', 'A2', 'A3'], 'factor_range': (0.2, 0.4)},
    'CO': {'name': 'NSR-10', 'zones': ['Low', 'Medium', 'High'], 'factor_range': (0.1, 0.35)},
    'EC': {'name': 'NEC', 'zones': ['I', 'II', 'III', 'IV', 'V', 'VI'], 'factor_range': (0.15, 0.5)},
    'MX': {'name': 'CFE', 'zones': ['A', 'B', 'C', 'D'], 'factor_range': (0.02, 0.4)},
    'US': {'name': 'ASCE7', 'zones': 'Continuous', 'factor_range': (0.0, 2.0)}
}


@dataclass
class LocationRecord:
    """Registro individual de ubicación geográfica"""
    # Información básica
    name: str = ""
    level: str = ""  # 'departamento', 'provincia', 'distrito', etc.
    parent_path: List[str] = field(default_factory=list)
    country_code: str = ""
    
    # Coordenadas geográficas
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    
    # Información demográfica
    population: int = 0
    area_km2: float = 0.0
    
    # Información sísmica
    seismic_zone: Union[int, str] = 0
    zone_factor: float = 0.0
    peak_acceleration: float = 0.0
    soil_type: str = ""
    
    # Metadatos
    source: str = ""
    last_updated: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validaciones post-inicialización"""
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()
        
        # Validar coordenadas
        if self.latitude != 0.0 and not (-90 <= self.latitude <= 90):
            logger.warning(f"Latitud fuera de rango válido: {self.latitude}")
        
        if self.longitude != 0.0 and not (-180 <= self.longitude <= 180):
            logger.warning(f"Longitud fuera de rango válido: {self.longitude}")
    
    def get_full_path(self) -> str:
        """Obtiene la ruta completa de la ubicación"""
        path_parts = self.parent_path + [self.name]
        return " > ".join(path_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return asdict(self)
    
    def from_dict(self, data: Dict[str, Any]):
        """Carga desde diccionario"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class LocationDatabase:
    """
    Base de datos centralizada de ubicaciones geográficas
    
    Gestiona ubicaciones jerárquicas con información geográfica y sísmica
    """
    
    def __init__(self, country_code: str = 'PE', enable_cache: bool = True):
        """
        Inicializa la base de datos de ubicaciones
        
        Parameters
        ----------
        country_code : str
            Código del país (PE, BO, CL, etc.)
        enable_cache : bool
            Habilitar cache para optimización
        """
        self.country_code = country_code.upper()
        self.hierarchy = COUNTRY_HIERARCHIES.get(self.country_code, ['LEVEL1', 'LEVEL2', 'LEVEL3'])
        self.normative_info = SEISMIC_NORMATIVES.get(self.country_code, {})
        
        # Datos principales
        self.data = pd.DataFrame()
        self.records = []  # Lista de LocationRecord
        
        # Cache para optimización
        self.enable_cache = enable_cache
        self._search_cache = {} if enable_cache else None
        self._hierarchy_cache = {} if enable_cache else None
        
        # Metadatos
        self.metadata = {
            'country_code': self.country_code,
            'hierarchy': self.hierarchy,
            'normative': self.normative_info.get('name', 'Unknown'),
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'total_records': 0,
            'data_sources': []
        }
        
        logger.info(f"LocationDatabase inicializada para {self.country_code}")
    
    def load_from_csv(self, csv_path: Union[str, Path], encoding: str = 'utf-8'):
        """
        Carga datos desde archivo CSV
        
        Parameters
        ----------
        csv_path : str or Path
            Ruta al archivo CSV
        encoding : str
            Codificación del archivo
        """
        csv_path = Path(csv_path)
        
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            self._load_from_dataframe(df)
            
            # Actualizar metadatos
            self.metadata['data_sources'].append({
                'type': 'csv',
                'path': str(csv_path),
                'loaded_at': datetime.now().isoformat(),
                'records': len(df)
            })
            
            logger.info(f"Datos cargados desde CSV: {csv_path} ({len(df)} registros)")
            
        except Exception as e:
            logger.error(f"Error cargando CSV {csv_path}: {e}")
            raise
    
    def load_from_dataframe(self, df: pd.DataFrame, source: str = "dataframe"):
        """
        Carga datos desde DataFrame
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de ubicaciones
        source : str
            Descripción de la fuente de datos
        """
        self._load_from_dataframe(df.copy())
        
        # Actualizar metadatos
        self.metadata['data_sources'].append({
            'type': 'dataframe',
            'source': source,
            'loaded_at': datetime.now().isoformat(),
            'records': len(df)
        })
        
        logger.info(f"Datos cargados desde DataFrame: {source} ({len(df)} registros)")
    
    def _load_from_dataframe(self, df: pd.DataFrame):
        """Carga datos internamente desde DataFrame"""
        # Normalizar nombres de columnas
        df = self._normalize_column_names(df)
        
        # Validar estructura jerárquica
        missing_hierarchy = [col for col in self.hierarchy if col not in df.columns]
        if missing_hierarchy:
            logger.warning(f"Columnas de jerarquía faltantes: {missing_hierarchy}")
        
        # Almacenar DataFrame principal
        self.data = df
        
        # Convertir a registros estructurados
        self._dataframe_to_records(df)
        
        # Limpiar cache
        self._clear_cache()
        
        # Actualizar metadatos
        self.metadata['last_modified'] = datetime.now().isoformat()
        self.metadata['total_records'] = len(df)
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres de columnas usando aliases"""
        column_mapping = {}
        
        for col in df.columns:
            col_upper = col.upper().strip()
            if col_upper in COLUMN_ALIASES:
                column_mapping[col] = COLUMN_ALIASES[col_upper]
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            logger.debug(f"Columnas normalizadas: {column_mapping}")
        
        return df
    
    def _dataframe_to_records(self, df: pd.DataFrame):
        """Convierte DataFrame a lista de LocationRecord"""
        self.records = []
        
        for idx, row in df.iterrows():
            record = LocationRecord(country_code=self.country_code)
            
            # Construir ruta jerárquica
            parent_path = []
            current_name = ""
            
            for level_idx, level_name in enumerate(self.hierarchy):
                if level_name in row:
                    value = str(row[level_name]).strip()
                    if value and value != 'nan':
                        if level_idx == len(self.hierarchy) - 1:
                            # Último nivel es el nombre
                            current_name = value
                        else:
                            parent_path.append(value)
            
            record.name = current_name
            record.parent_path = parent_path
            record.level = self.hierarchy[-1] if self.hierarchy else "unknown"
            
            # Mapear coordenadas geográficas
            if 'LATITUD' in row:
                try:
                    record.latitude = float(row['LATITUD'])
                except (ValueError, TypeError):
                    record.latitude = 0.0
            
            if 'LONGITUD' in row:
                try:
                    record.longitude = float(row['LONGITUD'])
                except (ValueError, TypeError):
                    record.longitude = 0.0
            
            if 'ALTITUD' in row:
                try:
                    record.altitude = float(row['ALTITUD'])
                except (ValueError, TypeError):
                    record.altitude = 0.0
            
            # Mapear información demográfica
            if 'POBLACION' in row:
                try:
                    record.population = int(row['POBLACION'])
                except (ValueError, TypeError):
                    record.population = 0
            
            if 'AREA_KM2' in row:
                try:
                    record.area_km2 = float(row['AREA_KM2'])
                except (ValueError, TypeError):
                    record.area_km2 = 0.0
            
            # Mapear información sísmica
            if 'ZONA_SISMICA' in row:
                record.seismic_zone = row['ZONA_SISMICA']
            
            if 'FACTOR_ZONA' in row:
                try:
                    record.zone_factor = float(row['FACTOR_ZONA'])
                except (ValueError, TypeError):
                    record.zone_factor = 0.0
            
            if 'ACELERACION_PICO' in row:
                try:
                    record.peak_acceleration = float(row['ACELERACION_PICO'])
                except (ValueError, TypeError):
                    record.peak_acceleration = 0.0
            
            if 'TIPO_SUELO' in row:
                record.soil_type = str(row['TIPO_SUELO'])
            
            # Datos adicionales
            additional_data = {}
            for col in row.index:
                if col not in self.hierarchy and col not in [
                    'LATITUD', 'LONGITUD', 'ALTITUD', 'POBLACION', 'AREA_KM2',
                    'ZONA_SISMICA', 'FACTOR_ZONA', 'ACELERACION_PICO', 'TIPO_SUELO'
                ]:
                    additional_data[col] = row[col]
            
            record.additional_data = additional_data
            record.source = "database_load"
            
            self.records.append(record)
        
        logger.debug(f"Convertidos {len(self.records)} registros de ubicaciones")
    
    def find_location(self, **kwargs) -> Optional[LocationRecord]:
        """
        Busca una ubicación específica
        
        Parameters
        ----------
        **kwargs
            Criterios de búsqueda usando nombres de jerarquía
            Ejemplo: departamento='CUSCO', provincia='CUSCO', distrito='CUSCO'
        
        Returns
        -------
        LocationRecord or None
            Registro de ubicación encontrado
        """
        # Intentar cache primero
        if self.enable_cache:
            cache_key = str(sorted(kwargs.items()))
            if cache_key in self._search_cache:
                return self._search_cache[cache_key]
        
        # Convertir kwargs a nombres de columna correctos
        search_criteria = {}
        for key, value in kwargs.items():
            # Buscar columna correspondiente en jerarquía
            column_name = None
            for hierarchy_level in self.hierarchy:
                if key.upper() == hierarchy_level or key.lower() == hierarchy_level.lower():
                    column_name = hierarchy_level
                    break
            
            if column_name:
                search_criteria[column_name] = value
            else:
                # Buscar en el DataFrame directamente
                search_criteria[key] = value
        
        # Buscar en DataFrame
        df_filtered = self.data
        for column, value in search_criteria.items():
            if column in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[column] == value]
        
        if df_filtered.empty:
            result = None
        else:
            # Convertir primera fila a LocationRecord
            row = df_filtered.iloc[0]
            result = self._row_to_location_record(row)
        
        # Guardar en cache
        if self.enable_cache:
            self._search_cache[cache_key] = result
        
        return result
    
    def _row_to_location_record(self, row: pd.Series) -> LocationRecord:
        """Convierte una fila de DataFrame a LocationRecord"""
        record = LocationRecord(country_code=self.country_code)
        
        # Construir información básica
        parent_path = []
        current_name = ""
        
        for level_idx, level_name in enumerate(self.hierarchy):
            if level_name in row:
                value = str(row[level_name]).strip()
                if value and value != 'nan':
                    if level_idx == len(self.hierarchy) - 1:
                        current_name = value
                    else:
                        parent_path.append(value)
        
        record.name = current_name
        record.parent_path = parent_path
        record.level = self.hierarchy[-1] if self.hierarchy else "unknown"
        
        # Mapear todos los campos disponibles
        field_mapping = {
            'LATITUD': 'latitude',
            'LONGITUD': 'longitude', 
            'ALTITUD': 'altitude',
            'POBLACION': 'population',
            'AREA_KM2': 'area_km2',
            'ZONA_SISMICA': 'seismic_zone',
            'FACTOR_ZONA': 'zone_factor',
            'ACELERACION_PICO': 'peak_acceleration',
            'TIPO_SUELO': 'soil_type'
        }
        
        for col_name, field_name in field_mapping.items():
            if col_name in row:
                try:
                    if field_name in ['latitude', 'longitude', 'altitude', 'zone_factor', 'peak_acceleration', 'area_km2']:
                        setattr(record, field_name, float(row[col_name]) if row[col_name] not in [None, '', 'nan'] else 0.0)
                    elif field_name == 'population':
                        setattr(record, field_name, int(row[col_name]) if row[col_name] not in [None, '', 'nan'] else 0)
                    else:
                        setattr(record, field_name, str(row[col_name]) if row[col_name] not in [None, '', 'nan'] else "")
                except (ValueError, TypeError):
                    # Usar valor por defecto si hay error de conversión
                    pass
        
        return record
    
    def get_hierarchy_options(self, level: int, filters: Dict[str, str] = None) -> List[str]:
        """
        Obtiene opciones disponibles para un nivel de jerarquía
        
        Parameters
        ----------
        level : int
            Nivel de jerarquía (0 = primer nivel)
        filters : Dict[str, str], optional
            Filtros para niveles superiores
        
        Returns
        -------
        List[str]
            Lista de opciones disponibles
        """
        if level >= len(self.hierarchy):
            return []
        
        column_name = self.hierarchy[level]
        
        # Aplicar filtros
        df_filtered = self.data
        if filters:
            for filter_level, filter_value in filters.items():
                if filter_level in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered[filter_level] == filter_value]
        
        # Obtener valores únicos
        if column_name in df_filtered.columns:
            options = df_filtered[column_name].dropna().unique().tolist()
            return sorted([str(opt) for opt in options])
        
        return []
    
    def get_locations_by_zone(self, zone: Union[int, str]) -> List[LocationRecord]:
        """
        Obtiene todas las ubicaciones en una zona sísmica específica
        
        Parameters
        ----------
        zone : int or str
            Zona sísmica
        
        Returns
        -------
        List[LocationRecord]
            Lista de ubicaciones en la zona
        """
        matching_records = []
        
        for record in self.records:
            if record.seismic_zone == zone:
                matching_records.append(record)
        
        return matching_records
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la base de datos
        
        Returns
        -------
        Dict[str, Any]
            Diccionario con estadísticas
        """
        stats = {
            'total_records': len(self.records),
            'country_code': self.country_code,
            'hierarchy_levels': len(self.hierarchy),
            'hierarchy': self.hierarchy,
            'normative': self.normative_info.get('name', 'Unknown'),
            'data_sources': len(self.metadata['data_sources']),
            'cache_enabled': self.enable_cache
        }
        
        if not self.data.empty:
            # Estadísticas por nivel de jerarquía
            hierarchy_stats = {}
            for level in self.hierarchy:
                if level in self.data.columns:
                    unique_count = self.data[level].nunique()
                    hierarchy_stats[level] = unique_count
            
            stats['hierarchy_stats'] = hierarchy_stats
            
            # Estadísticas de zonas sísmicas
            if 'ZONA_SISMICA' in self.data.columns:
                zone_counts = self.data['ZONA_SISMICA'].value_counts().to_dict()
                stats['seismic_zones'] = zone_counts
            
            # Estadísticas geográficas
            geographic_stats = {}
            if 'LATITUD' in self.data.columns:
                lat_data = pd.to_numeric(self.data['LATITUD'], errors='coerce').dropna()
                if not lat_data.empty:
                    geographic_stats['latitude_range'] = (lat_data.min(), lat_data.max())
            
            if 'LONGITUD' in self.data.columns:
                lon_data = pd.to_numeric(self.data['LONGITUD'], errors='coerce').dropna()
                if not lon_data.empty:
                    geographic_stats['longitude_range'] = (lon_data.min(), lon_data.max())
            
            if geographic_stats:
                stats['geographic_stats'] = geographic_stats
        
        return stats
    
    def validate_data(self) -> Tuple[bool, List[str]]:
        """
        Valida la integridad de los datos
        
        Returns
        -------
        Tuple[bool, List[str]]
            (es_válido, lista_errores)
        """
        errors = []
        
        # Validar que hay datos
        if self.data.empty:
            errors.append("No hay datos cargados")
            return False, errors
        
        # Validar jerarquía
        for level in self.hierarchy:
            if level not in self.data.columns:
                errors.append(f"Columna de jerarquía faltante: {level}")
        
        # Validar coordenadas si existen
        if 'LATITUD' in self.data.columns:
            lat_data = pd.to_numeric(self.data['LATITUD'], errors='coerce')
            invalid_lat = ((lat_data < -90) | (lat_data > 90)).sum()
            if invalid_lat > 0:
                errors.append(f"{invalid_lat} registros con latitud inválida")
        
        if 'LONGITUD' in self.data.columns:
            lon_data = pd.to_numeric(self.data['LONGITUD'], errors='coerce')
            invalid_lon = ((lon_data < -180) | (lon_data > 180)).sum()
            if invalid_lon > 0:
                errors.append(f"{invalid_lon} registros con longitud inválida")
        
        # Validar factores sísmicos si existen
        if 'FACTOR_ZONA' in self.data.columns and self.normative_info:
            factor_range = self.normative_info.get('factor_range', (0, 1))
            factor_data = pd.to_numeric(self.data['FACTOR_ZONA'], errors='coerce')
            invalid_factors = ((factor_data < factor_range[0]) | (factor_data > factor_range[1])).sum()
            if invalid_factors > 0:
                errors.append(f"{invalid_factors} registros con factor sísmico fuera de rango")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def export_to_csv(self, filepath: Union[str, Path], include_metadata: bool = True):
        """
        Exporta datos a archivo CSV
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo de salida
        include_metadata : bool
            Incluir metadatos como comentarios
        """
        filepath = Path(filepath)
        
        try:
            if include_metadata:
                # Crear encabezado con metadatos
                header_lines = [
                    f"# Base de Datos de Ubicaciones - {self.country_code}",
                    f"# Normativa: {self.normative_info.get('name', 'Unknown')}",
                    f"# Generado: {datetime.now().isoformat()}",
                    f"# Total registros: {len(self.data)}",
                    f"# Jerarquía: {' > '.join(self.hierarchy)}",
                    "#"
                ]
                
                # Escribir encabezado
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(header_lines) + '\n')
                
                # Agregar datos CSV
                self.data.to_csv(filepath, mode='a', index=False, encoding='utf-8')
            else:
                self.data.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.info(f"Datos exportados a CSV: {filepath}")
            
        except Exception as e:
            logger.error(f"Error exportando a CSV {filepath}: {e}")
            raise
    
    def save_to_json(self, filepath: Union[str, Path]):
        """
        Guarda base de datos completa a archivo JSON
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo JSON
        """
        filepath = Path(filepath)
        
        try:
            export_data = {
                'metadata': self.metadata,
                'hierarchy': self.hierarchy,
                'country_code': self.country_code,
                'normative_info': self.normative_info,
                'records': [record.to_dict() for record in self.records],
                'data_csv': self.data.to_dict('records') if not self.data.empty else []
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Base de datos guardada en JSON: {filepath}")
            
        except Exception as e:
            logger.error(f"Error guardando JSON {filepath}: {e}")
            raise
    
    def load_from_json(self, filepath: Union[str, Path]):
        """
        Carga base de datos desde archivo JSON
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del archivo JSON
        """
        filepath = Path(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Cargar metadatos
            self.metadata = data.get('metadata', {})
            self.hierarchy = data.get('hierarchy', [])
            self.country_code = data.get('country_code', 'PE')
            self.normative_info = data.get('normative_info', {})
            
            # Cargar registros
            self.records = []
            for record_data in data.get('records', []):
                record = LocationRecord()
                record.from_dict(record_data)
                self.records.append(record)
            
            # Cargar DataFrame
            csv_data = data.get('data_csv', [])
            if csv_data:
                self.data = pd.DataFrame(csv_data)
            else:
                self.data = pd.DataFrame()
            
            # Limpiar cache
            self._clear_cache()
            
            logger.info(f"Base de datos cargada desde JSON: {filepath}")
            
        except Exception as e:
            logger.error(f"Error cargando JSON {filepath}: {e}")
            raise
    
    def _clear_cache(self):
        """Limpia cache de búsqueda"""
        if self.enable_cache:
            if self._search_cache:
                self._search_cache.clear()
            if self._hierarchy_cache:
                self._hierarchy_cache.clear()


class SeismicZoneDatabase(LocationDatabase):
    """
    Base de datos especializada para zonificación sísmica
    
    Extiende LocationDatabase con funcionalidad específica para datos sísmicos
    """
    
    def __init__(self, country_code: str = 'PE', enable_cache: bool = True):
        """
        Inicializa la base de datos de zonificación sísmica
        
        Parameters
        ----------
        country_code : str
            Código del país
        enable_cache : bool
            Habilitar cache para optimización
        """
        super().__init__(country_code, enable_cache)
        
        # Columnas específicas para datos sísmicos
        self.seismic_columns = [
            'ZONA_SISMICA', 'FACTOR_ZONA', 'ACELERACION_PICO', 
            'TIPO_SUELO', 'CATEGORIA_EDIFICACION'
        ]
        
        # Información adicional de normativa
        self.normative_parameters = {}
        
        logger.info(f"SeismicZoneDatabase inicializada para {country_code}")
    
    def load_seismic_data(self, df: pd.DataFrame, normative: str = "", 
                         normative_params: Dict[str, Any] = None):
        """
        Carga datos de zonificación sísmica
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos de ubicaciones y zonificación sísmica
        normative : str
            Nombre de la normativa sísmica
        normative_params : Dict[str, Any], optional
            Parámetros específicos de la normativa
        """
        # Cargar datos base
        self.load_from_dataframe(df, f"seismic_data_{normative}")
        
        # Actualizar información de normativa
        if normative:
            self.normative_info['name'] = normative
        
        if normative_params:
            self.normative_parameters.update(normative_params)
        
        # Validar columnas sísmicas
        available_seismic = [col for col in self.seismic_columns if col in df.columns]
        if not available_seismic:
            logger.warning("No se encontraron columnas de datos sísmicos estándar")
        else:
            logger.info(f"Columnas sísmicas detectadas: {available_seismic}")
        
        # Actualizar metadatos específicos
        self.metadata.update({
            'seismic_normative': normative,
            'seismic_columns': available_seismic,
            'normative_parameters': self.normative_parameters
        })
    
    def get_seismic_zone_info(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Obtiene información completa de zona sísmica para una ubicación
        
        Parameters
        ----------
        **kwargs
            Criterios de búsqueda de ubicación
        
        Returns
        -------
        Dict[str, Any] or None
            Información completa de zona sísmica
        """
        location = self.find_location(**kwargs)
        
        if not location:
            return None
        
        seismic_info = {
            'location': location.get_full_path(),
            'seismic_zone': location.seismic_zone,
            'zone_factor': location.zone_factor,
            'peak_acceleration': location.peak_acceleration,
            'soil_type': location.soil_type,
            'normative': self.normative_info.get('name', ''),
            'country_code': self.country_code
        }
        
        # Agregar parámetros adicionales si existen
        if self.normative_parameters:
            seismic_info['normative_parameters'] = self.normative_parameters
        
        # Agregar descripción de zona si está disponible
        if isinstance(location.seismic_zone, (int, float)):
            zone_descriptions = {
                1: "Zona de sismicidad baja",
                2: "Zona de sismicidad media", 
                3: "Zona de sismicidad alta",
                4: "Zona de sismicidad muy alta"
            }
            seismic_info['zone_description'] = zone_descriptions.get(
                int(location.seismic_zone), 
                f"Zona {location.seismic_zone}"
            )
        
        return seismic_info
    
    def get_zone_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de zonificación sísmica
        
        Returns
        -------
        Dict[str, Any]
            Estadísticas de zonificación
        """
        stats = self.get_statistics()
        
        # Estadísticas adicionales sísmicas
        seismic_stats = {}
        
        if not self.data.empty:
            # Distribución por zonas sísmicas
            if 'ZONA_SISMICA' in self.data.columns:
                zone_distribution = self.data['ZONA_SISMICA'].value_counts().to_dict()
                seismic_stats['zone_distribution'] = zone_distribution
                seismic_stats['total_zones'] = len(zone_distribution)
            
            # Rangos de factores de zona
            if 'FACTOR_ZONA' in self.data.columns:
                factor_data = pd.to_numeric(self.data['FACTOR_ZONA'], errors='coerce').dropna()
                if not factor_data.empty:
                    seismic_stats['factor_range'] = (factor_data.min(), factor_data.max())
                    seismic_stats['factor_mean'] = factor_data.mean()
            
            # Tipos de suelo
            if 'TIPO_SUELO' in self.data.columns:
                soil_types = self.data['TIPO_SUELO'].value_counts().to_dict()
                seismic_stats['soil_type_distribution'] = soil_types
        
        stats['seismic_statistics'] = seismic_stats
        return stats
    
    def validate_seismic_data(self) -> Tuple[bool, List[str]]:
        """
        Valida específicamente los datos sísmicos
        
        Returns
        -------
        Tuple[bool, List[str]]
            (es_válido, lista_errores)
        """
        is_valid, errors = self.validate_data()
        
        # Validaciones adicionales sísmicas
        if not self.data.empty:
            # Validar zonas sísmicas
            if 'ZONA_SISMICA' in self.data.columns:
                valid_zones = self.normative_info.get('zones', [])
                if valid_zones and valid_zones != 'Continuous':
                    invalid_zones = ~self.data['ZONA_SISMICA'].isin(valid_zones)
                    invalid_count = invalid_zones.sum()
                    if invalid_count > 0:
                        errors.append(f"{invalid_count} registros con zona sísmica inválida")
            
            # Validar factores de zona
            if 'FACTOR_ZONA' in self.data.columns:
                factor_range = self.normative_info.get('factor_range', (0, 1))
                factor_data = pd.to_numeric(self.data['FACTOR_ZONA'], errors='coerce')
                out_of_range = ((factor_data < factor_range[0]) | 
                               (factor_data > factor_range[1])).sum()
                if out_of_range > 0:
                    errors.append(f"{out_of_range} registros con factor de zona fuera del rango {factor_range}")
            
            # Validar consistencia zona-factor
            if 'ZONA_SISMICA' in self.data.columns and 'FACTOR_ZONA' in self.data.columns:
                # Esta validación sería específica por normativa
                # Por ahora solo verificamos que ambos campos estén presentes cuando uno existe
                zone_nulls = self.data['ZONA_SISMICA'].isnull()
                factor_nulls = self.data['FACTOR_ZONA'].isnull()
                
                inconsistent = (zone_nulls & ~factor_nulls) | (~zone_nulls & factor_nulls)
                inconsistent_count = inconsistent.sum()
                if inconsistent_count > 0:
                    errors.append(f"{inconsistent_count} registros con inconsistencia zona-factor")
        
        is_valid = len(errors) == 0
        return is_valid, errors


# Funciones de migración desde código existente

def migrate_basedatos_zonas_sismicas(legacy_bd: Any, country_code: str = None) -> SeismicZoneDatabase:
    """
    Migra datos desde BaseDatos_Zonas_Sismicas heredado
    
    Parameters
    ----------
    legacy_bd : Any
        Objeto de base de datos heredado con atributo BD_Zonas_Sismicas
    country_code : str, optional
        Código de país (se detecta automáticamente si no se proporciona)
        
    Returns
    -------
    SeismicZoneDatabase
        Base de datos migrada
    """
    try:
        # Extraer DataFrame
        if hasattr(legacy_bd, 'BD_Zonas_Sismicas'):
            df = legacy_bd.BD_Zonas_Sismicas.copy()
        elif isinstance(legacy_bd, pd.DataFrame):
            df = legacy_bd.copy()
        else:
            raise ValueError("Formato de datos heredados no reconocido")
        
        # Detectar país automáticamente si no se proporciona
        if not country_code:
            if 'DEPARTAMENTO' in df.columns and 'DISTRITO' in df.columns:
                country_code = 'PE'  # Perú tiene DEPARTAMENTO-PROVINCIA-DISTRITO
            elif 'DEPARTAMENTO' in df.columns and 'MUNICIPIO' in df.columns:
                country_code = 'BO'  # Bolivia tiene DEPARTAMENTO-PROVINCIA-MUNICIPIO
            elif 'REGION' in df.columns:
                country_code = 'CL'  # Chile tiene REGION-PROVINCIA-COMUNA
            else:
                country_code = 'PE'  # Default
                logger.warning("No se pudo detectar el país, usando Perú por defecto")
        
        # Crear nueva base de datos
        seismic_db = SeismicZoneDatabase(country_code)
        
        # Mapear nombres de columnas heredadas
        column_mapping = {
            'ZONA(Z)': 'ZONA_SISMICA',
            'ZONA': 'ZONA_SISMICA',
            'Z': 'FACTOR_ZONA',
            'SUELO': 'TIPO_SUELO'
        }
        
        df_migrated = df.rename(columns=column_mapping)
        
        # Determinar normativa según país
        normative_map = {
            'PE': 'E.030',
            'BO': 'NBC', 
            'CL': 'NCh433',
            'CO': 'NSR-10',
            'EC': 'NEC'
        }
        
        normative = normative_map.get(country_code, 'Unknown')
        
        # Cargar datos
        seismic_db.load_seismic_data(df_migrated, normative)
        
        # Agregar información de migración a metadatos
        seismic_db.metadata['migration_info'] = {
            'source': 'BaseDatos_Zonas_Sismicas',
            'migrated_at': datetime.now().isoformat(),
            'original_columns': list(df.columns),
            'mapped_columns': column_mapping
        }
        
        logger.info(f"Migración exitosa desde BaseDatos_Zonas_Sismicas: {len(df)} registros → {country_code}")
        return seismic_db
        
    except Exception as e:
        logger.error(f"Error en migración de BaseDatos_Zonas_Sismicas: {e}")
        raise


def load_standard_location_database(country_code: str, data_path: str = None) -> LocationDatabase:
    """
    Carga base de datos estándar para un país específico
    
    Parameters
    ----------
    country_code : str
        Código del país
    data_path : str, optional
        Ruta personalizada de datos (usa ruta estándar si no se proporciona)
        
    Returns
    -------
    LocationDatabase
        Base de datos cargada
    """
    if not data_path:
        # Usar estructura estándar de archivos
        data_path = f"data/locations/{country_code.lower()}_locations.csv"
    
    db = LocationDatabase(country_code)
    
    try:
        # Intentar cargar datos
        if Path(data_path).exists():
            db.load_from_csv(data_path)
            logger.info(f"Base de datos estándar cargada para {country_code}")
        else:
            logger.warning(f"Archivo de datos no encontrado: {data_path}")
            # Crear base de datos vacía con estructura mínima
            empty_df = pd.DataFrame(columns=db.hierarchy)
            db.load_from_dataframe(empty_df, "empty_standard")
    
    except Exception as e:
        logger.error(f"Error cargando base de datos estándar para {country_code}: {e}")
        # Crear base de datos vacía como fallback
        empty_df = pd.DataFrame(columns=db.hierarchy)
        db.load_from_dataframe(empty_df, "fallback_empty")
    
    return db


# Clases de compatibilidad con código existente

class BaseDatos_Zonas_Sismicas:
    """
    Clase de compatibilidad con código heredado
    
    Mantiene la interfaz original mientras usa internamente LocationDatabase
    """
    
    def __init__(self, csv_path: str = None, country_code: str = 'PE'):
        """
        Inicializa con compatibilidad hacia atrás
        
        Parameters
        ----------
        csv_path : str, optional
            Ruta al archivo CSV con datos
        country_code : str
            Código del país
        """
        self._db = SeismicZoneDatabase(country_code)
        
        if csv_path and Path(csv_path).exists():
            self._db.load_from_csv(csv_path)
        else:
            # Crear DataFrame vacío con estructura esperada
            columns = self._db.hierarchy + ['ZONA', 'FACTOR_ZONA']
            empty_df = pd.DataFrame(columns=columns)
            self._db.load_from_dataframe(empty_df, "legacy_empty")
        
        logger.info(f"BaseDatos_Zonas_Sismicas (compatibilidad) inicializada para {country_code}")
    
    @property
    def BD_Zonas_Sismicas(self) -> pd.DataFrame:
        """Propiedad de compatibilidad que retorna el DataFrame"""
        return self._db.data
    
    @BD_Zonas_Sismicas.setter
    def BD_Zonas_Sismicas(self, df: pd.DataFrame):
        """Setter de compatibilidad"""
        self._db.load_from_dataframe(df, "legacy_setter")
    
    def get_location_info(self, **kwargs) -> Dict[str, Any]:
        """Método de compatibilidad para obtener información de ubicación"""
        location = self._db.find_location(**kwargs)
        if location:
            return location.to_dict()
        return {}
    
    def get_seismic_info(self, **kwargs) -> Dict[str, Any]:
        """Método de compatibilidad para obtener información sísmica"""
        return self._db.get_seismic_zone_info(**kwargs) or {}


# Funciones de utilidad adicionales

def create_location_tree(db: LocationDatabase) -> Dict[str, Any]:
    """
    Crea estructura de árbol de ubicaciones para interfaces
    
    Parameters
    ----------
    db : LocationDatabase
        Base de datos de ubicaciones
        
    Returns
    -------
    Dict[str, Any]
        Estructura de árbol jerárquica
    """
    tree = {}
    
    if db.data.empty or not db.hierarchy:
        return tree
    
    # Construir árbol nivel por nivel
    for _, row in db.data.iterrows():
        current_level = tree
        
        for level_name in db.hierarchy:
            if level_name in row and pd.notna(row[level_name]):
                value = str(row[level_name])
                
                if value not in current_level:
                    current_level[value] = {}
                
                current_level = current_level[value]
    
    return tree


def validate_location_database_integrity(db: LocationDatabase) -> Dict[str, Any]:
    """
    Valida integridad completa de base de datos de ubicaciones
    
    Parameters
    ----------
    db : LocationDatabase
        Base de datos a validar
        
    Returns
    -------
    Dict[str, Any]
        Reporte completo de validación
    """
    report = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {},
        'recommendations': []
    }
    
    try:
        # Validación básica
        is_valid, errors = db.validate_data()
        if not is_valid:
            report['is_valid'] = False
            report['errors'].extend(errors)
        
        # Estadísticas
        report['statistics'] = db.get_statistics()
        
        # Validaciones adicionales
        if not db.data.empty:
            # Verificar completitud de jerarquía
            for level in db.hierarchy:
                if level in db.data.columns:
                    null_count = db.data[level].isnull().sum()
                    if null_count > 0:
                        report['warnings'].append(f"{null_count} registros con {level} nulo")
            
            # Verificar duplicados
            if len(db.hierarchy) > 0:
                duplicates = db.data.duplicated(subset=db.hierarchy).sum()
                if duplicates > 0:
                    report['warnings'].append(f"{duplicates} registros duplicados en jerarquía")
        
        # Recomendaciones
        if 'LATITUD' not in db.data.columns or 'LONGITUD' not in db.data.columns:
            report['recommendations'].append("Agregar coordenadas geográficas para mejor funcionalidad")
        
        if isinstance(db, SeismicZoneDatabase):
            seismic_valid, seismic_errors = db.validate_seismic_data()
            if not seismic_valid:
                report['errors'].extend(seismic_errors)
                report['is_valid'] = False
    
    except Exception as e:
        report['is_valid'] = False
        report['errors'].append(f"Error en validación: {str(e)}")
    
    return report


def export_location_database_summary(db: LocationDatabase, 
                                   output_path: Union[str, Path] = None) -> str:
    """
    Exporta resumen completo de base de datos de ubicaciones
    
    Parameters
    ----------
    db : LocationDatabase
        Base de datos a exportar
    output_path : str or Path, optional
        Ruta del archivo de salida (usa stdout si no se proporciona)
        
    Returns
    -------
    str
        Resumen en formato texto
    """
    summary_lines = []
    
    # Encabezado
    summary_lines.append("=" * 60)
    summary_lines.append("RESUMEN DE BASE DE DATOS DE UBICACIONES")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    
    # Información básica
    stats = db.get_statistics()
    summary_lines.append("INFORMACIÓN BÁSICA:")
    summary_lines.append(f"  País: {db.country_code}")
    summary_lines.append(f"  Normativa: {stats.get('normative', 'N/A')}")
    summary_lines.append(f"  Total registros: {stats['total_records']}")
    summary_lines.append(f"  Jerarquía: {' > '.join(db.hierarchy)}")
    summary_lines.append("")
    
    # Estadísticas por nivel
    if 'hierarchy_stats' in stats:
        summary_lines.append("ESTADÍSTICAS POR NIVEL:")
        for level, count in stats['hierarchy_stats'].items():
            summary_lines.append(f"  {level}: {count} únicos")
        summary_lines.append("")
    
    # Zonificación sísmica
    if 'seismic_zones' in stats:
        summary_lines.append("ZONIFICACIÓN SÍSMICA:")
        for zone, count in stats['seismic_zones'].items():
            summary_lines.append(f"  Zona {zone}: {count} ubicaciones")
        summary_lines.append("")
    
    # Cobertura geográfica
    if 'geographic_stats' in stats:
        geo_stats = stats['geographic_stats']
        summary_lines.append("COBERTURA GEOGRÁFICA:")
        if 'latitude_range' in geo_stats:
            lat_range = geo_stats['latitude_range']
            summary_lines.append(f"  Latitud: {lat_range[0]:.4f} a {lat_range[1]:.4f}")
        if 'longitude_range' in geo_stats:
            lon_range = geo_stats['longitude_range']
            summary_lines.append(f"  Longitud: {lon_range[0]:.4f} a {lon_range[1]:.4f}")
        summary_lines.append("")
    
    # Fuentes de datos
    if db.metadata.get('data_sources'):
        summary_lines.append("FUENTES DE DATOS:")
        for source in db.metadata['data_sources']:
            source_type = source.get('type', 'unknown')
            records = source.get('records', 0)
            summary_lines.append(f"  {source_type}: {records} registros")
        summary_lines.append("")
    
    # Validación
    validation = validate_location_database_integrity(db)
    summary_lines.append("ESTADO DE VALIDACIÓN:")
    summary_lines.append(f"  Válido: {'SÍ' if validation['is_valid'] else 'NO'}")
    summary_lines.append(f"  Errores: {len(validation['errors'])}")
    summary_lines.append(f"  Advertencias: {len(validation['warnings'])}")
    
    if validation['errors']:
        summary_lines.append("  Errores encontrados:")
        for error in validation['errors'][:5]:  # Mostrar solo los primeros 5
            summary_lines.append(f"    - {error}")
    
    summary_lines.append("")
    summary_lines.append("=" * 60)
    
    summary_text = "\n".join(summary_lines)
    
    # Escribir a archivo si se especifica
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        logger.info(f"Resumen exportado a: {output_path}")
    
    return summary_text


# Funciones de validación del módulo

def validate_location_database_module() -> bool:
    """
    Valida el funcionamiento correcto del módulo location_database
    
    Returns
    -------
    bool
        True si todas las validaciones pasan
    """
    try:
        # Probar creación de base de datos básica
        db = LocationDatabase('PE')
        assert db.country_code == 'PE'
        assert db.hierarchy == ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']
        
        # Probar carga de datos de prueba
        test_data = pd.DataFrame({
            'DEPARTAMENTO': ['CUSCO', 'LIMA'],
            'PROVINCIA': ['CUSCO', 'LIMA'],
            'DISTRITO': ['CUSCO', 'MIRAFLORES'],
            'ZONA_SISMICA': [4, 4],
            'FACTOR_ZONA': [0.45, 0.45]
        })
        
        db.load_from_dataframe(test_data, "test")
        assert len(db.data) == 2
        assert len(db.records) == 2
        
        # Probar búsqueda
        location = db.find_location(DEPARTAMENTO='CUSCO', PROVINCIA='CUSCO', DISTRITO='CUSCO')
        assert location is not None
        assert location.name == 'CUSCO'
        assert location.seismic_zone == 4
        
        # Probar jerarquía
        options = db.get_hierarchy_options(0)
        assert 'CUSCO' in options
        assert 'LIMA' in options
        
        # Probar SeismicZoneDatabase
        seismic_db = SeismicZoneDatabase('PE')
        seismic_db.load_seismic_data(test_data, 'E.030')
        
        seismic_info = seismic_db.get_seismic_zone_info(DEPARTAMENTO='CUSCO', PROVINCIA='CUSCO', DISTRITO='CUSCO')
        assert seismic_info is not None
        assert seismic_info['seismic_zone'] == 4
        assert seismic_info['normative'] == 'E.030'
        
        # Probar estadísticas
        stats = db.get_statistics()
        assert stats['total_records'] == 2
        assert stats['country_code'] == 'PE'
        
        # Probar validación
        is_valid, errors = db.validate_data()
        assert is_valid
        assert len(errors) == 0
        
        # Probar migración
        legacy_df = pd.DataFrame({
            'DEPARTAMENTO': ['TEST'],
            'PROVINCIA': ['TEST'],
            'DISTRITO': ['TEST'],
            'ZONA(Z)': [3]
        })
        
        migrated_db = migrate_basedatos_zonas_sismicas(legacy_df, 'PE')
        assert isinstance(migrated_db, SeismicZoneDatabase)
        assert 'ZONA_SISMICA' in migrated_db.data.columns
        
        # Probar compatibilidad
        legacy_bd = BaseDatos_Zonas_Sismicas()
        assert hasattr(legacy_bd, 'BD_Zonas_Sismicas')
        
        logger.info("✓ Validación del módulo location_database exitosa")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error en validación del módulo location_database: {e}")
        return False


# Punto de entrada para pruebas del módulo
if __name__ == "__main__":
    # Configurar logging para pruebas
    logging.basicConfig(level=logging.INFO)
    
    # Ejecutar validación
    if validate_location_database_module():
        print("✓ Módulo location_database validado correctamente")
        
        # Ejemplo de uso básico
        print("\nEjemplo de uso:")
        
        # Crear base de datos de ejemplo
        db = LocationDatabase('PE')
        
        # Datos de ejemplo
        sample_data = pd.DataFrame({
            'DEPARTAMENTO': ['CUSCO', 'LIMA', 'AREQUIPA'],
            'PROVINCIA': ['CUSCO', 'LIMA', 'AREQUIPA'], 
            'DISTRITO': ['CUSCO', 'MIRAFLORES', 'CERRO COLORADO'],
            'ZONA_SISMICA': [4, 4, 3],
            'FACTOR_ZONA': [0.45, 0.45, 0.35],
            'LATITUD': [-13.5319, -12.1211, -16.4055],
            'LONGITUD': [-71.9675, -77.0218, -71.5310]
        })
        
        db.load_from_dataframe(sample_data, "ejemplo")
        print(f"Base de datos cargada con {len(db.records)} ubicaciones")
        
        # Buscar ubicación
        cusco = db.find_location(DEPARTAMENTO='CUSCO', PROVINCIA='CUSCO', DISTRITO='CUSCO')
        if cusco:
            print(f"Ubicación encontrada: {cusco.get_full_path()}")
            print(f"Zona sísmica: {cusco.seismic_zone}, Factor: {cusco.zone_factor}")
        
        # Mostrar estadísticas
        stats = db.get_statistics()
        print(f"\nEstadísticas:")
        print(f"- Total registros: {stats['total_records']}")
        print(f"- Jerarquía: {' > '.join(stats['hierarchy'])}")
        
        # Crear base de datos sísmica
        seismic_db = SeismicZoneDatabase('PE')
        seismic_db.load_seismic_data(sample_data, 'E.030')
        
        seismic_info = seismic_db.get_seismic_zone_info(DEPARTAMENTO='LIMA', PROVINCIA='LIMA', DISTRITO='MIRAFLORES')
        if seismic_info:
            print(f"\nInformación sísmica para {seismic_info['location']}:")
            print(f"- Zona: {seismic_info['seismic_zone']}")
            print(f"- Factor Z: {seismic_info['zone_factor']}")
            print(f"- Normativa: {seismic_info['normative']}")
        
    else:
        print("✗ Error en validación del módulo location_database")
        sys.exit(1)