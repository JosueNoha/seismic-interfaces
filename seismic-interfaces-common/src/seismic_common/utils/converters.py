"""
Módulo de convertidores centralizados para interfaces sísmicas
=============================================================

Este módulo centraliza todas las funciones de conversión utilizadas
en el proyecto de interfaces sísmicas, evitando duplicación de código
entre las diferentes normativas.

Tipos de convertidores incluidos:
- Conversores de unidades de ingeniería
- Convertidores de datos sísmicos entre normativas
- Convertidores de formatos de archivos
- Convertidores de sistemas de coordenadas
- Convertidores de bases de datos legacy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import json

# Configurar logger
logger = logging.getLogger(__name__)


class UnitSystem(Enum):
    """Enumeración de sistemas de unidades soportados"""
    SI = "SI"           # Sistema Internacional
    MKS = "MKS"         # Metro-Kilogramo-Segundo
    FPS = "FPS"         # Pie-Libra-Segundo
    CGS = "CGS"         # Centímetro-Gramo-Segundo


@dataclass
class ConversionResult:
    """Resultado de una conversión con metadatos"""
    original_value: float
    converted_value: float
    from_unit: str
    to_unit: str
    conversion_factor: float
    system_from: str
    system_to: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()


class UnitConverter:
    """
    Convertidor centralizado de unidades de ingeniería
    
    Soporta conversión entre diferentes sistemas de unidades
    comúnmente utilizados en ingeniería sísmica
    """
    
    def __init__(self, system: Union[str, UnitSystem] = UnitSystem.SI):
        """
        Inicializa el convertidor de unidades
        
        Parameters
        ----------
        system : str or UnitSystem
            Sistema de unidades base
        """
        if isinstance(system, str):
            self.system = UnitSystem(system)
        else:
            self.system = system
        
        self.conversion_history = []
        self._setup_base_units()
        self._setup_conversion_factors()
    
    def _setup_base_units(self):
        """Configura las unidades base según el sistema"""
        if self.system == UnitSystem.SI:
            self.m = 1.0
            self.kg = 1.0
            self.s = 1.0
        elif self.system == UnitSystem.MKS:
            self.m = 1.0
            self.kg = 1.0 / 9.8106
            self.s = 1.0
        elif self.system == UnitSystem.FPS:
            self.m = 100.0 / (2.54 * 12)
            self.kg = 1.0 / 2.20462
            self.s = 1.0
        elif self.system == UnitSystem.CGS:
            self.m = 100.0
            self.kg = 1000.0
            self.s = 1.0
        
        # Unidades derivadas
        self._calculate_derived_units()
    
    def _calculate_derived_units(self):
        """Calcula unidades derivadas basadas en las unidades base"""
        # Longitud
        self.mm = self.m / 1000.0
        self.cm = self.m / 100.0
        self.inch = 2.54 * self.cm
        self.ft = self.inch * 12.0
        
        # Masa y peso
        self.g = self.kg / 1000.0
        self.lb = 2.20462 * self.kg
        
        # Fuerza
        self.N = self.kg * self.m / (self.s ** 2)
        self.kN = 1000.0 * self.N
        self.kgf = self.N * 9.8106
        self.tonf = 1000.0 * self.kgf
        
        # Presión y esfuerzo
        self.Pa = self.N / (self.m ** 2)
        self.kPa = 1000.0 * self.Pa
        self.MPa = 1e6 * self.Pa
        self.psi = self.Pa * 6.895e3
        self.ksi = 1000.0 * self.psi
        
        # Unidades compuestas para ingeniería sísmica
        self.kgf_cm2 = self.kgf / (self.cm ** 2)
        self.tonf_m2 = self.tonf / (self.m ** 2)
        self.kgf_m = self.kgf * self.m
        self.tonf_m = self.tonf * self.m
        
        # Aceleración
        self.g_accel = 9.81  # Aceleración gravitacional (m/s²)
        
        # Adimensional
        self.adim = 1.0
        self.nu = 1.0
    
    def _setup_conversion_factors(self):
        """Configura factores de conversión adicionales"""
        self.conversion_factors = {
            # Factores especiales
            'kip': 4.4482 * (self.kN if hasattr(self, 'kN') else 4448.2),
            'kip_ft': 4.4482 * self.kN * 0.3048 * self.m,
            
            # Factores de zona sísmica (ejemplo)
            'zone_factor_1': 0.10,
            'zone_factor_2': 0.25,
            'zone_factor_3': 0.35,
            'zone_factor_4': 0.45,
            
            # Factores de suelo típicos
            'soil_s0': 0.80,
            'soil_s1': 1.00,
            'soil_s2': 1.05,
            'soil_s3': 1.15,
            'soil_s4': 1.20,
        }
    
    def convert(self, value: float, from_unit: str, to_unit: str) -> ConversionResult:
        """
        Convierte un valor entre unidades
        
        Parameters
        ----------
        value : float
            Valor a convertir
        from_unit : str
            Unidad origen
        to_unit : str
            Unidad destino
            
        Returns
        -------
        ConversionResult
            Resultado de la conversión con metadatos
        """
        # Obtener factores de conversión
        from_factor = self.get_unit_factor(from_unit)
        to_factor = self.get_unit_factor(to_unit)
        
        # Realizar conversión
        conversion_factor = from_factor / to_factor
        converted_value = value * conversion_factor
        
        # Crear resultado
        result = ConversionResult(
            original_value=value,
            converted_value=converted_value,
            from_unit=from_unit,
            to_unit=to_unit,
            conversion_factor=conversion_factor,
            system_from=self.system.value,
            system_to=self.system.value
        )
        
        # Registrar en historial
        self.conversion_history.append(result)
        
        return result
    
    def get_unit_factor(self, unit_name: str) -> float:
        """
        Obtiene el factor de conversión para una unidad
        
        Parameters
        ----------
        unit_name : str
            Nombre de la unidad
            
        Returns
        -------
        float
            Factor de conversión
        """
        # Verificar en unidades base y derivadas
        if hasattr(self, unit_name):
            return getattr(self, unit_name)
        
        # Verificar en factores adicionales
        if unit_name in self.conversion_factors:
            return self.conversion_factors[unit_name]
        
        raise ValueError(f"Unidad no reconocida: {unit_name}")
    
    def bulk_convert(self, values: Dict[str, float], 
                    unit_mapping: Dict[str, Tuple[str, str]]) -> Dict[str, ConversionResult]:
        """
        Convierte múltiples valores con mapeo de unidades
        
        Parameters
        ----------
        values : dict
            Diccionario {variable: valor}
        unit_mapping : dict
            Diccionario {variable: (from_unit, to_unit)}
            
        Returns
        -------
        dict
            Diccionario {variable: ConversionResult}
        """
        results = {}
        
        for var_name, value in values.items():
            if var_name in unit_mapping:
                from_unit, to_unit = unit_mapping[var_name]
                try:
                    results[var_name] = self.convert(value, from_unit, to_unit)
                except Exception as e:
                    logger.error(f"Error convirtiendo {var_name}: {e}")
                    results[var_name] = None
            else:
                logger.warning(f"No se encontró mapeo de unidades para {var_name}")
        
        return results
    
    def get_available_units(self) -> Dict[str, Dict[str, float]]:
        """
        Obtiene todas las unidades disponibles organizadas por tipo
        
        Returns
        -------
        dict
            Diccionario con unidades por categoría
        """
        return {
            'length': {
                'mm': self.mm, 'cm': self.cm, 'm': self.m,
                'inch': self.inch, 'ft': self.ft
            },
            'force': {
                'N': self.N, 'kN': self.kN, 'kgf': self.kgf,
                'tonf': self.tonf
            },
            'pressure': {
                'Pa': self.Pa, 'kPa': self.kPa, 'MPa': self.MPa,
                'psi': self.psi, 'ksi': self.ksi, 'kgf_cm2': self.kgf_cm2
            },
            'mass': {
                'g': self.g, 'kg': self.kg, 'lb': self.lb
            },
            'moment': {
                'kgf_m': self.kgf_m, 'tonf_m': self.tonf_m
            },
            'distributed_load': {
                'tonf_m2': self.tonf_m2
            }
        }


class SeismicDataConverter:
    """
    Convertidor de datos sísmicos entre diferentes normativas
    """
    
    def __init__(self):
        """Inicializa el convertidor de datos sísmicos"""
        self.unit_converter = UnitConverter()
        self.normative_mappings = self._setup_normative_mappings()
    
    def _setup_normative_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Configura mapeos entre normativas
        
        Returns
        -------
        dict
            Mapeos de campos entre normativas
        """
        return {
            'peru_to_bolivia': {
                'Z': 'Z',           # Factor de zona
                'U': 'I',           # Factor de uso/importancia
                'S': 'S',           # Factor de suelo
                'Tp': 'T0',         # Periodo característico
                'Tl': 'Ts',         # Periodo límite
                'R': 'Rd',          # Factor de reducción
            },
            'bolivia_to_peru': {
                'Z': 'Z',
                'I': 'U', 
                'S': 'S',
                'T0': 'Tp',
                'Ts': 'Tl',
                'Rd': 'R',
            },
            'peru_to_ecuador': {
                'Z': 'Z',
                'U': 'I',
                'S': 'Fa',  # Factor de sitio
                'R': 'R',
            },
            'ecuador_to_colombia': {
                'Z': 'Aa',          # Aceleración pico
                'I': 'I',
                'Fa': 'Fa',
                'R': 'R',
            }
        }
    
    def convert_normative_parameters(self, params: Dict[str, Any], 
                                   from_norm: str, to_norm: str) -> Dict[str, Any]:
        """
        Convierte parámetros entre normativas
        
        Parameters
        ----------
        params : dict
            Parámetros originales
        from_norm : str
            Normativa origen
        to_norm : str
            Normativa destino
            
        Returns
        -------
        dict
            Parámetros convertidos
        """
        mapping_key = f"{from_norm}_to_{to_norm}"
        
        if mapping_key not in self.normative_mappings:
            raise ValueError(f"Mapeo no disponible: {mapping_key}")
        
        mapping = self.normative_mappings[mapping_key]
        converted_params = {}
        
        for original_key, value in params.items():
            if original_key in mapping:
                new_key = mapping[original_key]
                converted_params[new_key] = value
            else:
                # Mantener parámetros no mapeados
                converted_params[original_key] = value
        
        return converted_params
    
    def convert_seismic_loads(self, loads: Dict[str, str], 
                            from_format: str, to_format: str) -> Dict[str, str]:
        """
        Convierte nomenclatura de cargas sísmicas
        
        Parameters
        ----------
        loads : dict
            Cargas sísmicas originales
        from_format : str
            Formato origen
        to_format : str
            Formato destino
            
        Returns
        -------
        dict
            Cargas con nomenclatura convertida
        """
        load_mappings = {
            'sap2000_to_etabs': {
                'SISMOX': 'EQX',
                'SISMOY': 'EQY',
                'ESTATICOX': 'ESTATX',
                'ESTATICOY': 'ESTATY'
            },
            'etabs_to_sap2000': {
                'EQX': 'SISMOX',
                'EQY': 'SISMOY',
                'ESTATX': 'ESTATICOX',
                'ESTATY': 'ESTATICOY'
            }
        }
        
        mapping_key = f"{from_format}_to_{to_format}"
        
        if mapping_key not in load_mappings:
            return loads  # Sin conversión disponible
        
        mapping = load_mappings[mapping_key]
        converted_loads = {}
        
        for load_name, load_value in loads.items():
            if load_name in mapping:
                new_name = mapping[load_name]
                converted_loads[new_name] = load_value
            else:
                converted_loads[load_name] = load_value
        
        return converted_loads


class DatabaseConverter:
    """
    Convertidor de bases de datos legacy a formato moderno
    """
    
    def __init__(self):
        """Inicializa el convertidor de bases de datos"""
        self.column_mappings = self._setup_column_mappings()
    
    def _setup_column_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Configura mapeos de columnas legacy
        
        Returns
        -------
        dict
            Mapeos de nombres de columnas
        """
        return {
            'legacy_to_modern': {
                'ZONA(Z)': 'ZONA_SISMICA',
                'FACTOR_Z': 'Z',
                'ACELERACION': 'Z',
                'DEPARTMENTO': 'DEPARTAMENTO',  # Corrección ortográfica
                'PROVNCIA': 'PROVINCIA',        # Corrección ortográfica
                'DSITRITO': 'DISTRITO',         # Corrección ortográfica
                'REGION': 'DEPARTAMENTO',
                'CIUDAD': 'DISTRITO'
            },
            'peru_legacy': {
                'ZONA': 'ZONA_SISMICA',
                'FACTOR': 'Z',
                'DEPT': 'DEPARTAMENTO',
                'PROV': 'PROVINCIA',
                'DIST': 'DISTRITO'
            },
            'bolivia_legacy': {
                'ZONA': 'ZONA_SISMICA', 
                'DEPTO': 'DEPARTAMENTO',
                'PCIA': 'PROVINCIA',
                'MUN': 'MUNICIPIO'
            }
        }
    
    def migrate_legacy_database(self, df: pd.DataFrame, 
                              country_code: str = 'PE') -> pd.DataFrame:
        """
        Migra base de datos legacy a formato moderno
        
        Parameters
        ----------
        df : DataFrame
            Base de datos legacy
        country_code : str
            Código del país
            
        Returns
        -------
        DataFrame
            Base de datos migrada
        """
        df_migrated = df.copy()
        
        # Seleccionar mapeo apropiado
        if country_code == 'PE':
            mapping = self.column_mappings.get('peru_legacy', {})
        elif country_code == 'BO':
            mapping = self.column_mappings.get('bolivia_legacy', {})
        else:
            mapping = self.column_mappings.get('legacy_to_modern', {})
        
        # Aplicar mapeo general
        general_mapping = self.column_mappings['legacy_to_modern']
        mapping.update(general_mapping)
        
        # Renombrar columnas
        df_migrated = df_migrated.rename(columns=mapping)
        
        # Limpiar y estandarizar datos
        df_migrated = self._clean_migrated_data(df_migrated)
        
        return df_migrated
    
    def _clean_migrated_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y estandariza datos migrados
        
        Parameters
        ----------
        df : DataFrame
            DataFrame a limpiar
            
        Returns
        -------
        DataFrame
            DataFrame limpio
        """
        df_clean = df.copy()
        
        # Limpiar campos de texto (mayúsculas, espacios)
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = (df_clean[col]
                                .astype(str)
                                .str.upper()
                                .str.strip()
                                .str.replace(r'\s+', ' ', regex=True))
        
        # Convertir columnas numéricas
        numeric_conversions = {
            'ZONA_SISMICA': 'int',
            'Z': 'float'
        }
        
        for col, dtype in numeric_conversions.items():
            if col in df_clean.columns:
                try:
                    if dtype == 'int':
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')
                    else:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Error convirtiendo columna {col}: {e}")
        
        # Eliminar filas completamente vacías
        df_clean = df_clean.dropna(how='all')
        
        # Eliminar duplicados
        df_clean = df_clean.drop_duplicates()
        
        return df_clean
    
    def convert_zone_format(self, zone_data: Union[int, float, str], 
                          from_format: str, to_format: str) -> Union[int, float]:
        """
        Convierte formatos de zonas sísmicas
        
        Parameters
        ----------
        zone_data : int, float, or str
            Datos de zona sísmica
        from_format : str
            Formato origen ('number', 'factor', 'acceleration')
        to_format : str
            Formato destino
            
        Returns
        -------
        int or float
            Zona convertida
        """
        # Mapeos de conversión de zonas
        zone_conversions = {
            'number_to_factor': {1: 0.10, 2: 0.25, 3: 0.35, 4: 0.45},
            'factor_to_number': {0.10: 1, 0.25: 2, 0.35: 3, 0.45: 4},
            'factor_to_acceleration': lambda x: x,  # Mismo valor
            'acceleration_to_factor': lambda x: x   # Mismo valor
        }
        
        conversion_key = f"{from_format}_to_{to_format}"
        
        if conversion_key in zone_conversions:
            conversion = zone_conversions[conversion_key]
            
            if callable(conversion):
                return conversion(zone_data)
            else:
                return conversion.get(zone_data, zone_data)
        
        return zone_data  # Sin conversión


class CoordinateConverter:
    """
    Convertidor de sistemas de coordenadas geográficas
    """
    
    def __init__(self):
        """Inicializa el convertidor de coordenadas"""
        self.datum_transformations = self._setup_datum_transformations()
    
    def _setup_datum_transformations(self) -> Dict[str, Dict[str, Any]]:
        """
        Configura transformaciones de datum
        
        Returns
        -------
        dict
            Parámetros de transformación por país
        """
        return {
            'PE': {
                'PSAD56_to_WGS84': {
                    'dx': -279.0, 'dy': 175.0, 'dz': -379.0
                },
                'WGS84_to_UTM18S': {
                    'zone': 18, 'hemisphere': 'S'
                },
                'WGS84_to_UTM19S': {
                    'zone': 19, 'hemisphere': 'S'
                }
            },
            'BO': {
                'PSAD56_to_WGS84': {
                    'dx': -270.0, 'dy': 188.0, 'dz': -388.0
                }
            }
        }
    
    def convert_coordinates(self, lat: float, lon: float, 
                          from_system: str, to_system: str,
                          country_code: str = 'PE') -> Tuple[float, float]:
        """
        Convierte coordenadas entre sistemas
        
        Parameters
        ----------
        lat : float
            Latitud
        lon : float
            Longitud
        from_system : str
            Sistema origen
        to_system : str
            Sistema destino
        country_code : str
            Código del país
            
        Returns
        -------
        tuple
            Coordenadas convertidas (lat, lon) o (x, y)
        """
        # Implementación básica - para uso completo se requeriría pyproj
        logger.info(f"Conversión de coordenadas: {from_system} → {to_system}")
        
        if from_system == to_system:
            return lat, lon
        
        # Aplicar transformaciones básicas
        if country_code in self.datum_transformations:
            transforms = self.datum_transformations[country_code]
            transform_key = f"{from_system}_to_{to_system}"
            
            if transform_key in transforms:
                # Aplicar transformación (simplificada)
                logger.info(f"Aplicando transformación {transform_key}")
                # En implementación real usaría pyproj o similar
                return lat, lon
        
        logger.warning(f"Transformación no disponible: {from_system} → {to_system}")
        return lat, lon


# ============================================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================================

def create_unit_converter(system: str = 'SI') -> UnitConverter:
    """
    Crea un convertidor de unidades con configuración estándar
    
    Parameters
    ----------
    system : str
        Sistema de unidades ('SI', 'MKS', 'FPS', 'CGS')
        
    Returns
    -------
    UnitConverter
        Instancia configurada del convertidor
    """
    return UnitConverter(system)


def convert_legacy_seismic_database(df: pd.DataFrame, 
                                   country_code: str = 'PE') -> pd.DataFrame:
    """
    Función de conveniencia para migrar bases de datos sísmicas legacy
    
    Parameters
    ----------
    df : DataFrame
        Base de datos legacy
    country_code : str
        Código del país
        
    Returns
    -------
    DataFrame
        Base de datos migrada
    """
    converter = DatabaseConverter()
    return converter.migrate_legacy_database(df, country_code)


def bulk_unit_conversion(values: Dict[str, float], 
                        from_units: Dict[str, str],
                        to_units: Dict[str, str],
                        system: str = 'SI') -> Dict[str, float]:
    """
    Conversión masiva de unidades
    
    Parameters
    ----------
    values : dict
        Valores a convertir {variable: valor}
    from_units : dict
        Unidades origen {variable: unidad}
    to_units : dict
        Unidades destino {variable: unidad}
    system : str
        Sistema de unidades base
        
    Returns
    -------
    dict
        Valores convertidos {variable: valor_convertido}
    """
    converter = UnitConverter(system)
    results = {}
    
    for var_name, value in values.items():
        if var_name in from_units and var_name in to_units:
            try:
                conversion_result = converter.convert(
                    value, from_units[var_name], to_units[var_name]
                )
                results[var_name] = conversion_result.converted_value
            except Exception as e:
                logger.error(f"Error convirtiendo {var_name}: {e}")
                results[var_name] = value
        else:
            results[var_name] = value
    
    return results


# ============================================================================
# EXPORTACIÓN DE FUNCIONES PRINCIPALES
# ============================================================================

__all__ = [
    'UnitSystem',
    'ConversionResult',
    'UnitConverter',
    'SeismicDataConverter',
    'DatabaseConverter',
    'CoordinateConverter',
    'create_unit_converter',
    'convert_legacy_seismic_database',
    'bulk_unit_conversion'
]