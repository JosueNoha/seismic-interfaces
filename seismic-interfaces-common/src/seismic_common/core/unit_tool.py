"""
Sistema de unidades centralizado para proyectos de ingeniería estructural
Permite conversiones entre diferentes sistemas de unidades
"""

import math
from typing import Dict, Optional, Union, Any
from enum import Enum


class UnitSystem(Enum):
    """Sistemas de unidades disponibles"""
    SI = "SI"           # Sistema Internacional (m, kg, s)
    MKS = "MKS"         # Metro-Kilogramo-Segundo (con gravedad)
    FPS = "FPS"         # Pie-Libra-Segundo
    METRIC = "METRIC"   # Métrico tradicional (cm, kgf, s)


class Units:
    """
    Clase principal para manejo de unidades y conversiones
    
    Permite trabajar con diferentes sistemas de unidades y realizar
    conversiones automáticas para cálculos de ingeniería estructural.
    """
    
    # Definición de unidades básicas
    m: float
    cm: float
    mm: float
    inch: float
    ft: float
    N: float
    kN: float
    kg: float
    g: float
    lb: float
    kgf: float
    tonf: float
    Pa: float
    MPa: float
    ksi: float
    psi: float
    s: float
    
    def __init__(self, system: Union[str, UnitSystem] = UnitSystem.SI):
        """
        Inicializa el sistema de unidades
        
        Parameters
        ----------
        system : str or UnitSystem
            Sistema de unidades a utilizar ('SI', 'MKS', 'FPS', 'METRIC')
        """
        if isinstance(system, str):
            try:
                self.system = UnitSystem(system.upper())
            except ValueError:
                raise ValueError(f"Sistema de unidades no válido: {system}. "
                               f"Opciones: {[s.value for s in UnitSystem]}")
        else:
            self.system = system
            
        self.set_units(self.system)
    
    def get_system(self) -> UnitSystem:
        """
        Obtiene el sistema de unidades actual
        
        Returns
        -------
        UnitSystem
            Sistema de unidades en uso
        """
        return self.system
    
    def set_units(self, u_system: Union[str, UnitSystem] = UnitSystem.SI) -> None:
        """
        Establece los factores de conversión según el sistema de unidades
        
        Parameters
        ----------
        u_system : str or UnitSystem
            Sistema de unidades ('SI', 'MKS', 'FPS', 'METRIC')
        """
        if isinstance(u_system, str):
            try:
                u_system = UnitSystem(u_system.upper())
            except ValueError:
                raise ValueError(f"Sistema de unidades no válido: {u_system}")
        
        self.system = u_system
        
        # Establecer unidades base según el sistema
        if u_system == UnitSystem.SI:
            # Sistema Internacional (N, m, kg, s, Pa)
            self.m = 1.0
            self.kg = 1.0
            self.s = 1.0
            self.N = 1.0  # kg⋅m⋅s⁻²
            
        elif u_system == UnitSystem.MKS:
            # Metro-Kilogramo-Segundo con fuerza gravitacional
            self.m = 1.0
            self.kg = 1.0 / 9.80665  # kgf a kg
            self.s = 1.0
            self.N = 1.0 / 9.80665   # N a kgf
            
        elif u_system == UnitSystem.FPS:
            # Pie-Libra-Segundo
            self.m = 3.28084         # m a ft
            self.kg = 2.20462        # kg a lb
            self.s = 1.0
            self.N = 0.224809        # N a lbf
            
        elif u_system == UnitSystem.METRIC:
            # Sistema métrico tradicional (cm, kgf, s)
            self.m = 100.0           # m a cm
            self.kg = 1.0 / 9.80665  # kg a kgf
            self.s = 1.0
            self.N = 1.0 / 9.80665   # N a kgf
        
        # Derivar todas las demás unidades
        self._derive_units()
    
    def _derive_units(self) -> None:
        """Calcula todas las unidades derivadas basándose en las unidades base"""
        
        # Unidades de longitud
        self.cm = self.m / 100.0
        self.mm = self.m / 1000.0
        self.inch = 2.54 * self.cm
        self.ft = 12.0 * self.inch
        
        # Unidades de masa y peso
        self.g = self.kg / 1000.0
        self.lb = 2.20462 * self.kg
        
        # Unidades de fuerza
        self.kN = 1000.0 * self.N
        self.kgf = 9.80665 * self.N
        self.tonf = 1000.0 * self.kgf
        self.lbf = 4.44822 * self.N
        self.kip = 1000.0 * self.lbf
        
        # Unidades de presión/esfuerzo
        self.Pa = self.N / (self.m ** 2)
        self.MPa = 1e6 * self.Pa
        self.kPa = 1000.0 * self.Pa
        self.psi = 6894.76 * self.Pa
        self.ksi = 1000.0 * self.psi
        self.kgf_cm2 = self.kgf / (self.cm ** 2)
        self.tonf_m2 = self.tonf / (self.m ** 2)
        
        # Unidades adicionales comunes en ingeniería
        self.kN_m = self.kN * self.m      # Momento
        self.tonf_m = self.tonf * self.m  # Momento
        self.kgf_m = self.kgf * self.m    # Momento
        
        # Unidades de área
        self.m2 = self.m ** 2
        self.cm2 = self.cm ** 2
        self.mm2 = self.mm ** 2
        self.in2 = self.inch ** 2
        
        # Unidades de volumen
        self.m3 = self.m ** 3
        self.cm3 = self.cm ** 3
        self.mm3 = self.mm ** 3
        
        # Unidades de densidad
        self.kg_m3 = self.kg / self.m3
        self.kgf_m3 = self.kgf / self.m3
        self.tonf_m3 = self.tonf / self.m3
    
    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        Convierte un valor de una unidad a otra
        
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
        float
            Valor convertido
            
        Examples
        --------
        >>> u = Units('SI')
        >>> u.convert(1000, 'mm', 'm')
        1.0
        """
        if not hasattr(self, from_unit):
            raise ValueError(f"Unidad origen no reconocida: {from_unit}")
        
        if not hasattr(self, to_unit):
            raise ValueError(f"Unidad destino no reconocida: {to_unit}")
        
        from_factor = getattr(self, from_unit)
        to_factor = getattr(self, to_unit)
        
        return value * from_factor / to_factor
    
    def get_unit_factor(self, unit_name: str) -> float:
        """
        Obtiene el factor de conversión para una unidad específica
        
        Parameters
        ----------
        unit_name : str
            Nombre de la unidad
            
        Returns
        -------
        float
            Factor de conversión
        """
        if not hasattr(self, unit_name):
            raise ValueError(f"Unidad no reconocida: {unit_name}")
        
        return getattr(self, unit_name)
    
    def list_available_units(self) -> Dict[str, float]:
        """
        Lista todas las unidades disponibles y sus factores
        
        Returns
        -------
        Dict[str, float]
            Diccionario con nombres de unidades y sus factores
        """
        units = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                try:
                    value = getattr(self, attr_name)
                    if isinstance(value, (int, float)) and attr_name != 'system':
                        units[attr_name] = value
                except:
                    continue
        return units
    
    def get_common_units(self) -> Dict[str, Dict[str, float]]:
        """
        Obtiene un diccionario con las unidades más comunes agrupadas por tipo
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Unidades agrupadas por categoría
        """
        return {
            'length': {
                'mm': self.mm,
                'cm': self.cm,
                'm': self.m,
                'inch': self.inch,
                'ft': self.ft
            },
            'force': {
                'N': self.N,
                'kN': self.kN,
                'kgf': self.kgf,
                'tonf': self.tonf,
                'lbf': getattr(self, 'lbf', 0),
                'kip': getattr(self, 'kip', 0)
            },
            'pressure': {
                'Pa': self.Pa,
                'kPa': getattr(self, 'kPa', 0),
                'MPa': self.MPa,
                'psi': self.psi,
                'ksi': self.ksi,
                'kgf_cm2': getattr(self, 'kgf_cm2', 0)
            },
            'mass': {
                'g': self.g,
                'kg': self.kg,
                'lb': self.lb
            }
        }
    
    def format_value(self, value: float, unit: str, decimals: int = 3) -> str:
        """
        Formatea un valor con su unidad
        
        Parameters
        ----------
        value : float
            Valor numérico
        unit : str
            Nombre de la unidad
        decimals : int
            Número de decimales
            
        Returns
        -------
        str
            Valor formateado con unidad
        """
        return f"{value:.{decimals}f} {unit}"
    
    def __repr__(self) -> str:
        """Representación string del objeto Units"""
        return f"Units(system={self.system.value})"
    
    def __str__(self) -> str:
        """String del objeto Units"""
        return f"Sistema de unidades: {self.system.value}"


# Funciones de utilidad para compatibilidad con código existente
def create_unit_dict(units_obj: Units) -> Dict[str, float]:
    """
    Crea un diccionario de unidades comunes para compatibilidad
    
    Parameters
    ----------
    units_obj : Units
        Objeto Units inicializado
        
    Returns
    -------
    Dict[str, float]
        Diccionario con unidades comunes
    """
    return {
        'mm': units_obj.mm,
        'm': units_obj.m,
        'cm': units_obj.cm,
        'pies': units_obj.ft,
        'pulg': units_obj.inch,
        'tonf': units_obj.tonf,
        'kN': units_obj.kN,
        'kgf': units_obj.kgf,
        'kip': getattr(units_obj, 'kip', 4.4482 * units_obj.kN),
        'Pa': units_obj.Pa,
        'MPa': units_obj.MPa,
        'psi': units_obj.psi,
        'ksi': units_obj.ksi
    }


# Constantes útiles para ingeniería estructural
class EngineeringConstants:
    """Constantes comunes en ingeniería estructural"""
    
    # Propiedades del acero
    STEEL_ELASTIC_MODULUS_SI = 200000e6  # Pa (200 GPa)
    STEEL_ELASTIC_MODULUS_IMPERIAL = 29000000  # psi (29 ksi)
    STEEL_DENSITY_SI = 7850  # kg/m³
    STEEL_DENSITY_IMPERIAL = 490  # lb/ft³
    
    # Propiedades del concreto
    CONCRETE_DENSITY_SI = 2400  # kg/m³
    CONCRETE_DENSITY_IMPERIAL = 150  # lb/ft³
    
    # Factores de seguridad típicos
    SAFETY_FACTORS = {
        'concrete_compression': 0.65,
        'concrete_shear': 0.75,
        'steel_tension': 0.90,
        'steel_compression': 0.90
    }
    
    # Conversiones útiles
    GRAVITY_SI = 9.80665  # m/s²
    GRAVITY_IMPERIAL = 32.174  # ft/s²


# Clase para configuración específica por país/normativa
class RegionalUnits:
    """Configuraciones de unidades específicas por región/normativa"""
    
    @staticmethod
    def peru_e030() -> Units:
        """Configuración típica para Perú (E-030)"""
        units = Units(UnitSystem.SI)
        return units
    
    @staticmethod
    def bolivia_cnbds() -> Units:
        """Configuración típica para Bolivia (CNBDS)"""
        units = Units(UnitSystem.SI)
        return units
    
    @staticmethod
    def usa_asce() -> Units:
        """Configuración típica para USA (ASCE)"""
        units = Units(UnitSystem.FPS)
        return units
    
    @staticmethod
    def get_preferred_units(country_code: str) -> Units:
        """
        Obtiene configuración preferida por país
        
        Parameters
        ----------
        country_code : str
            Código de país ('PE', 'BO', 'US', etc.)
            
        Returns
        -------
        Units
            Objeto Units configurado para el país
        """
        country_configs = {
            'PE': RegionalUnits.peru_e030,
            'BO': RegionalUnits.bolivia_cnbds,
            'US': RegionalUnits.usa_asce,
            'USA': RegionalUnits.usa_asce
        }
        
        config_func = country_configs.get(country_code.upper())
        if config_func:
            return config_func()
        else:
            # Default a SI
            return Units(UnitSystem.SI)


if __name__ == '__main__':
    # Ejemplos de uso
    print("=== Sistema de Unidades - Ejemplos ===\n")
    
    # Crear instancia con sistema SI
    u_si = Units('SI')
    print(f"Sistema SI: {u_si}")
    print(f"1 metro = {u_si.m}")
    print(f"1 kN = {u_si.kN}")
    print(f"1 MPa = {u_si.MPa}\n")
    
    # Crear instancia con sistema MKS
    u_mks = Units('MKS')
    print(f"Sistema MKS: {u_mks}")
    print(f"1 metro = {u_mks.m}")
    print(f"1 kN = {u_mks.kN}")
    print(f"1 MPa = {u_mks.MPa}\n")
    
    # Conversiones
    print("=== Conversiones ===")
    print(f"1000 mm a metros: {u_si.convert(1000, 'mm', 'm')} m")
    print(f"1 kN a N: {u_si.convert(1, 'kN', 'N')} N")
    print(f"25 MPa a Pa: {u_si.convert(25, 'MPa', 'Pa')} Pa\n")
    
    # Diccionario de unidades comunes
    unit_dict = create_unit_dict(u_si)
    print("=== Diccionario de Unidades Comunes ===")
    for name, factor in list(unit_dict.items())[:5]:
        print(f"{name}: {factor}")
    
    print(f"\n=== Unidades por Categorías ===")
    common_units = u_si.get_common_units()
    for category, units in common_units.items():
        print(f"\n{category.title()}:")
        for unit_name, factor in list(units.items())[:3]:
            print(f"  {unit_name}: {factor}")
    
    # Configuraciones regionales
    print(f"\n=== Configuraciones Regionales ===")
    peru_units = RegionalUnits.get_preferred_units('PE')
    bolivia_units = RegionalUnits.get_preferred_units('BO')
    usa_units = RegionalUnits.get_preferred_units('US')
    
    print(f"Perú: {peru_units.system.value}")
    print(f"Bolivia: {bolivia_units.system.value}")
    print(f"USA: {usa_units.system.value}")