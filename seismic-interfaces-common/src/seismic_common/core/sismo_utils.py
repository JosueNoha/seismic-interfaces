"""
Utilidades centralizadas para análisis sísmico
Clase base común para proyectos de análisis sísmico según diferentes normativas
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Any, Tuple
from abc import ABC, abstractmethod

# Importaciones locales (se configurarán según el proyecto específico)
try:
    from . import etabs_utils as etb
    from . import unit_tool
except ImportError:
    # Fallback para desarrollo
    import etabs_utils as etb
    import unit_tool

u = unit_tool.Units()

# Diccionario de unidades común
UNIT_DICT = {
    'mm': u.mm,
    'm': u.m,
    'cm': u.cm,
    'pies': u.ft,
    'pulg': u.inch,
    'tonf': u.tonf,
    'kN': u.kN,
    'kgf': u.kgf,
    'kip': 4.4482 * u.kN
}


class SeismicLoads:
    """Clase para manejo de cargas sísmicas"""
    
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
            Ejemplo: {'SX': 'EQX', 'SY': 'EQY', 'SDX': 'SPECX', 'SDY': 'SPECY'}
        """
        self.seism_loads = seism_loads
    

class SeismicTables:
    """Clase para almacenamiento de tablas de resultados sísmicos"""
    
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


class SeismicData:
    """Clase base para almacenamiento de datos sísmicos básicos"""
    
    def __init__(self):
        # Factores de reducción (común a la mayoría de normativas)
        self.Rx = 8.0  # Factor de reducción X
        self.Ry = 8.0  # Factor de reducción Y
        
        # Irregularidades (común a la mayoría de normativas)
        self.Ia = 1.0  # Factor de irregularidad en altura
        self.Ip = 1.0  # Factor de irregularidad en planta
        
        # Periodos fundamentales (calculados, no definidos por normativa)
        self.Tx = 0.0  # Periodo fundamental X
        self.Ty = 0.0  # Periodo fundamental Y
        
        # Masas participativas (calculadas, no definidas por normativa)
        self.MP_x = 0.0  # Masa participativa X
        self.MP_y = 0.0  # Masa participativa Y
        
        # Parámetros de deriva (común a la mayoría de normativas)
        self.max_drift_x = 0.007  # Deriva máxima permitida X
        self.max_drift_y = 0.007  # Deriva máxima permitida Y
        
        # Información del proyecto (independiente de normativa)
        self.proyecto = ""
        self.ubicacion = ""
        self.autor = ""
        self.fecha = ""
        self.descripcion = ""
        self.modelamiento = ""
        self.cargas = ""
        
        # Diccionario para parámetros específicos de cada normativa
        self.normative_params = {}


class BaseSeismicAnalysis(ABC):
    """Clase base abstracta para análisis sísmico"""
    
    def __init__(self):
        self.loads = SeismicLoads()
        self.tables = SeismicTables()
        self.data = SeismicData()
        
        # Configuración de unidades por defecto
        self.u_h = 'm'    # Unidad de altura/longitud
        self.u_d = 'mm'   # Unidad de desplazamiento
        self.u_f = 'kN'   # Unidad de fuerza
        
        # Piso base para análisis
        self.base_story = "Base"
        
        # Figuras para gráficos
        self.fig_drifts: Optional[Figure] = None
        self.fig_displacements: Optional[Figure] = None
        self.dynamic_shear_fig: Optional[Figure] = None
        self.static_shear_fig: Optional[Figure] = None
        self.spectrum_fig: Optional[Figure] = None
    
    def set_base_story(self, base_story: str) -> None:
        """Establece el piso base para análisis"""
        self.base_story = base_story
    
    def set_units(self, u_h: str = 'm', u_d: str = 'mm', u_f: str = 'kN') -> None:
        """
        Establece las unidades de trabajo
        
        Parameters
        ----------
        u_h : str
            Unidad de altura/longitud
        u_d : str
            Unidad de desplazamiento
        u_f : str
            Unidad de fuerza
        """
        self.u_h = u_h
        self.u_d = u_d
        self.u_f = u_f
    
    # Métodos de análisis modal
    def ana_modal(self, SapModel: Any) -> None:
        """
        Análisis modal de la estructura
        
        Parameters
        ----------
        SapModel : Any
            Objeto SapModel de ETABS
        """
        try:
            # Obtener datos modales usando etabs_utils
            modal_data = etb.get_modal_data(SapModel, clean_data=True)
            modal, MP_x, MP_y, period_x, period_y, Ux, Uy = modal_data
            
            # Procesar datos modales
            modal_processed = modal.copy()
            if 'Case' in modal_processed.columns:
                modal_processed = modal_processed[modal_processed['Case'] == modal_processed['Case'].iloc[0]]
            
            # Seleccionar columnas relevantes
            columns = ['Period', 'Ux', 'Uy', 'SumUx', 'SumUy']
            available_columns = [col for col in columns if col in modal_processed.columns]
            modal_processed = modal_processed[available_columns]
            
            # Convertir a numérico
            for col in available_columns:
                modal_processed[col] = pd.to_numeric(modal_processed[col], errors='coerce')
            
            # Almacenar resultados
            self.tables.modal = modal_processed
            self.data.MP_x = float(MP_x)
            self.data.MP_y = float(MP_y)
            self.data.Tx = float(period_x)
            self.data.Ty = float(period_y)
            
            # Encontrar modos fundamentales
            if 'Ux' in modal_processed.columns:
                self.modex = modal_processed[modal_processed.Ux == modal_processed.Ux.max()].index
            if 'Uy' in modal_processed.columns:
                self.modey = modal_processed[modal_processed.Uy == modal_processed.Uy.max()].index
                
        except Exception as e:
            print(f"Error en análisis modal: {str(e)}")
            # Crear DataFrame vacío en caso de error
            self.tables.modal = pd.DataFrame()
    
    @abstractmethod
    def get_k_factor(self, T: float) -> float:
        """
        Calcula el exponente k relacionado con el periodo según normativa específica
        
        Parameters
        ----------
        T : float
            Periodo fundamental de vibración (seg)
            
        Returns
        -------
        float
            Factor k para distribución de fuerzas
        """
        pass
    
    def irregularidad_rigidez(self, SapModel: Any, combo: bool = False) -> None:
        """
        Análisis de irregularidad de rigidez (piso blando)
        
        Parameters
        ----------
        SapModel : Any
            Objeto SapModel de ETABS
        combo : bool
            Si usar combinaciones en lugar de casos individuales
        """
        try:
            # Obtener datos de derivas de piso
            story_drifts = etb.get_story_drifts(SapModel)
            
            if story_drifts.empty:
                print("No se pudieron obtener datos de derivas")
                self.tables.rigidez_table = pd.DataFrame()
                return
            
            # Procesar datos para irregularidad de rigidez
            # (La implementación específica dependerá de cada normativa)
            self.tables.rigidez_table = story_drifts
            
        except Exception as e:
            print(f"Error en análisis de irregularidad de rigidez: {str(e)}")
            self.tables.rigidez_table = pd.DataFrame()
    
    def irregularidad_torsion(self, SapModel: Any, half_condition: bool = False, 
                             disp_combo: bool = False) -> None:
        """
        Análisis de irregularidad torsional
        
        Parameters
        ----------
        SapModel : Any
            Objeto SapModel de ETABS
        half_condition : bool
            Si considerar condición del 50%
        disp_combo : bool
            Si mostrar combinaciones de desplazamiento
        """
        try:
            # Obtener desplazamientos de juntas
            joint_disps = etb.get_joint_displacements(SapModel)
            
            if joint_disps.empty:
                print("No se pudieron obtener desplazamientos")
                self.tables.torsion_table = pd.DataFrame()
                return
            
            # Procesar datos para análisis torsional
            # (La implementación específica dependerá de cada normativa)
            self.tables.torsion_table = joint_disps
            
        except Exception as e:
            print(f"Error en análisis torsional: {str(e)}")
            self.tables.torsion_table = pd.DataFrame()
    
    def irregularidad_masa(self, SapModel: Any) -> None:
        """
        Análisis de irregularidad de masa
        
        Parameters
        ----------
        SapModel : Any
            Objeto SapModel de ETABS
        """
        try:
            # Obtener datos de masa por piso
            _, mass_data = etb.get_table(SapModel, 'Centers of Mass and Rigidity')
            
            if mass_data.empty:
                print("No se pudieron obtener datos de masa")
                self.tables.masa_table = pd.DataFrame()
                return
            
            # Procesar datos de masa
            self.tables.masa_table = mass_data
            
        except Exception as e:
            print(f"Error en análisis de masa: {str(e)}")
            self.tables.masa_table = pd.DataFrame()
    
    def derivas(self, SapModel: Any, disp_combo: bool = False) -> None:
        """
        Calcula y grafica las derivas de entrepiso
        
        Parameters
        ----------
        SapModel : Any
            Objeto SapModel de ETABS
        disp_combo : bool
            Si usar combinaciones para desplazamientos
        """
        try:
            # Obtener derivas
            story_drifts = etb.get_story_drifts(SapModel)
            
            if story_drifts.empty:
                print("No se pudieron obtener datos de derivas")
                return
            
            self.tables.story_drifts = story_drifts
            
            # Crear gráfico de derivas
            self._create_drift_plot(story_drifts)
            
        except Exception as e:
            print(f"Error en cálculo de derivas: {str(e)}")
    
    def desplazamientos(self, SapModel: Any, disp_combo: bool = False) -> None:
        """
        Calcula y grafica los desplazamientos laterales
        
        Parameters
        ----------
        SapModel : Any
            Objeto SapModel de ETABS
        disp_combo : bool
            Si usar combinaciones para desplazamientos
        """
        try:
            # Obtener desplazamientos
            joint_disps = etb.get_joint_displacements(SapModel)
            
            if joint_disps.empty:
                print("No se pudieron obtener desplazamientos")
                return
            
            self.tables.joint_displacements = joint_disps
            
            # Crear gráfico de desplazamientos
            self._create_displacement_plot(joint_disps)
            
        except Exception as e:
            print(f"Error en cálculo de desplazamientos: {str(e)}")
    
    def graph_shear(self, SapModel: Any, s_type: str = 'dynamic') -> None:
        """
        Grafica fuerzas cortantes por piso
        
        Parameters
        ----------
        SapModel : Any
            Objeto SapModel de ETABS
        s_type : str
            Tipo de análisis ('dynamic' o 'static')
        """
        try:
            # Obtener reacciones en la base
            base_reactions = etb.get_base_reactions(SapModel)
            
            if base_reactions.empty:
                print("No se pudieron obtener reacciones en la base")
                return
            
            # Crear gráfico de cortantes
            self._create_shear_plot(base_reactions, s_type)
            
        except Exception as e:
            print(f"Error en gráfico de cortantes: {str(e)}")
    
    def _create_drift_plot(self, story_drifts: pd.DataFrame) -> None:
        """Crea gráfico de derivas de entrepiso"""
        try:
            fig = Figure(figsize=(10, 8), dpi=100)
            ax = fig.add_subplot(111)
            
            # Configurar gráfico básico
            ax.set_xlabel(f'Deriva')
            ax.set_ylabel(f'Piso')
            ax.set_title('Derivas de Entrepiso')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            self.fig_drifts = fig
            
        except Exception as e:
            print(f"Error creando gráfico de derivas: {str(e)}")
    
    def _create_displacement_plot(self, joint_disps: pd.DataFrame) -> None:
        """Crea gráfico de desplazamientos laterales"""
        try:
            fig = Figure(figsize=(10, 8), dpi=100)
            ax = fig.add_subplot(111)
            
            # Configurar gráfico básico
            ax.set_xlabel(f'Desplazamiento ({self.u_d})')
            ax.set_ylabel(f'Altura ({self.u_h})')
            ax.set_title('Desplazamientos Laterales')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            self.fig_displacements = fig
            
        except Exception as e:
            print(f"Error creando gráfico de desplazamientos: {str(e)}")
    
    def _create_shear_plot(self, base_reactions: pd.DataFrame, s_type: str) -> None:
        """Crea gráfico de fuerzas cortantes"""
        try:
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # Configurar gráfico básico
            ax.set_xlabel(f'Fuerza Cortante ({self.u_f})')
            ax.set_ylabel(f'Altura ({self.u_h})')
            title = 'Fuerzas Cortantes ' + ('Dinámicas' if s_type == 'dynamic' else 'Estáticas')
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            if s_type == 'dynamic':
                self.dynamic_shear_fig = fig
            else:
                self.static_shear_fig = fig
                
        except Exception as e:
            print(f"Error creando gráfico de cortantes: {str(e)}")
    
    def min_shear(self, SapModel: Any, base_story: Optional[str] = None) -> None:
        """
        Verifica cortante mínima según normativa
        
        Parameters
        ----------
        SapModel : Any
            Objeto SapModel de ETABS
        base_story : str, optional
            Piso base para análisis
        """
        try:
            if base_story:
                self.base_story = base_story
            
            # Obtener reacciones en la base
            base_reactions = etb.get_base_reactions(SapModel)
            
            if not base_reactions.empty:
                self.tables.base_reactions = base_reactions
            
        except Exception as e:
            print(f"Error en verificación de cortante mínima: {str(e)}")
    
    # Métodos abstractos que deben implementarse en cada normativa específica
    @abstractmethod
    def espectro_respuesta(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera espectro de respuesta según normativa específica
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tupla con (periodos, aceleraciones)
        """
        pass
    
    @abstractmethod
    def sismo_estatico(self, SapModel: Any) -> None:
        """
        Análisis sísmico estático según normativa específica
        
        Parameters
        ----------
        SapModel : Any
            Objeto SapModel de ETABS
        """
        pass
    
    @abstractmethod
    def generate_memory(self, output_dir: str, **kwargs) -> None:
        """
        Genera memoria de cálculo según normativa específica
        
        Parameters
        ----------
        output_dir : str
            Directorio de salida para la memoria
        **kwargs
            Argumentos adicionales específicos de cada normativa
        """
        pass
    
    # Métodos de utilidad
    def validate_modal_analysis(self) -> Dict[str, bool]:
        """
        Valida si el análisis modal cumple con los requisitos
        
        Returns
        -------
        Dict[str, bool]
            Diccionario con resultados de validación
        """
        results = {
            'mp_x_sufficient': self.data.MP_x >= 0.9,
            'mp_y_sufficient': self.data.MP_y >= 0.9,
            'periods_valid': self.data.Tx > 0 and self.data.Ty > 0
        }
        
        return results
    
    def get_summary_data(self) -> Dict[str, Any]:
        """
        Obtiene resumen de datos principales del análisis
        
        Returns
        -------
        Dict[str, Any]
            Diccionario con datos de resumen
        """
        return {
            'periods': {'Tx': self.data.Tx, 'Ty': self.data.Ty},
            'participating_mass': {'MP_x': self.data.MP_x, 'MP_y': self.data.MP_y},
            'site_factors': {'Z': self.data.Z, 'U': self.data.U, 'S': self.data.S},
            'reduction_factors': {'Rx': self.data.Rx, 'Ry': self.data.Ry},
            'irregularity_factors': {'Ia': self.data.Ia, 'Ip': self.data.Ip},
            'units': {'height': self.u_h, 'displacement': self.u_d, 'force': self.u_f}
        }
    
    def export_tables_to_excel(self, filename: str) -> None:
        """
        Exporta todas las tablas a un archivo Excel
        
        Parameters
        ----------
        filename : str
            Nombre del archivo Excel
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                if not self.tables.modal.empty:
                    self.tables.modal.to_excel(writer, sheet_name='Modal', index=False)
                
                if not self.tables.story_drifts.empty:
                    self.tables.story_drifts.to_excel(writer, sheet_name='Derivas', index=False)
                
                if not self.tables.joint_displacements.empty:
                    self.tables.joint_displacements.to_excel(writer, sheet_name='Desplazamientos', index=False)
                
                if not self.tables.rigidez_table.empty:
                    self.tables.rigidez_table.to_excel(writer, sheet_name='Irregularidad_Rigidez', index=False)
                
                if not self.tables.torsion_table.empty:
                    self.tables.torsion_table.to_excel(writer, sheet_name='Torsion', index=False)
                
                # Crear hoja de resumen
                summary_df = pd.DataFrame([self.get_summary_data()])
                summary_df.to_excel(writer, sheet_name='Resumen', index=False)
                
            print(f"Tablas exportadas exitosamente a {filename}")
            
        except Exception as e:
            print(f"Error exportando tablas: {str(e)}")



if __name__ == '__main__':
    pass