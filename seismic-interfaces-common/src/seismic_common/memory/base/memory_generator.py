"""
Generador base de memorias de cálculo sísmico
Centraliza la funcionalidad común para generar memorias LaTeX
"""

import os
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ...core.latex_utils import (
    escape_text, dataframe_to_latex, process_latex_variables,
    compile_latex, create_table_wrapper
)
from .template_manager import TemplateManager
from ..processors.table_processor import TableProcessor
from ..processors.graph_processor import GraphProcessor
from ..processors.variable_processor import VariableProcessor


class BaseMemoryGenerator(ABC):
    """
    Clase base para generadores de memorias de cálculo sísmico
    
    Proporciona funcionalidad común que puede ser heredada por 
    implementaciones específicas por país/norma
    """
    
    def __init__(self, template_path: Optional[Union[str, Path]] = None):
        """
        Inicializa el generador de memorias
        
        Parameters
        ----------
        template_path : str or Path, optional
            Ruta al archivo de plantilla LaTeX personalizada
        """
        self.template_manager = TemplateManager()
        self.table_processor = TableProcessor()
        self.graph_processor = GraphProcessor()
        self.variable_processor = VariableProcessor()
        
        # Cargar plantilla personalizada si se proporciona
        if template_path:
            self.template_path = Path(template_path)
        else:
            self.template_path = self._get_default_template()
    
    @abstractmethod
    def _get_default_template(self) -> Path:
        """Obtiene la ruta de la plantilla por defecto para la implementación específica"""
        pass
    
    @abstractmethod
    def get_country_specific_variables(self, seismic_data: Any) -> Dict[str, Any]:
        """Obtiene variables específicas del país/norma"""
        pass
    
    def load_template(self, template_path: Optional[Union[str, Path]] = None) -> str:
        """
        Carga el contenido de una plantilla LaTeX
        
        Parameters
        ----------
        template_path : str or Path, optional
            Ruta personalizada de plantilla
            
        Returns
        -------
        str
            Contenido de la plantilla
        """
        path = Path(template_path) if template_path else self.template_path
        return self.template_manager.load_template(path)
    
    def save_variables(self, variables: Dict[str, Any], content: str) -> str:
        """
        Procesa y guarda variables en el contenido LaTeX
        
        Parameters
        ----------
        variables : Dict[str, Any]
            Diccionario con variables y valores
        content : str
            Contenido del template LaTeX
            
        Returns
        -------
        str
            Contenido con variables procesadas
        """
        return self.variable_processor.process_variables(variables, content)
    
    def create_modal_table(self, modal_data: pd.DataFrame, content: str) -> str:
        """
        Crea tabla de períodos y porcentajes de masa participativa
        
        Parameters
        ----------
        modal_data : pd.DataFrame
            Datos modales con columnas: Period, UX, UY, RZ, SumUX, SumUY, SumRZ
        content : str
            Contenido del template LaTeX
            
        Returns
        -------
        str
            Contenido con tabla modal insertada
        """
        return self.table_processor.create_modal_table(modal_data, content)
    
    def create_torsion_table(self, torsion_x: pd.DataFrame, torsion_y: pd.DataFrame, 
                           content: str) -> str:
        """
        Crea tabla de irregularidad torsional
        
        Parameters
        ----------
        torsion_x : pd.DataFrame
            Datos de torsión en dirección X
        torsion_y : pd.DataFrame
            Datos de torsión en dirección Y
        content : str
            Contenido del template LaTeX
            
        Returns
        -------
        str
            Contenido con tabla de torsión insertada
        """
        return self.table_processor.create_torsion_table(torsion_x, torsion_y, content)
    
    def create_mass_table(self, mass_data: pd.DataFrame, content: str) -> str:
        """
        Crea tabla de irregularidad de masa
        
        Parameters
        ----------
        mass_data : pd.DataFrame
            Datos de masa por piso
        content : str
            Contenido del template LaTeX
            
        Returns
        -------
        str
            Contenido con tabla de masa insertada
        """
        return self.table_processor.create_mass_table(mass_data, content)
    
    def create_stiffness_table(self, stiffness_x: pd.DataFrame, stiffness_y: pd.DataFrame,
                             content: str) -> str:
        """
        Crea tabla de irregularidad de rigidez
        
        Parameters
        ----------
        stiffness_x : pd.DataFrame
            Datos de rigidez en dirección X
        stiffness_y : pd.DataFrame
            Datos de rigidez en dirección Y
        content : str
            Contenido del template LaTeX
            
        Returns
        -------
        str
            Contenido con tabla de rigidez insertada
        """
        return self.table_processor.create_stiffness_table(stiffness_x, stiffness_y, content)
    
    def create_response_spectrum_figure(self, seismic_data: Any, output_dir: Union[str, Path]) -> None:
        """
        Genera figura del espectro de respuesta
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos
        output_dir : str or Path
            Directorio de salida para las imágenes
        """
        self.graph_processor.create_response_spectrum(seismic_data, output_dir)
    
    def create_displacement_figure(self, seismic_data: Any, output_dir: Union[str, Path]) -> None:
        """
        Genera figura de desplazamientos laterales
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos
        output_dir : str or Path
            Directorio de salida para las imágenes
        """
        self.graph_processor.create_displacement_figure(seismic_data, output_dir)
    
    def create_shear_figures(self, seismic_data: Any, output_dir: Union[str, Path]) -> None:
        """
        Genera figuras de cortantes dinámico y estático
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos
        output_dir : str or Path
            Directorio de salida para las imágenes
        """
        self.graph_processor.create_shear_figures(seismic_data, output_dir)
    
    def update_images(self, seismic_data: Any, output_dir: Union[str, Path]) -> None:
        """
        Actualiza todas las imágenes para la memoria
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos
        output_dir : str or Path
            Directorio de salida para las imágenes
        """
        output_path = Path(output_dir)
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Generar figuras
        self.create_response_spectrum_figure(seismic_data, images_dir)
        self.create_displacement_figure(seismic_data, images_dir)
        self.create_shear_figures(seismic_data, images_dir)
        
        # Copiar imágenes estáticas desde recursos
        self._copy_static_images(images_dir)
    
    def _copy_static_images(self, output_images_dir: Path) -> None:
        """
        Copia imágenes estáticas desde el directorio de recursos
        
        Parameters
        ----------
        output_images_dir : Path
            Directorio de destino para las imágenes
        """
        # Buscar directorio de recursos (puede ser específico de la implementación)
        source_dirs = self._get_resource_directories()
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.pdf')
        
        for source_dir in source_dirs:
            if source_dir.exists():
                for file_path in source_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                        shutil.copy2(file_path, output_images_dir / file_path.name)
    
    def _get_resource_directories(self) -> List[Path]:
        """
        Obtiene los directorios de recursos donde buscar imágenes
        Puede ser sobrescrito por implementaciones específicas
        
        Returns
        -------
        List[Path]
            Lista de directorios de recursos
        """
        # Directorio base de recursos comunes
        common_resources = Path(__file__).parent.parent.parent / "resources" / "images" / "common"
        return [common_resources]
    
    def process_numeric_variables(self, seismic_data: Any, content: Optional[str] = None) -> str:
        """
        Procesa variables numéricas generales
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos
        content : str, optional
            Contenido del template, si no se proporciona se carga
            
        Returns
        -------
        str
            Contenido con variables procesadas
        """
        if content is None:
            content = self.load_template()
        
        # Variables comunes base
        base_variables = self._get_base_variables(seismic_data)
        
        # Variables específicas del país
        country_variables = self.get_country_specific_variables(seismic_data)
        
        # Combinar variables
        all_variables = {**base_variables, **country_variables}
        
        return self.save_variables(all_variables, content)
    
    def _get_base_variables(self, seismic_data: Any) -> Dict[str, Any]:
        """
        Obtiene variables base comunes a todas las implementaciones
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos
            
        Returns
        -------
        Dict[str, Any]
            Diccionario con variables base
        """
        # Variables básicas que deberían existir en cualquier análisis sísmico
        base_vars = {}
        
        # Información del proyecto
        if hasattr(seismic_data, 'proyecto'):
            base_vars['proyecto'] = seismic_data.proyecto
        if hasattr(seismic_data, 'ubicacion'):
            base_vars['ubicacion'] = seismic_data.ubicacion
        if hasattr(seismic_data, 'autor'):
            base_vars['autor'] = seismic_data.autor
        if hasattr(seismic_data, 'fecha'):
            base_vars['fecha'] = seismic_data.fecha
        
        # Períodos fundamentales
        if hasattr(seismic_data, 'Tx'):
            base_vars['Tx'] = seismic_data.Tx
        if hasattr(seismic_data, 'Ty'):
            base_vars['Ty'] = seismic_data.Ty
        
        # Cortantes
        if hasattr(seismic_data, 'Vsx'):
            base_vars['Vsx'] = seismic_data.Vsx
        if hasattr(seismic_data, 'Vsy'):
            base_vars['Vsy'] = seismic_data.Vsy
        if hasattr(seismic_data, 'Vdx'):
            base_vars['Vdx'] = seismic_data.Vdx
        if hasattr(seismic_data, 'Vdy'):
            base_vars['Vdy'] = seismic_data.Vdy
        
        # Cálculo de porcentajes de cortante dinámico vs estático
        if 'Vdx' in base_vars and 'Vsx' in base_vars and base_vars['Vsx'] != 0:
            base_vars['perVdsx'] = (base_vars['Vdx'] / base_vars['Vsx']) * 100
        if 'Vdy' in base_vars and 'Vsy' in base_vars and base_vars['Vsy'] != 0:
            base_vars['perVdsy'] = (base_vars['Vdy'] / base_vars['Vsy']) * 100
        
        # Factores de escala
        if hasattr(seismic_data, 'FEx'):
            base_vars['FEx'] = seismic_data.FEx
        if hasattr(seismic_data, 'FEy'):
            base_vars['FEy'] = seismic_data.FEy
        
        return base_vars
    
    def generate_memory(self, seismic_data: Any, output_dir: Union[str, Path], 
                       custom_template: Optional[Union[str, Path]] = None,
                       delete_tex: bool = False, run_twice: bool = False) -> bool:
        """
        Genera la memoria completa de cálculo sísmico
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos del análisis sísmico
        output_dir : str or Path
            Directorio de salida
        custom_template : str or Path, optional
            Plantilla personalizada a usar
        delete_tex : bool
            Si eliminar el archivo .tex después de compilar
        run_twice : bool
            Si ejecutar LaTeX dos veces (para referencias cruzadas)
            
        Returns
        -------
        bool
            True si la generación fue exitosa
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Cargar contenido base
            content = self.load_template(custom_template)
            
            # Actualizar imágenes
            self.update_images(seismic_data, output_path)
            
            # Procesar contenido paso a paso
            content = self._process_all_content(seismic_data, content)
            
            # Guardar archivo LaTeX
            latex_file = output_path / "memoria.tex"
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Compilar a PDF
            os.chdir(output_path)  # Cambiar al directorio de salida
            runs = 2 if run_twice else 1
            success = compile_latex("memoria.tex", clean_aux=True, runs=runs)
            
            # Eliminar archivo .tex si se solicita
            if delete_tex and success:
                latex_file.unlink(missing_ok=True)
            
            return success
            
        except Exception as e:
            print(f"Error generando memoria: {e}")
            return False
    
    def _process_all_content(self, seismic_data: Any, content: str) -> str:
        """
        Procesa todo el contenido de la memoria
        Puede ser sobrescrito por implementaciones específicas
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos
        content : str
            Contenido base del template
            
        Returns
        -------
        str
            Contenido completamente procesado
        """
        # Procesar variables numéricas
        content = self.process_numeric_variables(seismic_data, content)
        
        # Procesar tablas si existen los datos
        if hasattr(seismic_data, 'tables'):
            tables = seismic_data.tables
            
            if hasattr(tables, 'modal'):
                content = self.create_modal_table(tables.modal, content)
            
            if hasattr(tables, 'torsion_table'):
                # Separar por direcciones si es necesario
                torsion_data = tables.torsion_table
                if hasattr(seismic_data, 'loads') and hasattr(seismic_data.loads, 'seism_loads'):
                    loads = seismic_data.loads.seism_loads
                    if 'tx' in loads and 'ty' in loads:
                        torsion_x = torsion_data[torsion_data['OutputCase'].str.startswith(loads['tx'])]
                        torsion_y = torsion_data[torsion_data['OutputCase'].str.startswith(loads['ty'])]
                        content = self.create_torsion_table(torsion_x, torsion_y, content)
            
            if hasattr(tables, 'mass_table'):
                content = self.create_mass_table(tables.mass_table, content)
            
            if hasattr(tables, 'stiffness_x') and hasattr(tables, 'stiffness_y'):
                content = self.create_stiffness_table(tables.stiffness_x, tables.stiffness_y, content)
        
        return content