"""
Procesador de gráficos para memorias de cálculo sísmico
Centraliza la creación y exportación de gráficos matplotlib para memorias sísmicas
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para generación de archivos
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class GraphProcessor:
    """
    Procesador centralizado de gráficos para memorias sísmicas
    
    Proporciona métodos para crear, formatear y exportar gráficos
    comunes en análisis sísmico según diferentes normativas
    """
    
    def __init__(self):
        """Inicializa el procesador de gráficos"""
        self.default_dpi = 300
        self.default_format = 'pdf'
        self.default_bbox = 'tight'
        self.default_figsize = (10, 8)
        
        # Configuración de estilos por defecto
        self.style_config = {
            'grid': True,
            'grid_alpha': 0.3,
            'linewidth': 2,
            'markersize': 6,
            'fontsize_title': 14,
            'fontsize_labels': 12,
            'fontsize_legend': 10,
            'colors': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e', 
                'accent': '#2ca02c',
                'warning': '#d62728',
                'info': '#9467bd'
            }
        }
    
    def setup_matplotlib(self):
        """Configura matplotlib para generación de figuras de alta calidad"""
        plt.rcParams.update({
            'figure.dpi': self.default_dpi,
            'savefig.dpi': self.default_dpi,
            'savefig.bbox': self.default_bbox,
            'font.size': self.style_config['fontsize_labels'],
            'axes.titlesize': self.style_config['fontsize_title'],
            'axes.labelsize': self.style_config['fontsize_labels'],
            'legend.fontsize': self.style_config['fontsize_legend'],
            'figure.figsize': self.default_figsize
        })
    
    def create_figure(self, figsize: Optional[Tuple[float, float]] = None) -> Tuple[Figure, Axes]:
        """
        Crea una nueva figura matplotlib
        
        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Tamaño de la figura (ancho, alto)
            
        Returns
        -------
        Tuple[Figure, Axes]
            Figura y ejes matplotlib
        """
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
        return fig, ax
    
    def apply_default_styling(self, ax: Axes, title: str = None, 
                            xlabel: str = None, ylabel: str = None) -> None:
        """
        Aplica estilos por defecto a los ejes
        
        Parameters
        ----------
        ax : Axes
            Ejes matplotlib
        title : str, optional
            Título del gráfico
        xlabel : str, optional
            Etiqueta del eje X
        ylabel : str, optional
            Etiqueta del eje Y
        """
        if self.style_config['grid']:
            ax.grid(True, alpha=self.style_config['grid_alpha'])
        
        if title:
            ax.set_title(title, fontsize=self.style_config['fontsize_title'])
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.style_config['fontsize_labels'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.style_config['fontsize_labels'])
    
    def save_figure(self, fig: Figure, filename: Union[str, Path],
                   output_dir: Union[str, Path] = None, 
                   format: str = None, dpi: int = None) -> str:
        """
        Guarda una figura en archivo
        
        Parameters
        ----------
        fig : Figure
            Figura matplotlib a guardar
        filename : str or Path
            Nombre del archivo (sin extensión)
        output_dir : str or Path, optional
            Directorio de salida
        format : str, optional
            Formato del archivo ('pdf', 'png', 'svg', etc.)
        dpi : int, optional
            Resolución en DPI
            
        Returns
        -------
        str
            Ruta completa del archivo guardado
        """
        # Configurar parámetros
        fmt = format or self.default_format
        resolution = dpi or self.default_dpi
        
        # Construir ruta completa
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            full_path = output_path / f"{filename}.{fmt}"
        else:
            full_path = Path(f"{filename}.{fmt}")
        
        # Guardar figura
        fig.savefig(
            full_path,
            dpi=resolution,
            bbox_inches=self.default_bbox,
            format=fmt,
            facecolor='white',
            edgecolor='none'
        )
        
        return str(full_path)
    
    def create_response_spectrum(self, seismic_data: Any, 
                               output_dir: Union[str, Path] = None) -> str:
        """
        Crea gráfico del espectro de respuesta sísmica
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos que contenga información del espectro
        output_dir : str or Path, optional
            Directorio donde guardar el gráfico
            
        Returns
        -------
        str
            Ruta del archivo generado
        """
        fig, ax = self.create_figure(figsize=(12, 8))
        
        try:
            # Extraer datos del espectro
            if hasattr(seismic_data, 'spectrum_periods') and hasattr(seismic_data, 'spectrum_accelerations'):
                periods = seismic_data.spectrum_periods
                accelerations = seismic_data.spectrum_accelerations
            elif hasattr(seismic_data, 'fig_spectrum'):
                # Si ya existe una figura, usar esa
                if output_dir:
                    return self.save_figure(seismic_data.fig_spectrum, 
                                          "espectro_respuestas", output_dir)
                else:
                    return "espectro_respuestas.pdf"
            else:
                # Crear espectro genérico si no hay datos específicos
                periods = np.linspace(0, 4, 100)
                accelerations = self._create_generic_spectrum(periods)
            
            # Graficar espectro
            ax.plot(periods, accelerations, 
                   color=self.style_config['colors']['primary'],
                   linewidth=self.style_config['linewidth'],
                   label='Espectro de Diseño')
            
            # Rellenar área bajo la curva
            ax.fill_between(periods, accelerations, alpha=0.3, 
                          color=self.style_config['colors']['primary'])
            
            # Marcar períodos característicos si están disponibles
            if hasattr(seismic_data, 'Tp') and seismic_data.Tp:
                ax.axvline(x=seismic_data.Tp, color=self.style_config['colors']['warning'],
                          linestyle='--', alpha=0.7, 
                          label=f'Tp = {seismic_data.Tp:.2f} s')
            
            if hasattr(seismic_data, 'Tl') and seismic_data.Tl:
                ax.axvline(x=seismic_data.Tl, color=self.style_config['colors']['accent'],
                          linestyle='--', alpha=0.7,
                          label=f'TL = {seismic_data.Tl:.2f} s')
            
            # Configurar ejes y título
            self.apply_default_styling(ax, 
                                     title='Espectro de Respuesta de Aceleraciones',
                                     xlabel='Periodo T (s)',
                                     ylabel='Aceleración Espectral Sa (g)')
            
            ax.legend()
            ax.set_xlim(0, max(periods) * 1.05)
            ax.set_ylim(0, max(accelerations) * 1.1)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creando espectro de respuesta:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Error en Espectro de Respuesta')
        
        plt.tight_layout()
        
        # Guardar y retornar ruta
        if output_dir:
            return self.save_figure(fig, "espectro_respuestas", output_dir)
        else:
            return "espectro_respuestas.pdf"
    
    def create_displacement_figure(self, seismic_data: Any, 
                                 output_dir: Union[str, Path] = None) -> str:
        """
        Crea gráfico de desplazamientos laterales
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos que contenga desplazamientos
        output_dir : str or Path, optional
            Directorio donde guardar el gráfico
            
        Returns
        -------
        str
            Ruta del archivo generado
        """
        fig, ax = self.create_figure(figsize=(10, 12))
        
        try:
            # Extraer datos de desplazamientos
            if hasattr(seismic_data, 'fig_displacements'):
                # Si ya existe una figura, usar esa
                if output_dir:
                    return self.save_figure(seismic_data.fig_displacements, 
                                          "desplazamientos_laterales", output_dir)
                else:
                    return "desplazamientos_laterales.pdf"
            
            # Intentar extraer datos de diferentes fuentes posibles
            heights = []
            disp_x = []
            disp_y = []
            
            if hasattr(seismic_data, 'disp_h') and hasattr(seismic_data, 'disp_x'):
                heights = seismic_data.disp_h
                disp_x = seismic_data.disp_x
                disp_y = seismic_data.disp_y if hasattr(seismic_data, 'disp_y') else []
            elif hasattr(seismic_data, 'displacement_data'):
                data = seismic_data.displacement_data
                heights = data.get('Height', [])
                disp_x = data.get('DisplacementX', [])
                disp_y = data.get('DisplacementY', [])
            
            if heights and (disp_x or disp_y):
                # Graficar desplazamientos
                if disp_x:
                    ax.plot(disp_x, heights, 'ro-', 
                           linewidth=self.style_config['linewidth'],
                           markersize=self.style_config['markersize'],
                           label='Desplazamiento X')
                
                if disp_y:
                    ax.plot(disp_y, heights, 'bs-',
                           linewidth=self.style_config['linewidth'], 
                           markersize=self.style_config['markersize'],
                           label='Desplazamiento Y')
                
                # Configurar ejes
                self.apply_default_styling(ax,
                                         title='Desplazamientos Laterales Inelásticos',
                                         xlabel='Desplazamiento (mm)',
                                         ylabel='Altura (m)')
                
                ax.legend()
                
                # Configurar límites
                if disp_x or disp_y:
                    max_disp = max((max(disp_x) if disp_x else 0), 
                                 (max(disp_y) if disp_y else 0))
                    ax.set_xlim(0, max_disp * 1.1)
                
                ax.set_ylim(0, max(heights) * 1.05)
            else:
                ax.text(0.5, 0.5, 'No hay datos de desplazamientos disponibles',
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Desplazamientos Laterales - Sin Datos')
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creando gráfico de desplazamientos:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Error en Desplazamientos Laterales')
        
        plt.tight_layout()
        
        # Guardar y retornar ruta
        if output_dir:
            return self.save_figure(fig, "desplazamientos_laterales", output_dir)
        else:
            return "desplazamientos_laterales.pdf"
    
    def create_drift_figure(self, seismic_data: Any, 
                          output_dir: Union[str, Path] = None) -> str:
        """
        Crea gráfico de derivas de entrepiso
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos que contenga derivas
        output_dir : str or Path, optional
            Directorio donde guardar el gráfico
            
        Returns
        -------
        str
            Ruta del archivo generado
        """
        fig, ax = self.create_figure(figsize=(10, 12))
        
        try:
            # Extraer datos de derivas
            if hasattr(seismic_data, 'fig_drifts'):
                # Si ya existe una figura, usar esa
                if output_dir:
                    return self.save_figure(seismic_data.fig_drifts, 
                                          "derivas", output_dir)
                else:
                    return "derivas.pdf"
            
            # Intentar extraer datos de diferentes fuentes posibles
            heights = []
            drift_x = []
            drift_y = []
            
            if hasattr(seismic_data, 'drift_data'):
                data = seismic_data.drift_data
                heights = data.get('Height', [])
                drift_x = data.get('DriftX', [])
                drift_y = data.get('DriftY', [])
            elif hasattr(seismic_data, 'tables') and hasattr(seismic_data.tables, 'drift_table'):
                # Extraer de tabla de derivas si existe
                drift_table = seismic_data.tables.drift_table
                if 'Story' in drift_table.columns:
                    heights = drift_table['Story'].tolist()
                if 'drift x' in drift_table.columns:
                    drift_x = drift_table['drift x'].tolist()
                if 'drift y' in drift_table.columns:
                    drift_y = drift_table['drift y'].tolist()
            
            if heights and (drift_x or drift_y):
                # Graficar derivas
                if drift_x:
                    ax.plot(drift_x, heights, 'ro-',
                           linewidth=self.style_config['linewidth'],
                           markersize=self.style_config['markersize'],
                           label='Deriva X')
                
                if drift_y:
                    ax.plot(drift_y, heights, 'bs-',
                           linewidth=self.style_config['linewidth'],
                           markersize=self.style_config['markersize'], 
                           label='Deriva Y')
                
                # Línea de límite de deriva (ejemplo: 0.007 para estructuras comunes)
                drift_limit = 0.007
                ax.axvline(x=drift_limit, color=self.style_config['colors']['warning'],
                          linestyle='--', alpha=0.7, 
                          label=f'Límite = {drift_limit:.3f}')
                
                # Configurar ejes
                self.apply_default_styling(ax,
                                         title='Derivas de Entrepiso',
                                         xlabel='Deriva',
                                         ylabel='Piso')
                
                ax.legend()
                
                # Configurar límites
                if drift_x or drift_y:
                    max_drift = max((max(drift_x) if drift_x else 0), 
                                  (max(drift_y) if drift_y else 0))
                    ax.set_xlim(0, max(max_drift, drift_limit) * 1.2)
                
                ax.set_ylim(0, max(heights) * 1.05)
            else:
                ax.text(0.5, 0.5, 'No hay datos de derivas disponibles',
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Derivas de Entrepiso - Sin Datos')
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creando gráfico de derivas:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Error en Derivas de Entrepiso')
        
        plt.tight_layout()
        
        # Guardar y retornar ruta
        if output_dir:
            return self.save_figure(fig, "derivas", output_dir)
        else:
            return "derivas.pdf"
    
    def create_shear_figures(self, seismic_data: Any, 
                           output_dir: Union[str, Path] = None) -> List[str]:
        """
        Crea gráficos de cortantes dinámico y estático
        
        Parameters
        ----------
        seismic_data : Any
            Objeto con datos sísmicos que contenga cortantes
        output_dir : str or Path, optional
            Directorio donde guardar los gráficos
            
        Returns
        -------
        List[str]
            Lista de rutas de archivos generados
        """
        generated_files = []
        
        # Cortante dinámico
        try:
            fig_dynamic, ax_dynamic = self.create_figure(figsize=(10, 12))
            
            if hasattr(seismic_data, 'dynamic_shear_fig'):
                # Si ya existe figura dinámica, usar esa
                if output_dir:
                    path = self.save_figure(seismic_data.dynamic_shear_fig, 
                                          "cortante_dinamico", output_dir)
                    generated_files.append(path)
            else:
                # Crear figura dinámica desde datos
                self._create_shear_figure(ax_dynamic, seismic_data, 'dynamic')
                self.apply_default_styling(ax_dynamic,
                                         title='Cortante Dinámico',
                                         xlabel='Cortante (tonf)',
                                         ylabel='Piso')
                plt.tight_layout()
                
                if output_dir:
                    path = self.save_figure(fig_dynamic, "cortante_dinamico", output_dir)
                    generated_files.append(path)
        
        except Exception as e:
            print(f"Error creando cortante dinámico: {e}")
        
        # Cortante estático
        try:
            fig_static, ax_static = self.create_figure(figsize=(10, 12))
            
            if hasattr(seismic_data, 'static_shear_fig'):
                # Si ya existe figura estática, usar esa
                if output_dir:
                    path = self.save_figure(seismic_data.static_shear_fig, 
                                          "cortante_estatico", output_dir)
                    generated_files.append(path)
            else:
                # Crear figura estática desde datos
                self._create_shear_figure(ax_static, seismic_data, 'static')
                self.apply_default_styling(ax_static,
                                         title='Cortante Estático',
                                         xlabel='Cortante (tonf)',
                                         ylabel='Piso')
                plt.tight_layout()
                
                if output_dir:
                    path = self.save_figure(fig_static, "cortante_estatico", output_dir)
                    generated_files.append(path)
        
        except Exception as e:
            print(f"Error creando cortante estático: {e}")
        
        return generated_files
    
    def _create_shear_figure(self, ax: Axes, seismic_data: Any, analysis_type: str) -> None:
        """
        Función auxiliar para crear gráficos de cortante
        
        Parameters
        ----------
        ax : Axes
            Ejes matplotlib
        seismic_data : Any
            Datos sísmicos
        analysis_type : str
            Tipo de análisis ('dynamic' o 'static')
        """
        try:
            # Intentar extraer datos de cortante
            stories = []
            shear_x = []
            shear_y = []
            
            if hasattr(seismic_data, 'shear_data'):
                data = seismic_data.shear_data.get(analysis_type, {})
                stories = data.get('Story', [])
                shear_x = data.get('V_x', [])
                shear_y = data.get('V_y', [])
            elif hasattr(seismic_data, 'tables'):
                # Buscar en las tablas disponibles
                table_name = f'{analysis_type}_shear_table'
                if hasattr(seismic_data.tables, table_name):
                    table = getattr(seismic_data.tables, table_name)
                    stories = table['Story'].tolist() if 'Story' in table.columns else []
                    shear_x = table['V_x'].tolist() if 'V_x' in table.columns else []
                    shear_y = table['V_y'].tolist() if 'V_y' in table.columns else []
            
            if stories and (shear_x or shear_y):
                # Graficar cortantes
                if shear_x:
                    ax.plot(shear_x, stories, 'ro-',
                           linewidth=self.style_config['linewidth'],
                           markersize=self.style_config['markersize'],
                           label='Cortante X')
                
                if shear_y:
                    ax.plot(shear_y, stories, 'bs-',
                           linewidth=self.style_config['linewidth'],
                           markersize=self.style_config['markersize'],
                           label='Cortante Y')
                
                ax.legend()
                
                # Configurar límites
                if shear_x or shear_y:
                    max_shear = max((max(shear_x) if shear_x else 0), 
                                  (max(shear_y) if shear_y else 0))
                    ax.set_xlim(0, max_shear * 1.1)
                
                ax.set_ylim(0, max(stories) * 1.05)
            else:
                ax.text(0.5, 0.5, f'No hay datos de cortante {analysis_type} disponibles',
                       transform=ax.transAxes, ha='center', va='center')
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creando cortante {analysis_type}:\n{str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
    
    def _create_generic_spectrum(self, periods: np.ndarray, 
                               Z: float = 0.35, U: float = 1.0, 
                               S: float = 1.2, R: float = 6.0) -> np.ndarray:
        """
        Crea un espectro de respuesta genérico
        
        Parameters
        ----------
        periods : np.ndarray
            Períodos del espectro
        Z : float
            Factor de zona
        U : float
            Factor de uso
        S : float
            Factor de suelo
        R : float
            Factor de reducción
            
        Returns
        -------
        np.ndarray
            Aceleraciones espectrales
        """
        # Parámetros típicos
        Tp = 0.6  # Período de inicio de la meseta
        Tl = 2.0  # Período límite
        
        accelerations = np.zeros_like(periods)
        
        for i, T in enumerate(periods):
            if T < Tp:
                Sa = Z * U * S * (1 + 2.5 * T / Tp) / R
            elif T <= Tl:
                Sa = Z * U * S * 2.5 / R
            else:
                Sa = Z * U * S * 2.5 * Tl / (T * R)
            
            accelerations[i] = Sa
        
        return accelerations
    
    def copy_static_images(self, source_dir: Union[str, Path], 
                         output_dir: Union[str, Path]) -> List[str]:
        """
        Copia imágenes estáticas desde directorio de recursos
        
        Parameters
        ----------
        source_dir : str or Path
            Directorio fuente con imágenes
        output_dir : str or Path
            Directorio destino
            
        Returns
        -------
        List[str]
            Lista de archivos copiados
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        
        if not source_path.exists():
            return []
        
        # Crear directorio de destino
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extensiones de imagen soportadas
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.pdf')
        
        copied_files = []
        
        for file_path in source_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                dest_path = output_path / file_path.name
                shutil.copy2(file_path, dest_path)
                copied_files.append(str(dest_path))
        
        return copied_files


# Funciones de utilidad para compatibilidad con código existente
def create_spectrum_figure_legacy(seismic_data: Any, output_dir: str) -> None:
    """
    Función de compatibilidad para crear espectro de respuesta
    Compatible con el código existente de Bolivia/Perú
    """
    processor = GraphProcessor()
    processor.create_response_spectrum(seismic_data, output_dir)


def create_displacement_figure_legacy(seismic_data: Any, output_dir: str) -> None:
    """
    Función de compatibilidad para crear desplazamientos laterales
    Compatible con el código existente de Bolivia/Perú
    """
    processor = GraphProcessor()
    processor.create_displacement_figure(seismic_data, output_dir)


def create_drift_figure_legacy(seismic_data: Any, output_dir: str) -> None:
    """
    Función de compatibilidad para crear derivas
    Compatible con el código existente de Bolivia/Perú
    """
    processor = GraphProcessor()
    processor.create_drift_figure(seismic_data, output_dir)


def create_shear_figures_legacy(seismic_data: Any, output_dir: str) -> None:
    """
    Función de compatibilidad para crear cortantes
    Compatible con el código existente de Bolivia/Perú
    """
    processor = GraphProcessor()
    processor.create_shear_figures(seismic_data, output_dir)


def actualize_images_legacy(seismic_data: Any, output_dir: str) -> None:
    """
    Función de compatibilidad para actualizar todas las imágenes
    Compatible con el código existente de Bolivia/Perú
    """
    processor = GraphProcessor()
    
    # Crear todos los gráficos
    processor.create_response_spectrum(seismic_data, output_dir)
    processor.create_displacement_figure(seismic_data, output_dir) 
    processor.create_drift_figure(seismic_data, output_dir)
    processor.create_shear_figures(seismic_data, output_dir)
    
    # Copiar imágenes estáticas si están disponibles
    if hasattr(seismic_data, 'source_dir'):
        images_dir = Path(output_dir) / "images"
        processor.copy_static_images(seismic_data.source_dir, images_dir)