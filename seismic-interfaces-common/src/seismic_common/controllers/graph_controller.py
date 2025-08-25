"""
Controlador centralizado para manejo de gráficos sísmicos
Proporciona funcionalidades avanzadas para crear, mostrar y exportar gráficos
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from abc import ABC, abstractmethod

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QComboBox, QCheckBox, QGroupBox, QSpinBox, 
    QDoubleSpinBox, QFileDialog, QMessageBox, QSplitter,
    QTabWidget, QScrollArea, QToolBar, QAction, QMenu,
    QColorDialog, QFontDialog, QInputDialog
)
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Importaciones del sistema centralizado
try:
    from .base_controller import BaseController
    from seismic_common.core import (
        get_story_drifts,
        get_joint_displacements,
        get_base_reactions,
        get_modal_data
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("⚠️ Importaciones centralizadas no disponibles para graph_controller")


class GraphType:
    """Enumeración de tipos de gráficos disponibles"""
    DRIFT = "drift"
    DISPLACEMENT = "displacement"
    SHEAR = "shear"
    MODAL = "modal"
    SPECTRUM = "spectrum"
    COMPARISON = "comparison"
    IRREGULARITY = "irregularity"


class BaseGraphDialog(QDialog):
    """
    Diálogo base avanzado para mostrar gráficos matplotlib con herramientas
    """
    
    # Señales
    graph_updated = pyqtSignal()
    graph_exported = pyqtSignal(str)  # ruta del archivo exportado
    
    def __init__(self, figure: Optional[Figure] = None, title: str = "Gráfico", 
                 parent=None, graph_type: str = None):
        """
        Inicializa diálogo de gráfico avanzado
        
        Parameters
        ----------
        figure : Figure, optional
            Figura de matplotlib a mostrar
        title : str
            Título de la ventana
        parent : QWidget, optional
            Widget padre
        graph_type : str, optional
            Tipo de gráfico para personalización
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(900, 700)
        
        self.figure = figure
        self.graph_type = graph_type
        self._original_figure = None  # Para resetear cambios
        
        # Configuración de estilo
        self.graph_config = {
            'grid': True,
            'legend': True,
            'tight_layout': True,
            'font_size': 12,
            'line_width': 2,
            'marker_size': 6,
            'colors': ['red', 'blue', 'green', 'orange', 'purple', 'brown'],
            'line_styles': ['-', '--', '-.', ':'],
            'markers': ['o', 's', '^', 'v', 'D', 'x']
        }
        
        self.setup_ui()
        
        if figure:
            self.set_figure(figure)
    
    def setup_ui(self) -> None:
        """Configura la interfaz de usuario"""
        # Layout principal
        main_layout = QVBoxLayout(self)
        
        # Toolbar
        self.create_toolbar()
        main_layout.addWidget(self.toolbar)
        
        # Splitter para gráfico y controles
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Frame del gráfico
        graph_widget = QFrame()
        self.graph_layout = QVBoxLayout(graph_widget)
        
        # Crear canvas placeholder
        if not self.figure:
            self.figure = Figure(figsize=(10, 8), dpi=100)
            self.figure.add_subplot(111)
        
        self.canvas = FigureCanvas(self.figure)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        
        self.graph_layout.addWidget(self.nav_toolbar)
        self.graph_layout.addWidget(self.canvas)
        
        splitter.addWidget(graph_widget)
        
        # Panel de controles
        self.controls_panel = self.create_controls_panel()
        splitter.addWidget(self.controls_panel)
        
        # Configurar proporción del splitter
        splitter.setSizes([700, 200])
        
        # Botones inferiores
        button_layout = QHBoxLayout()
        
        self.btn_reset = QPushButton("Restablecer")
        self.btn_export = QPushButton("Exportar")
        self.btn_close = QPushButton("Cerrar")
        
        self.btn_reset.clicked.connect(self.reset_figure)
        self.btn_export.clicked.connect(self.export_figure)
        self.btn_close.clicked.connect(self.accept)
        
        button_layout.addStretch()
        button_layout.addWidget(self.btn_reset)
        button_layout.addWidget(self.btn_export)
        button_layout.addWidget(self.btn_close)
        
        main_layout.addLayout(button_layout)
    
    def create_toolbar(self) -> None:
        """Crea toolbar con acciones comunes"""
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(24, 24))
        
        # Acción: Actualizar gráfico
        action_refresh = QAction("🔄", self)
        action_refresh.setToolTip("Actualizar gráfico")
        action_refresh.triggered.connect(self.update_figure)
        self.toolbar.addAction(action_refresh)
        
        self.toolbar.addSeparator()
        
        # Acción: Guardar
        action_save = QAction("💾", self)
        action_save.setToolTip("Guardar gráfico")
        action_save.triggered.connect(self.export_figure)
        self.toolbar.addAction(action_save)
        
        # Acción: Copiar al clipboard
        action_copy = QAction("📋", self)
        action_copy.setToolTip("Copiar al portapapeles")
        action_copy.triggered.connect(self.copy_to_clipboard)
        self.toolbar.addAction(action_copy)
        
        self.toolbar.addSeparator()
        
        # Acción: Configurar colores
        action_colors = QAction("🎨", self)
        action_colors.setToolTip("Configurar colores")
        action_colors.triggered.connect(self.configure_colors)
        self.toolbar.addAction(action_colors)
        
        # Acción: Configurar fuente
        action_font = QAction("🅰", self)
        action_font.setToolTip("Configurar fuente")
        action_font.triggered.connect(self.configure_font)
        self.toolbar.addAction(action_font)
    
    def create_controls_panel(self) -> QFrame:
        """Crea panel de controles lateral"""
        panel = QFrame()
        panel.setMaximumWidth(250)
        layout = QVBoxLayout(panel)
        
        # Configuraciones generales
        general_group = QGroupBox("Configuración General")
        general_layout = QVBoxLayout(general_group)
        
        # Grid
        self.check_grid = QCheckBox("Mostrar rejilla")
        self.check_grid.setChecked(self.graph_config['grid'])
        self.check_grid.toggled.connect(self.update_grid)
        general_layout.addWidget(self.check_grid)
        
        # Leyenda
        self.check_legend = QCheckBox("Mostrar leyenda")
        self.check_legend.setChecked(self.graph_config['legend'])
        self.check_legend.toggled.connect(self.update_legend)
        general_layout.addWidget(self.check_legend)
        
        # Tamaño de fuente
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Tamaño fuente:"))
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(8, 20)
        self.spin_font_size.setValue(self.graph_config['font_size'])
        self.spin_font_size.valueChanged.connect(self.update_font_size)
        font_layout.addWidget(self.spin_font_size)
        general_layout.addLayout(font_layout)
        
        layout.addWidget(general_group)
        
        # Configuraciones específicas según el tipo
        if self.graph_type:
            specific_group = self.create_specific_controls()
            if specific_group:
                layout.addWidget(specific_group)
        
        layout.addStretch()
        return panel
    
    def create_specific_controls(self) -> Optional[QGroupBox]:
        """Crea controles específicos según el tipo de gráfico"""
        if self.graph_type == GraphType.DRIFT:
            return self.create_drift_controls()
        elif self.graph_type == GraphType.DISPLACEMENT:
            return self.create_displacement_controls()
        elif self.graph_type == GraphType.SHEAR:
            return self.create_shear_controls()
        return None
    
    def create_drift_controls(self) -> QGroupBox:
        """Crea controles específicos para gráficos de deriva"""
        group = QGroupBox("Configuración Derivas")
        layout = QVBoxLayout(group)
        
        # Límite de deriva
        limit_layout = QHBoxLayout()
        limit_layout.addWidget(QLabel("Límite deriva:"))
        self.spin_drift_limit = QDoubleSpinBox()
        self.spin_drift_limit.setRange(0.001, 0.020)
        self.spin_drift_limit.setSingleStep(0.001)
        self.spin_drift_limit.setDecimals(4)
        self.spin_drift_limit.setValue(0.007)
        limit_layout.addWidget(self.spin_drift_limit)
        layout.addLayout(limit_layout)
        
        # Mostrar línea de límite
        self.check_drift_limit = QCheckBox("Mostrar línea límite")
        self.check_drift_limit.setChecked(True)
        layout.addWidget(self.check_drift_limit)
        
        return group
    
    def create_displacement_controls(self) -> QGroupBox:
        """Crea controles específicos para gráficos de desplazamiento"""
        group = QGroupBox("Configuración Desplazamientos")
        layout = QVBoxLayout(group)
        
        # Mostrar envolvente
        self.check_envelope = QCheckBox("Mostrar envolvente")
        layout.addWidget(self.check_envelope)
        
        # Normalizar por altura
        self.check_normalize = QCheckBox("Normalizar por altura")
        layout.addWidget(self.check_normalize)
        
        return group
    
    def create_shear_controls(self) -> QGroupBox:
        """Crea controles específicos para gráficos de cortante"""
        group = QGroupBox("Configuración Cortantes")
        layout = QVBoxLayout(group)
        
        # Mostrar estático/dinámico
        self.combo_analysis = QComboBox()
        self.combo_analysis.addItems(["Dinámico", "Estático", "Ambos"])
        layout.addWidget(QLabel("Mostrar análisis:"))
        layout.addWidget(self.combo_analysis)
        
        return group
    
    def set_figure(self, figure: Figure) -> None:
        """
        Establece la figura a mostrar
        
        Parameters
        ----------
        figure : Figure
            Figura de matplotlib
        """
        self.figure = figure
        self._original_figure = figure  # Backup
        
        # Recrear canvas
        if hasattr(self, 'canvas'):
            self.graph_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        
        self.canvas = FigureCanvas(self.figure)
        
        # Recrear navigation toolbar
        if hasattr(self, 'nav_toolbar'):
            self.graph_layout.removeWidget(self.nav_toolbar)
            self.nav_toolbar.deleteLater()
        
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        
        # Añadir al layout
        self.graph_layout.insertWidget(0, self.nav_toolbar)
        self.graph_layout.insertWidget(1, self.canvas)
        
        self.update_figure()
    
    def update_figure(self) -> None:
        """Actualiza el canvas con configuraciones actuales"""
        if not self.figure:
            return
        
        # Aplicar configuraciones
        for ax in self.figure.get_axes():
            ax.grid(self.graph_config['grid'])
            
            if hasattr(ax, 'legend_') and ax.legend_:
                ax.legend_.set_visible(self.graph_config['legend'])
        
        if self.graph_config['tight_layout']:
            self.figure.tight_layout()
        
        self.canvas.draw()
        self.graph_updated.emit()
    
    def update_grid(self, checked: bool) -> None:
        """Actualiza visibilidad de rejilla"""
        self.graph_config['grid'] = checked
        self.update_figure()
    
    def update_legend(self, checked: bool) -> None:
        """Actualiza visibilidad de leyenda"""
        self.graph_config['legend'] = checked
        self.update_figure()
    
    def update_font_size(self, size: int) -> None:
        """Actualiza tamaño de fuente"""
        self.graph_config['font_size'] = size
        # Aplicar a todos los elementos de texto
        if self.figure:
            for ax in self.figure.get_axes():
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
                           ax.get_xticklabels() + ax.get_yticklabels()):
                    if hasattr(item, 'set_fontsize'):
                        item.set_fontsize(size)
        self.update_figure()
    
    def configure_colors(self) -> None:
        """Abre diálogo de configuración de colores"""
        # Implementación básica - se puede extender
        color = QColorDialog.getColor(Qt.blue, self)
        if color.isValid():
            # Aplicar color a las líneas del gráfico
            if self.figure:
                for ax in self.figure.get_axes():
                    for line in ax.get_lines():
                        line.set_color(color.name())
            self.update_figure()
    
    def configure_font(self) -> None:
        """Abre diálogo de configuración de fuente"""
        font, ok = QFontDialog.getFont(self)
        if ok:
            # Aplicar fuente seleccionada
            self.graph_config['font_size'] = font.pointSize()
            self.spin_font_size.setValue(font.pointSize())
            self.update_figure()
    
    def reset_figure(self) -> None:
        """Restablece la figura a su estado original"""
        if self._original_figure:
            self.set_figure(self._original_figure)
    
    def export_figure(self) -> None:
        """Exporta la figura a archivo"""
        if not self.figure:
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Exportar gráfico",
            "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;JPG (*.jpg)"
        )
        
        if file_path:
            try:
                dpi = 300 if file_path.lower().endswith(('.png', '.jpg')) else None
                self.figure.savefig(file_path, dpi=dpi, bbox_inches='tight')
                
                QMessageBox.information(
                    self,
                    "Éxito",
                    f"Gráfico exportado exitosamente:\n{file_path}"
                )
                
                self.graph_exported.emit(file_path)
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error al exportar gráfico:\n{str(e)}"
                )
    
    def copy_to_clipboard(self) -> None:
        """Copia la figura al portapapeles"""
        if not self.figure:
            return
        
        try:
            # Crear imagen en memoria
            from io import BytesIO
            buffer = BytesIO()
            self.figure.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            
            # Copiar al clipboard
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue())
            
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setPixmap(pixmap)
            
            QMessageBox.information(
                self,
                "Éxito",
                "Gráfico copiado al portapapeles"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error al copiar gráfico:\n{str(e)}"
            )


class GraphController:
    """
    Controlador centralizado para manejo de gráficos sísmicos
    """
    
    def __init__(self, parent=None):
        """
        Inicializa el controlador de gráficos
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget padre
        """
        self.parent = parent
        self.current_figures = {}  # Cache de figuras
        self.graph_config = {
            'default_size': (10, 8),
            'default_dpi': 100,
            'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'line_styles': ['-', '--', '-.', ':'],
            'markers': ['o', 's', '^', 'v', 'D']
        }
    
    def create_drift_graph(self, story_drifts: pd.DataFrame, 
                          config: Optional[Dict[str, Any]] = None) -> Figure:
        """
        Crea gráfico de derivas de entrepiso
        
        Parameters
        ----------
        story_drifts : pd.DataFrame
            DataFrame con datos de derivas
        config : Dict, optional
            Configuración específica del gráfico
            
        Returns
        -------
        Figure
            Figura de matplotlib
        """
        config = config or {}
        
        fig = Figure(figsize=self.graph_config['default_size'], 
                    dpi=self.graph_config['default_dpi'])
        ax = fig.add_subplot(111)
        
        if story_drifts.empty:
            ax.text(0.5, 0.5, 'No hay datos de derivas disponibles', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        try:
            # Configurar datos
            heights = story_drifts.get('Height', [])
            drift_x = story_drifts.get('DriftX', story_drifts.get('Drifts_x', []))
            drift_y = story_drifts.get('DriftY', story_drifts.get('Drifts_y', []))
            
            # Graficar
            if len(drift_x) > 0:
                ax.plot(drift_x, heights, 'r-o', label='Dirección X', 
                       linewidth=2, markersize=6)
                
            if len(drift_y) > 0:
                ax.plot(drift_y, heights, 'b-s', label='Dirección Y', 
                       linewidth=2, markersize=6)
            
            # Línea de límite si se especifica
            drift_limit = config.get('drift_limit', 0.007)
            if drift_limit and len(heights) > 0:
                ax.axvline(x=drift_limit, color='green', linestyle='--', 
                          alpha=0.7, label=f'Límite ({drift_limit:.3f})')
            
            # Configurar ejes y labels
            ax.set_xlabel('Deriva')
            ax.set_ylabel('Altura (m)')
            ax.set_title('Derivas de Entrepiso')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend()
            
            # Limites
            if len(drift_x) > 0 or len(drift_y) > 0:
                max_drift = max([max(drift_x) if len(drift_x) > 0 else 0,
                               max(drift_y) if len(drift_y) > 0 else 0,
                               drift_limit * 1.2])
                ax.set_xlim(0, max_drift)
                
            if len(heights) > 0:
                ax.set_ylim(0, max(heights) * 1.05)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creando gráfico: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
        
        fig.tight_layout()
        self.current_figures[GraphType.DRIFT] = fig
        return fig
    
    def create_displacement_graph(self, joint_disps: pd.DataFrame,
                                config: Optional[Dict[str, Any]] = None) -> Figure:
        """
        Crea gráfico de desplazamientos laterales
        
        Parameters
        ----------
        joint_disps : pd.DataFrame
            DataFrame con desplazamientos
        config : Dict, optional
            Configuración específica
            
        Returns
        -------
        Figure
            Figura de matplotlib
        """
        config = config or {}
        
        fig = Figure(figsize=self.graph_config['default_size'],
                    dpi=self.graph_config['default_dpi'])
        ax = fig.add_subplot(111)
        
        if joint_disps.empty:
            ax.text(0.5, 0.5, 'No hay datos de desplazamientos disponibles',
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        try:
            # Procesar datos básicos
            heights = joint_disps.get('Height', [])
            disp_x = joint_disps.get('UX', joint_disps.get('DisplacementX', []))
            disp_y = joint_disps.get('UY', joint_disps.get('DisplacementY', []))
            
            # Graficar
            if len(disp_x) > 0:
                ax.plot(disp_x, heights, 'r-o', label='Desplazamiento X',
                       linewidth=2, markersize=6)
                
            if len(disp_y) > 0:
                ax.plot(disp_y, heights, 'b-s', label='Desplazamiento Y',
                       linewidth=2, markersize=6)
            
            # Configurar
            ax.set_xlabel('Desplazamiento (mm)')
            ax.set_ylabel('Altura (m)')
            ax.set_title('Desplazamientos Laterales')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creando gráfico: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
        
        fig.tight_layout()
        self.current_figures[GraphType.DISPLACEMENT] = fig
        return fig
    
    def create_shear_graph(self, base_reactions: pd.DataFrame, analysis_type: str = 'dynamic',
                          config: Optional[Dict[str, Any]] = None) -> Figure:
        """
        Crea gráfico de fuerzas cortantes
        
        Parameters
        ----------
        base_reactions : pd.DataFrame
            DataFrame con reacciones en la base
        analysis_type : str
            Tipo de análisis ('dynamic' o 'static')
        config : Dict, optional
            Configuración específica
            
        Returns
        -------
        Figure
            Figura de matplotlib
        """
        config = config or {}
        
        fig = Figure(figsize=self.graph_config['default_size'],
                    dpi=self.graph_config['default_dpi'])
        ax = fig.add_subplot(111)
        
        if base_reactions.empty:
            ax.text(0.5, 0.5, 'No hay datos de cortantes disponibles',
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        try:
            # Procesar datos
            heights = base_reactions.get('Height', [])
            shear_x = base_reactions.get('VX', base_reactions.get('ShearX', []))
            shear_y = base_reactions.get('VY', base_reactions.get('ShearY', []))
            
            # Graficar
            if len(shear_x) > 0:
                ax.plot(shear_x, heights, 'r-o', label='Cortante X',
                       linewidth=2, markersize=6)
                
            if len(shear_y) > 0:
                ax.plot(shear_y, heights, 'b-s', label='Cortante Y',
                       linewidth=2, markersize=6)
            
            # Configurar
            ax.set_xlabel('Fuerza Cortante (tonf)')
            ax.set_ylabel('Altura (m)')
            title = f'Fuerzas Cortantes {"Dinámicas" if analysis_type == "dynamic" else "Estáticas"}'
            ax.set_title(title)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creando gráfico: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
        
        fig.tight_layout()
        self.current_figures[f"{GraphType.SHEAR}_{analysis_type}"] = fig
        return fig
    
    def create_modal_graph(self, modal_data: pd.DataFrame,
                          config: Optional[Dict[str, Any]] = None) -> Figure:
        """
        Crea gráfico de parámetros modales
        
        Parameters
        ----------
        modal_data : pd.DataFrame
            DataFrame con datos modales
        config : Dict, optional
            Configuración específica
            
        Returns
        -------
        Figure
            Figura de matplotlib
        """
        config = config or {}
        
        fig = Figure(figsize=(12, 8), dpi=self.graph_config['default_dpi'])
        
        if modal_data.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No hay datos modales disponibles',
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        try:
            # Crear subgráficos
            ax1 = fig.add_subplot(221)  # Periodos
            ax2 = fig.add_subplot(222)  # Masas participativas
            ax3 = fig.add_subplot(223)  # Frecuencias
            ax4 = fig.add_subplot(224)  # Factores de participación
            
            modes = modal_data.get('Mode', range(1, len(modal_data) + 1))
            periods = modal_data.get('Period', [])
            frequencies = modal_data.get('Frequency', [])
            mass_x = modal_data.get('UX', modal_data.get('MassX', []))
            mass_y = modal_data.get('UY', modal_data.get('MassY', []))
            
            # Gráfico de periodos
            if len(periods) > 0:
                ax1.bar(modes, periods, alpha=0.7, color=self.graph_config['color_palette'][0])
                ax1.set_xlabel('Modo')
                ax1.set_ylabel('Periodo (s)')
                ax1.set_title('Periodos de Vibración')
                ax1.grid(True, alpha=0.3)
            
            # Gráfico de masas participativas
            if len(mass_x) > 0 and len(mass_y) > 0:
                width = 0.35
                x_pos = np.arange(len(modes))
                ax2.bar(x_pos - width/2, mass_x, width, label='UX', alpha=0.7,
                       color=self.graph_config['color_palette'][1])
                ax2.bar(x_pos + width/2, mass_y, width, label='UY', alpha=0.7,
                       color=self.graph_config['color_palette'][2])
                ax2.set_xlabel('Modo')
                ax2.set_ylabel('Masa Participativa')
                ax2.set_title('Masas Participativas')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(modes)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Línea de 90%
                ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90%')
            
            # Gráfico de frecuencias
            if len(frequencies) > 0:
                ax3.plot(modes, frequencies, 'o-', linewidth=2, markersize=6,
                        color=self.graph_config['color_palette'][3])
                ax3.set_xlabel('Modo')
                ax3.set_ylabel('Frecuencia (Hz)')
                ax3.set_title('Frecuencias de Vibración')
                ax3.grid(True, alpha=0.3)
            
            # Gráfico adicional según disponibilidad
            participation_x = modal_data.get('PartFactorUX', [])
            participation_y = modal_data.get('PartFactorUY', [])
            
            if len(participation_x) > 0 and len(participation_y) > 0:
                ax4.plot(modes, participation_x, 'o-', label='Factor X', 
                        linewidth=2, markersize=6, color=self.graph_config['color_palette'][1])
                ax4.plot(modes, participation_y, 's-', label='Factor Y',
                        linewidth=2, markersize=6, color=self.graph_config['color_palette'][2])
                ax4.set_xlabel('Modo')
                ax4.set_ylabel('Factor de Participación')
                ax4.set_title('Factores de Participación Modal')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Datos adicionales\nno disponibles',
                        transform=ax4.transAxes, ha='center', va='center')
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error creando gráfico modal: {str(e)}',
                    transform=ax1.transAxes, ha='center', va='center')
        
        fig.tight_layout()
        self.current_figures[GraphType.MODAL] = fig
        return fig
    
    def create_comparison_graph(self, comparison_data: List[Dict[str, Any]],
                              config: Optional[Dict[str, Any]] = None) -> Figure:
        """
        Crea gráfico comparativo entre diferentes configuraciones
        
        Parameters
        ----------
        comparison_data : List[Dict]
            Lista de diccionarios con datos de comparación
        config : Dict, optional
            Configuración específica
            
        Returns
        -------
        Figure
            Figura de matplotlib
        """
        config = config or {}
        
        fig = Figure(figsize=(14, 10), dpi=self.graph_config['default_dpi'])
        
        if not comparison_data:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No hay datos de comparación disponibles',
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        try:
            # Crear subgráficos para comparación
            ax1 = fig.add_subplot(221)  # Periodos fundamentales
            ax2 = fig.add_subplot(222)  # Masas participativas
            ax3 = fig.add_subplot(223)  # Relación de periodos
            ax4 = fig.add_subplot(224)  # Scatter Tx vs Ty
            
            # Extraer datos
            configurations = [item.get('config_name', f"Config {i+1}") 
                            for i, item in enumerate(comparison_data)]
            tx_values = [item.get('Tx', 0) for item in comparison_data]
            ty_values = [item.get('Ty', 0) for item in comparison_data]
            mpx_values = [item.get('MP_x', 0) for item in comparison_data]
            mpy_values = [item.get('MP_y', 0) for item in comparison_data]
            
            # Gráfico 1: Periodos fundamentales
            x_pos = np.arange(len(configurations))
            width = 0.35
            
            ax1.bar(x_pos - width/2, tx_values, width, label='Tx', alpha=0.7,
                   color=self.graph_config['color_palette'][1])
            ax1.bar(x_pos + width/2, ty_values, width, label='Ty', alpha=0.7,
                   color=self.graph_config['color_palette'][2])
            ax1.set_xlabel('Configuración')
            ax1.set_ylabel('Periodo (s)')
            ax1.set_title('Periodos Fundamentales')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(configurations, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Masas participativas
            ax2.bar(x_pos - width/2, mpx_values, width, label='MP_x', alpha=0.7,
                   color=self.graph_config['color_palette'][1])
            ax2.bar(x_pos + width/2, mpy_values, width, label='MP_y', alpha=0.7,
                   color=self.graph_config['color_palette'][2])
            ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Mínimo 90%')
            ax2.set_xlabel('Configuración')
            ax2.set_ylabel('Masa Participativa')
            ax2.set_title('Masas Participativas Acumuladas')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(configurations, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.0)
            
            # Gráfico 3: Relación de periodos
            period_ratios = [tx/ty if ty != 0 else 0 for tx, ty in zip(tx_values, ty_values)]
            ax3.bar(x_pos, period_ratios, alpha=0.7, color=self.graph_config['color_palette'][3])
            ax3.set_xlabel('Configuración')
            ax3.set_ylabel('Tx / Ty')
            ax3.set_title('Relación de Periodos')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(configurations, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
            
            # Gráfico 4: Scatter plot Tx vs Ty
            colors = [self.graph_config['color_palette'][i % len(self.graph_config['color_palette'])] 
                     for i in range(len(configurations))]
            ax4.scatter(tx_values, ty_values, s=100, alpha=0.7, c=colors)
            
            # Añadir etiquetas a los puntos
            for i, config in enumerate(configurations):
                ax4.annotate(config, (tx_values[i], ty_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax4.set_xlabel('Tx (s)')
            ax4.set_ylabel('Ty (s)')
            ax4.set_title('Tx vs Ty')
            ax4.grid(True, alpha=0.3)
            
            # Línea de igualdad
            max_period = max(max(tx_values), max(ty_values)) if tx_values and ty_values else 1
            ax4.plot([0, max_period], [0, max_period], 'k--', alpha=0.5, label='Tx = Ty')
            ax4.legend()
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error creando gráfico comparativo: {str(e)}',
                    transform=ax1.transAxes, ha='center', va='center')
        
        fig.tight_layout()
        self.current_figures[GraphType.COMPARISON] = fig
        return fig
    
    def create_spectrum_graph(self, spectrum_data: pd.DataFrame, 
                            config: Optional[Dict[str, Any]] = None) -> Figure:
        """
        Crea gráfico del espectro de respuesta
        
        Parameters
        ----------
        spectrum_data : pd.DataFrame
            DataFrame con datos del espectro
        config : Dict, optional
            Configuración específica
            
        Returns
        -------
        Figure
            Figura de matplotlib
        """
        config = config or {}
        
        fig = Figure(figsize=self.graph_config['default_size'],
                    dpi=self.graph_config['default_dpi'])
        ax = fig.add_subplot(111)
        
        if spectrum_data.empty:
            ax.text(0.5, 0.5, 'No hay datos de espectro disponibles',
                   transform=ax.transAxes, ha='center', va='center')
            return fig
        
        try:
            periods = spectrum_data.get('Period', spectrum_data.get('T', []))
            accelerations = spectrum_data.get('Acceleration', spectrum_data.get('Sa', []))
            
            if len(periods) > 0 and len(accelerations) > 0:
                ax.plot(periods, accelerations, 'b-', linewidth=3, label='Espectro de Diseño')
                ax.fill_between(periods, accelerations, alpha=0.3, color='lightblue')
                
                # Marcar puntos característicos si están disponibles
                tp_value = config.get('Tp', None)
                tl_value = config.get('TL', None)
                
                if tp_value:
                    ax.axvline(x=tp_value, color='red', linestyle='--', alpha=0.7, 
                              label=f'Tp = {tp_value:.2f} s')
                
                if tl_value:
                    ax.axvline(x=tl_value, color='green', linestyle='--', alpha=0.7,
                              label=f'TL = {tl_value:.2f} s')
                
                ax.set_xlabel('Periodo T (s)')
                ax.set_ylabel('Aceleración Espectral Sa (g)')
                ax.set_title('Espectro de Respuesta Sísmica')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_xlim(0, max(periods) * 1.1)
                ax.set_ylim(0, max(accelerations) * 1.1)
            else:
                ax.text(0.5, 0.5, 'Datos de espectro incompletos',
                       transform=ax.transAxes, ha='center', va='center')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creando espectro: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center')
        
        fig.tight_layout()
        self.current_figures[GraphType.SPECTRUM] = fig
        return fig
    
    def show_graph(self, figure: Figure, title: str = "Gráfico", 
                  graph_type: str = None) -> BaseGraphDialog:
        """
        Muestra un gráfico en un diálogo
        
        Parameters
        ----------
        figure : Figure
            Figura de matplotlib
        title : str
            Título del diálogo
        graph_type : str, optional
            Tipo de gráfico para funcionalidades específicas
            
        Returns
        -------
        BaseGraphDialog
            Diálogo del gráfico
        """
        dialog = BaseGraphDialog(figure, title, self.parent, graph_type)
        dialog.exec_()
        return dialog
    
    def show_drift_graph(self, story_drifts: pd.DataFrame, 
                        config: Optional[Dict[str, Any]] = None) -> None:
        """
        Crea y muestra gráfico de derivas
        
        Parameters
        ----------
        story_drifts : pd.DataFrame
            DataFrame con datos de derivas
        config : Dict, optional
            Configuración del gráfico
        """
        figure = self.create_drift_graph(story_drifts, config)
        self.show_graph(figure, "Derivas de Entrepiso", GraphType.DRIFT)
    
    def show_displacement_graph(self, joint_disps: pd.DataFrame,
                               config: Optional[Dict[str, Any]] = None) -> None:
        """
        Crea y muestra gráfico de desplazamientos
        
        Parameters
        ----------
        joint_disps : pd.DataFrame
            DataFrame con desplazamientos
        config : Dict, optional
            Configuración del gráfico
        """
        figure = self.create_displacement_graph(joint_disps, config)
        self.show_graph(figure, "Desplazamientos Laterales", GraphType.DISPLACEMENT)
    
    def show_shear_graph(self, base_reactions: pd.DataFrame, analysis_type: str = 'dynamic',
                        config: Optional[Dict[str, Any]] = None) -> None:
        """
        Crea y muestra gráfico de cortantes
        
        Parameters
        ----------
        base_reactions : pd.DataFrame
            DataFrame con reacciones
        analysis_type : str
            Tipo de análisis
        config : Dict, optional
            Configuración del gráfico
        """
        figure = self.create_shear_graph(base_reactions, analysis_type, config)
        title = f'Cortantes {"Dinámicas" if analysis_type == "dynamic" else "Estáticas"}'
        self.show_graph(figure, title, GraphType.SHEAR)
    
    def show_modal_graph(self, modal_data: pd.DataFrame,
                        config: Optional[Dict[str, Any]] = None) -> None:
        """
        Crea y muestra gráfico modal
        
        Parameters
        ----------
        modal_data : pd.DataFrame
            DataFrame con datos modales
        config : Dict, optional
            Configuración del gráfico
        """
        figure = self.create_modal_graph(modal_data, config)
        self.show_graph(figure, "Análisis Modal", GraphType.MODAL)
    
    def show_comparison_graph(self, comparison_data: List[Dict[str, Any]],
                            config: Optional[Dict[str, Any]] = None) -> None:
        """
        Crea y muestra gráfico comparativo
        
        Parameters
        ----------
        comparison_data : List[Dict]
            Datos de comparación
        config : Dict, optional
            Configuración del gráfico
        """
        figure = self.create_comparison_graph(comparison_data, config)
        self.show_graph(figure, "Comparación de Configuraciones", GraphType.COMPARISON)
    
    def show_spectrum_graph(self, spectrum_data: pd.DataFrame,
                           config: Optional[Dict[str, Any]] = None) -> None:
        """
        Crea y muestra gráfico de espectro
        
        Parameters
        ----------
        spectrum_data : pd.DataFrame
            Datos del espectro
        config : Dict, optional
            Configuración del gráfico
        """
        figure = self.create_spectrum_graph(spectrum_data, config)
        self.show_graph(figure, "Espectro de Respuesta", GraphType.SPECTRUM)
    
    def export_all_graphs(self, output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """
        Exporta todas las figuras actuales
        
        Parameters
        ----------
        output_dir : str
            Directorio de salida
        formats : List[str], optional
            Formatos de exportación (por defecto ['png', 'pdf'])
            
        Returns
        -------
        Dict[str, str]
            Mapeo de nombres de gráficos a rutas de archivos
        """
        formats = formats or ['png', 'pdf']
        exported_files = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        for graph_type, figure in self.current_figures.items():
            for fmt in formats:
                filename = f"{graph_type}_graph.{fmt}"
                filepath = os.path.join(output_dir, filename)
                
                try:
                    dpi = 300 if fmt in ['png', 'jpg'] else None
                    figure.savefig(filepath, dpi=dpi, bbox_inches='tight')
                    exported_files[f"{graph_type}_{fmt}"] = filepath
                except Exception as e:
                    print(f"Error exportando {filename}: {str(e)}")
        
        return exported_files
    
    def get_figure(self, graph_type: str) -> Optional[Figure]:
        """
        Obtiene una figura específica del cache
        
        Parameters
        ----------
        graph_type : str
            Tipo de gráfico
            
        Returns
        -------
        Figure or None
            Figura si existe en cache
        """
        return self.current_figures.get(graph_type)
    
    def clear_cache(self) -> None:
        """Limpia el cache de figuras"""
        self.current_figures.clear()
    
    def set_default_config(self, config: Dict[str, Any]) -> None:
        """
        Establece configuración por defecto
        
        Parameters
        ----------
        config : Dict
            Nueva configuración por defecto
        """
        self.graph_config.update(config)
    
    def get_available_graph_types(self) -> List[str]:
        """
        Obtiene lista de tipos de gráficos disponibles
        
        Returns
        -------
        List[str]
            Lista de tipos disponibles
        """
        return [
            GraphType.DRIFT,
            GraphType.DISPLACEMENT,
            GraphType.SHEAR,
            GraphType.MODAL,
            GraphType.SPECTRUM,
            GraphType.COMPARISON,
            GraphType.IRREGULARITY
        ]