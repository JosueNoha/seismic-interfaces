"""
Diálogo centralizado para mostrar gráficos de análisis sísmicos
Centraliza la funcionalidad común de mostrar gráficos matplotlib en diálogos
"""

from typing import Optional, Dict, Any, List, Union, Callable
import os
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QPushButton, QFileDialog, QMessageBox, QApplication,
    QSplitter, QToolBar, QAction, QMenu, QColorDialog,
    QFontDialog, QInputDialog, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
    QScrollArea, QTabWidget
)
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import pandas as pd
import numpy as np


class SeismicGraphDialog(QDialog):
    """
    Diálogo centralizado para mostrar gráficos de análisis sísmico
    Reemplaza las clases FormCanvas y similares de los proyectos existentes
    """
    
    # Señales
    graph_exported = pyqtSignal(str)  # ruta del archivo exportado
    graph_updated = pyqtSignal()      # gráfico actualizado
    graph_closed = pyqtSignal()       # diálogo cerrado
    
    def __init__(self, figure: Optional[Figure] = None, 
                 title: str = "Gráfico", 
                 graph_title: str = "Gráfico", 
                 parent=None,
                 show_controls: bool = False):
        """
        Inicializa el diálogo de gráfico sísmico
        
        Parameters
        ----------
        figure : Figure, optional
            Figura de matplotlib a mostrar
        title : str
            Título de la ventana del diálogo
        graph_title : str
            Título mostrado encima del gráfico
        parent : QWidget, optional
            Widget padre
        show_controls : bool
            Si mostrar panel de controles avanzados
        """
        super().__init__(parent)
        
        self.figure = figure
        self.graph_title = graph_title
        self.show_controls = show_controls
        self._original_figure = None  # Para restablecer cambios
        
        # Configuración de la ventana
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(800, 600)
        
        # Configuraciones por defecto
        self.graph_config = {
            'grid': True,
            'legend': True,
            'tight_layout': True,
            'font_size': 12,
            'colors': ['red', 'blue', 'green', 'orange', 'purple', 'brown'],
            'line_styles': ['-', '--', '-.', ':'],
            'markers': ['o', 's', '^', 'v', 'D', 'x']
        }
        
        # Configurar interfaz
        self.setup_ui()
        self.setup_canvas()
        self.setup_connections()
        
        # Configurar figura inicial
        if self.figure:
            self.set_figure(self.figure)
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Layout principal
        if self.show_controls:
            main_layout = QHBoxLayout(self)
            
            # Splitter para gráfico y controles
            splitter = QSplitter(Qt.Horizontal)
            main_layout.addWidget(splitter)
            
            # Widget del gráfico
            graph_widget = self.create_graph_widget()
            splitter.addWidget(graph_widget)
            
            # Panel de controles
            controls_widget = self.create_controls_panel()
            splitter.addWidget(controls_widget)
            
            # Configurar proporción del splitter
            splitter.setSizes([700, 200])
            
        else:
            # Layout simple
            self.main_layout = QHBoxLayout(self)
            self.vertical_layout = QVBoxLayout()
            
            # Label de título
            self.title_label = QLabel(self.graph_title)
            self.setup_title_font()
            self.title_label.setAlignment(Qt.AlignCenter)
            self.title_label.setMaximumSize(QtCore.QSize(16777215, 40))
            self.title_label.setObjectName("title_label")
            
            # Frame para el gráfico
            self.frame_plot = QFrame()
            self.frame_plot.setFrameShape(QFrame.StyledPanel)
            self.frame_plot.setFrameShadow(QFrame.Raised)
            self.frame_plot.setObjectName("framePlot")  # Compatible con código existente
            
            # Botones
            self.buttons_layout = QHBoxLayout()
            self.setup_buttons()
            
            # Agregar widgets al layout
            self.vertical_layout.addWidget(self.title_label)
            self.vertical_layout.addWidget(self.frame_plot)
            self.vertical_layout.addLayout(self.buttons_layout)
            
            self.main_layout.addLayout(self.vertical_layout)
    
    def create_graph_widget(self):
        """Crea el widget contenedor del gráfico"""
        graph_widget = QFrame()
        graph_layout = QVBoxLayout(graph_widget)
        
        # Título del gráfico
        self.title_label = QLabel(self.graph_title)
        self.setup_title_font()
        self.title_label.setAlignment(Qt.AlignCenter)
        graph_layout.addWidget(self.title_label)
        
        # Frame para el canvas
        self.frame_plot = QFrame()
        self.frame_plot.setFrameShape(QFrame.StyledPanel)
        self.frame_plot.setFrameShadow(QFrame.Raised)
        graph_layout.addWidget(self.frame_plot)
        
        return graph_widget
    
    def create_controls_panel(self):
        """Crea el panel de controles avanzados"""
        controls_widget = QFrame()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Título del panel
        controls_title = QLabel("Controles")
        controls_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        controls_layout.addWidget(controls_title)
        
        # Scroll area para controles
        scroll = QScrollArea()
        scroll_widget = QFrame()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Grupo: Apariencia
        appearance_group = QGroupBox("Apariencia")
        appearance_layout = QGridLayout(appearance_group)
        
        # Grid
        self.grid_check = QCheckBox("Mostrar grilla")
        self.grid_check.setChecked(self.graph_config['grid'])
        appearance_layout.addWidget(self.grid_check, 0, 0, 1, 2)
        
        # Legend
        self.legend_check = QCheckBox("Mostrar leyenda")
        self.legend_check.setChecked(self.graph_config['legend'])
        appearance_layout.addWidget(self.legend_check, 1, 0, 1, 2)
        
        # Font size
        appearance_layout.addWidget(QLabel("Tamaño fuente:"), 2, 0)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(self.graph_config['font_size'])
        appearance_layout.addWidget(self.font_size_spin, 2, 1)
        
        scroll_layout.addWidget(appearance_group)
        
        # Grupo: Exportación
        export_group = QGroupBox("Exportación")
        export_layout = QVBoxLayout(export_group)
        
        self.export_btn = QPushButton("Exportar Imagen")
        self.save_data_btn = QPushButton("Guardar Datos")
        
        export_layout.addWidget(self.export_btn)
        export_layout.addWidget(self.save_data_btn)
        
        scroll_layout.addWidget(export_group)
        
        # Grupo: Acciones
        actions_group = QGroupBox("Acciones")
        actions_layout = QVBoxLayout(actions_group)
        
        self.reset_btn = QPushButton("Restablecer")
        self.refresh_btn = QPushButton("Actualizar")
        
        actions_layout.addWidget(self.reset_btn)
        actions_layout.addWidget(self.refresh_btn)
        
        scroll_layout.addWidget(actions_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        controls_layout.addWidget(scroll)
        
        return controls_widget
    
    def setup_title_font(self):
        """Configura la fuente del título para mantener consistencia"""
        font = QFont()
        font.setFamily("Montserrat")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.title_label.setFont(font)
    
    def setup_buttons(self):
        """Configura los botones del diálogo simple"""
        # Botón exportar
        self.export_btn = QPushButton("Exportar")
        self.export_btn.setToolTip("Exportar gráfico como imagen")
        
        # Botón cerrar
        self.close_btn = QPushButton("Cerrar")
        
        # Agregar botones al layout
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.export_btn)
        self.buttons_layout.addWidget(self.close_btn)
    
    def setup_canvas(self):
        """Configura el canvas de matplotlib"""
        # Crear figura por defecto si no se proporciona
        if not self.figure:
            self.figure = Figure(figsize=(10, 8), dpi=100)
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Gráfico vacío', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=16, alpha=0.5)
        
        # Crear canvas
        self.canvas = FigureCanvas(self.figure)
        
        # Crear toolbar de navegación solo si hay controles
        if self.show_controls:
            self.nav_toolbar = NavigationToolbar(self.canvas, self)
        
        # Layout del canvas
        self.canvas_layout = QHBoxLayout(self.frame_plot)
        
        if hasattr(self, 'nav_toolbar'):
            canvas_widget = QFrame()
            canvas_widget_layout = QVBoxLayout(canvas_widget)
            canvas_widget_layout.addWidget(self.nav_toolbar)
            canvas_widget_layout.addWidget(self.canvas)
            self.canvas_layout.addWidget(canvas_widget)
        else:
            self.canvas_layout.addWidget(self.canvas)
    
    def setup_connections(self):
        """Configura las conexiones de señales y slots"""
        # Botones básicos
        self.export_btn.clicked.connect(self.export_figure)
        if hasattr(self, 'close_btn'):
            self.close_btn.clicked.connect(self.close_dialog)
        
        # Controles avanzados si existen
        if self.show_controls:
            self.grid_check.toggled.connect(self.toggle_grid)
            self.legend_check.toggled.connect(self.toggle_legend)
            self.font_size_spin.valueChanged.connect(self.change_font_size)
            self.reset_btn.clicked.connect(self.reset_figure)
            self.refresh_btn.clicked.connect(self.refresh_figure)
            if hasattr(self, 'save_data_btn'):
                self.save_data_btn.clicked.connect(self.save_figure_data)
        
        # Señal de cierre del diálogo
        self.finished.connect(lambda: self.graph_closed.emit())
    
    def set_figure(self, figure: Figure):
        """
        Establece una nueva figura en el diálogo
        
        Parameters
        ----------
        figure : Figure
            Nueva figura de matplotlib
        """
        self.figure = figure
        self._original_figure = figure  # Guardar para reset
        
        # Limpiar layout anterior
        if self.canvas_layout:
            while self.canvas_layout.count():
                child = self.canvas_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        
        # Crear nuevo canvas
        self.canvas = FigureCanvas(self.figure)
        
        if self.show_controls and hasattr(self, 'nav_toolbar'):
            self.nav_toolbar = NavigationToolbar(self.canvas, self)
            canvas_widget = QFrame()
            canvas_widget_layout = QVBoxLayout(canvas_widget)
            canvas_widget_layout.addWidget(self.nav_toolbar)
            canvas_widget_layout.addWidget(self.canvas)
            self.canvas_layout.addWidget(canvas_widget)
        else:
            self.canvas_layout.addWidget(self.canvas)
        
        # Dibujar
        self.canvas.draw()
        self.graph_updated.emit()
    
    def update_canvas(self):
        """Actualiza el canvas redibujándolo"""
        if hasattr(self, 'canvas'):
            self.canvas.draw()
    
    def set_title(self, title: str):
        """
        Establece el título del gráfico
        
        Parameters
        ----------
        title : str
            Nuevo título
        """
        self.graph_title = title
        self.title_label.setText(title)
    
    def clear_figure(self):
        """Limpia la figura actual"""
        if self.figure:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Gráfico limpiado', transform=ax.transAxes,
                   ha='center', va='center', fontsize=16, alpha=0.5)
            self.update_canvas()
    
    def toggle_grid(self, checked: bool):
        """Alterna la visibilidad de la grilla"""
        if self.figure and self.figure.axes:
            for ax in self.figure.axes:
                ax.grid(checked)
            self.update_canvas()
    
    def toggle_legend(self, checked: bool):
        """Alterna la visibilidad de la leyenda"""
        if self.figure and self.figure.axes:
            for ax in self.figure.axes:
                if checked:
                    ax.legend()
                else:
                    legend = ax.get_legend()
                    if legend:
                        legend.remove()
            self.update_canvas()
    
    def change_font_size(self, size: int):
        """Cambia el tamaño de fuente del gráfico"""
        if self.figure:
            plt.rcParams.update({'font.size': size})
            self.update_canvas()
    
    def reset_figure(self):
        """Restablece la figura al estado original"""
        if self._original_figure:
            self.set_figure(self._original_figure)
    
    def refresh_figure(self):
        """Actualiza/refresca la figura"""
        self.update_canvas()
        self.graph_updated.emit()
    
    def export_figure(self):
        """Exporta el gráfico como imagen"""
        if not self.figure:
            QMessageBox.warning(self, "Advertencia", "No hay gráfico para exportar.")
            return
        
        # Diálogo para seleccionar archivo
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar gráfico",
            f"{self.graph_title.replace(' ', '_')}.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;JPG Files (*.jpg)"
        )
        
        if file_path:
            try:
                # Configurar DPI para alta calidad
                dpi = 300 if file_path.endswith('.png') else 'figure'
                
                self.figure.savefig(file_path, dpi=dpi, bbox_inches='tight', 
                                  facecolor='white', edgecolor='none')
                
                QMessageBox.information(
                    self, "Éxito", 
                    f"Gráfico exportado correctamente a:\n{file_path}"
                )
                self.graph_exported.emit(file_path)
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", 
                    f"Error al exportar el gráfico:\n{str(e)}"
                )
    
    def save_figure_data(self):
        """Guarda los datos del gráfico (si están disponibles)"""
        QMessageBox.information(
            self, "Información", 
            "Funcionalidad de guardado de datos no implementada."
        )
    
    def close_dialog(self):
        """Cierra el diálogo"""
        self.accept()
    
    # Métodos de compatibilidad con código existente
    def exec_(self) -> int:
        """Ejecuta el diálogo de forma modal - Compatible con código existente"""
        return super().exec_()
    
    @property
    def ui(self):
        """
        Propiedad de compatibilidad para acceder a elementos UI
        Permite usar dialog.ui.framePlot como en el código existente
        """
        return self
    
    @property 
    def framePlot(self):
        """Compatibilidad: acceso directo al frame del gráfico"""
        return self.frame_plot
    
    @property
    def label_2(self):
        """Compatibilidad: acceso directo al label de título"""
        return self.title_label


# Funciones de conveniencia para crear diálogos específicos
def create_drift_graph_dialog(figure: Figure, parent=None) -> SeismicGraphDialog:
    """
    Crea un diálogo para mostrar gráfico de derivas
    
    Parameters
    ----------
    figure : Figure
        Figura con el gráfico de derivas
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicGraphDialog
        Diálogo configurado para derivas
    """
    dialog = SeismicGraphDialog(
        figure=figure,
        title="Derivas de Entrepiso",
        graph_title="Derivas",
        parent=parent
    )
    return dialog


def create_displacement_graph_dialog(figure: Figure, parent=None) -> SeismicGraphDialog:
    """
    Crea un diálogo para mostrar gráfico de desplazamientos
    
    Parameters
    ----------
    figure : Figure
        Figura con el gráfico de desplazamientos
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicGraphDialog
        Diálogo configurado para desplazamientos
    """
    dialog = SeismicGraphDialog(
        figure=figure,
        title="Desplazamientos Laterales",
        graph_title="Desplazamientos",
        parent=parent
    )
    return dialog


def create_shear_graph_dialog(figure: Figure, analysis_type: str = "dynamic", parent=None) -> SeismicGraphDialog:
    """
    Crea un diálogo para mostrar gráfico de cortantes
    
    Parameters
    ----------
    figure : Figure
        Figura con el gráfico de cortantes
    analysis_type : str
        Tipo de análisis ('dynamic' o 'static')
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicGraphDialog
        Diálogo configurado para cortantes
    """
    title = f"Cortantes {'Dinámicos' if analysis_type == 'dynamic' else 'Estáticos'}"
    dialog = SeismicGraphDialog(
        figure=figure,
        title=title,
        graph_title="Cortantes",
        parent=parent
    )
    return dialog


def create_spectrum_graph_dialog(figure: Figure, parent=None) -> SeismicGraphDialog:
    """
    Crea un diálogo para mostrar espectro de respuesta
    
    Parameters
    ----------
    figure : Figure
        Figura con el espectro de respuesta
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicGraphDialog
        Diálogo configurado para espectro
    """
    dialog = SeismicGraphDialog(
        figure=figure,
        title="Espectro de Respuesta",
        graph_title="Espectro",
        parent=parent
    )
    return dialog


def create_advanced_graph_dialog(figure: Figure, title: str = "Gráfico Avanzado", parent=None) -> SeismicGraphDialog:
    """
    Crea un diálogo avanzado con controles
    
    Parameters
    ----------
    figure : Figure
        Figura a mostrar
    title : str
        Título del diálogo
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicGraphDialog
        Diálogo avanzado con controles
    """
    dialog = SeismicGraphDialog(
        figure=figure,
        title=title,
        graph_title=title,
        parent=parent,
        show_controls=True
    )
    return dialog


# Clase de compatibilidad para código existente
class FormCanvas(SeismicGraphDialog):
    """
    Clase de compatibilidad para el código existente
    Permite usar FormCanvas(figure) como antes sin cambios
    """
    
    def __init__(self, figure: Optional[Figure] = None, parent=None):
        # Si no se proporciona figura, crear una figura vacía (comportamiento original)
        if figure is None:
            figure = plt.figure()
        
        super().__init__(
            figure=figure,
            title="Gráfico",
            graph_title="Gráfico",
            parent=parent
        )
        
        # Propiedades adicionales para compatibilidad total
        self.Hlayout = self.canvas_layout


# Alias para compatibilidad adicional
GraphDialog = SeismicGraphDialog
SeismicGraphCanvas = FormCanvas