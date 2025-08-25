"""
Diálogo centralizado para mostrar tablas de datos sísmicos
Centraliza la funcionalidad común de mostrar tablas pandas en diálogos
"""

from typing import Optional, Dict, Any, List, Union, Callable
import os
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QAbstractTableModel, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableView, 
    QFrame, QPushButton, QFileDialog, QMessageBox, QHeaderView,
    QApplication, QSizePolicy
)
from PyQt5.QtGui import QFont, QIcon

import pandas as pd


class PandasTableModel(QAbstractTableModel):
    """
    Modelo estándar para mostrar DataFrames de pandas en QTableView
    Compatible con todos los proyectos sísmicos existentes
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el modelo con un DataFrame
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a mostrar en la tabla
        """
        QAbstractTableModel.__init__(self)
        self._data = data.copy() if data is not None else pd.DataFrame()
        self._original_data = self._data.copy()
    
    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        """Retorna el número de filas"""
        return self._data.shape[0]
    
    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        """Retorna el número de columnas"""
        return self._data.shape[1]
    
    def data(self, index, role=Qt.DisplayRole):
        """Retorna los datos de la celda especificada"""
        if index.isValid():
            if role == Qt.DisplayRole:
                value = self._data.iloc[index.row(), index.column()]
                return str(value)
        return None
    
    def headerData(self, section, orientation, role):
        """Retorna los headers de filas y columnas"""
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self._data.columns[section])
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return str(self._data.index[section])
        return None
    
    def get_dataframe(self) -> pd.DataFrame:
        """Retorna el DataFrame actual"""
        return self._data.copy()
    
    def update_data(self, new_data: pd.DataFrame):
        """Actualiza los datos del modelo"""
        self.beginResetModel()
        self._data = new_data.copy()
        self.endResetModel()


class SeismicTableDialog(QDialog):
    """
    Diálogo centralizado para mostrar tablas de análisis sísmico
    Reemplaza las clases diagTable, FormTabla y similares de los proyectos existentes
    """
    
    # Señales
    table_exported = pyqtSignal(str)  # ruta del archivo exportado
    table_closed = pyqtSignal()       # diálogo cerrado
    
    def __init__(self, data: Optional[pd.DataFrame] = None, 
                 title: str = "Tabla", 
                 table_title: str = "Datos", 
                 parent=None):
        """
        Inicializa el diálogo de tabla sísmica
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            DataFrame a mostrar
        title : str
            Título de la ventana del diálogo
        table_title : str
            Título mostrado encima de la tabla
        parent : QWidget, optional
            Widget padre
        """
        super().__init__(parent)
        
        self.data = data if data is not None else pd.DataFrame()
        self.table_title = table_title
        
        # Configuración de la ventana
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(800, 600)
        
        # Configurar interfaz
        self.setup_ui()
        self.setup_table()
        self.setup_connections()
    
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Layout principal
        self.main_layout = QHBoxLayout(self)
        
        # Layout vertical interno
        self.vertical_layout = QVBoxLayout()
        
        # Label de título
        self.title_label = QLabel(self.table_title)
        self.setup_title_font()
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setObjectName("title_label")
        
        # Vista de tabla
        self.table_view = QTableView()
        self.table_view.setObjectName("tabla_modal")  # Compatible con código existente
        
        # Botones
        self.buttons_layout = QHBoxLayout()
        self.setup_buttons()
        
        # Agregar widgets al layout
        self.vertical_layout.addWidget(self.title_label)
        self.vertical_layout.addWidget(self.table_view)
        self.vertical_layout.addLayout(self.buttons_layout)
        
        self.main_layout.addLayout(self.vertical_layout)
    
    def setup_title_font(self):
        """Configura la fuente del título para mantener consistencia"""
        font = QFont()
        font.setFamily("Montserrat")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.title_label.setFont(font)
    
    def setup_buttons(self):
        """Configura los botones del diálogo"""
        # Botón exportar
        self.export_btn = QPushButton("Exportar")
        self.export_btn.setToolTip("Exportar tabla a Excel")
        
        # Botón cerrar
        self.close_btn = QPushButton("Cerrar")
        
        # Agregar botones al layout
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.export_btn)
        self.buttons_layout.addWidget(self.close_btn)
    
    def setup_table(self):
        """Configura la tabla y el modelo"""
        # Crear y asignar modelo
        self.model = PandasTableModel(self.data)
        self.table_view.setModel(self.model)
        
        # Configuraciones de la tabla
        self.table_view.setSortingEnabled(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setAlternatingRowColors(True)
        
        # Configurar headers
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.verticalHeader().setVisible(True)
    
    def setup_connections(self):
        """Configura las conexiones de señales y slots"""
        self.export_btn.clicked.connect(self.export_table)
        self.close_btn.clicked.connect(self.close_dialog)
        
        # Conectar señal de cierre del diálogo
        self.finished.connect(lambda: self.table_closed.emit())
    
    def set_data(self, data: pd.DataFrame, table_title: str = None):
        """
        Actualiza los datos mostrados en la tabla
        
        Parameters
        ----------
        data : pd.DataFrame
            Nuevos datos a mostrar
        table_title : str, optional
            Nuevo título para la tabla
        """
        self.data = data.copy()
        self.model.update_data(self.data)
        
        if table_title:
            self.table_title = table_title
            self.title_label.setText(table_title)
        
        # Auto-ajustar columnas para tablas pequeñas
        if len(data.columns) <= 10:
            self.table_view.resizeColumnsToContents()
    
    def set_column_widths(self, widths: List[int]):
        """
        Establece anchos específicos para las columnas
        Compatible con el código existente que usa esta funcionalidad
        
        Parameters
        ----------
        widths : List[int]
            Lista de anchos para cada columna en píxeles
        """
        for index, width in enumerate(widths):
            if index < self.model.columnCount():
                self.table_view.setColumnWidth(index, width)
    
    def resize_columns_to_contents(self):
        """Ajusta el tamaño de todas las columnas al contenido"""
        self.table_view.resizeColumnsToContents()
    
    def export_table(self):
        """Exporta la tabla a un archivo Excel"""
        if self.data.empty:
            QMessageBox.warning(self, "Advertencia", "No hay datos para exportar.")
            return
        
        # Diálogo para seleccionar archivo
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar tabla",
            f"{self.table_title.replace(' ', '_')}.xlsx",
            "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # Exportar según extensión
                if file_path.endswith('.csv'):
                    self.data.to_csv(file_path, index=False)
                else:
                    self.data.to_excel(file_path, index=False, sheet_name=self.table_title)
                
                QMessageBox.information(
                    self, "Éxito", 
                    f"Tabla exportada correctamente a:\n{file_path}"
                )
                self.table_exported.emit(file_path)
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", 
                    f"Error al exportar la tabla:\n{str(e)}"
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
        Permite usar dialog.ui.tabla_modal como en el código existente
        """
        return self
    
    @property 
    def tabla_modal(self):
        """Compatibilidad: acceso directo a la tabla"""
        return self.table_view
    
    @property
    def label_2(self):
        """Compatibilidad: acceso directo al label de título"""
        return self.title_label


# Funciones de conveniencia para crear diálogos específicos
def create_modal_table_dialog(data: pd.DataFrame, parent=None) -> SeismicTableDialog:
    """
    Crea un diálogo para mostrar tabla modal
    
    Parameters
    ----------
    data : pd.DataFrame
        Datos modales a mostrar
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicTableDialog
        Diálogo configurado para tabla modal
    """
    dialog = SeismicTableDialog(
        data=data,
        title="Tabla Modal",
        table_title="Tabla Modal",
        parent=parent
    )
    
    # Anchos específicos para tabla modal (compatibilidad)
    modal_widths = [60, 70, 70, 70, 70, 70, 70, 70]
    dialog.set_column_widths(modal_widths)
    
    return dialog


def create_static_analysis_dialog(data: pd.DataFrame, parent=None) -> SeismicTableDialog:
    """
    Crea un diálogo para mostrar análisis estático
    
    Parameters
    ----------
    data : pd.DataFrame
        Datos de análisis estático
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicTableDialog
        Diálogo configurado para análisis estático
    """
    dialog = SeismicTableDialog(
        data=data,
        title="Análisis Estático",
        table_title="Análisis Estático",
        parent=parent
    )
    
    # Anchos específicos para análisis estático
    static_widths = [60, 50, 45, 45, 45, 60, 60, 50, 45, 45, 45]
    dialog.set_column_widths(static_widths)
    
    return dialog


def create_soft_story_dialog(data: pd.DataFrame, parent=None) -> SeismicTableDialog:
    """
    Crea un diálogo para mostrar análisis de piso blando
    
    Parameters
    ----------
    data : pd.DataFrame
        Datos de piso blando
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicTableDialog
        Diálogo configurado para piso blando
    """
    dialog = SeismicTableDialog(
        data=data,
        title="Piso Blando",
        table_title="Piso Blando",
        parent=parent
    )
    
    # Ajustar columnas al contenido para piso blando
    dialog.resize_columns_to_contents()
    dialog.resize(600, dialog.height())
    
    return dialog


def create_mass_irregularity_dialog(data: pd.DataFrame, parent=None) -> SeismicTableDialog:
    """
    Crea un diálogo para mostrar irregularidad de masa
    
    Parameters
    ----------
    data : pd.DataFrame
        Datos de irregularidad de masa
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicTableDialog
        Diálogo configurado para irregularidad de masa
    """
    dialog = SeismicTableDialog(
        data=data,
        title="Irregularidad de Masa",
        table_title="Irregularidad Masa",
        parent=parent
    )
    
    dialog.resize_columns_to_contents()
    return dialog


def create_torsion_irregularity_dialog(data: pd.DataFrame, parent=None) -> SeismicTableDialog:
    """
    Crea un diálogo para mostrar irregularidad torsional
    
    Parameters
    ----------
    data : pd.DataFrame
        Datos de irregularidad torsional
    parent : QWidget, optional
        Widget padre
        
    Returns
    -------
    SeismicTableDialog
        Diálogo configurado para irregularidad torsional
    """
    dialog = SeismicTableDialog(
        data=data,
        title="Irregularidad Torsional",
        table_title="Irregularidad Torsión",
        parent=parent
    )
    
    dialog.resize_columns_to_contents()
    dialog.resize(450, dialog.height())
    
    return dialog


# Clase de compatibilidad para código existente
class diagTable(SeismicTableDialog):
    """
    Clase de compatibilidad para el código existente
    Permite usar diagTable() como antes sin cambios
    """
    
    def __init__(self, parent=None):
        super().__init__(
            data=pd.DataFrame(),
            title="Tabla",
            table_title="Datos",
            parent=parent
        )


# Alias para compatibilidad adicional
FormTabla = diagTable
TableDialog = SeismicTableDialog