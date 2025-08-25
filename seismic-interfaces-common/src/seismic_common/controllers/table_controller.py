"""
Controlador centralizado para manejo de tablas de datos sísmicos
Proporciona funcionalidades avanzadas para mostrar, exportar y manipular tablas
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import json
import csv

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QAbstractTableModel, QSortFilterProxyModel, pyqtSignal, QModelIndex
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableView, 
    QHeaderView, QPushButton, QLineEdit, QComboBox, QCheckBox,
    QGroupBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
    QSplitter, QTextEdit, QTabWidget, QProgressBar, QMenu, QAction
)
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QKeySequence

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Importaciones del sistema centralizado
try:
    from .base_controller import BaseController, BaseTableDialog
    from seismic_common.core import (
        dataframe_to_latex,
        create_default_unit_dict,
        Units
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("⚠️ Importaciones centralizadas no disponibles para table_controller")


class AdvancedTableModel(QAbstractTableModel):
    """
    Modelo avanzado para tablas con funcionalidades extendidas
    """
    
    # Señales
    dataChanged = pyqtSignal()
    
    def __init__(self, data: pd.DataFrame, parent=None):
        """
        Inicializa el modelo avanzado de tabla
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a mostrar
        parent : QObject, optional
            Objeto padre
        """
        super().__init__(parent)
        self._data = data.copy()
        self._original_data = data.copy()  # Backup para resetear
        self._numeric_precision = 4
        self._highlight_cells = {}  # {(row, col): color}
        self._editable_columns = set()
        self._column_formats = {}  # {col: format_function}
        
    def rowCount(self, parent=QModelIndex()) -> int:
        """Número de filas"""
        return len(self._data)
    
    def columnCount(self, parent=QModelIndex()) -> int:
        """Número de columnas"""
        return len(self._data.columns)
    
    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        """Datos de celda con formato avanzado"""
        if not index.isValid():
            return None
        
        row = index.row()
        col = index.column()
        value = self._data.iloc[row, col]
        
        if role == Qt.DisplayRole or role == Qt.EditRole:
            # Aplicar formato personalizado si existe
            column_name = self._data.columns[col]
            if column_name in self._column_formats:
                formatter = self._column_formats[column_name]
                return formatter(value)
            
            # Formato por defecto según tipo de dato
            if pd.isna(value):
                return ""
            elif isinstance(value, (int, np.integer)):
                return str(value)
            elif isinstance(value, (float, np.floating)):
                if abs(value) < 1e-10:  # Valores muy pequeños como cero
                    return "0"
                return f"{value:.{self._numeric_precision}f}"
            else:
                return str(value)
        
        elif role == Qt.BackgroundRole:
            # Resaltado de celdas
            if (row, col) in self._highlight_cells:
                color = self._highlight_cells[(row, col)]
                return QColor(color)
        
        elif role == Qt.TextAlignmentRole:
            # Alineación según tipo de dato
            if isinstance(value, (int, float, np.integer, np.floating)):
                return Qt.AlignRight | Qt.AlignVCenter
            return Qt.AlignLeft | Qt.AlignVCenter
        
        elif role == Qt.FontRole:
            # Fuente para números
            if isinstance(value, (int, float, np.integer, np.floating)):
                font = QFont("Consolas", 9)  # Fuente monoespaciada para números
                return font
        
        return None
    
    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        """Permite edición de celdas si está habilitada"""
        if not index.isValid() or role != Qt.EditRole:
            return False
        
        col = index.column()
        column_name = self._data.columns[col]
        
        # Verificar si la columna es editable
        if column_name not in self._editable_columns:
            return False
        
        row = index.row()
        
        try:
            # Convertir valor según tipo original
            original_type = type(self._data.iloc[row, col])
            if original_type in (int, np.integer):
                converted_value = int(float(value))
            elif original_type in (float, np.floating):
                converted_value = float(value)
            else:
                converted_value = str(value)
            
            # Actualizar dato
            self._data.iloc[row, col] = converted_value
            
            # Emitir señales
            self.dataChanged.emit(index, index, [Qt.DisplayRole])
            self.dataChanged.emit()
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """Flags de celda (editable, seleccionable, etc.)"""
        if not index.isValid():
            return Qt.NoItemFlags
        
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        
        # Permitir edición si la columna está marcada como editable
        col = index.column()
        column_name = self._data.columns[col]
        if column_name in self._editable_columns:
            flags |= Qt.ItemIsEditable
        
        return flags
    
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        """Headers con formato mejorado"""
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(self._data.index[section])
        
        elif role == Qt.FontRole and orientation == Qt.Horizontal:
            # Fuente en negrita para headers
            font = QFont()
            font.setBold(True)
            return font
        
        elif role == Qt.BackgroundRole and orientation == Qt.Horizontal:
            # Color de fondo para headers
            return QColor(240, 240, 240)
        
        return None
    
    # Métodos de configuración
    def set_numeric_precision(self, precision: int) -> None:
        """Establece precisión decimal para números"""
        self._numeric_precision = precision
        self.layoutChanged.emit()
    
    def set_editable_columns(self, columns: List[str]) -> None:
        """Establece qué columnas son editables"""
        self._editable_columns = set(columns)
        self.layoutChanged.emit()
    
    def set_column_format(self, column: str, formatter: Callable[[Any], str]) -> None:
        """Establece formato personalizado para una columna"""
        self._column_formats[column] = formatter
        self.layoutChanged.emit()
    
    def highlight_cell(self, row: int, col: int, color: str = "#FFFF99") -> None:
        """Resalta una celda específica"""
        self._highlight_cells[(row, col)] = color
        index = self.createIndex(row, col)
        self.dataChanged.emit(index, index, [Qt.BackgroundRole])
    
    def clear_highlights(self) -> None:
        """Limpia todos los resaltados"""
        self._highlight_cells.clear()
        self.layoutChanged.emit()
    
    def reset_data(self) -> None:
        """Resetea datos a valores originales"""
        self.beginResetModel()
        self._data = self._original_data.copy()
        self.endResetModel()
    
    def get_dataframe(self) -> pd.DataFrame:
        """Obtiene el DataFrame actual"""
        return self._data.copy()


class TableFilterWidget(QtWidgets.QWidget):
    """
    Widget para filtrado avanzado de tablas
    """
    
    # Señales
    filter_changed = pyqtSignal(dict)  # {column: filter_value}
    
    def __init__(self, dataframe: pd.DataFrame, parent=None):
        """
        Inicializa widget de filtros
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            DataFrame para generar filtros
        parent : QWidget, optional
            Widget padre
        """
        super().__init__(parent)
        self.df = dataframe
        self.filters = {}
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Configura la interfaz de filtros"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Título
        title_label = QLabel("Filtros:")
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title_label)
        
        # Crear filtros para cada columna
        for col in self.df.columns:
            self.create_column_filter(col, layout)
        
        # Botón limpiar filtros
        clear_btn = QPushButton("Limpiar")
        clear_btn.clicked.connect(self.clear_filters)
        clear_btn.setMaximumWidth(60)
        layout.addWidget(clear_btn)
        
        layout.addStretch()
    
    def create_column_filter(self, column: str, layout: QHBoxLayout) -> None:
        """Crea filtro para una columna específica"""
        col_data = self.df[column]
        
        # Crear label
        label = QLabel(f"{column}:")
        label.setMaximumWidth(80)
        layout.addWidget(label)
        
        # Determinar tipo de filtro según tipo de dato
        if col_data.dtype in ['object', 'category']:
            # ComboBox para datos categóricos
            combo = QComboBox()
            combo.addItem("Todos")
            
            unique_values = sorted(col_data.dropna().unique().astype(str))
            combo.addItems(unique_values)
            combo.currentTextChanged.connect(
                lambda text, col=column: self.update_filter(col, text if text != "Todos" else None)
            )
            combo.setMaximumWidth(120)
            layout.addWidget(combo)
            
        elif col_data.dtype in ['int64', 'float64']:
            # Rangos para datos numéricos
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            
            # SpinBox mínimo
            min_spin = QDoubleSpinBox()
            min_spin.setRange(min_val - abs(min_val), max_val + abs(max_val))
            min_spin.setValue(min_val)
            min_spin.setDecimals(4)
            min_spin.setMaximumWidth(80)
            
            # SpinBox máximo  
            max_spin = QDoubleSpinBox()
            max_spin.setRange(min_val - abs(min_val), max_val + abs(max_val))
            max_spin.setValue(max_val)
            max_spin.setDecimals(4)
            max_spin.setMaximumWidth(80)
            
            # Conectar señales
            min_spin.valueChanged.connect(
                lambda val, col=column: self.update_numeric_filter(col, val, max_spin.value())
            )
            max_spin.valueChanged.connect(
                lambda val, col=column: self.update_numeric_filter(col, min_spin.value(), val)
            )
            
            layout.addWidget(min_spin)
            layout.addWidget(QLabel("-"))
            layout.addWidget(max_spin)
        
        else:
            # LineEdit para otros tipos
            line_edit = QLineEdit()
            line_edit.setPlaceholderText("Filtrar...")
            line_edit.textChanged.connect(
                lambda text, col=column: self.update_filter(col, text if text else None)
            )
            line_edit.setMaximumWidth(100)
            layout.addWidget(line_edit)
    
    def update_filter(self, column: str, value: Any) -> None:
        """Actualiza filtro de columna"""
        if value is None:
            self.filters.pop(column, None)
        else:
            self.filters[column] = {'type': 'exact', 'value': value}
        
        self.filter_changed.emit(self.filters)
    
    def update_numeric_filter(self, column: str, min_val: float, max_val: float) -> None:
        """Actualiza filtro numérico"""
        self.filters[column] = {'type': 'range', 'min': min_val, 'max': max_val}
        self.filter_changed.emit(self.filters)
    
    def clear_filters(self) -> None:
        """Limpia todos los filtros"""
        self.filters.clear()
        self.filter_changed.emit(self.filters)
        
        # Resetear widgets
        for child in self.findChildren((QComboBox, QLineEdit)):
            if isinstance(child, QComboBox):
                child.setCurrentIndex(0)
            elif isinstance(child, QLineEdit):
                child.clear()


class TableStatisticsWidget(QtWidgets.QWidget):
    """
    Widget para mostrar estadísticas de tabla
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self) -> None:
        """Configura interfaz de estadísticas"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("Estadísticas")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Área de texto para estadísticas
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setReadOnly(True)
        layout.addWidget(self.stats_text)
        
        # Botón actualizar
        refresh_btn = QPushButton("Actualizar")
        refresh_btn.clicked.connect(self.refresh_stats)
        layout.addWidget(refresh_btn)
    
    def update_statistics(self, dataframe: pd.DataFrame) -> None:
        """Actualiza estadísticas con nuevo DataFrame"""
        self.df = dataframe
        self.refresh_stats()
    
    def refresh_stats(self) -> None:
        """Refresca las estadísticas mostradas"""
        if not hasattr(self, 'df') or self.df.empty:
            self.stats_text.setText("No hay datos disponibles")
            return
        
        stats_html = self.generate_statistics_html()
        self.stats_text.setHtml(stats_html)
    
    def generate_statistics_html(self) -> str:
        """Genera HTML con estadísticas"""
        html = "<h3>Resumen de Datos</h3>"
        html += f"<b>Filas:</b> {len(self.df)}<br>"
        html += f"<b>Columnas:</b> {len(self.df.columns)}<br>"
        html += f"<b>Memoria:</b> {self.df.memory_usage(deep=True).sum() / 1024:.1f} KB<br><br>"
        
        # Estadísticas por columna
        html += "<h4>Por Columna:</h4>"
        html += "<table border='1' cellpadding='3'>"
        html += "<tr><th>Columna</th><th>Tipo</th><th>No Nulos</th><th>Únicos</th></tr>"
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            non_null = self.df[col].count()
            unique = self.df[col].nunique()
            
            html += f"<tr><td>{col}</td><td>{dtype}</td><td>{non_null}</td><td>{unique}</td></tr>"
        
        html += "</table><br>"
        
        # Estadísticas numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            html += "<h4>Estadísticas Numéricas:</h4>"
            html += "<table border='1' cellpadding='3'>"
            html += "<tr><th>Columna</th><th>Min</th><th>Max</th><th>Media</th><th>Desv.Est</th></tr>"
            
            for col in numeric_cols:
                data = self.df[col].dropna()
                if len(data) > 0:
                    html += f"<tr><td>{col}</td>"
                    html += f"<td>{data.min():.4f}</td>"
                    html += f"<td>{data.max():.4f}</td>" 
                    html += f"<td>{data.mean():.4f}</td>"
                    html += f"<td>{data.std():.4f}</td></tr>"
            
            html += "</table>"
        
        return html


class AdvancedTableDialog(BaseTableDialog):
    """
    Diálogo avanzado de tabla con funcionalidades extendidas
    """
    
    # Señales
    data_exported = pyqtSignal(str)  # ruta del archivo exportado
    data_modified = pyqtSignal()     # datos fueron modificados
    
    def __init__(self, data: pd.DataFrame, title: str = "Tabla Avanzada", 
                 show_filters: bool = True, show_statistics: bool = True,
                 editable_columns: List[str] = None, parent=None):
        """
        Inicializa diálogo avanzado de tabla
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a mostrar
        title : str
            Título de la ventana
        show_filters : bool
            Si mostrar panel de filtros
        show_statistics : bool
            Si mostrar panel de estadísticas
        editable_columns : List[str], optional
            Columnas que pueden ser editadas
        parent : QWidget, optional
            Widget padre
        """
        # No llamar al __init__ del padre directamente
        QDialog.__init__(self, parent)
        
        self.original_data = data.copy()
        self.filtered_data = data.copy()
        self.show_filters = show_filters
        self.show_statistics = show_statistics
        self.editable_columns = editable_columns or []
        
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(1000, 700)
        
        self.setup_advanced_ui()
        self.setup_model_and_view()
        self.setup_connections()
    
    def setup_advanced_ui(self) -> None:
        """Configura interfaz avanzada"""
        layout = QVBoxLayout(self)
        
        # Barra de herramientas
        toolbar_layout = QHBoxLayout()
        
        # Botones de acción
        self.export_btn = QPushButton("Exportar")
        self.export_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.print_btn = QPushButton("Imprimir")
        self.print_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        self.reset_btn = QPushButton("Resetear")
        self.reset_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton))
        
        toolbar_layout.addWidget(self.export_btn)
        toolbar_layout.addWidget(self.print_btn)
        toolbar_layout.addWidget(self.reset_btn)
        toolbar_layout.addStretch()
        
        # Configuración de visualización
        precision_label = QLabel("Decimales:")
        self.precision_spin = QSpinBox()
        self.precision_spin.setRange(0, 10)
        self.precision_spin.setValue(4)
        
        self.auto_resize_cb = QCheckBox("Auto-ajustar columnas")
        self.auto_resize_cb.setChecked(True)
        
        toolbar_layout.addWidget(precision_label)
        toolbar_layout.addWidget(self.precision_spin)
        toolbar_layout.addWidget(self.auto_resize_cb)
        
        layout.addLayout(toolbar_layout)
        
        # Panel principal con splitter
        main_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(main_splitter)
        
        # Panel superior: filtros + tabla
        upper_widget = QtWidgets.QWidget()
        upper_layout = QVBoxLayout(upper_widget)
        
        # Panel de filtros
        if self.show_filters:
            self.filter_widget = TableFilterWidget(self.original_data)
            upper_layout.addWidget(self.filter_widget)
        
        # Vista de tabla
        self.table_view = QTableView()
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        upper_layout.addWidget(self.table_view)
        
        main_splitter.addWidget(upper_widget)
        
        # Panel inferior: estadísticas (si está habilitado)
        if self.show_statistics:
            self.stats_widget = TableStatisticsWidget()
            main_splitter.addWidget(self.stats_widget)
            
            # Configurar proporción del splitter
            main_splitter.setSizes([500, 200])
        
        # Botones de acción del diálogo
        button_layout = QHBoxLayout()
        
        self.accept_btn = QPushButton("Aceptar")
        self.cancel_btn = QPushButton("Cancelar")
        
        button_layout.addStretch()
        button_layout.addWidget(self.accept_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def setup_model_and_view(self) -> None:
        """Configura modelo y vista de tabla"""
        # Crear modelo avanzado
        self.table_model = AdvancedTableModel(self.filtered_data)
        self.table_model.set_editable_columns(self.editable_columns)
        
        # Crear proxy model para filtrado y ordenamiento
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.table_model)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        
        # Configurar vista
        self.table_view.setModel(self.proxy_model)
        
        # Configurar headers
        header = self.table_view.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        if self.auto_resize_cb.isChecked():
            self.auto_resize_columns()
    
    def setup_connections(self) -> None:
        """Configura conexiones de señales"""
        # Botones principales
        self.accept_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        # Botones de acción
        self.export_btn.clicked.connect(self.export_data)
        self.print_btn.clicked.connect(self.print_table)
        self.reset_btn.clicked.connect(self.reset_data)
        
        # Configuración
        self.precision_spin.valueChanged.connect(self.update_precision)
        self.auto_resize_cb.toggled.connect(self.toggle_auto_resize)
        
        # Filtros
        if self.show_filters:
            self.filter_widget.filter_changed.connect(self.apply_filters)
        
        # Modelo
        self.table_model.dataChanged.connect(self.on_data_changed)
        
        # Estadísticas
        if self.show_statistics:
            self.stats_widget.update_statistics(self.filtered_data)
    
    def apply_filters(self, filters: Dict[str, Any]) -> None:
        """Aplica filtros a la tabla"""
        filtered_df = self.original_data.copy()
        
        for column, filter_config in filters.items():
            if column not in filtered_df.columns:
                continue
            
            filter_type = filter_config.get('type', 'exact')
            
            if filter_type == 'exact':
                value = filter_config['value']
                filtered_df = filtered_df[
                    filtered_df[column].astype(str).str.contains(str(value), na=False, case=False)
                ]
            
            elif filter_type == 'range':
                min_val = filter_config['min']
                max_val = filter_config['max']
                filtered_df = filtered_df[
                    (filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)
                ]
        
        # Actualizar datos filtrados
        self.filtered_data = filtered_df
        self.table_model._data = filtered_df.copy()
        self.table_model.layoutChanged.emit()
        
        # Actualizar estadísticas
        if self.show_statistics:
            self.stats_widget.update_statistics(filtered_df)
        
        # Auto-redimensionar si está habilitado
        if self.auto_resize_cb.isChecked():
            self.auto_resize_columns()
    
    def update_precision(self, precision: int) -> None:
        """Actualiza precisión de números"""
        self.table_model.set_numeric_precision(precision)
    
    def toggle_auto_resize(self, enabled: bool) -> None:
        """Activa/desactiva auto-redimensionado"""
        if enabled:
            self.auto_resize_columns()
    
    def auto_resize_columns(self) -> None:
        """Redimensiona automáticamente las columnas"""
        self.table_view.resizeColumnsToContents()
        
        # Limitar ancho máximo
        for col in range(self.table_model.columnCount()):
            current_width = self.table_view.columnWidth(col)
            if current_width > 200:
                self.table_view.setColumnWidth(col, 200)
    
    def export_data(self) -> None:
        """Exporta datos a varios formatos"""
        # Crear menú de opciones de exportación
        export_menu = QMenu(self)
        
        excel_action = export_menu.addAction("Excel (.xlsx)")
        csv_action = export_menu.addAction("CSV (.csv)")
        latex_action = export_menu.addAction("LaTeX (.tex)")
        json_action = export_menu.addAction("JSON (.json)")
        
        # Mostrar menú
        action = export_menu.exec_(self.export_btn.mapToGlobal(self.export_btn.rect().bottomLeft()))
        
        if not action:
            return
        
        # Seleccionar archivo
        if action == excel_action:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Exportar Excel", "tabla.xlsx", "Excel Files (*.xlsx)"
            )
            if file_path:
                self._export_excel(file_path)
        
        elif action == csv_action:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Exportar CSV", "tabla.csv", "CSV Files (*.csv)"
            )
            if file_path:
                self._export_csv(file_path)
        
        elif action == latex_action:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Exportar LaTeX", "tabla.tex", "LaTeX Files (*.tex)"
            )
            if file_path:
                self._export_latex(file_path)
        
        elif action == json_action:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Exportar JSON", "tabla.json", "JSON Files (*.json)"
            )
            if file_path:
                self._export_json(file_path)
    
    def _export_excel(self, file_path: str) -> None:
        """Exporta a Excel"""
        try:
            current_data = self.table_model.get_dataframe()
            current_data.to_excel(file_path, index=False)
            self.data_exported.emit(file_path)
            QMessageBox.information(self, "Éxito", f"Datos exportados a:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exportando Excel:\n{str(e)}")
    
    def _export_csv(self, file_path: str) -> None:
        """Exporta a CSV"""
        try:
            current_data = self.table_model.get_dataframe()
            current_data.to_csv(file_path, index=False, encoding='utf-8')
            self.data_exported.emit(file_path)
            QMessageBox.information(self, "Éxito", f"Datos exportados a:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exportando CSV:\n{str(e)}")
    
    def _export_latex(self, file_path: str) -> None:
        """Exporta a LaTeX"""
        try:
            current_data = self.table_model.get_dataframe()
            
            if CORE_AVAILABLE:
                latex_content = dataframe_to_latex(
                    current_data,
                    caption="Tabla de Datos",
                    decimals=self.precision_spin.value()
                )
            else:
                # Fallback básico
                latex_content = current_data.to_latex(index=False)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            self.data_exported.emit(file_path)
            QMessageBox.information(self, "Éxito", f"Tabla LaTeX exportada a:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exportando LaTeX:\n{str(e)}")
    
    def _export_json(self, file_path: str) -> None:
        """Exporta a JSON"""
        try:
            current_data = self.table_model.get_dataframe()
            current_data.to_json(file_path, orient='records', indent=2, force_ascii=False)
            self.data_exported.emit(file_path)
            QMessageBox.information(self, "Éxito", f"Datos exportados a:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exportando JSON:\n{str(e)}")
    
    def print_table(self) -> None:
        """Imprime la tabla"""
        try:
            from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
            
            printer = QPrinter()
            dialog = QPrintDialog(printer, self)
            
            if dialog.exec_() == QPrintDialog.Accepted:
                # Crear HTML para impresión
                html = self.generate_print_html()
                
                # Imprimir
                from PyQt5.QtGui import QTextDocument
                document = QTextDocument()
                document.setHtml(html)
                document.print_(printer)
                
                QMessageBox.information(self, "Éxito", "Tabla enviada a impresora")
                
        except ImportError:
            QMessageBox.warning(self, "No Disponible", "Funcionalidad de impresión no disponible")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error imprimiendo:\n{str(e)}")
    
    def generate_print_html(self) -> str:
        """Genera HTML para impresión"""
        current_data = self.table_model.get_dataframe()
        
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; font-size: 10pt; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; font-weight: bold; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .numeric { text-align: right; }
                h2 { color: #333; border-bottom: 2px solid #333; }
            </style>
        </head>
        <body>
        """
        
        html += f"<h2>{self.windowTitle()}</h2>"
        html += f"<p>Generado el: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        html += f"<p>Filas: {len(current_data)} | Columnas: {len(current_data.columns)}</p>"
        
        # Convertir DataFrame a HTML
        df_html = current_data.to_html(
            classes='print-table',
            table_id='data-table',
            escape=False,
            index=False
        )
        
        # Agregar clases CSS a columnas numéricas
        for col in current_data.select_dtypes(include=[np.number]).columns:
            df_html = df_html.replace(f'<td>{col}</td>', f'<td class="numeric">{col}</td>')
        
        html += df_html
        html += "</body></html>"
        
        return html
    
    def reset_data(self) -> None:
        """Resetea datos a valores originales"""
        reply = QMessageBox.question(
            self, "Confirmar Reset",
            "¿Está seguro de que desea resetear todos los datos a los valores originales?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.table_model.reset_data()
            self.filtered_data = self.original_data.copy()
            
            # Limpiar filtros
            if self.show_filters:
                self.filter_widget.clear_filters()
            
            # Actualizar estadísticas
            if self.show_statistics:
                self.stats_widget.update_statistics(self.filtered_data)
            
            QMessageBox.information(self, "Éxito", "Datos reseteados exitosamente")
    
    def on_data_changed(self) -> None:
        """Maneja cambios en los datos"""
        self.data_modified.emit()
        
        # Actualizar estadísticas si están visibles
        if self.show_statistics:
            current_data = self.table_model.get_dataframe()
            self.stats_widget.update_statistics(current_data)
    
    def get_current_data(self) -> pd.DataFrame:
        """Obtiene los datos actuales (filtrados y modificados)"""
        return self.table_model.get_dataframe()


class TableController:
    """
    Controlador principal para manejo avanzado de tablas
    
    Proporciona funcionalidades para:
    - Crear tablas avanzadas con filtros y estadísticas
    - Exportar a múltiples formatos
    - Edición in-situ de datos
    - Análisis estadístico básico
    - Formateo personalizado
    """
    
    def __init__(self):
        """Inicializa el controlador de tablas"""
        self.units = None
        self.unit_dict = {}
        
        if CORE_AVAILABLE:
            self.units = Units()
            self.unit_dict = create_default_unit_dict()
    
    def show_advanced_table(self, 
                          data: pd.DataFrame, 
                          title: str = "Tabla de Datos",
                          show_filters: bool = True,
                          show_statistics: bool = True,
                          editable_columns: List[str] = None,
                          custom_formatters: Dict[str, Callable] = None) -> AdvancedTableDialog:
        """
        Muestra tabla avanzada con funcionalidades completas
        
        Parameters
        ----------
        data : pd.DataFrame
            Datos a mostrar
        title : str
            Título de la ventana
        show_filters : bool
            Si mostrar panel de filtros
        show_statistics : bool
            Si mostrar panel de estadísticas  
        editable_columns : List[str], optional
            Columnas editables
        custom_formatters : Dict[str, Callable], optional
            Formateadores personalizados por columna
            
        Returns
        -------
        AdvancedTableDialog
            Diálogo de tabla configurado
        """
        dialog = AdvancedTableDialog(
            data=data,
            title=title,
            show_filters=show_filters,
            show_statistics=show_statistics,
            editable_columns=editable_columns or []
        )
        
        # Aplicar formateadores personalizados
        if custom_formatters:
            for column, formatter in custom_formatters.items():
                dialog.table_model.set_column_format(column, formatter)
        
        return dialog
    
    def show_modal_table(self, data: pd.DataFrame, title: str = "Análisis Modal") -> None:
        """Muestra tabla modal con formato específico"""
        # Formateadores específicos para tabla modal
        formatters = {
            'Period': lambda x: f"{x:.4f}" if pd.notnull(x) else "",
            'Ux': lambda x: f"{x:.3f}" if pd.notnull(x) else "",
            'Uy': lambda x: f"{x:.3f}" if pd.notnull(x) else "", 
            'SumUx': lambda x: f"{x:.3f}" if pd.notnull(x) else "",
            'SumUy': lambda x: f"{x:.3f}" if pd.notnull(x) else ""
        }
        
        dialog = self.show_advanced_table(
            data=data,
            title=title,
            show_filters=False,  # Modal no necesita filtros
            show_statistics=True,
            custom_formatters=formatters
        )
        
        # Resaltar filas importantes (modos fundamentales)
        if 'Ux' in data.columns and 'Uy' in data.columns:
            # Encontrar modo fundamental X
            max_ux_idx = data['Ux'].idxmax()
            if pd.notnull(max_ux_idx):
                for col in range(len(data.columns)):
                    dialog.table_model.highlight_cell(max_ux_idx, col, "#FFE6E6")  # Rojo claro
            
            # Encontrar modo fundamental Y  
            max_uy_idx = data['Uy'].idxmax()
            if pd.notnull(max_uy_idx) and max_uy_idx != max_ux_idx:
                for col in range(len(data.columns)):
                    dialog.table_model.highlight_cell(max_uy_idx, col, "#E6E6FF")  # Azul claro
        
        dialog.exec_()
    
    def show_irregularity_table(self, 
                               data: pd.DataFrame, 
                               irregularity_type: str,
                               direction: str = None) -> None:
        """
        Muestra tabla de irregularidades con formato específico
        
        Parameters
        ----------
        data : pd.DataFrame
            Datos de irregularidades
        irregularity_type : str
            Tipo de irregularidad ('rigidez', 'torsion', 'masa')
        direction : str, optional
            Dirección de análisis ('X' o 'Y')
        """
        title_map = {
            'rigidez': 'Irregularidad de Rigidez (Piso Blando)',
            'torsion': 'Irregularidad Torsional',
            'masa': 'Irregularidad de Masa'
        }
        
        title = title_map.get(irregularity_type, f"Irregularidad - {irregularity_type.title()}")
        if direction:
            title += f" - Dirección {direction}"
        
        # Formateadores específicos según tipo
        formatters = {}
        
        if irregularity_type == 'rigidez':
            formatters.update({
                'drift': lambda x: f"{x:.5f}" if pd.notnull(x) else "",
                'stiff': lambda x: f"{x:.3f}" if pd.notnull(x) else "",
                '70%k_prev': lambda x: f"{x:.3f}" if pd.notnull(x) else "",
                '80%k_3': lambda x: f"{x:.3f}" if pd.notnull(x) else ""
            })
        
        elif irregularity_type == 'torsion':
            formatters.update({
                'Drifts': lambda x: f"{x:.5f}" if pd.notnull(x) else "",
                'Ratio': lambda x: f"{x:.3f}" if pd.notnull(x) else ""
            })
        
        elif irregularity_type == 'masa':
            formatters.update({
                'Mass': lambda x: f"{x:.5f}" if pd.notnull(x) else "",
                '1.5 Mass': lambda x: f"{x:.5f}" if pd.notnull(x) else ""
            })
        
        dialog = self.show_advanced_table(
            data=data,
            title=title,
            show_filters=True,
            show_statistics=True,
            custom_formatters=formatters
        )
        
        # Resaltar filas irregulares si existe la columna
        if 'is_irregular' in data.columns:
            irregular_rows = data[data['is_irregular'] == True].index
            for row_idx in irregular_rows:
                for col in range(len(data.columns)):
                    dialog.table_model.highlight_cell(row_idx, col, "#FFB6C1")  # Rosa claro
        
        dialog.exec_()
    
    def show_drift_table(self, data: pd.DataFrame, max_drift: float = 0.007) -> None:
        """Muestra tabla de derivas con validación de límites"""
        formatters = {
            'DriftX': lambda x: f"{x:.4f}" if pd.notnull(x) else "",
            'DriftY': lambda x: f"{x:.4f}" if pd.notnull(x) else "",
            'Drift Ratio X': lambda x: f"{x:.2%}" if pd.notnull(x) else "",
            'Drift Ratio Y': lambda x: f"{x:.2%}" if pd.notnull(x) else ""
        }
        
        dialog = self.show_advanced_table(
            data=data,
            title="Derivas de Entrepiso",
            show_filters=True,
            show_statistics=True,
            custom_formatters=formatters
        )
        
        # Resaltar derivas que exceden el límite
        for col_name in ['DriftX', 'DriftY']:
            if col_name in data.columns:
                col_idx = list(data.columns).index(col_name)
                for row_idx, value in enumerate(data[col_name]):
                    if pd.notnull(value) and abs(value) > max_drift:
                        dialog.table_model.highlight_cell(row_idx, col_idx, "#FF6B6B")  # Rojo
        
        dialog.exec_()
    
    def export_multiple_tables(self, 
                              tables_data: Dict[str, pd.DataFrame],
                              output_file: str,
                              format_type: str = 'excel') -> bool:
        """
        Exporta múltiples tablas a un archivo
        
        Parameters
        ----------
        tables_data : Dict[str, pd.DataFrame]
            Diccionario con nombre_hoja: dataframe
        output_file : str
            Archivo de salida
        format_type : str
            Formato ('excel', 'latex', 'json')
            
        Returns
        -------
        bool
            True si la exportación fue exitosa
        """
        try:
            if format_type == 'excel':
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    for sheet_name, df in tables_data.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
            elif format_type == 'latex':
                latex_content = "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n"
                latex_content += "\\usepackage{booktabs}\n\\usepackage{longtable}\n"
                latex_content += "\\begin{document}\n\n"
                
                for table_name, df in tables_data.items():
                    latex_content += f"\\section{{{table_name}}}\n\n"
                    if CORE_AVAILABLE:
                        table_latex = dataframe_to_latex(df, caption=table_name)
                    else:
                        table_latex = df.to_latex(index=False)
                    latex_content += table_latex + "\n\n"
                
                latex_content += "\\end{document}"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(latex_content)
            
            elif format_type == 'json':
                json_data = {}
                for table_name, df in tables_data.items():
                    json_data[table_name] = df.to_dict('records')
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error exportando tablas: {str(e)}")
            return False
    
    def create_comparison_table(self, 
                               tables_list: List[Tuple[str, pd.DataFrame]],
                               comparison_columns: List[str]) -> pd.DataFrame:
        """
        Crea tabla comparativa de múltiples DataFrames
        
        Parameters
        ----------
        tables_list : List[Tuple[str, pd.DataFrame]]
            Lista de (nombre, dataframe) a comparar
        comparison_columns : List[str]
            Columnas a incluir en la comparación
            
        Returns
        -------
        pd.DataFrame
            Tabla comparativa
        """
        comparison_data = []
        
        for name, df in tables_list:
            row_data = {'Configuración': name}
            
            for col in comparison_columns:
                if col in df.columns:
                    # Estadísticas básicas de la columna
                    if df[col].dtype in ['float64', 'int64']:
                        row_data[f'{col}_Mean'] = df[col].mean()
                        row_data[f'{col}_Max'] = df[col].max()
                        row_data[f'{col}_Min'] = df[col].min()
                        row_data[f'{col}_Std'] = df[col].std()
                    else:
                        row_data[f'{col}_Count'] = df[col].count()
                        row_data[f'{col}_Unique'] = df[col].nunique()
            
            comparison_data.append(row_data)
        
        return pd.DataFrame(comparison_data)


if __name__ == '__main__':
    # Ejemplo de uso
    print("=== Table Controller - Información ===")
    
    if CORE_AVAILABLE:
        print("✓ Sistema centralizado disponible")
    else:
        print("⚠️ Sistema centralizado parcialmente disponible")
    
    print("\nCaracterísticas del TableController:")
    print("• Tablas avanzadas con filtros y estadísticas")
    print("• Exportación a múltiples formatos (Excel, CSV, LaTeX, JSON)")
    print("• Edición in-situ de datos")
    print("• Formateo personalizado por columna")
    print("• Resaltado condicional de celdas")
    print("• Análisis estadístico integrado")
    print("• Impresión de tablas")
    print("• Comparación entre múltiples tablas")
    
    # Crear datos de ejemplo
    sample_data = pd.DataFrame({
        'Modo': [1, 2, 3, 4, 5],
        'Periodo': [1.25, 0.85, 0.65, 0.45, 0.35],
        'Ux': [0.65, 0.15, 0.10, 0.05, 0.05],
        'Uy': [0.10, 0.70, 0.15, 0.03, 0.02],
        'SumUx': [0.65, 0.80, 0.90, 0.95, 1.00],
        'SumUy': [0.10, 0.80, 0.95, 0.98, 1.00]
    })
    
    print(f"\nDatos de ejemplo creados: {len(sample_data)} filas, {len(sample_data.columns)} columnas")
    print("Listo para usar con TableController")