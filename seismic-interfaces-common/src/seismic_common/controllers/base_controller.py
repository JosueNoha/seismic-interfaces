"""
Controlador base centralizado para interfaces sísmicas
Proporciona funcionalidad común para todos los proyectos específicos por país
"""

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QAbstractTableModel, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QMessageBox, QFileDialog, QDialog, QVBoxLayout, 
    QHBoxLayout, QLabel, QTableView, QFrame
)
from PyQt5.QtGui import QIcon, QDoubleValidator, QIntValidator

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Importaciones del core centralizado
try:
    from seismic_common.core import (
        connect_to_etabs,
        BaseSeismicAnalysis,
        Units,
        create_default_unit_dict,
        validate_environment
    )
    CORE_AVAILABLE = True
except ImportError:
    # Fallback para desarrollo sin core instalado
    CORE_AVAILABLE = False
    print("⚠️ Core centralizado no disponible - usando importaciones locales")


class PandasTableModel(QAbstractTableModel):
    """
    Modelo de tabla estándar para mostrar DataFrames de pandas en QTableView
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el modelo con un DataFrame
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a mostrar
        """
        QAbstractTableModel.__init__(self)
        self._data = data
    
    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        """Número de filas"""
        return self._data.shape[0]
    
    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        """Número de columnas"""
        return self._data.shape[1]
    
    def data(self, index, role=Qt.DisplayRole):
        """Datos de celda"""
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None
    
    def headerData(self, col, orientation, role):
        """Headers de columnas"""
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class BaseGraphDialog(QDialog):
    """
    Diálogo base para mostrar gráficos matplotlib
    """
    
    def __init__(self, figure: Optional[Figure] = None, title: str = "Gráfico"):
        """
        Inicializa diálogo de gráfico
        
        Parameters
        ----------
        figure : Figure, optional
            Figura de matplotlib a mostrar
        title : str
            Título de la ventana
        """
        super().__init__()
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(800, 600)
        
        # Layout principal
        layout = QVBoxLayout(self)
        
        # Frame para el gráfico
        self.graph_frame = QFrame(self)
        layout.addWidget(self.graph_frame)
        
        # Configurar canvas
        if figure:
            self.set_figure(figure)
        else:
            # Crear figura vacía
            self.figure = Figure(figsize=(10, 8), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            graph_layout = QHBoxLayout(self.graph_frame)
            graph_layout.addWidget(self.canvas)
    
    def set_figure(self, figure: Figure) -> None:
        """
        Establece la figura a mostrar
        
        Parameters
        ----------
        figure : Figure
            Figura de matplotlib
        """
        self.figure = figure
        self.canvas = FigureCanvas(self.figure)
        
        # Limpiar layout anterior si existe
        if self.graph_frame.layout():
            while self.graph_frame.layout().count():
                child = self.graph_frame.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        
        # Crear nuevo layout
        graph_layout = QHBoxLayout(self.graph_frame)
        graph_layout.addWidget(self.canvas)
    
    def update_figure(self) -> None:
        """Actualiza el canvas"""
        self.canvas.draw()


class BaseTableDialog(QDialog):
    """
    Diálogo base para mostrar tablas de datos
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None, title: str = "Tabla"):
        """
        Inicializa diálogo de tabla
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            DataFrame a mostrar
        title : str
            Título de la ventana y tabla
        """
        super().__init__()
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(800, 500)
        
        # Layout principal
        layout = QVBoxLayout(self)
        
        # Título
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            QLabel {
                font-family: 'Montserrat', Arial, sans-serif;
                font-size: 18px;
                font-weight: bold;
                color: #333;
                padding: 10px;
                text-align: center;
            }
        """)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Tabla
        self.table_view = QTableView(self)
        layout.addWidget(self.table_view)
        
        # Establecer datos si se proporcionan
        if data is not None:
            self.set_data(data)
    
    def set_data(self, data: pd.DataFrame, title: Optional[str] = None) -> None:
        """
        Establece los datos de la tabla
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a mostrar
        title : str, optional
            Nuevo título para la tabla
        """
        model = PandasTableModel(data)
        self.table_view.setModel(model)
        self.table_view.resizeColumnsToContents()
        
        if title:
            self.title_label.setText(title)
            self.setWindowTitle(title)
    
    def set_column_widths(self, widths: List[int]) -> None:
        """
        Establece anchos específicos para columnas
        
        Parameters
        ----------
        widths : List[int]
            Lista de anchos para cada columna
        """
        for index, width in enumerate(widths):
            if index < self.table_view.model().columnCount():
                self.table_view.setColumnWidth(index, width)


class BaseController(ABC):
    """
    Controlador base abstracto para interfaces sísmicas
    
    Proporciona funcionalidad común para:
    - Conexión a ETABS
    - Manejo de análisis sísmico
    - Mostrar tablas y gráficos
    - Generar reportes
    - Validaciones
    """
    
    # Señales PyQt
    etabs_connected = pyqtSignal(bool)
    analysis_completed = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)  # title, message
    
    def __init__(self, main_window: QtWidgets.QMainWindow):
        """
        Inicializa el controlador base
        
        Parameters
        ----------
        main_window : QtWidgets.QMainWindow
            Ventana principal de la aplicación
        """
        self.main_window = main_window
        self.ui = main_window.ui if hasattr(main_window, 'ui') else None
        
        # Estado de ETABS
        self.etabs_object = None
        self.sap_model = None
        self.is_connected = False
        
        # Análisis sísmico (debe ser configurado por subclases)
        self.seismic_analysis = None
        
        # Sistema de unidades
        self.units = None
        self.unit_dict = {}
        
        # Configuración
        self.config = {
            'auto_connect': True,
            'show_connection_status': True,
            'default_output_dir': str(Path.home() / "SeismicReports"),
            'table_precision': 4,
            'graph_dpi': 100
        }
        
        # Inicialización
        self._initialize_units()
        self._setup_validators()
        if self.config['auto_connect']:
            self.connect_to_etabs()
    
    def _initialize_units(self) -> None:
        """Inicializa el sistema de unidades"""
        if CORE_AVAILABLE:
            self.units = Units()
            self.unit_dict = create_default_unit_dict()
        else:
            # Fallback básico
            self.unit_dict = {
                'mm': 0.001, 'm': 1.0, 'cm': 0.01,
                'kN': 1000.0, 'tonf': 9806.65, 'kgf': 9.80665
            }
    
    def _setup_validators(self) -> None:
        """Configura validadores para campos numéricos"""
        self.double_validator = QDoubleValidator()
        self.int_validator = QIntValidator()
    
    # Métodos de conexión ETABS
    def connect_to_etabs(self) -> bool:
        """
        Conecta a ETABS
        
        Returns
        -------
        bool
            True si la conexión fue exitosa
        """
        try:
            if CORE_AVAILABLE:
                self.etabs_object, self.sap_model = connect_to_etabs()
            else:
                # Implementación fallback local
                self.etabs_object, self.sap_model = self._connect_to_etabs_local()
            
            if self.sap_model is not None:
                self.is_connected = True
                self.etabs_connected.emit(True)
                
                if self.config['show_connection_status']:
                    self.show_info_message("Conexión ETABS", "Conexión exitosa a ETABS")
                
                return True
            else:
                self.is_connected = False
                self.etabs_connected.emit(False)
                self.show_error_message("Error ETABS", "No se pudo conectar a ETABS")
                return False
                
        except Exception as e:
            self.is_connected = False
            self.etabs_connected.emit(False)
            self.show_error_message("Error ETABS", f"Error conectando: {str(e)}")
            return False
    
    def _connect_to_etabs_local(self) -> Tuple[Any, Any]:
        """Implementación local de conexión (fallback)"""
        try:
            import comtypes.client
            helper = comtypes.client.CreateObject('ETABSv1.Helper')
            helper = helper.QueryInterface(comtypes.gen.ETABSv1.cHelper)
            etabs_obj = helper.GetObject("CSI.ETABS.API.ETABSObject")
            sap_model = etabs_obj.SapModel
            return etabs_obj, sap_model
        except:
            return None, None
    
    def validate_etabs_connection(self) -> bool:
        """
        Valida que la conexión a ETABS esté activa
        
        Returns
        -------
        bool
            True si la conexión es válida
        """
        if not self.is_connected or self.sap_model is None:
            self.show_error_message(
                "ETABS no conectado", 
                "Primero debe conectarse a ETABS"
            )
            return False
        
        try:
            # Probar operación simple
            self.sap_model.GetModelFilename()
            return True
        except:
            self.is_connected = False
            self.etabs_connected.emit(False)
            self.show_error_message(
                "Conexión perdida", 
                "Se perdió la conexión con ETABS"
            )
            return False
    
    # Métodos de análisis sísmico (deben implementarse en subclases)
    @abstractmethod
    def initialize_seismic_analysis(self) -> None:
        """Inicializa el análisis sísmico específico del país/normativa"""
        pass
    
    @abstractmethod
    def set_seismic_parameters(self) -> None:
        """Configura parámetros sísmicos desde la interfaz"""
        pass
    
    # Métodos comunes de análisis
    def run_modal_analysis(self) -> bool:
        """
        Ejecuta análisis modal
        
        Returns
        -------
        bool
            True si el análisis fue exitoso
        """
        if not self.validate_etabs_connection():
            return False
        
        try:
            if self.seismic_analysis:
                self.seismic_analysis.ana_modal(self.sap_model)
                self.analysis_completed.emit("modal")
                return True
            else:
                self.show_error_message("Error", "Análisis sísmico no inicializado")
                return False
        except Exception as e:
            self.show_error_message("Error Análisis Modal", str(e))
            return False
    
    def run_drift_analysis(self) -> bool:
        """
        Ejecuta análisis de derivas
        
        Returns
        -------
        bool
            True si el análisis fue exitoso
        """
        if not self.validate_etabs_connection():
            return False
        
        try:
            if self.seismic_analysis:
                self.seismic_analysis.derivas(self.sap_model)
                self.analysis_completed.emit("drifts")
                return True
            else:
                self.show_error_message("Error", "Análisis sísmico no inicializado")
                return False
        except Exception as e:
            self.show_error_message("Error Análisis Derivas", str(e))
            return False
    
    def run_irregularity_analysis(self, analysis_type: str) -> bool:
        """
        Ejecuta análisis de irregularidades
        
        Parameters
        ----------
        analysis_type : str
            Tipo de análisis ('rigidez', 'torsion', 'masa')
            
        Returns
        -------
        bool
            True si el análisis fue exitoso
        """
        if not self.validate_etabs_connection():
            return False
        
        try:
            if not self.seismic_analysis:
                self.show_error_message("Error", "Análisis sísmico no inicializado")
                return False
            
            if analysis_type == 'rigidez':
                self.seismic_analysis.irregularidad_rigidez(self.sap_model)
            elif analysis_type == 'torsion':
                self.seismic_analysis.irregularidad_torsion(self.sap_model)
            elif analysis_type == 'masa':
                self.seismic_analysis.irregularidad_masa(self.sap_model)
            else:
                self.show_error_message("Error", f"Tipo de análisis no válido: {analysis_type}")
                return False
            
            self.analysis_completed.emit(analysis_type)
            return True
            
        except Exception as e:
            self.show_error_message(f"Error Análisis {analysis_type.title()}", str(e))
            return False
    
    # Métodos de mostrar resultados
    def show_modal_table(self) -> None:
        """Muestra tabla de análisis modal"""
        if not self.run_modal_analysis():
            return
        
        if self.seismic_analysis and not self.seismic_analysis.tables.modal.empty:
            dialog = BaseTableDialog(
                self.seismic_analysis.tables.modal, 
                "Análisis Modal"
            )
            # Anchos de columna típicos para modal
            dialog.set_column_widths([60, 70, 70, 70, 70, 70, 70, 70])
            dialog.exec_()
        else:
            self.show_warning_message("Sin Datos", "No hay datos modales disponibles")
    
    def show_irregularity_table(self, irregularity_type: str, direction: str = 'X') -> None:
        """
        Muestra tabla de irregularidades
        
        Parameters
        ----------
        irregularity_type : str
            Tipo de irregularidad ('rigidez', 'torsion', 'masa')
        direction : str
            Dirección de análisis ('X' o 'Y')
        """
        if not self.run_irregularity_analysis(irregularity_type):
            return
        
        # Mapeo de tipos a tablas
        table_map = {
            'rigidez': self.seismic_analysis.tables.rigidez_table,
            'torsion': self.seismic_analysis.tables.torsion_table,
            'masa': self.seismic_analysis.tables.masa_table
        }
        
        table_data = table_map.get(irregularity_type)
        if table_data is None or table_data.empty:
            self.show_warning_message("Sin Datos", f"No hay datos de {irregularity_type} disponibles")
            return
        
        # Filtrar por dirección si corresponde
        if irregularity_type in ['rigidez', 'torsion'] and 'OutputCase' in table_data.columns:
            cases = self.get_seismic_cases_for_direction(direction)
            if cases:
                regex = '^(' + '|'.join(cases) + ')'
                table_data = table_data[table_data['OutputCase'].str.contains(regex, na=False)]
        
        # Formatear datos numéricos
        numeric_columns = table_data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            table_data[col] = table_data[col].round(self.config['table_precision'])
        
        dialog = BaseTableDialog(table_data, f"Irregularidad - {irregularity_type.title()}")
        dialog.exec_()
    
    def show_drift_graph(self) -> None:
        """Muestra gráfico de derivas"""
        if not self.run_drift_analysis():
            return
        
        if self.seismic_analysis and self.seismic_analysis.fig_drifts:
            dialog = BaseGraphDialog(
                self.seismic_analysis.fig_drifts, 
                "Derivas de Entrepiso"
            )
            dialog.exec_()
        else:
            self.show_warning_message("Sin Gráfico", "No se pudo generar el gráfico de derivas")
    
    def show_displacement_graph(self) -> None:
        """Muestra gráfico de desplazamientos"""
        if not self.validate_etabs_connection():
            return
        
        try:
            if self.seismic_analysis:
                self.seismic_analysis.desplazamientos(self.sap_model)
                if self.seismic_analysis.fig_displacements:
                    dialog = BaseGraphDialog(
                        self.seismic_analysis.fig_displacements, 
                        "Desplazamientos Laterales"
                    )
                    dialog.exec_()
                else:
                    self.show_warning_message("Sin Gráfico", "No se pudo generar el gráfico de desplazamientos")
        except Exception as e:
            self.show_error_message("Error Desplazamientos", str(e))
    
    def show_shear_graph(self, analysis_type: str = 'dynamic') -> None:
        """
        Muestra gráfico de cortantes
        
        Parameters
        ----------
        analysis_type : str
            Tipo de análisis ('dynamic' o 'static')
        """
        if not self.validate_etabs_connection():
            return
        
        try:
            if self.seismic_analysis:
                self.seismic_analysis.graph_shear(self.sap_model, analysis_type)
                
                if analysis_type == 'dynamic' and self.seismic_analysis.dynamic_shear_fig:
                    figure = self.seismic_analysis.dynamic_shear_fig
                    title = "Cortantes Dinámicas"
                elif analysis_type == 'static' and self.seismic_analysis.static_shear_fig:
                    figure = self.seismic_analysis.static_shear_fig
                    title = "Cortantes Estáticas"
                else:
                    self.show_warning_message("Sin Gráfico", f"No se pudo generar el gráfico de cortantes {analysis_type}")
                    return
                
                dialog = BaseGraphDialog(figure, title)
                dialog.exec_()
        except Exception as e:
            self.show_error_message("Error Cortantes", str(e))
    
    # Métodos de generación de reportes
    def generate_seismic_report(self) -> None:
        """Genera reporte sísmico completo"""
        if not self.validate_etabs_connection():
            return
        
        # Seleccionar directorio de salida
        output_dir = QFileDialog.getExistingDirectory(
            self.main_window, 
            "Seleccionar carpeta de destino",
            self.config['default_output_dir']
        )
        
        if not output_dir:
            return
        
        try:
            # Crear directorio único
            base_dir = Path(output_dir)
            report_dir = base_dir / "reporte_sismico"
            count = 1
            while report_dir.exists():
                report_dir = base_dir / f"reporte_sismico_{count}"
                count += 1
            
            report_dir.mkdir(exist_ok=True)
            
            # Configurar parámetros del análisis
            self.set_seismic_parameters()
            
            # Generar reporte usando el análisis sísmico
            if self.seismic_analysis:
                self.seismic_analysis.generate_memory(str(report_dir))
                
                self.show_success_message(
                    "Reporte Generado", 
                    f"El reporte se generó exitosamente en:\n{report_dir}"
                )
            else:
                self.show_error_message("Error", "Análisis sísmico no inicializado")
                
        except Exception as e:
            self.show_error_message("Error Generación", f"Error generando reporte: {str(e)}")
    
    # Métodos de utilidad
    def get_seismic_cases_for_direction(self, direction: str) -> List[str]:
        """
        Obtiene casos sísmicos para una dirección específica
        
        Parameters
        ----------
        direction : str
            Dirección ('X' o 'Y')
            
        Returns
        -------
        List[str]
            Lista de casos sísmicos
        """
        if self.seismic_analysis and hasattr(self.seismic_analysis, 'loads'):
            return self.seismic_analysis.loads.get_seismic_cases(direction)
        return []
    
    def format_numeric_value(self, value: float, decimals: int = None) -> str:
        """
        Formatea valor numérico
        
        Parameters
        ----------
        value : float
            Valor a formatear
        decimals : int, optional
            Número de decimales
            
        Returns
        -------
        str
            Valor formateado
        """
        if decimals is None:
            decimals = self.config['table_precision']
        
        try:
            return f"{float(value):.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    # Métodos de mensajes de usuario
    def show_info_message(self, title: str, message: str) -> None:
        """Muestra mensaje informativo"""
        msg = QMessageBox(self.main_window)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
    
    def show_success_message(self, title: str, message: str) -> None:
        """Muestra mensaje de éxito"""
        msg = QMessageBox(self.main_window)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
    
    def show_warning_message(self, title: str, message: str) -> None:
        """Muestra mensaje de advertencia"""
        msg = QMessageBox(self.main_window)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        msg.exec_()
    
    def show_error_message(self, title: str, message: str) -> None:
        """Muestra mensaje de error"""
        self.error_occurred.emit(title, message)
        msg = QMessageBox(self.main_window)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()
    
    def show_question(self, title: str, message: str) -> bool:
        """
        Muestra pregunta al usuario
        
        Returns
        -------
        bool
            True si el usuario acepta
        """
        reply = QMessageBox.question(
            self.main_window, title, message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return reply == QMessageBox.Yes
    
    # Métodos de configuración
    def set_config(self, key: str, value: Any) -> None:
        """Establece valor de configuración"""
        self.config[key] = value
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Obtiene valor de configuración"""
        return self.config.get(key, default)


# Funciones de utilidad para crear controladores específicos
def create_controller_for_country(country_code: str, main_window: QtWidgets.QMainWindow) -> BaseController:
    """
    Crea un controlador específico para un país
    
    Parameters
    ----------
    country_code : str
        Código del país ('PE', 'BO', etc.)
    main_window : QtWidgets.QMainWindow
        Ventana principal
        
    Returns
    -------
    BaseController
        Controlador configurado para el país
    """
    # Esta función debe ser implementada en cada proyecto específico
    # Ejemplo:
    # if country_code == 'PE':
    #     return PeruController(main_window)
    # elif country_code == 'BO':
    #     return BoliviaController(main_window)
    
    # Por ahora retorna la clase base
    class GenericController(BaseController):
        def initialize_seismic_analysis(self):
            pass
        def set_seismic_parameters(self):
            pass
    
    return GenericController(main_window)


if __name__ == '__main__':
    # Ejemplo de uso básico
    print("=== Base Controller - Información ===")
    
    if CORE_AVAILABLE:
        print("✓ Core centralizado disponible")
        env = validate_environment()
        print(f"Python: {env['python_version'].split()[0]}")
        
        deps = env['dependencies']
        print("\nDependencias:")
        for dep, status in deps.items():
            status_str = "✓" if status else "✗"
            print(f"  {dep}: {status_str}")
    else:
        print("⚠️ Core centralizado no disponible")
        print("Usando funcionalidad básica")
    
    print("\nBase Controller listo para herencia en proyectos específicos")