"""
Utilidades centralizadas para conexión y manejo de ETABS API
Módulo común para proyectos de análisis sísmico
"""

import comtypes.client
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Any


def connect_to_csi(prog: str) -> Tuple[Any, Any]:
    """
    Conecta a un programa CSI (ETABS, SAFE, etc.)
    
    Parameters
    ----------
    prog : str
        Nombre del programa CSI ('ETABS', 'SAFE', etc.)
        
    Returns
    -------
    Tuple[Any, Any]
        Tupla con (CSIObject, SapModel) o (None, None) si falla
    """
    try:
        # Crear objeto helper de API
        helper = comtypes.client.CreateObject(f'{prog}v1.Helper')
        exec(f'helper = helper.QueryInterface(comtypes.gen.{prog}v1.cHelper)')
        
        # Adjuntar a instancia en ejecución de ETABS
        CSIObject = helper.GetObject(f"CSI.{prog}.API.ETABSObject")
        
        # Crear objeto SapModel
        SapModel = CSIObject.SapModel
        
        try:
            set_envelopes_for_display(SapModel)
        except:
            # Intentar obtener objeto activo como respaldo
            CSIObject = comtypes.client.GetActiveObject(f"CSI.{prog}.API.ETABSObject")
            SapModel = CSIObject.SapModel
            try:
                set_envelopes_for_display(SapModel)
            except:
                print(f'Lo sentimos no es posible conectarse al API de {prog}')  
                return None, None
        
        return CSIObject, SapModel
        
    except Exception as e:
        print(f'No es posible conectarse a {prog}: {str(e)}')
        return None, None


def connect_to_etabs() -> Tuple[Any, Any]:
    """
    Conecta específicamente a ETABS
    
    Returns
    -------
    Tuple[Any, Any]
        Tupla con (EtabsObject, SapModel) o (None, None) si falla
    """
    return connect_to_csi('ETABS')


def connect_to_safe() -> Tuple[Any, Any]:
    """
    Conecta específicamente a SAFE
    
    Returns
    -------
    Tuple[Any, Any]
        Tupla con (SafeObject, SapModel) o (None, None) si falla
    """
    return connect_to_csi('SAFE')


def set_units(SapModel: Any, unit: str = 'Ton_m_C') -> None:
    """
    Establece las unidades del modelo
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
    unit : str
        Código de unidad ('Ton_m_C', 'kgf_cm_C', 'N_m_C', 'Ton_mm_C')
    """
    units = {
        'Ton_m_C': 12, 
        'kgf_cm_C': 14,
        'N_m_C': 10,
        'Ton_mm_C': 11
    }
    
    if unit in units:
        SapModel.SetPresentUnits(units[unit])
    else:
        raise ValueError(f"Unidad no reconocida: {unit}. Opciones válidas: {list(units.keys())}")


def get_modal_data(SapModel: Any, clean_data: bool = True) -> Tuple[pd.DataFrame, float, float, float, float, float, float]:
    """
    Obtiene datos del análisis modal
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
    clean_data : bool, optional
        Si True, elimina columnas no necesarias (default: True)
        
    Returns
    -------
    Tuple[pd.DataFrame, float, float, float, float, float, float]
        Tupla con (modal_df, MP_x, MP_y, period_x, period_y, Ux, Uy)
    """
    # Configurar salida para análisis modal
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    SapModel.Results.Setup.SetCaseSelectedForOutput("Modal")
    
    # Obtener datos modales
    modal = SapModel.Results.ModalParticipatingMassRatios()
    
    # Convertir a DataFrame
    modal = pd.DataFrame(
        modal[1:17],
        index=['LoadCase', 'StepType', 'StepNum', 'Period', 'Ux', 'Uy', 'Uz', 
               'SumUx', 'SumUy', 'SumUz', 'Rx', 'Ry', 'Rz', 'SumRx', 'SumRy', 'SumRz']
    ).transpose()
    
    if clean_data:
        modal = modal.drop(
            ['LoadCase', 'StepType', 'StepNum', 'Rx', 'Ry', 'Rz', 'SumRx', 'SumRy', 'SumRz'],
            axis=1
        )
    
    # Calcular masas participativas máximas
    MP_x = max(modal.SumUx)
    MP_y = max(modal.SumUy)
    
    # Encontrar periodos fundamentales
    mode_x = modal[modal.Ux == max(modal.Ux)].index
    period_x = modal.Period[mode_x[0]]
    Ux = modal.Ux[mode_x[0]]
    
    mode_y = modal[modal.Uy == max(modal.Uy)].index
    period_y = modal.Period[mode_y[0]]
    Uy = modal.Uy[mode_y[0]]
    
    return (modal, MP_x, MP_y, period_x, period_y, Ux, Uy)


def set_envelopes_for_display(SapModel: Any, set_envelopes: bool = True) -> None:
    """
    Configura las opciones de visualización de envolventes
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
    set_envelopes : bool, optional
        Si True, configura para mostrar envolventes (default: True)
    """
    IsUserBaseReactionLocation = False
    UserBaseReactionX = 0
    UserBaseReactionY = 0
    UserBaseReactionZ = 0
    IsAllModes = True
    StartMode = 0
    EndMode = 0
    IsAllBucklingModes = True
    StartBucklingMode = 0
    EndBucklingMode = 0
    MultistepStatic = 1 if set_envelopes else 2
    NonlinearStatic = 1 if set_envelopes else 2
    ModalHistory = 1
    DirectHistory = 1
    Combo = 2
    
    SapModel.DataBaseTables.SetOutputOptionsForDisplay(
        IsUserBaseReactionLocation, UserBaseReactionX,
        UserBaseReactionY, UserBaseReactionZ, IsAllModes,
        StartMode, EndMode, IsAllBucklingModes, StartBucklingMode,
        EndBucklingMode, MultistepStatic, NonlinearStatic,
        ModalHistory, DirectHistory, Combo
    )


def get_table(SapModel: Any, table_name: str, set_envelopes: bool = True) -> Tuple[List[str], pd.DataFrame]:
    """
    Obtiene una tabla de la base de datos de ETABS
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
    table_name : str
        Nombre de la tabla a obtener
    set_envelopes : bool, optional
        Si True, configura envolventes antes de obtener tabla (default: True)
        
    Returns
    -------
    Tuple[List[str], pd.DataFrame]
        Tupla con (nombres_columnas, tabla_dataframe)
    """
    set_envelopes_for_display(SapModel, set_envelopes)
    data = SapModel.DatabaseTables.GetTableForDisplayArray(table_name, FieldKeyList='', GroupName='')
    
    # Si no hay datos, ejecutar análisis
    if not data[2][0]:
        SapModel.Analyze.RunAnalysis()
        data = SapModel.DatabaseTables.GetTableForDisplayArray(table_name, FieldKeyList='', GroupName='')
    
    columns = data[2]
    # Reemplazar valores None por cadena vacía
    data = [i if i is not None else '' for i in data[4]]
    
    # Reshape data
    data = pd.DataFrame(data)
    num_rows = int(len(data) / len(columns))
    data = data.values.reshape(num_rows, len(columns))
    table = pd.DataFrame(data, columns=columns)
    
    return columns, table


def get_unique_cases(SapModel: Any, combo_name: str) -> List[str]:
    """
    Obtiene todos los casos únicos que componen una combinación
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
    combo_name : str
        Nombre de la combinación
        
    Returns
    -------
    List[str]
        Lista de casos únicos en la combinación
    """
    # Obtener información de la combinación
    combo_info = SapModel.RespCombo.GetCaseList(combo_name)
    combo_types = combo_info[1]
    combo_cases = combo_info[2]
    
    unique_cases = []
    
    for c_type, combo_case in zip(combo_types, combo_cases):
        if c_type == 0:  # Caso de carga
            unique_cases.append(combo_case)
        elif c_type == 1:  # Combinación
            unique_cases.extend(get_unique_cases(SapModel, combo_case))
    
    return list(set(unique_cases))


def get_story_data(SapModel: Any) -> pd.DataFrame:
    """
    Obtiene datos de pisos/niveles
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
        
    Returns
    -------
    pd.DataFrame
        DataFrame con datos de pisos
    """
    _, story_data = get_table(SapModel, 'Story Definitions')
    return story_data


def get_joint_displacements(SapModel: Any, load_cases: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Obtiene desplazamientos de nodos
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
    load_cases : List[str], optional
        Lista de casos de carga específicos (default: None para todos)
        
    Returns
    -------
    pd.DataFrame
        DataFrame con desplazamientos de nodos
    """
    _, displacements = get_table(SapModel, 'Joint Displacements')
    
    if load_cases:
        displacements = displacements[displacements['OutputCase'].isin(load_cases)]
    
    return displacements


def get_story_drifts(SapModel: Any, load_cases: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Obtiene derivas de entrepiso
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
    load_cases : List[str], optional
        Lista de casos de carga específicos (default: None para todos)
        
    Returns
    -------
    pd.DataFrame
        DataFrame con derivas de entrepiso
    """
    _, drifts = get_table(SapModel, 'Story Drifts')
    
    if load_cases:
        drifts = drifts[drifts['OutputCase'].isin(load_cases)]
    
    return drifts


def get_base_reactions(SapModel: Any, load_cases: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Obtiene reacciones en la base
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
    load_cases : List[str], optional
        Lista de casos de carga específicos (default: None para todos)
        
    Returns
    -------
    pd.DataFrame
        DataFrame con reacciones en la base
    """
    _, reactions = get_table(SapModel, 'Base Reactions')
    
    if load_cases:
        reactions = reactions[reactions['OutputCase'].isin(load_cases)]
    
    return reactions


def check_model_is_analyzed(SapModel: Any) -> bool:
    """
    Verifica si el modelo ha sido analizado
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
        
    Returns
    -------
    bool
        True si el modelo está analizado, False en caso contrario
    """
    try:
        data = SapModel.DatabaseTables.GetTableForDisplayArray(
            'Analysis Messages', FieldKeyList='', GroupName=''
        )
        return bool(data[2][0])
    except:
        return False


def run_analysis(SapModel: Any) -> bool:
    """
    Ejecuta el análisis del modelo
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
        
    Returns
    -------
    bool
        True si el análisis fue exitoso, False en caso contrario
    """
    try:
        result = SapModel.Analyze.RunAnalysis()
        return result == 0  # 0 indica éxito en la API de ETABS
    except Exception as e:
        print(f"Error ejecutando análisis: {str(e)}")
        return False


def get_load_cases(SapModel: Any) -> List[str]:
    """
    Obtiene lista de todos los casos de carga
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
        
    Returns
    -------
    List[str]
        Lista con nombres de casos de carga
    """
    try:
        _, load_cases_table = get_table(SapModel, 'Load Case Definitions - Summary')
        return load_cases_table['Name'].unique().tolist()
    except:
        return []


def get_load_combinations(SapModel: Any) -> List[str]:
    """
    Obtiene lista de todas las combinaciones de carga
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
        
    Returns
    -------
    List[str]
        Lista con nombres de combinaciones de carga
    """
    try:
        _, combos_table = get_table(SapModel, 'Load Combination Definitions')
        return combos_table['Name'].unique().tolist()
    except:
        return []


# Funciones de utilidad adicionales
def validate_connection(SapModel: Any) -> bool:
    """
    Valida que la conexión a ETABS sea funcional
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
        
    Returns
    -------
    bool
        True si la conexión es válida, False en caso contrario
    """
    try:
        # Intentar una operación simple
        SapModel.GetModelFilename()
        return True
    except:
        return False


def get_model_info(SapModel: Any) -> dict:
    """
    Obtiene información general del modelo
    
    Parameters
    ----------
    SapModel : Any
        Objeto SapModel de ETABS
        
    Returns
    -------
    dict
        Diccionario con información del modelo
    """
    info = {}
    
    try:
        info['filename'] = SapModel.GetModelFilename()
        info['units'] = SapModel.GetPresentUnits()
        info['is_locked'] = SapModel.GetModelIsLocked()
        info['load_cases'] = len(get_load_cases(SapModel))
        info['load_combinations'] = len(get_load_combinations(SapModel))
        info['is_analyzed'] = check_model_is_analyzed(SapModel)
    except Exception as e:
        info['error'] = str(e)
    
    return info


if __name__ == '__main__':
    # Ejemplo de uso
    print("Intentando conectar a ETABS...")
    etabs_obj, sap_model = connect_to_etabs()
    
    if sap_model:
        print("Conexión exitosa!")
        model_info = get_model_info(sap_model)
        print("Información del modelo:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
    else:
        print("No se pudo conectar a ETABS")