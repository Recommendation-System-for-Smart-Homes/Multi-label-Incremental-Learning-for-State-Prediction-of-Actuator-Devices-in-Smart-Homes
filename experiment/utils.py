import itertools
import os
from datetime import datetime

from river import linear_model as lm
from river import preprocessing as pp
from river import tree, neighbors, forest
from river.multioutput import ClassifierChain

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


def create_time_variables(timestamp: datetime) -> dict:
    """Calcula features cíclicas (sin/cos) a partir de um timestamp."""
    # Períodos para normalização
    seconds_in_day = 24 * 60 * 60
    days_in_week = 7
    months_in_year = 12

    # Posição no ciclo
    s = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    d = timestamp.weekday()  # Monday=0, Sunday=6
    m = timestamp.month

    # Cálculo das features
    features = {
        'day_sin': np.sin(2 * np.pi * s / seconds_in_day),
        'day_cos': np.cos(2 * np.pi * s / seconds_in_day),
        'weekday_sin': np.sin(2 * np.pi * d / days_in_week),
        'weekday_cos': np.cos(2 * np.pi * d / days_in_week),
        'month_sin': np.sin(2 * np.pi * m / months_in_year),
        'month_cos': np.cos(2 * np.pi * m / months_in_year),
    }
    return features

def monta_modelo(parametros: dict, nome_modelo: str):
    modelo_arvore = None
    match nome_modelo:
        case 'HoeffdingAdaptiveTreeClassifier':
            modelo_arvore = tree.HoeffdingAdaptiveTreeClassifier(**parametros)
        case 'ExtremelyFastDecisionTreeClassifier':
            modelo_arvore = tree.ExtremelyFastDecisionTreeClassifier(**parametros)
        case 'HoeffdingTreeClassifier':
            modelo_arvore = tree.HoeffdingTreeClassifier(**parametros)
        case 'KNNClassifier':
            modelo_arvore = neighbors.KNNClassifier(**parametros)
        case 'AMFClassifier':
            modelo_arvore = forest.AMFClassifier(**parametros)
        case 'ARFClassifier':
            modelo_arvore = forest.ARFClassifier(**parametros)
        case 'ALMAClassifier':
            modelo_arvore = (pp.StandardScaler() | lm.ALMAClassifier(**parametros))
    mult_output = ClassifierChain(model=modelo_arvore)
    return mult_output

def combinacao_parametros(param_grid: dict):
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    params_completo = []

    # Iterar sobre as combinações de hiperparâmetros
    for param_comb in param_combinations:
        # Criar um dicionário de parâmetros para o modelo atual
        params = dict(zip(param_names, param_comb))
        params_completo.append(params)
    return params_completo[:50000]

class IncrementalLabelEncoder(LabelEncoder):
    """
    transformador incremental usado para codificar valores alvo. Utilize o método transform() para realizar a codificaçõ e o inverse_transform() desfazer a codificacao.
    """

    def __init__(self):
        super()
        self.encoder_file = None
        self.classes_ = np.array([])
        self.name = "my_encoder"

    def transform(self, x) -> list[int]:
        """
        realiza a transformacao de dados recebidos, sendo que o encoder pode ser feito de forma incremental.

        Example:
            encoder.transform(["batata", "pera", "morango"]) # -> [0, 1, 2]
            encoder.inverse_transform([2]) # -> ["morango"]

        :param x: lista de elementos que devem ser transformados.
        :return: lista de elementos já tranformados
        """
        E = [str(e) for e in x]
        check_is_fitted(self)
        for e in E:
            if e not in self.classes_:
                self.classes_ = np.concatenate([self.classes_, np.array([e])])
        return super().transform(x).tolist()

    def __check_folder(self) -> None:
        """Verifica e cria estrtura de pasta necessária para salvar os arquivos e encoders"""
        if not os.path.isdir(self.encoder_file):
            os.makedirs(self.encoder_file)
