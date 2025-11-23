import os
from copy import deepcopy

import pandas as pd
from joblib import Parallel, delayed
from river.neighbors import SWINN
from sklearn.model_selection import ParameterSampler

from experiment.cenarios import Cenarios
from experiment.intelligence_incremental import IntelligenceIncremental
from experiment.parametros import param_grid
from experiment.utils import monta_modelo
from experiment.validacao import CalculoMetricas


def save_hiperparametros(caminho_base: str, nome_modelo: str, metrica: dict):
    caminho_arquivo = f"{caminho_base}/metricas_por_hiperparametros_{nome_modelo}.csv"
    if not os.path.exists(caminho_arquivo):
        os.makedirs(caminho_base, exist_ok=True)
    with open(caminho_arquivo, 'a') as f:
        # nome_modelo, parametros, mediaMicro
        f.write(f"{metrica['nome_modelo']}, \"{str(metrica['parametros'])}\", {metrica['mediaMicro']}\n")


def processa_modelos_dia_a_dia(nome_modelo: str, parametros: dict):
    todos_cenarios = Cenarios()
    metrica_dia = []
    max_window_size = parametros.pop('max_window_size')
    min_window_size = parametros.pop('min_window_size')
    try:
        for index_cenario, cenario in enumerate(todos_cenarios.todos_cenarios_experimento()):
            experimento = f'{nome_modelo}_cenario_{index_cenario}'
            print(f'######### {nome_modelo}_cenario_{index_cenario} #########')

            modelo = monta_modelo(parametros, nome_modelo)
            inteligence = IntelligenceIncremental(experimento, model=modelo, max_window_size=max_window_size,
                                                  min_window_size=min_window_size, save=True)

            i = 0
            for dto, linha in cenario:
                dto = list(dto.values())[0]
                list_rec, tempo = inteligence.generate_incremental_recommendation(dto)
                print(f'{nome_modelo} - linha: {i}')
                i += 1

            df_X_Y = inteligence.df_treino_incremental_model.set_index('X_index')
            df_REC = inteligence.df_recommendation.set_index('X_index')
            pd.options.display.max_columns = None
            print(df_X_Y.head())
            print(df_REC.head())

            metrica_dia += CalculoMetricas.metricas_por_dia(nome_modelo, index_cenario, df_REC, df_X_Y)
    except Exception as e:
        print(e)

    save_metrica_por_dia('./modelo_treinado', nome_modelo, metrica_dia)


def save_metrica_por_dia(caminho_base: str, nome_modelo: str, metrica: list):
    caminho_arquivo = f"{caminho_base}/metricas_por_dia_{nome_modelo}.csv"
    if not os.path.exists(caminho_arquivo):
        os.makedirs(caminho_base, exist_ok=True)
    df = pd.DataFrame(metrica)
    df.to_csv(caminho_arquivo)


def processa_modelo_hiperparametros(index_parametro, tamanho_total_lote, parametros, nome_modelo):
    print(f'########### {index_parametro} de {tamanho_total_lote} ###########')
    todos_cenarios = Cenarios()

    media_micro = 0.0
    parametro_teste = deepcopy(parametros)
    max_window_size = parametro_teste.pop('max_window_size')
    min_window_size = parametro_teste.pop('min_window_size')

    for index_cenario, cenario in enumerate(todos_cenarios.todos_cenarios_hiperparametro()):
        experimento = f'{nome_modelo}_cenario_{index_cenario}'
        print(f'######### {nome_modelo}_cenario_{index_cenario} #########')
        modelo = monta_modelo(parametro_teste, nome_modelo)
        inteligence = IntelligenceIncremental(experimento, max_window_size=max_window_size,
                                              min_window_size=min_window_size, model=modelo, save=True)
        linha = 0
        for dto, _ in cenario:
            print(f'linha: {linha}')
            dto = list(dto.values())[0]
            inteligence.generate_incremental_recommendation(dto)
            linha += 1
        df_X_Y = inteligence.df_treino_incremental_model.set_index('X_index')
        df_REC = inteligence.df_recommendation.set_index('X_index')
        media_micro += CalculoMetricas.metricas_por_hiperparametros(df_REC, df_X_Y)

    media_micro = media_micro / len(todos_cenarios.todos_cenarios_hiperparametro())
    metrica = {'nome_modelo': nome_modelo,
               'parametros': str(parametros),
               'mediaMicro': media_micro}

    save_hiperparametros('./modelo_treinado', nome_modelo, metrica)


def experimento_hiperparametros(nomes_modelos_arvore: list):
    for nome_modelo in nomes_modelos_arvore:
        param_grid_model = param_grid[nome_modelo]
        lista_parametros = ParameterSampler(param_grid_model, 3500)
        tamanho_total_lote = len(lista_parametros)
        print(tamanho_total_lote)

        resultado = Parallel(n_jobs=42)(
            delayed(processa_modelo_hiperparametros)(index, tamanho_total_lote, n, nome_modelo) for index, n in
            enumerate(lista_parametros))
        # for index_parametro, p in enumerate(lista_parametros):
        #     processa_modelo_hiperparametros(index_parametro, tamanho_total_lote, p, nome_modelo)


if __name__ == '__main__':
    tipo = 'modelo'  # 'hiperparametros', 'modelo'

    if tipo == 'hiperparametros':
        modelos_arvore = [
            'ExtremelyFastDecisionTreeClassifier',
            'HoeffdingAdaptiveTreeClassifier',
            'HoeffdingTreeClassifier',
            'AMFClassifier',
            'ARFClassifier',
            # 'ALMAClassifier',
            'KNNClassifier'
        ]
        experimento_hiperparametros(modelos_arvore)
    elif tipo == 'modelo':
        melhores_parametros_modelos_arvore = {
            'ExtremelyFastDecisionTreeClassifier': {'tau': 0.04, 'split_criterion': 'info_gain', 'nb_threshold': 1,
                                                    'min_window_size': 30, 'max_window_size': 60,
                                                    'min_samples_reevaluate': 25, 'min_branch_fraction': 0.01,
                                                    'merit_preprune': False, 'max_share_to_split': 0.98, 'max_depth': 1,
                                                    'leaf_prediction': 'nb', 'grace_period': 100, 'delta': 1e-07},
            'HoeffdingAdaptiveTreeClassifier': {'tau': 0.05, 'switch_significance': 0.5, 'stop_mem_management': True,
                                                'split_criterion': 'info_gain', 'nb_threshold': 0,
                                                'min_window_size': 30, 'min_branch_fraction': 0.01,
                                                'merit_preprune': True, 'max_window_size': 60,
                                                'max_share_to_split': 0.97, 'max_depth': None, 'leaf_prediction': 'nba',
                                                'grace_period': 25, 'drift_window_threshold': 350, 'delta': 1e-05,
                                                'bootstrap_sampling': True, 'binary_split': False},
            'HoeffdingTreeClassifier': {'tau': 0.07, 'stop_mem_management': False, 'split_criterion': 'info_gain',
                                        'remove_poor_attrs': True, 'nb_threshold': 1, 'min_window_size': 30,
                                        'min_branch_fraction': 0.01, 'merit_preprune': False, 'max_window_size': 60,
                                        'max_share_to_split': 0.98, 'max_depth': 1, 'leaf_prediction': 'nb',
                                        'grace_period': 25, 'delta': 1e-07, 'binary_split': False},
            'AMFClassifier': {'use_aggregation': False, 'split_pure': False, 'n_estimators': 20, 'min_window_size': 30,
                              'max_window_size': 60, 'dirichlet': 0.5},
            'ARFClassifier': {'tau': 0.03, 'split_criterion': 'info_gain', 'remove_poor_attrs': False,
                              'nb_threshold': 3, 'n_models': 25, 'min_window_size': 15, 'min_branch_fraction': 0.01,
                              'merit_preprune': True, 'max_window_size': 60, 'max_share_to_split': 0.99,
                              'max_features': 'sqrt', 'leaf_prediction': 'nb', 'lambda_value': 8, 'grace_period': 50,
                              'disable_weighted_vote': False, 'delta': 1e-08},
            # 'ALMAClassifier': {'p': 5, 'min_window_size': 60, 'max_window_size': 60, 'alpha': 0.8, 'C': 1.1, 'B': 1.2},
            'KNNClassifier': {'weighted': True, 'n_neighbors': 3, 'min_window_size': 30, 'max_window_size': 60,
                              'engine': SWINN()}
        }
        # for nome_modelo, parametros in melhores_parametros_modelos_arvore.items():
        #     processa_modelos_dia_a_dia(nome_modelo, parametros)

        resultado = Parallel(n_jobs=8)(
            delayed(processa_modelos_dia_a_dia)(nome_modelo, parametros) for nome_modelo, parametros in
            melhores_parametros_modelos_arvore.items())
