import os
import pandas as pd


def dividi_dados(dados: pd.DataFrame):
    """
    Divide os dados para os conjuntos de treinamento(70%), validação(20%) e teste(10%). Observe que os dados não estão sendo embaralhados
    aleatoriamente antes da divisão para garantir que o conjunto de dados seja sequencial.
    """
    n = len(dados)
    treino_df = dados[0:int(n * 0.7)]
    validacao_df = dados[int(n * 0.7):]
    return treino_df.reset_index(drop=True), validacao_df.reset_index(drop=True)




if __name__ == '__main__':
    # carrega dos dados e realiza asa transformações necessárias
    file_name = "Grafo_teste-grupo(3usuarios)(20_dias).csv"

    dados_carregado = pd.read_csv(f"../../rotinas/{file_name}")  # 60dias

    # separa dados, sendo (70%, 40%) para os conjuntos de treinamento, validação
    treino_df, validacao_df = dividi_dados(dados_carregado)

    if not os.path.exists(f"./dados/{file_name}"):
        os.makedirs(f"./dados/{file_name}")
    validacao_df.to_csv(f"./dados/{file_name}/db_selecao_hiperparametro.csv")
    treino_df.to_csv(f"./dados/{file_name}/db_experimento.csv")