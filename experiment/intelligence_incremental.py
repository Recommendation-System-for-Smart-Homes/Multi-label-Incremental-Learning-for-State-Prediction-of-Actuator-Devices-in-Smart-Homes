import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from river import tree
from river.multioutput import ClassifierChain

from experiment.dto import DeviceStatusDTO
from experiment.utils import IncrementalLabelEncoder, create_time_variables


class IntelligenceIncremental:
    """
    classe responsavel pelo modelo de arvore incremental.
    """

    def __init__(self, key_experimento, max_window_size, min_window_size, model=None, save=False):
        self.key_experimento = key_experimento  # identificador do experimento
        self.dir_to_save = os.path.abspath('/'.join(["modelo_treinado", self.key_experimento]))
        self.caminho_arquivo_instancia = Path(f'{self.dir_to_save}/IntelligenceIncremental.pkl')
        self.model = model
        self.save = save

        self.max_window_size = max_window_size
        self.max_window_density = 30
        self.min_window_size = min_window_size
        self.min_window_density = 10
        self.t_lookback = 600  # Período de Observação da Densidade
        self.current_density = 0
        self.registry_density: list[DeviceStatusDTO] = []

        # Esta varaivel representa o db que armazena o status de todos os dispositivos do ambiente. A ultima linha é o status mais recente
        self.environment_status_incremental = pd.DataFrame(
            columns=["interval_sin", "interval_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"])

        self.buffer_x = []  # entrada
        self.buffer_y = []  # alvo
        self.buffer_w = []  # janela dinamica
        self.buffer_d = []  # densidade
        self.model_file = None
        self.encoder = IncrementalLabelEncoder()
        self.encoder_columns = []
        self.encoder_classes = []
        self.registered_sensors = []
        self.registered_actuators = []

        self.file_treino = None
        self.file_recommendation = None
        self.df_treino_incremental_model = pd.DataFrame()
        self.df_recommendation = pd.DataFrame()

        self.cont_treino = 0
        self.cont_recommendation = 0


    def _carregar_estado(self) -> bool:
        """Verifica a existência do arquivo e tenta carregar o estado salvo."""

        if self.caminho_arquivo_instancia.is_file():
            try:
                with open(self.caminho_arquivo_instancia, 'rb') as f:
                    # Carrega todos os atributos salvos e os atribui à instância atual
                    dados = pickle.load(f)

                    # Atualiza a instância com os dados do arquivo
                    self.__dict__.update(dados.__dict__)

                    return True
            except Exception as e:
                print(f"[ERRO] Falha ao carregar o arquivo pickle. Ignorando e inicializando padrão. Erro: {e}")
                return False
        return False

    def salvar_estado(self):
        try:
            with open(self.caminho_arquivo_instancia, 'wb') as f:
                pickle.dump(self, f)
            print(f"[SUCESSO] Estado atualizado e salvo em: {self.caminho_arquivo_instancia}")
        except Exception as e:
            print(f"[ERRO] Não foi possível salvar o arquivo: {e}")

    def update_density(self, new_status_dto: DeviceStatusDTO):
        self.registry_density.append(new_status_dto)
        i = 0
        while i < len(self.registry_density):
            delta_seconds = (new_status_dto.timeStamp - self.registry_density[i].timeStamp).total_seconds()
            if delta_seconds > self.t_lookback:
                self.registry_density.pop(i)
                # Não incrementa i, pois pop() desloca o próximo para a posição atual
            else:
                i += 1
        self.current_density = len(self.registry_density)

    def load_encoder(self) -> IncrementalLabelEncoder:
        return self.encoder

    def save_model(self) -> None:
        """ Salva o modelo e se necessário cria estrutura de pastas necessária """
        if self.save:
            model_file_name = "incremental_model.pkl"
            path_to_save_model = Path(os.path.abspath('/'.join([self.dir_to_save, model_file_name])))
            os.makedirs(self.dir_to_save, exist_ok=True)

            with open(path_to_save_model, 'wb') as f:
                pickle.dump(self.model, f)
                self.model_file = str(path_to_save_model)

    def load_model(self) -> None:
        if self.model_file is None:
            self.model = ClassifierChain(
                # model=forest.AMFClassifier(n_estimators=10, use_aggregation=True, dirichlet=0.1, split_pure=False)
                model=tree.ExtremelyFastDecisionTreeClassifier(grace_period=25, max_depth=4,
                                                               min_samples_reevaluate=15, split_criterion='info_gain',
                                                               delta=1e-05, tau=0.04, leaf_prediction='mc',
                                                               nb_threshold=1, min_branch_fraction=0.012,
                                                               max_share_to_split=0.995, max_size=100.0,
                                                               merit_preprune=False)
            )
            self.save_model()
        else:
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
        return self.model

    def train_incremental(self, x_a: dict, y_a: dict) -> None:
        """Implementação do treinamento incremental do modelo"""
        # self.load_model()
        self.model.learn_one(x_a, y_a)
        self.save_model()

    def register_device(self, data: DeviceStatusDTO) -> None:
        if data.sensorType == "actuator":
            if not data.devId in self.registered_actuators:
                self.registered_actuators.append(data.devId)

        elif data.sensorType == "sensor":
            if not data.devId in self.registered_sensors:
                self.registered_sensors.append(data.devId)

    def current_window_size_linear(self):
        """
        Calcula current_window_size usando um mapeamento linear inverso.
        A janela diminui linearmente à medida que a densidade aumenta.
        Returns:
            float: O tamanho calculado da janela, W_current.
        """
        # Garante que a densidade não exceda os limites definidos
        if self.current_density <= self.min_window_density:
            return self.max_window_size
        if self.current_density >= self.max_window_density:
            return self.min_window_size

        # Aplica a fórmula de interpolação linear inversa
        fracao_densidade = (self.current_density - self.min_window_density) / (
                self.max_window_density - self.min_window_density)
        return self.max_window_size - fracao_densidade * (self.max_window_size - self.min_window_size)

    def current_window_size_exponencial(self):
        """
        Calcula current_window_size usando uma função de decaimento exponencial.
        A janela diminui rapidamente no início e depois suaviza.

        Returns:
            float: O tamanho calculado da janela, W_current.
        """
        k = -0.1  # Fator de decaimento para o metodo exponencial
        return self.min_window_size + (self.max_window_size - self.min_window_size) * math.exp(k * self.current_density)

    def generate_incremental_recommendation(self, status_dto: DeviceStatusDTO) -> (list, int):
        """
        Função para gerar a recomendação baseado no Status atual da casa.
        """

        if status_dto.sensorType == "atuador":
            print(status_dto.device)
        full_recommendation = []

        self.update_density(status_dto)

        # 0) Realiza procedimentos antes de realizar a predição
        # Realiza o encoder das mensagens do dispositivo
        message_encoded = self.encoder.transform([status_dto.message])[0]
        # Atualiza a variavel que mantem o dataFrame com o status atual de toda a casa
        self.update_status_environment(status_dto, message_encoded)
        # Atualiza lista de dispositivos presentes no ambiente
        self.register_device(status_dto)

        # ============================================================================================================

        # 1) Carrega instância do ambiente e status atual
        x_environment_current_status = self.environment_status_incremental.iloc[-1].to_frame().T

        # 2) Extrai o Y atual (atuadores). Caso exista algum atuador que ainda não foi registrado no Environment.registered_actuators ele não irá produzir recomendação.
        y_actuators_status = x_environment_current_status[self.registered_actuators]

        # 3) Indica a atualização do buffer e do modelo incremental.
        self.update_buffer_and_model(x_environment_current_status, y_actuators_status)

        # 4) Realiza predição com base no status atual da casa
        atuctuators_prev, inference_elapsed_time = self.predict_state(
            x_environment_current_status.reset_index(drop=True).loc[0].to_dict())

        # 5) Compara o status preditos dos atuadores com os status atuais e define quais dispositivos devem ter seu status atualizado
        atuctuators = list(set(x_environment_current_status.columns.values) & set(atuctuators_prev.keys()))

        self.salva_dados_recomendacao(x_environment_current_status, atuctuators_prev, inference_elapsed_time)

        if len(atuctuators) > 1:
            atuctuators_current_status = x_environment_current_status[atuctuators].to_dict(orient='records')[0]
            modify_actuators = {id_disp: atuctuators_prev[id_disp] for id_disp in atuctuators if
                                atuctuators_prev[id_disp] != atuctuators_current_status[id_disp]}

            # 6) Monta estrutura de lista de recomendações
            if len(modify_actuators) > 0:
                for actuator_id, actuator_rec in modify_actuators.items():
                    full_recommendation.append({"device_id": actuator_id, "message": actuator_rec})
        return full_recommendation, inference_elapsed_time

    def predict_state(self, x_i: dict) -> (dict, float):
        """Implementação da predicao do modelo"""
        # self.load_model()
        start_time = time.time_ns()
        Y_prev = self.model.predict_one(x_i)
        end_time = time.time_ns()
        inference_elapsed_time = (end_time - start_time) / 1_000_000
        if Y_prev is None:
            return {}, inference_elapsed_time
        return Y_prev, inference_elapsed_time

    def update_buffer_and_model(self, x_environment_current_status: DataFrame, y_actuators_status: DataFrame) -> None:
        """ Este metodo gerencia um fluxo contínuo de dados, armazenando e atualizando informações em um buffer antes de treiná-las em um modelo incremental """
        format_time = "%Y-%m-%d %H:%M:%S.%f"

        # 1) **INSERIR (X, Y) SIMULTANEAMENTE** para garantir índice 1:1
        self.append_buffer_x(x_environment_current_status)
        self.append_buffer_y(y_actuators_status)
        self.append_buffer_w(self.current_window_size_exponencial())
        self.append_buffer_d(self.current_density)

        # 2) **ATUALIZAR Y** para todos os X no buffer que ainda estejam na janela
        x_current_time = pd.to_datetime(x_environment_current_status.index[0])
        for i in range(len(self.get_buffer_x())):
            x_df = self.get_index_buffer_x(i)
            timestamp_buffer = pd.to_datetime(x_df.index.values[0])
            delta_seconds = (x_current_time - timestamp_buffer).total_seconds()

            # Se o X ainda não ultrapassou a janela, "atualizamos" seu Y
            if delta_seconds <= self.get_index_buffer_w(i):
                self.update_buffer_y(i, y_actuators_status)
            # Caso contrário, deixamos como está, pois logo iremos treinar e remover
        # 3) **TREINAR E REMOVER** amostras que ultrapassaram a janela
        i = 0
        while i < len(self.get_buffer_x()):
            try:
                x_df = self.get_index_buffer_x(i)
                y_df = self.get_index_buffer_y(i)
                w_for_x = self.get_index_buffer_w(i)
                d_for_x = self.get_index_buffer_d(i)

                timestamp_buffer = pd.to_datetime(x_df.index.values[0])
                delta_seconds = (x_current_time - timestamp_buffer).total_seconds()

                # Se passou da janela, treinamos e removemos
                if delta_seconds > w_for_x:
                    self.train_incremental(
                        x_df.reset_index(drop=True).squeeze('index').to_dict(),
                        y_df.reset_index(drop=True).squeeze('index').to_dict()
                    )
                    self.del_buffer_x(i)
                    self.del_buffer_y(i)
                    self.del_buffer_w(i)
                    self.del_buffer_d(i)

                    self.salva_dados_para_treino(x_df, y_df, w_for_x, d_for_x)
                    # Não incrementa i, pois pop() desloca o próximo para a posição atual
                else:
                    i += 1

            except Exception as e:
                print(f"Erro modelo incremental: {e}")
                # Para evitar loop travado em caso de erro, avance
                i += 1

    def get_buffer_x(self) -> DataFrame:
        return DataFrame.from_dict({list(item.keys())[0]: list(item.values())[0] for item in self.buffer_x},
                                   orient='index')

    def get_index_buffer_x(self, index) -> DataFrame:
        return DataFrame.from_dict(self.buffer_x[index], orient='index')

    def append_buffer_x(self, x_i: DataFrame) -> bool:
        self.buffer_x.append(x_i.to_dict(orient='index'))
        return True

    def update_buffer_x(self, index, x_i) -> bool:
        self.buffer_x[index] = x_i.to_dict(orient='index')
        return True

    def del_buffer_x(self, index) -> bool:
        self.buffer_x.pop(index)
        return True

    def get_buffer_y(self) -> DataFrame:
        return DataFrame.from_dict({list(item.keys())[0]: list(item.values())[0] for item in self.buffer_y},
                                   orient='index')

    def get_index_buffer_y(self, index) -> DataFrame:
        return DataFrame.from_dict(self.buffer_y[index], orient='index')

    def append_buffer_y(self, y_i: DataFrame) -> bool:
        self.buffer_y.append(y_i.to_dict(orient='index'))
        return True

    def update_buffer_y(self, index, y_i) -> bool:
        self.buffer_y[index] = y_i.to_dict(orient='index')
        return True

    def del_buffer_y(self, index) -> bool:
        self.buffer_y.pop(index)
        return True

    def get_buffer_w(self) -> list[int]:
        return self.buffer_w

    def get_index_buffer_w(self, index) -> int:
        return self.buffer_w[index]

    def append_buffer_w(self, w_i) -> bool:
        self.buffer_w.append(w_i)
        return True

    def update_buffer_w(self, index, w_i) -> bool:
        self.buffer_w[index] = w_i
        return True

    def del_buffer_w(self, index) -> bool:
        self.buffer_w.pop(index)
        return True

    def get_index_buffer_d(self, index) -> int:
        return self.buffer_d[index]

    def append_buffer_d(self, w_i) -> bool:
        self.buffer_d.append(w_i)
        return True

    def del_buffer_d(self, index) -> bool:
        self.buffer_d.pop(index)
        return True

    def update_status_environment(self, dto: DeviceStatusDTO, message_encoded: int):
        """
        Recebe um DeviceStatusDTO, processa-o e adiciona uma nova linha
        ao DataFrame 'self.df'.

        A estrutura do DataFrame é:
        [interval_sin, interval_cos, weekday_sin, weekday_cos, month_sin, month_cos, X_1, ..., X_n]
        onde X_n são colunas baseadas no devId.
        """

        ts_index = pd.to_datetime(dto.timeStamp)
        new_row_data = create_time_variables(ts_index)
        current_dev_id = dto.devId
        current_value = message_encoded

        if self.environment_status_incremental.empty:
            new_row_data[current_dev_id] = current_value
            self.environment_status_incremental = pd.DataFrame([new_row_data], index=[ts_index])
        else:
            last_row = self.environment_status_incremental.iloc[-1]
            device_cols = last_row.drop(create_time_variables(ts_index).keys())
            new_row_data.update(device_cols)

            if current_dev_id not in self.environment_status_incremental.columns:
                self.environment_status_incremental[current_dev_id] = np.nan
                self.environment_status_incremental.sort_index(axis=1, inplace=True)

            new_row_data[current_dev_id] = current_value
            new_row_df = pd.DataFrame([new_row_data], index=[ts_index])
            self.environment_status_incremental = pd.concat([self.environment_status_incremental, new_row_df])

        self.environment_status_incremental.ffill(inplace=True)
        self.environment_status_incremental.fillna(-1, inplace=True)

    def salva_dados_para_treino(self, x_df: DataFrame, y_df: DataFrame, w_for_x: int, d_for_x: int):
        linha = pd.concat([x_df.reset_index().add_prefix('X_'), y_df.reset_index().add_prefix('Y_'),
                           DataFrame([[w_for_x, d_for_x]], columns=['janela', 'densidade'])], axis=1)
        self.df_treino_incremental_model = pd.concat([self.df_treino_incremental_model, linha]).sort_index(axis=1,
                                                                                                           ascending=True)
        self.cont_treino += 1
        if self.save:
            self.cont_treino = 0
            if self.file_treino is None:
                df_file_name = "treino_incremental_model.csv"
                dir_to_save = os.path.abspath('/'.join(["modelo_treinado", self.key_experimento]))
                path_to_save_df = Path(os.path.abspath('/'.join([dir_to_save, df_file_name])))
                self.file_treino = str(path_to_save_df)
                os.makedirs(dir_to_save, exist_ok=True)
            self.df_treino_incremental_model.to_csv(self.file_treino, index=False)


    def salva_dados_recomendacao(self, x_df: DataFrame, rec: dict, inference_elapsed_time: float):
        rec.update({'inference_elapsed_time': inference_elapsed_time})
        linha = pd.concat([x_df.reset_index().add_prefix('X_'), pd.DataFrame([rec]).add_prefix('REC_')], axis=1)
        self.df_recommendation = pd.concat([self.df_recommendation, linha]).sort_index(axis=1, ascending=True)

        self.cont_recommendation += 1
        if self.save:
            self.cont_recommendation = 0
            if self.file_recommendation is None:
                df_file_name = "recomendacao_incremental_model.csv"
                dir_to_save = os.path.abspath('/'.join(["modelo_treinado", self.key_experimento]))
                path_to_save_df = Path(os.path.abspath('/'.join([dir_to_save, df_file_name])))
                self.file_recommendation = str(path_to_save_df)
                os.makedirs(dir_to_save, exist_ok=True)
            self.df_recommendation.to_csv(self.file_recommendation, index=False)
