import ast
import copy
import re

import pandas as pd
from river import base, stream

from experiment.dto import DeviceStatusDTO, MessageDTO, CodeDTO


class Cenarios:

    def cenario_00_hiperparametro(self) -> base.typing.Stream:
        df = pd.read_csv(
            './dados/Cenario_validacao_Artigo(1usuarios-casa-mayki)(20_dias)/db_selecao_hiperparametro.csv')
        df['timeStamp'] = pd.to_datetime(df['timeStamp'])
        df.sort_values(by=["timeStamp"], ascending=True, inplace=True)

        entrada_modificada = {}
        for linha, data in df.iterrows():
            message = DeviceStatusDTO(
                environment="00000000",
                devId=data["devId"],
                device=data["device"],
                space=data["space"],
                message=copy.deepcopy(
                    MessageDTO([CodeDTO(**item) for item in ast.literal_eval(data["message"])["status"]])),
                sensorType="sensor" if data["sensorType"] == "sensor" else "actuator",
                timeStamp=data["timeStamp"]
            )
            entrada_modificada[linha] = message
        return stream.iter_pandas(pd.DataFrame([entrada_modificada]).T)

    def cenario_00_experimento(self) -> base.typing.Stream:
        df = pd.read_csv('./dados/Cenario_validacao_Artigo(1usuarios-casa-mayki)(20_dias)/db_experimento.csv')
        df['timeStamp'] = pd.to_datetime(df['timeStamp'])
        df.sort_values(by=["timeStamp"], ascending=True, inplace=True)

        entrada_modificada = {}
        for linha, data in df.iterrows():
            message = DeviceStatusDTO(
                environment="00000000",
                devId=data["devId"],
                device=data["device"],
                space=data["space"],
                message=copy.deepcopy(
                    MessageDTO([CodeDTO(**item) for item in ast.literal_eval(data["message"])["status"]])),
                sensorType="sensor" if data["sensorType"] == "sensor" else "actuator",
                timeStamp=data["timeStamp"]
            )
            entrada_modificada[linha] = message
        return stream.iter_pandas(pd.DataFrame([entrada_modificada]).T)

    def cenario_01_hiperparametro(self) -> base.typing.Stream:
        df = pd.read_csv('./dados/Grafo_casa-ROTINA_SIMPLES(Dividindo_ap)(20_dias)/db_selecao_hiperparametro.csv')
        df['timeStamp'] = pd.to_datetime(df['timeStamp'])
        df.sort_values(by=["timeStamp"], ascending=True, inplace=True)

        entrada_modificada = {}
        for linha, data in df.iterrows():
            message = DeviceStatusDTO(
                environment="11111111",
                devId=data["devId"],
                device=data["device"],
                space=data["space"],
                message=copy.deepcopy(
                    MessageDTO([CodeDTO(**item) for item in ast.literal_eval(data["message"])["status"]])),
                sensorType="sensor" if data["sensorType"] == "sensor" else "actuator",
                timeStamp=data["timeStamp"]
            )
            entrada_modificada[linha] = message
        return stream.iter_pandas(pd.DataFrame([entrada_modificada]).T)

    def cenario_01_experimento(self) -> base.typing.Stream:
        df = pd.read_csv('./dados/Grafo_casa-ROTINA_SIMPLES(Dividindo_ap)(20_dias)/db_experimento.csv')
        df['timeStamp'] = pd.to_datetime(df['timeStamp'])
        df.sort_values(by=["timeStamp"], ascending=True, inplace=True)

        entrada_modificada = {}
        for linha, data in df.iterrows():
            message = DeviceStatusDTO(
                environment="11111111",
                devId=data["devId"],
                device=data["device"],
                space=data["space"],
                message=copy.deepcopy(
                    MessageDTO([CodeDTO(**item) for item in ast.literal_eval(data["message"])["status"]])),
                sensorType="sensor" if data["sensorType"] == "sensor" else "actuator",
                timeStamp=data["timeStamp"]
            )
            entrada_modificada[linha] = message
        return stream.iter_pandas(pd.DataFrame([entrada_modificada]).T)

    def cenario_02_hiperparametro(self) -> base.typing.Stream:
        df = pd.read_csv('./dados/Grafo_teste-grupo(3usuarios)(20_dias)/db_selecao_hiperparametro.csv')
        df['timeStamp'] = pd.to_datetime(df['timeStamp'])
        df.sort_values(by=["timeStamp"], ascending=True, inplace=True)

        entrada_modificada = {}
        for linha, data in df.iterrows():
            message = DeviceStatusDTO(
                environment="22222222",
                devId=data["devId"],
                device=data["device"],
                space=data["space"],
                message=copy.deepcopy(
                    MessageDTO([CodeDTO(**item) for item in ast.literal_eval(data["message"])["status"]])),
                sensorType="sensor" if data["sensorType"] == "sensor" else "actuator",
                timeStamp=data["timeStamp"]
            )
            entrada_modificada[linha] = message
        return stream.iter_pandas(pd.DataFrame([entrada_modificada]).T)

    def cenario_02_experimento(self) -> base.typing.Stream:
        df = pd.read_csv('./dados/Grafo_teste-grupo(3usuarios)(20_dias)/db_experimento.csv')
        df['timeStamp'] = pd.to_datetime(df['timeStamp'])
        df.sort_values(by=["timeStamp"], ascending=True, inplace=True)

        entrada_modificada = {}
        for linha, data in df.iterrows():
            message = DeviceStatusDTO(
                environment="22222222",
                devId=data["devId"],
                device=data["device"],
                space=data["space"],
                message=copy.deepcopy(
                    MessageDTO([CodeDTO(**item) for item in ast.literal_eval(data["message"])["status"]])),
                sensorType="sensor" if data["sensorType"] == "sensor" else "actuator",
                timeStamp=data["timeStamp"]
            )
            entrada_modificada[linha] = message
        return stream.iter_pandas(pd.DataFrame([entrada_modificada]).T)

    def cenario_03_hiperparametro(self) -> base.typing.Stream:
        df = pd.read_csv('./dados/2011-07/db_selecao_hiperparametro.csv')
        df['timeStamp'] = pd.to_datetime(df['data'] + ' ' + df['hora'])
        df.sort_values(by=["timeStamp"], ascending=True, inplace=True)

        entrada_modificada = {}
        for linha, data in df.iterrows():
            message = DeviceStatusDTO(
                environment="33333333",
                devId=f"disp{data["id_dispositivo"]}",
                device=f'actuator-{data["id_dispositivo"]}' if re.search(pattern=r"(^L\d)|(^E+\d)|(^F+\d)", string=data[
                    "id_dispositivo"]) else f'sensor-{data["id_dispositivo"]}',
                space="none",
                message=data["status"],
                sensorType='actuator' if re.search(pattern=r"(^L\d)|(^E+\d)|(^F+\d)",
                                                   string=data["id_dispositivo"]) else 'sensor',
                timeStamp=data["timeStamp"]
            )
            entrada_modificada[linha] = message
        return stream.iter_pandas(pd.DataFrame([entrada_modificada]).T)

    def cenario_03_experimento(self) -> base.typing.Stream:
        df = pd.read_csv('./dados/2011-07/db_experimento.csv')
        df = df[df['id_dispositivo'].str.contains(
            r"(?:^I+\d)|(?:^L\d)|(?:^E+\d)|(?:^F+\d)|(?:D+\d)|(?:M+\d)|(?:T+\d)|(?:MA+\d)|(?:^A)")]
        df['timeStamp'] = pd.to_datetime(df['data'] + ' ' + df['hora'])
        df.sort_values(by=["timeStamp"], ascending=True, inplace=True)

        entrada_modificada = {}
        for linha, data in df.iterrows():
            if re.search(pattern=r"(^P+\d)|(^BA\d)|(^LS\d)|(^SG\d)|(^E\d)", string=data["id_dispositivo"]):
                continue  # Pula as mensagens irrelevantes do dataset para diminuir o tamanho total

            message = DeviceStatusDTO(
                environment="33333333",
                devId=f"disp{data["id_dispositivo"]}",
                device=f'actuator-{data["id_dispositivo"]}' if re.search(pattern=r"(^I+\d)|(^L\d)|(^E+\d)|(^F+\d)",
                                                                         string=data[
                                                                             "id_dispositivo"]) else f'sensor-{data["id_dispositivo"]}',
                space="none",
                message=data["status"],
                sensorType='actuator' if re.search(pattern=r"(^I+\d)|(^L\d)|(^E+\d)|(^F+\d)",
                                                   string=data["id_dispositivo"]) else 'sensor',
                timeStamp=data["timeStamp"]
            )
            entrada_modificada[linha] = message
        print(f"Qauntidade de dados do cenário: {pd.DataFrame([entrada_modificada]).T.shape}")
        return stream.iter_pandas(pd.DataFrame([entrada_modificada]).T)

    def todos_cenarios_hiperparametro(self) -> list:
        return [self.cenario_00_hiperparametro(), self.cenario_01_hiperparametro(), self.cenario_02_hiperparametro(), self.cenario_03_hiperparametro()]

    def todos_cenarios_experimento(self) -> list:
        return [self.cenario_00_experimento(), self.cenario_01_experimento(), self.cenario_02_experimento(), self.cenario_03_experimento()]
