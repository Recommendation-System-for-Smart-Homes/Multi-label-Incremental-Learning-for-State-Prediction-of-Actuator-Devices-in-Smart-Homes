from copy import deepcopy

from pandas import DataFrame
from river import metrics


class CalculoMetricas:

    @staticmethod
    def metricas_por_dia(nome_modelo, index_cenario, df_REC, df_X_Y):
        df_X = df_X_Y.filter(regex='^X_', axis=1)
        df_Y = df_X_Y.filter(regex='^Y_', axis=1)
        metricas = []

        metric_0 = metrics.multioutput.PerOutput(metrics.MacroF1())
        metric_M_0 = metrics.multioutput.MicroAverage(metrics.MacroF1())
        metric_cm = metrics.multioutput.MultiLabelConfusionMatrix()
        dia = None
        dia_experimento = 0
        tempo_inferencia = 0
        cont = 0
        for i in df_X.index.tolist():
            y = df_Y.loc[[i]].filter(regex='^Y_disp', axis=1).rename(columns=lambda coluna: coluna.removeprefix('Y_'))
            y = {} if y.dropna(axis=1).empty else y.dropna(axis=1).to_dict(orient='records')[0]

            rec = df_REC.loc[[i]].filter(regex='^REC_disp', axis=1).rename(
                columns=lambda coluna: coluna.removeprefix('REC_'))
            rec = {} if rec.dropna(axis=1).empty else rec.dropna(axis=1).to_dict(orient='records')[0]

            atuadores = list(y.keys() & rec.keys())

            new_rec = {a: rec.get(a) for a in atuadores}
            new_y = {a: y.get(a) for a in atuadores}

            dia_atual = i.day
            metric_0.update(new_y, new_rec)
            metric_M_0.update(new_y, new_rec)
            metric_cm.update(new_y, new_rec)

            if len(new_rec) > 0:
                tempo_inferencia += df_REC.loc[[i]]['REC_inference_elapsed_time'].values[0]
                cont += 1
            if dia_atual != dia:
                dia = dia_atual
                dia_experimento += 1
                metricas.append({
                    'nome_modelo': nome_modelo,
                    'index_cenario': index_cenario,
                    'dia_dados': dia,
                    'dia_experimento': dia_experimento,
                    'tempoMedioInferencia': 0 if cont == 0 else (tempo_inferencia / cont),
                    'deviceMacroF1PerDevice': deepcopy(metric_0.get()),
                    'mediaMicro': deepcopy(metric_M_0.get()),
                    'matrizConfisao': deepcopy(metric_cm),
                })
        return metricas

    @staticmethod
    def metricas_por_hiperparametros(df_REC: DataFrame, df_X_Y: DataFrame):
        df_X = df_X_Y.filter(regex='^X_', axis=1)
        df_Y = df_X_Y.filter(regex='^Y_', axis=1)

        metric_M_0 = metrics.multioutput.MicroAverage(metrics.MacroF1())
        for i in df_X.index.tolist():
            y = df_Y.loc[[i]].filter(regex='^Y_disp', axis=1).rename(columns=lambda coluna: coluna.removeprefix('Y_'))
            y = {} if y.dropna(axis=1).empty else y.dropna(axis=1).to_dict(orient='records')[0]

            rec = df_REC.loc[[i]].filter(regex='^REC_disp', axis=1).rename(
                columns=lambda coluna: coluna.removeprefix('REC_'))
            rec = {} if rec.dropna(axis=1).empty else rec.dropna(axis=1).to_dict(orient='records')[0]
            atuadores = list(y.keys() & rec.keys())

            new_rec = {a: rec.get(a) for a in atuadores}
            new_y = {a: y.get(a) for a in atuadores}

            metric_M_0.update(new_y, new_rec)

        mediaMicro = metric_M_0.get()

        return mediaMicro
