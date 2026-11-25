"""Reproduz os modelos e as tabelas de sensibilidade do lote prioritário."""

import argparse
import io
import warnings
from pathlib import Path
from statistics import NormalDist
from zipfile import ZipFile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


CENTROS = {
    "Cleveland": "processed.cleveland.data",
    "Hungria": "processed.hungarian.data",
    "Suíça": "processed.switzerland.data",
    "Long Beach VA": "processed.va.data",
}
COLUNAS = "age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal num".split()
PASSOS = {"age": 10, "sex": 1, "chol": 50, "trestbps": 10, "exang": 1}
BINARIAS = {"sex", "exang"}
Z = NormalDist().inv_cdf(0.975)


def carregar(arquivo):
    tabelas = []
    with ZipFile(arquivo) as zip_dados:
        for centro, nome in CENTROS.items():
            linhas = zip_dados.read(nome)
            tabela = pd.read_csv(io.BytesIO(linhas), header=None, na_values="?", dtype=float)
            tabela.columns = COLUNAS
            tabela["centro"] = centro
            tabelas.append(tabela)
    dados = pd.concat(tabelas, ignore_index=True)
    if not dados["num"].isin(range(5)).all():
        raise ValueError("O desfecho contém código inválido ou ausente.")
    for nome in BINARIAS:
        if not dados[nome].dropna().isin([0, 1]).all():
            raise ValueError(f"Código inesperado em {nome}.")
    for nome in ("chol", "trestbps"):
        dados[nome] = dados[nome].mask(dados[nome] <= 0)
    dados["positivo"] = dados["num"].gt(0).astype(int)
    return dados


def wilson(k, n):
    p = k / n
    fator = 1 + Z**2 / n
    meio = (p + Z**2 / (2 * n)) / fator
    raio = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / fator
    return max(0, meio - raio), min(1, meio + raio)


def sensibilidade_codigo(tabela, centro):
    contagens = tabela["num"].value_counts().reindex(range(5), fill_value=0)
    n = len(tabela)
    k = int(contagens[1])
    li, ls = wilson(k, n)
    return {
        "id": "EXT-01", "centro": centro, "n": n,
        **{f"num_{codigo}": int(qtd) for codigo, qtd in contagens.items()},
        "n_maior_zero": n - int(contagens[0]), "n_pelo_menos_2": int(contagens[2:].sum()),
        "p_maior_zero": 1 - contagens[0] / n, "p_pelo_menos_2": contagens[2:].sum() / n,
        "diferenca_pp": 100 * k / n, "ic95_inf_pp": 100 * li, "ic95_sup_pp": 100 * ls,
    }


def preparar(tabela, exposicao, combinado):
    ajuste = [nome for nome in ("age", "sex") if nome != exposicao]
    amostra = tabela.dropna(subset=[exposicao, *ajuste, "positivo"]).copy()
    matriz = pd.DataFrame(index=amostra.index)
    matriz["intercepto"] = 1.0
    for nome in [exposicao, *ajuste]:
        centro = 0 if nome in BINARIAS else amostra[nome].median()
        matriz[nome] = (amostra[nome] - centro) / PASSOS[nome]
    if combinado:
        indicadores = pd.get_dummies(amostra["centro"], prefix="centro", drop_first=True, dtype=float)
        matriz = pd.concat([matriz, indicadores], axis=1)
    return amostra, matriz, ajuste


def registrar_suporte(tabela, amostra, nome, grupo):
    saida = []
    for desfecho in (0, 1):
        todos = tabela.loc[tabela["positivo"].eq(desfecho)]
        valores = amostra.loc[amostra["positivo"].eq(desfecho), nome]
        linha = {
            "id": f"{nome}->num|", "centro": grupo, "num_maior_que_zero": desfecho,
            "n_origem": len(todos), "n_medicao_valida": int(todos[nome].notna().sum()),
            "n_modelo": len(valores), "valores_distintos": valores.nunique(),
        }
        if nome in BINARIAS:
            linha.update(n_codigo_0=int(valores.eq(0).sum()), n_codigo_1=int(valores.eq(1).sum()))
        elif not valores.empty:
            linha.update(minimo=valores.min(), q25=valores.quantile(.25), mediana=valores.median(),
                         q75=valores.quantile(.75), maximo=valores.max())
        saida.append(linha)
    return saida


def estimar_modelo(matriz, y):
    if y.nunique() < 2:
        raise ValueError("Sem positivos e negativos disponíveis.")
    for nome, valores in matriz.items():
        if nome == "intercepto":
            continue
        if valores.nunique() < 2:
            raise ValueError(f"Sem contraste em {nome}.")
        if valores.nunique() == 2 and (pd.crosstab(valores, y) == 0).any().any():
            raise ValueError(f"Célula vazia em {nome} por desfecho, com separação.")
    if np.linalg.matrix_rank(matriz) < matriz.shape[1]:
        raise ValueError("Preditores linearmente dependentes.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerfectSeparationWarning)
        warnings.simplefilter("error", RuntimeWarning)
        modelo = sm.GLM(y, matriz, family=sm.families.Binomial()).fit(maxiter=150)
    if not modelo.converged or not np.isfinite(modelo.bse).all() or modelo.bse.max() > 10:
        raise ValueError("Ajuste instável, erro-padrão excessivo ou falta de convergência.")
    return modelo


def calcular_or(beta, erro):
    return tuple(np.exp([beta, beta - Z * erro, beta + Z * erro]))


def checar_curvatura(nome, amostra, matriz, modelo):
    quad = matriz.assign(quadratico=matriz[nome] ** 2)
    modelo_quad = estimar_modelo(quad, amostra["positivo"])
    saida = []
    for percentil in (.25, .75):
        inicio = amostra[nome].quantile(percentil)
        fim = inicio + PASSOS[nome]
        linha = {
            "id": f"{nome}->num|", "n": len(amostra), "percentil_inicial": percentil,
            "valor_inicial": inicio, "valor_final": fim, "aic_linear": modelo.aic,
            "aic_quadratico": modelo_quad.aic, "delta_aic_quad_menos_linear": modelo_quad.aic - modelo.aic,
        }
        if fim <= amostra[nome].max():
            x = (inicio - amostra[nome].median()) / PASSOS[nome]
            contraste = pd.Series(0., index=quad.columns)
            contraste[nome] = 1
            contraste["quadratico"] = 2 * x + 1
            beta = contraste @ modelo_quad.params
            erro = np.sqrt(contraste @ modelo_quad.cov_params() @ contraste)
            razao, li, ls = calcular_or(beta, erro)
            linha.update(estimativa_or=razao, ic95_inf=li, ic95_sup=ls, estado="estimada")
        else:
            linha["estado"] = "incremento fora da faixa observada"
        saida.append(linha)
    return saida


def analisar_grupo(tabela, nome, grupo):
    combinado = grupo == "Todos"
    amostra, matriz, ajuste = preparar(tabela, nome, combinado)
    y = amostra["positivo"]
    linha = {
        "id": f"{nome}->num|", "centro": grupo, "passo": PASSOS[nome],
        "ajuste": ", ".join(ajuste + (["centro"] if combinado else [])),
        "n_origem": len(tabela), "n": len(amostra), "positivos": int(y.sum()),
        "negativos": int(len(y) - y.sum()), "n_centros": amostra["centro"].nunique(),
        "centros_incluidos": ", ".join(amostra["centro"].unique()),
    }
    suporte = registrar_suporte(tabela, amostra, nome, grupo)
    if nome in BINARIAS:
        for codigo in (0, 1):
            desfechos = y[amostra[nome].eq(codigo)]
            linha[f"n_codigo_{codigo}"] = len(desfechos)
            linha[f"positivos_codigo_{codigo}"] = int(desfechos.sum())
            if len(desfechos):
                linha[f"pct_positivos_codigo_{codigo}"] = 100 * desfechos.mean()
    elif all("minimo" in grupo_desfecho for grupo_desfecho in suporte):
        inferior = max(grupo_desfecho["minimo"] for grupo_desfecho in suporte)
        superior = min(grupo_desfecho["maximo"] for grupo_desfecho in suporte)
        if inferior <= superior:
            linha["sobreposicao_inferior"] = inferior
            linha["sobreposicao_superior"] = superior
    curvatura = []
    try:
        modelo = estimar_modelo(matriz, y)
    except (ValueError, np.linalg.LinAlgError, Warning) as erro:
        linha.update(estado="bloqueada", observacao=str(erro))
        return linha, suporte, curvatura
    escasso = []
    for coluna in matriz:
        if matriz[coluna].nunique() == 2 and pd.crosstab(matriz[coluna], y).min().min() < 5:
            escasso.append(f"célula abaixo de 5 para {coluna}")
    if y.value_counts().min() < 10 * (matriz.shape[1] - 1):
        escasso.append("menos de 10 no menor grupo de desfecho por inclinação")
    razao, li, ls = calcular_or(modelo.params[nome], modelo.bse[nome])
    linha.update(estimativa_or=razao, ic95_inf=li, ic95_sup=ls, estado="estimada",
                 observacao=", ".join(escasso))
    if combinado and nome not in BINARIAS:
        try:
            curvatura = checar_curvatura(nome, amostra, matriz, modelo)
        except (ValueError, np.linalg.LinAlgError, Warning) as erro:
            curvatura = [{"id": linha["id"], "n": len(amostra), "estado": str(erro)}]
    return linha, suporte, curvatura


def executar(arquivo, destino):
    dados = carregar(arquivo)
    modelos, suporte, curvaturas, codigos = [], [], [], []
    for centro, tabela in [("Todos", dados), *dados.groupby("centro", sort=False)]:
        codigos.append(sensibilidade_codigo(tabela, centro))
        for exposicao in PASSOS:
            resultado, distribuicao, checagem = analisar_grupo(tabela, exposicao, centro)
            modelos.append(resultado)
            suporte.extend(distribuicao)
            curvaturas.extend(checagem)
    destino.mkdir(parents=True, exist_ok=True)
    for nome, registros in [
        ("estimativas", modelos), ("suporte_modelos", suporte),
        ("curvatura", curvaturas), ("sensibilidade_num", codigos),
    ]:
        pd.DataFrame(registros).to_csv(destino / f"{nome}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arquivo", type=Path, help="ZIP original do UCI Heart Disease")
    parser.add_argument("destino", type=Path, help="Pasta dos resultados")
    argumentos = parser.parse_args()
    executar(argumentos.arquivo, argumentos.destino)
