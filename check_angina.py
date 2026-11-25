"""Execute com python rodada_angina/calcular.py na pasta do projeto."""
import csv
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from math import sqrt

import sys
totais = Counter()
padronizada = [0., 0.]
variancia = 0.
with ZipFile(Path(sys.argv[1])) as z:
    for nome in sorted(n for n in z.namelist() if n.startswith("processed.") and n.endswith(".data")):
        grupo = Counter()
        for linha in csv.reader(z.read(nome).decode().splitlines()):
            if not linha:
                continue
            angina = linha[8] if linha[8] == "?" else str(int(float(linha[8])))
            grupo[angina, int(float(linha[13]) > 0)] += 1
        totais.update(grupo)
        tamanhos = [grupo[str(a), 1] + grupo[str(a), 0] for a in range(2)]
        for a, n in enumerate(tamanhos):
            p = grupo[str(a), 1] / n
            padronizada[a] += sum(tamanhos) * p
            variancia += sum(tamanhos)**2 * p * (1-p) / n
        print(nome, grupo)
n_obs = sum(n for (a, y), n in totais.items() if a != "?")
p0, p1 = [s / n_obs for s in padronizada]
erro = sqrt(variancia) / n_obs
print("Sem, com, diferenca e IC aproximado:", p0, p1, p1-p0, p1-p0-1.96*erro, p1-p0+1.96*erro)
for extremo in ("menor", "maior"):
    g = totais.copy()
    for y in (0, 1):
        a = y if extremo == "maior" else 1-y
        g[str(a), y] += g["?", y]
    def proporcao(a):
        return g[a, 1] / (g[a, 0] + g[a, 1])
    print(extremo, proporcao("1") - proporcao("0"))

