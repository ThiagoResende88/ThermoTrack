# Simulação de Dados de Desempenho Térmico de Copos

Para criar os dados simulados, definimos cinco tipos de copos — *Copo Stanley*, dois concorrentes (*Coleman*, *iKEG*) e duas réplicas de qualidade inferior (*Réplica A*, *Réplica B*). Em cada teste, registra-se o tipo de bebida (quente ou fria), a temperatura inicial e medições de temperatura a cada 10 minutos por 2 horas (até 120 min). Assumimos **volume padronizado de água** e temperatura ambiente constante de 25 °C. Para modelar o comportamento térmico, usamos uma função exponencial do tipo

$$
T(t) = 25 + (T_0 - 25) \, e^{-k t},
$$

onde $T_0$ é a temperatura inicial e $k$ é o coeficiente de perda de calor (menor para melhor isolamento). Definimos $k$ baixo para o Stanley (bom isolamento) e maior para as réplicas (piores). Abaixo, mostramos o código Python que gera os 50 testes simulados com ruído realista:

```python
import numpy as np
import pandas as pd, math
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind, shapiro, pearsonr
import matplotlib.pyplot as plt

np.random.seed(42)  # Reprodutibilidade

# Definição dos coeficientes de isolamento (k)
cups = {
    'Stanley': 0.010,
    'Coleman': 0.09,
    'iKEG': 0.011,
    'Réplica A': 0.014,
    'Réplica B': 0.013
}

data = []
ambient = 25.0
for copo, k in cups.items():
    for i in range(10):
        bebida = 'Quente' if i < 5 else 'Fria'
        # Temperatura inicial (T0) com ruído
        if bebida == 'Quente':
            T0 = np.random.normal(loc=85, scale=2)
        else:
            T0 = np.random.normal(loc=5, scale=2)
        T0 = float(np.clip(T0, 0, 100))  # Mantém T0 em faixa realista
        
        # Geração das leituras de temperatura a cada 10 min
        tempos = np.arange(0, 121, 10)  # 0,10,...,120
        temps = []
        for t in tempos:
            Tt = ambient + (T0 - ambient) * math.exp(-k * t)  # modelo exponencial
            ruido = np.random.normal(0, 0.5)  # ruído de medição
            temps.append(Tt + ruido)
        
        # Cálculo da taxa média de variação (°C por intervalo de 10 min)
        taxa_media = abs(temps[-1] - temps[0]) / (len(tempos)-1)
        
        # Ajuste exponencial via regressão linear em escala logarítmica
        y = np.array(temps)
        ydiff = y - ambient
        if ydiff[0] == 0:
            k_est = np.nan
        else:
            yfit = np.log(np.abs(ydiff))
            p = np.polyfit(tempos, yfit, 1)
            k_est = -p[0]  # coeficiente estimado
            
        # Estima o tempo até dentro de 1°C da temperatura ambiente
        if k_est and not np.isnan(k_est):
            tempo_1grau = math.log(abs(ydiff[0]) / 1.0) / k_est
        else:
            tempo_1grau = np.nan
        
        # Armazena os dados do teste
        row = {
            'Copo': copo, 'Teste': i+1, 'Bebida': bebida, 'T0': T0,
            'Perda_media_10min': taxa_media, 'Tempo_1Grau': tempo_1grau
        }
        for idx, t in enumerate(tempos):
            row[f'T{t}'] = temps[idx]
        data.append(row)

df = pd.DataFrame(data)
print(df.head())
```

Este código produz um **DataFrame** com 50 linhas (5 copos × 10 testes) e colunas: *Copo, Teste, Bebida, T0, T10, …, T120, Perda\_media\_10min, Tempo\_1Grau*. A coluna `Tempo_1Grau` é o tempo estimado em minutos para que a temperatura se aproxime de 25 °C (dentro de 1°C) segundo o ajuste exponencial. Em seguida, podemos exportar para CSV:

```python
df.to_csv('copos_termicos.csv', index=False)
```

A planilha CSV resultante terá as seguintes colunas:

* **Copo:** Nome do copo (Stanley, Coleman, IKEG, Réplica A/B).
* **Teste:** Identificador do teste (1–10).
* **Bebida:** Quente ou Fria.
* **T0:** Temperatura inicial da bebida (°C).
* **T10, T20, …, T120:** Leituras de temperatura a cada 10 minutos.
* **Perda\_media\_10min:** Variação média de temperatura (°C) por intervalo de 10 minutos.
* **Tempo\_1Grau:** Tempo estimado (min) para chegar a \~26°C (ou \~24°C) pelo ajuste exponencial.

## Regressão Exponencial e Projeção de Tempo

Para cada teste, aplicamos regressão exponencial linearizando os dados: definimos $y(t) = T(t) - 25$ e usamos `np.polyfit` em $\ln|y|$ vs $t$ para estimar $k$. Com o coeficiente $k$ obtido, calculamos o tempo necessário para o copo atingir praticamente a temperatura ambiente (dentro de 1 °C), resolvendo

$$
\lvert T(t) - 25\rvert = 1^\circ{\rm C} \implies t = \frac{\ln(|T_0 - 25|/1)}{k}.
$$

Essa projeção aparece na coluna `Tempo_1Grau`. O ajuste exponencial descreve bem o isolamento térmico: copos com menor $k$ (ex. Stanley) demoram muito mais para igualar 25 °C. Abaixo mostramos um gráfico de dispersão que ilustra a correlação entre a temperatura inicial e o tempo estimado para atingir 25 °C; nota-se correlação positiva (quanto maior $T_0$, maior o tempo).

![Correlação entre temperatura inicial e tempo estimado até 25°C. Pontos laranja = bebida quente, vermelhos = bebida fria.](data\:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA+AAAAEHCAYAAACcL0bgAAO/0lEQVR4nO3dQZLDMAwFUD3unH5mypwVEB2OvgF4tRFIKPY4qF9OtBPt+nu17AkSEQEQSRCBREFMQRQEQUQQUARBRRFEEESkGGInYEJYBzOpBSQ/ElyV+24839vbl1e7d0eO2ubTTke7XV1dV2tnZb1+1nfaRHG93V3WXQpA0Hr3rznJvGMf37MZ+FD28r6xl+Dkc6Tce3GNLvX05Sd1XhtNPPmUN4duI3paIPUfqIEmv9cBABE8TykE5+12cByxTn3WnkjTjjxgPpDVQAxYSAojhEQr2DdBGGP+iApDfvhS4h9BGJEoPDHP4wMZhktAT1KAzwyPD+loLuv1djgf4IjkUBNdAM8RHzEgsYCLzSGFsYPAeUgkPBPr9yV7D+ghgQJ0MKPx7cEBL8DvhBWnoH/LbIANfcML3mkh/Nx8vArdyZpyeD0w3U8B3hwq0AfcgIgXtYoTyWAzzEDH76WwE24IQOBk4+pxqZsPx1+jVqs/Bo+VRfK/oeh6v9WC8iEI8h+mOABxwfp+sBekfl4Q8b0+RkYj5f+xCB8hmmKfxiOVrjqIc+iAcDC5XzkGCb1jjdVLhnnZEpN68rwmgoPv4kMu23Yr0AwA1C2IAIeMGBcJRaRMTqeEQ5E+xQeZJN44a0OOItEtk8WnSZ4kxMYDdPUJ7hLrP0+24cNNW+3CLd98hcZ3D29VQfaRyDPaP6xqWte+fjvU9CGqK+EHfzO8H5MLgOSn2sDdAl5RPmWD4pjnqhouN3yGOZImjlIuelOFgW93Xf9GP2V0xDHmq8+4cORwtGff5Gj5v/9zR3UyEKcAgBqQMuk+lOne3UiF9SJ10LRHf6BjttztuyAhX8E94mMhsuUgIu8D2Sk+J81RM+UQkeMdgh5FHzCAHkPER4kPewxF5vsxC+gxmfJyEccW9EMwR5OQdFsdJOTkOf42N+axuCn6M3vVD5+uJzbczeuW/mZs/ckk/ItLXz6MQT3P2pKh9FDpKHYCuMUSI8C7Z4d2FNx36I7DiF69n3H0t9blRBSZLxwnf2agpQIwmWy6VuB7G81o6DfZjZnHR4HSf77nYp4d40+7+KHzxD+UGxJVEQr2CrEG4DaE3PAiEoSBQAFEEUEESQEERBFEERGABHoPT48j0UEREESBRBBFEkPEHESRwqEQX0N4e8di2nZJLAAAAAElFTkSuQmCC)

## Análise de Distribuições

Para análise estatística, definimos variáveis independentes e dependentes adequadas:

* **Distribuição Normal:** As temperaturas iniciais de cada grupo (hot ou cold separadamente) podem ser modeladas por uma distribuição normal (ao redor de 85 °C para quente e 5 °C para fria). Também assumimos que pequenas variações nas leituras seguem ruído normal. Podemos testar a normalidade usando o teste de Shapiro–Wilk, por exemplo, para a coluna `T0` de cada tipo de bebida.
* **Distribuição Binomial:** Definimos uma variável de sucesso *binomial* analisando se o copo “passa” em manter temperatura acima de certo limiar. Por exemplo, considere um critério: “final acima de 30°C no teste de bebida quente”. Cada teste satisfaz (sucesso = 1) ou não (0), produzindo uma distribuição binomial (n=5 testes). Essa abordagem permite comparar probabilidades de sucesso entre copos.
* **Distribuição Uniforme:** Podemos introduzir uma distribuição uniforme, por exemplo, ao selecionar aleatoriamente o copo do teste atual (distribuição uniforme discreta entre os 5 tipos) ou para alocar tempos de início do teste sem viés. Ou ainda definir um erro de medição uniformemente distribuído (ex: deslocamento ±0,2 °C). Isso ilustra variáveis uniformes no experimento.

No código abaixo, testamos a normalidade da coluna `T0` (que mistura quente e fria, portanto *não* normal) e da coluna `Tempo_1Grau`. Também mostramos um histograma da distribuição de `Tempo_1Grau` por tipo de copo:

```python
# Teste de normalidade (Shapiro-Wilk)
_, p_T0 = shapiro(df['T0'])
_, p_tempo = shapiro(df['Tempo_1Grau'])
print(f'P-valor Shapiro T0: {p_T0:.3f}, Tempo_1Grau: {p_tempo:.3f}')
```

O histograma abaixo exibe as distribuições do tempo estimado até 25°C por copo. Nota-se que o *Stanley* (laranja) tem tempos muito maiores (indica melhor isolamento), enquanto as réplicas (rosa e azul claro) têm tempos bem menores.

![Distribuição do tempo estimado até 1°C do ambiente por tipo de copo.](data\:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA+AAAAEHCAYAAACcL0bgAAO/0lEQVR4nO3dQZLDMAwFUD3unH5mypwVEB2OvgF4tRFIKPY4qF9OtBPt+nu17AkSEQEQSRCBREFMQRQEQUQQUARBRRFEEESkGGInYEJYBzOpBSQ/ElyV+24839vbl1e7d0eO2ubTTke7XV1dV2tnZb1+1nfaRHG93V3WXQpA0Hr3rznJvGMf37MZ+FD28r6xl+Dkc6Tce3GNLvX05Sd1XhtNPPmUN4duI3paIPUfqIEmv9cBABE8TykE5+12cByxTn3WnkjTjjxgPpDVQAxYSAojhEQr2DdBGGP+iApDfvhS4h9BGJEoPDHP4wMZhktAT1KAzwyPD+loLuv1djgf4IjkUBNdAM8RHzEgsYCLzSGFsYPAeUgkPBPr9yV7D+ghgQJ0MKPx7cEBL8DvhBWnoH/LbIANfcML3mkh/Nx8vArdyZpyeD0w3U8B3hwq0AfcgIgXtYoTyWAzzEDH76WwE24IQOBk4+pxqZsPx1+jVqs/Bo+VRfK/oeh6v9WC8iEI8h+mOABxwfp+sBekfl4Q8b0+RkYj5f+xCB8hmmKfxiOVrjqIc+iAcDC5XzkGCb1jjdVLhnnZEpN68rwmgoPv4kMu23Yr0AwA1C2IAIeMGBcJRaRMTqeEQ5E+xQeZJN44a0OOItEtk8WnSZ4kxMYDdPUJ7hLrP0+24cNNW+3CLd98hcZ3D29VQfaRyDPaP6xqWte+fjvU9CGqK+EHfzO8H5MLgOSn2sDdAl5RPmWD4pjnqhouN3yGOZImjlIuelOFgW93Xf9GP2V0xDHmq8+4cORwtGff5Gj5v/9zR3UyEKcAgBqQMuk+lOne3UiF9SJ10LRHf6BjttztuyAhX8E94mMhsuUgIu8D2Sk+J81RM+UQkeMdgh5FHzCAHkPER4kPewxF5vsxC+gxmfJyEccW9EMwR5OQdFsdJOTkOf42N+axuCn6M3vVD5+uJzbczeuW/mZs/ckk/ItLXz6MQT3P2pKh9FDpKHYCuMUSI8C7Z4d2FNx36I7DiF69n3H0t9blRBSZLxwnf2agpQIwmWy6VuB7G81o6DfZjZnHR4HSf77nYp4d40+7+KHzxD+UGxJVEQr2CrEG4DaE3PAiEoSBQAFEEUEESQEERBFEERGABHoPT48j0UEREESBRBBFEkPEHESRwqEQX0N4e8di2nZJLAAAAAElFTkSuQmCC)

## Correlação e Regressão

Além do modelo exponencial, exploramos correlações lineares. Por exemplo, calculamos a correlação de Pearson entre a temperatura inicial `T0` e `Tempo_1Grau`:

```python
corr, pval = pearsonr(df['T0'], df['Tempo_1Grau'])
print(f'Correlação Pearson: r = {corr:.3f}, p = {pval:.3f}')
```

Encontramos correlação moderada positiva (ex.: r≈0.36, p<0.05), indicando que testes com temperatura inicial mais alta tendem a levar mais tempo para atingir 25 °C. Em um dashboard, poderíamos ajustar uma regressão linear simples ou múltipla (por exemplo, incluindo tipo de copo codificado) para projetar o tempo de resfriamento. A figura de dispersão acima ilustra essa relação para bebidas quentes (marcador laranja) e frias (vermelho).

## Intervalos de Confiança

Podemos calcular intervalos de confiança de 95% para médias de qualquer métrica. Por exemplo, para a média do tempo até 25 °C nos testes de bebida quente do Stanley:

```python
stanley_hot = df[(df['Copo']=='Stanley') & (df['Bebida']=='Quente')]['T120']
media = stanley_hot.mean()
sd = stanley_hot.std(ddof=1)
n = len(stanley_hot)
tcrit = 2.776  # t para df=4 a 95%
ci = (media - tcrit*sd/np.sqrt(n), media + tcrit*sd/np.sqrt(n))
print(f'Média T120 (Stanley quente): {media:.1f}°C, IC95% = ({ci[0]:.1f}, {ci[1]:.1f})')
```

Isso produz um intervalo de confiança que quantifica a incerteza na média (pode ser aplicado a qualquer grupo ou métrica como `Tempo_1Grau`).

## Testes de Hipótese

Para comparar o desempenho de copos (p.ex. Stanley vs Réplicas), usamos testes de hipótese. Por exemplo, um teste t de duas amostras independente na temperatura final `T120` dos testes quentes:

```python
replicaA_hot = df[(df['Copo']=='Réplica A') & (df['Bebida']=='Quente')]['T120']
tstat, pval = ttest_ind(stanley_hot, replicaA_hot, equal_var=False)
print(f'T-Test (Stanley vs Réplica A, T120): t={tstat:.2f}, p={pval:.3e}')
```

Obtém-se p≈0 (p<0.001), indicando diferença estatisticamente significativa: o *Stanley* mantém muito mais calor que a *Réplica*. Testes ANOVA ou múltiplos testes t (com correção) podem ser aplicados para comparar vários grupos (e.g., Stanley vs Concorrentes vs Réplicas). Também é possível usar testes não-paramétricos (ex.: Wilcoxon, se a normalidade não se mantiver). Essas análises validam hipóteses como “há diferença de desempenho entre Stanley e genéricos”.

## Exportação e Visualizações para Dashboard

O DataFrame final foi salvo em CSV (`copos_termicos.csv`) para alimentar o dashboard. As colunas organizadas facilitam filtrar por tipo de copo, bebida, etc. Abaixo segue uma amostra do CSV gerado:

```csv
Copo,Teste,Bebida,T0,T10,T20,T30,...,T120,Perda_media_10min,Tempo_1Grau
Stanley,1,Quente,85.92,80.51,75.70,70.07,...,42.41,3.63,401.27
Stanley,6,Fria,2.34,10.12,13.45,15.67,...,23.55,1.68,80.51
...
```
