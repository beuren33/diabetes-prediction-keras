# Rede Neural para Previsão de Diabetes

Este projeto usa uma rede neural para prever se uma pessoa tem diabetes a partir de dados clínicos, um problema de classificação binária. A base é a conhecida Pima Indians Diabetes, onde cada linha representa uma paciente com medidas como número de gestações, nível de glicose, pressão sanguínea, índice de massa corporal e idade, entre outras, e a resposta é apenas sim ou não para a presença da doença. A ideia prática é receber esses exames de uma pessoa e o modelo devolver uma probabilidade de ela ser diabética, funcionando como um apoio à triagem e nunca como um diagnóstico definitivo.

## Como funciona

Antes de treinar, os dados passam por uma preparação importante. As features são padronizadas com o StandardScaler, que coloca todas as medidas numa mesma escala. Esse passo faz diferença numa rede neural, pois medidas em unidades muito diferentes, como glicose na casa das centenas e número de gestações em unidades, podem desequilibrar o aprendizado se entrarem na rede sem tratamento. Com tudo na mesma escala, a rede aprende de forma mais estável.

O modelo é uma rede neural densa montada com Keras, e o ponto mais interessante do projeto está na escolha dos hiperparâmetros. Em vez de chutar valores, foi usado o GridSearch para testar de forma sistemática várias combinações de configuração da rede, variando coisas como a função de ativação, o inicializador dos pesos, o otimizador, o número de neurônios, o tamanho do lote e a quantidade de épocas. O GridSearch treina e avalia cada combinação com validação cruzada e aponta qual delas rende o melhor resultado médio, o que dá uma escolha bem mais justificada do que o ajuste manual.

## Dados

A base tem 768 registros de pacientes, cada um com oito features clínicas e o resultado indicando presença ou ausência de diabetes. É um conjunto pequeno, o que reforça a importância da padronização e da validação cruzada para tirar conclusões confiáveis.

## Resultados

A melhor combinação encontrada pelo GridSearch usou ativação relu, inicialização normal, otimizador adam, quatro neurônios, lote de dez e cinquenta épocas, alcançando por volta de 76,7% de acurácia na validação cruzada. Ao testar o modelo já treinado com um caso novo, ele retornou a probabilidade da pessoa ser diabética, mostrando o uso prático do modelo ponta a ponta, do dado clínico até a previsão. O modelo treinado foi salvo no formato Keras e acompanha o projeto, pronto para carregar e usar sem precisar treinar de novo.

| Métrica | Valor |
| --- | --- |
| Tipo | Classificação binária |
| Registros | 768 |
| Features | 8 medidas clínicas |
| Melhores hiperparâmetros | relu, normal, adam, 4 neurônios, batch 10, 50 épocas |
| Acurácia (validação cruzada) | por volta de 76,7% |

## Como rodar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Depois abra o notebook e execute as células em ordem:

```bash
jupyter notebook notebook/classificacao_diabetes.ipynb
```

Os dados estão na pasta `data` e o modelo já treinado fica em `modelo`, então dá para tanto refazer todo o processo quanto apenas carregar o modelo pronto e fazer previsões.

## Estrutura do projeto

```
rede-neural-diabetes/
├── notebook/
│   └── classificacao_diabetes.ipynb   # preparo, GridSearch, treino e teste
├── data/
│   └── diabetes.csv
├── modelo/
│   └── modelo_diabetes.keras          # rede ja treinada
├── requirements.txt
└── .gitignore
```

## Observações e próximos passos

Num problema de saúde, a acurácia sozinha não conta a história toda, pois errar um caso positivo é bem mais grave do que um alarme falso. Um próximo passo importante é olhar métricas como recall e precisão, além da matriz de confusão, para entender que tipo de erro o modelo comete. Também vale investigar a base, já que algumas colunas trazem zeros que na prática representam dados ausentes, e tratar esses valores tende a melhorar bastante a qualidade das previsões.
