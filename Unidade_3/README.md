# PLN 2026/2 — Trabalho da Unidade 3

## 1) Atividades realizadas

| Tarefa                             | BoW | TF-IDF | Word2Vec | BERT |
| ---------------------------------- | :-: | :----: | :------: | :--: |
| Clusterização + PCA e visualização |  ✔  |   ✔    |    ✔     |  ✔   |
| Busca e recomendação               |  ✔  |   ✔    |    ✔     |  ✔   |
| Classificação                      |  ✔  |   ✔    |    ✔     |  ✔   |

| Tarefa                               | Situação                                |
| ------------------------------------ | --------------------------------------- |
| Análise de sentimento                | (x) Foi realizada ( ) Não foi realizada |
| Reconhecimento de Entidades Nomeadas | (x) Foi realizado ( ) Não foi realizado |
| Extração de Relações                 | (x) Foi realizada ( ) Não foi realizada |

Sentimento, NER e relações usam um modelo pré-treinado (pysentimiento) e modelos/regras do spaCy, e não dependem das quatro representações vetoriais. A projeção 2D usou **TruncatedSVD** (variante do PCA para matrizes esparsas, sem centralização), não o `PCA` do scikit-learn.

## 2) Clusterização

K-Means (`n_init=10`, `random_state=42`) com k = 2, 3, 4, 5, 8 e 10. O melhor silhouette foi **k=2 nas quatro representações** (figuras abaixo). Tabelas completas: [`metricas_clusterizacao.csv`](resultados_trabalho_unidade3/metricas_clusterizacao.csv).

<table>
  <tr>
    <td align="center"><img src="imagens/fig02_clusters_bow_k2.png" width="380"><br><sub>Fig. 2 — BoW, k=2</sub></td>
    <td align="center"><img src="imagens/fig03_clusters_tfidf_k2.png" width="380"><br><sub>Fig. 3 — TF-IDF, k=2</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="imagens/fig04_clusters_word2vec_k2.png" width="380"><br><sub>Fig. 4 — Word2Vec, k=2</sub></td>
    <td align="center"><img src="imagens/fig05_clusters_bert_k2.png" width="380"><br><sub>Fig. 5 — BERT, k=2</sub></td>
  </tr>
</table>

| Representação | Silhouette k=2 | Davies-Bouldin k=2 | Calinski-Harabasz k=2 | Silhouette k=3 / 4 / 5 / 8 / 10       |
| ------------- | -------------: | -----------------: | --------------------: | ------------------------------------- |
| BoW           |           0,18 |               2,07 |                    69 | -0,03 / -0,10 / -0,09 / -0,12 / -0,19 |
| TF-IDF        |           0,05 |               2,96 |                    44 | 0,03 / 0,01 / 0,03 / 0,00 / 0,00      |
| Word2Vec      |           0,81 |               0,59 |                   933 | 0,51 / 0,40 / 0,36 / 0,11 / 0,10      |
| BERT          |           0,69 |               0,79 |                   519 | 0,45 / 0,39 / 0,23 / 0,15 / 0,12      |

<details>
<summary>Davies-Bouldin e Calinski-Harabasz para todos os k</summary>

**Davies-Bouldin — menor é melhor**

| Representação |  k=2 |  k=3 |  k=4 |  k=5 |  k=8 | k=10 |
| ------------- | ---: | ---: | ---: | ---: | ---: | ---: |
| BoW           | 2,07 | 1,86 | 2,00 | 1,99 | 2,33 | 1,95 |
| TF-IDF        | 2,96 | 4,33 | 4,83 | 6,02 | 5,15 | 3,98 |
| Word2Vec      | 0,59 | 2,26 | 2,49 | 2,26 | 2,67 | 2,69 |
| BERT          | 0,79 | 1,86 | 2,16 | 2,53 | 2,44 | 2,63 |

**Calinski-Harabasz — maior é melhor**

| Representação |   k=2 |   k=3 |   k=4 |   k=5 |   k=8 |  k=10 |
| ------------- | ----: | ----: | ----: | ----: | ----: | ----: |
| BoW           |  69,0 |  53,3 |  42,9 |  41,6 |  28,1 |  25,6 |
| TF-IDF        |  44,4 |  25,3 |  19,3 |  15,1 |   9,8 |   8,4 |
| Word2Vec      | 932,8 | 503,3 | 353,2 | 272,0 | 165,7 | 135,2 |
| BERT          | 518,8 | 294,3 | 222,2 | 174,3 | 117,8 |  96,0 |

</details>

**Métricas.** O silhouette combina coesão intracluster e separação intercluster (maior é melhor). O Davies-Bouldin é a razão entre dispersão interna e distância entre centróides (menor é melhor). O Calinski-Harabasz é a razão entre dispersão inter e intracluster (maior é melhor). Word2Vec e BERT têm valores muito melhores que BoW e TF-IDF. No BoW, o silhouette é negativo a partir de k=3 (clusters sobrepostos), e no TF-IDF fica perto de zero. Em geral, as métricas pioram com k maior. Davies-Bouldin e Calinski-Harabasz não são comparáveis entre representações: BoW e TF-IDF foram avaliados após redução a 100 componentes por SVD, Word2Vec em 100 dimensões e BERT em 768. No Word2Vec, só o resultado de k=2 é estável entre execuções.

**Leitura qualitativa: a divisão em k=2 é por idioma.** Os termos mais frequentes dos clusters de BoW e TF-IDF são “you, the, and, me…” de um lado e “que, eu, de, nao, voce…” do outro. Os primeiros exemplos do cluster em inglês são “Training Season”, “Ordinary” e “Die With A Smile” (todas Pop); os do outro são “Peão Todo Tatuado”, “Evidências” e “Te Esperando” (todas Sertanejo). Essas partições foram reproduzidas (tamanhos 141/326 e 318/149, iguais aos do notebook) e cruzadas com o idioma estimado por uma heurística simples de palavras funcionais:

| Representação | Cluster | Músicas | Em inglês | Em português |
| ------------- | ------: | ------: | --------: | -----------: |
| BoW           |       0 |     141 |       141 |            0 |
| BoW           |       1 |     326 |        14 |          312 |
| TF-IDF        |       0 |     318 |         6 |          312 |
| TF-IDF        |       1 |     149 |       149 |            0 |

Pela heurística, o corpus tem 312 letras em português e 155 em inglês (Pop e Rock quase todos em inglês; Rap e Reggae, 32 de 47 cada). Para Word2Vec e BERT, os clusters também têm 316 e 151 músicas. Uma amostra aleatória de 5 músicas por cluster (célula de leitura qualitativa do notebook, `random_state=42`) mostrou, nas duas representações, o cluster maior só com letras em português (Pagode, Sertanejo e MPB) e o menor só com letras em inglês (Rock, Pop e um Reggae que mistura hebraico transliterado e inglês): 20 de 20 coerentes com a separação por idioma. Os títulos sorteados são os mesmos nas duas representações, o que sugere agrupamentos muito parecidos. A amostra é pequena e serve como evidência qualitativa.

Com k=10, a pureza por gênero foi 0,27 no BoW e no TF-IDF.

**Conclusão.** Pelas métricas, Word2Vec e depois BERT geraram os clusters mais coesos, com k=2. Mas, nos casos verificados (BoW e TF-IDF por cruzamento completo; Word2Vec e BERT por amostra), esses clusters separam o idioma, e não tema ou gênero, e nenhum k recuperou os 10 gêneros. Logo, k=2 é o melhor numericamente, mas não expressa as características originais dos dados. Uma explicação plausível para os valores altos do Word2Vec é que ele foi treinado em 467 letras de dois idiomas com vocabulários quase disjuntos. Agrupar por tema exige separar o idioma das letras.

<details>
<summary>Código da clusterização, das métricas e da projeção (trechos do notebook)</summary>

```python
def clusterizar_e_avaliar(X, nome, ks=(2,3,4,5,8,10)):
    resultados = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, metric="cosine", sample_size=min(X.shape[0], 2000),
                               random_state=RANDOM_STATE)
        X_eval = X.toarray() if hasattr(X, "toarray") else X
        if X_eval.shape[1] > 3000:     # BoW/TF-IDF: reduz a 100 componentes antes de DB e CH
            X_eval = TruncatedSVD(n_components=min(100, X_eval.shape[1]-1),
                                  random_state=RANDOM_STATE).fit_transform(X)
        db = davies_bouldin_score(X_eval, labels)
        ch = calinski_harabasz_score(X_eval, labels)
        resultados.append({"representação": nome, "k": k, "silhouette_coseno": sil,
                           "davies_bouldin": db, "calinski_harabasz": ch})
    return pd.DataFrame(resultados)

# Projeção 2D usada nas figuras (TruncatedSVD, não PCA)
X_plot = TruncatedSVD(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
```

</details>

## 3) Busca e recomendação

Similaridade de cosseno entre a consulta (vetorizada com a mesma representação) e as 467 letras. Abaixo, o top-5 de cada consulta (os 3 primeiros formam o top-3), com o gênero entre parênteses. Os valores de similaridade não são comparáveis entre representações, então só a ordem foi analisada. Similaridades e trechos completos estão na seção 6 do [notebook](PLN_Trabalho_Unidade_3_Notebook_Integrado.ipynb).

<details>
<summary>Código da busca (trecho do notebook)</summary>

```python
def buscar(consulta, nome, k=5):
    X = representacoes[nome]
    q = representar_consulta(consulta, nome)      # vetoriza a consulta com a mesma representação
    sims = cosine_similarity(q, X).ravel()
    ordem = np.argsort(sims)[::-1][:k]
    return pd.DataFrame({
        "posição": np.arange(1, len(ordem)+1), "similaridade": sims[ordem],
        "titulo": df.iloc[ordem]["titulo"].values, "artista": df.iloc[ordem]["artista"].values,
        "gênero": df.iloc[ordem]["genero"].values,
        "trecho": df.iloc[ordem]["letra"].str.replace("\n", " ", regex=False).str.slice(0, 220).values
    })

CONSULTAS = ["saudade de um amor que acabou", "dançar na festa com os amigos", "fé em Deus e esperança"]
```

</details>

**Consulta 1: “saudade de um amor que acabou”**

| #   | BoW                                  | TF-IDF                          | Word2Vec                             | BERT                              |
| --- | ------------------------------------ | ------------------------------- | ------------------------------------ | --------------------------------- |
| 1   | Flor E O Beija-Flor (Sertanejo)      | Seja Para Mim (Reggae)          | Flor E O Beija-Flor (Sertanejo)      | Tu Mandas No Meu Coração (Pagode) |
| 2   | COMO É QUE FALA DE AMOR? (Sertanejo) | Flor E O Beija-Flor (Sertanejo) | COMO É QUE FALA DE AMOR? (Sertanejo) | La Mour (Agamenon) (Forró)        |
| 3   | Seja Para Mim (Reggae)               | La Mour (Agamenon) (Forró)      | Defeito Meu (Pagode)                 | Girassol (Rap)                    |
| 4   | La Mour (Agamenon) (Forró)           | Defeito Meu (Pagode)            | A Banda (MPB)                        | Displicente (Forró)               |
| 5   | É Tarde Demais (Pagode)              | É Tarde Demais (Pagode)         | Que Se Chama Amor (Pagode)           | Fim de Noite (Pagode)             |

**Consulta 2: “dançar na festa com os amigos”**

| #   | BoW                                      | TF-IDF                  | Word2Vec                                              | BERT                                        |
| --- | ---------------------------------------- | ----------------------- | ----------------------------------------------------- | ------------------------------------------- |
| 1   | Ainda Bem (MPB)                          | Voando Pro Pará (Forró) | Voando Pro Pará (Forró)                               | Era Uma Vez (Poesia Acústica #6) (Rap)      |
| 2   | Gnarly (Pop)                             | LET’S GO 4 (Funk)       | Te Esperando (Sertanejo)                              | Laço na Fivela (Funk)                       |
| 3   | Os Guerreiros Se Preparam - 212 (Gospel) | Não Sei Dançar (MPB)    | Vida Loka, Pt. 1 (Rap)                                | Set Boca de 09 / Mostra Esses Peito… (Funk) |
| 4   | LET’S GO 4 (Funk)                        | Vida Loka, Pt. 1 (Rap)  | Os Guerreiros Se Preparam - 212 (Gospel)              | Cuida do Pet (Funk)                         |
| 5   | Ta Pedindo Toma (Funk)                   | Laço na Fivela (Funk)   | MDT (part. MC Jvila, DJ Glenner e Caio Passos) (Funk) | GRWM (Funk)                                 |

**Consulta 3: “fé em Deus e esperança”**

| #   | BoW                                | TF-IDF                             | Word2Vec                      | BERT                               |
| --- | ---------------------------------- | ---------------------------------- | ----------------------------- | ---------------------------------- |
| 1   | A Alma Abatida - 193 (Gospel)      | A Alma Abatida - 193 (Gospel)      | A Alma Abatida - 193 (Gospel) | Toda Bíblia É Comunicação (Gospel) |
| 2   | Toda Bíblia É Comunicação (Gospel) | Selado Na Luz (Reggae)             | Conversão - 15 (Gospel)       | Ó Desce Fogo Santo - 5 (Gospel)    |
| 3   | Clareou (Pagode)                   | Toda Bíblia É Comunicação (Gospel) | Oração de São Francisco (MPB) | Porque Ele Vive - 545 (Gospel)     |
| 4   | Bondade de Deus (Gospel)           | Clareou (Pagode)                   | Agindo Diferente (Gospel)     | Campeões da luz - 305 (Gospel)     |
| 5   | Ó Desce Fogo Santo - 5 (Gospel)    | Vida Loka, Pt. 1 (Rap)             | Selado Na Luz (Reggae)        | Firme nas Promessas - 107 (Gospel) |

**Avaliação de relevância.** Cada resultado foi julgado pela leitura da letra (início do texto e contagem de palavras-chave na letra inteira), com nota 1 (relevante), 0,5 (parcial) ou 0 (irrelevante). Critérios: consulta 1, a letra trata de saudade ou do fim de um relacionamento; consulta 2, de festa, dança ou amigos (parcial quando aparece de forma secundária); consulta 3, de fé, Deus ou esperança. As 60 notas (39 músicas distintas) e a justificativa de cada uma estão em [`avaliacao_busca.csv`](resultados_trabalho_unidade3/avaliacao_busca.csv).

| Precisão@3 / Precisão@5 |         BoW |      TF-IDF |    Word2Vec |            BERT |
| ----------------------- | ----------: | ----------: | ----------: | --------------: |
| Consulta 1 (saudade)    | 0,33 / 0,60 | 0,67 / 0,80 | 0,67 / 0,60 |     1,00 / 1,00 |
| Consulta 2 (festa)      | 0,17 / 0,20 | 0,67 / 0,60 | 0,33 / 0,20 |     0,67 / 0,50 |
| Consulta 3 (fé)         | 1,00 / 1,00 | 1,00 / 0,80 | 1,00 / 1,00 |     1,00 / 1,00 |
| **Média**               | 0,50 / 0,60 | 0,78 / 0,73 | 0,67 / 0,60 | **0,89 / 0,83** |

**Análise crítica**

- **O BERT foi a melhor representação na média** (P@3 0,89 e P@5 0,83), seguido de TF-IDF (0,78 e 0,73), Word2Vec (0,67 e 0,60) e BoW (0,50 e 0,60). Com 3 consultas e um avaliador, é um indício, não uma comparação estatística.
- **Consulta 3 (vocabulário religioso explícito):** todas acertaram o top-3. O único erro do top-5 é “Vida Loka, Pt. 1” no TF-IDF, uma narrativa de crime em que “Deus” aparece pouco. O rótulo de gênero subestima a qualidade: o Word2Vec tem só 3 de 5 Gospel pelo rótulo, mas 5 de 5 relevantes, porque “Oração de São Francisco” (MPB) e “Selado Na Luz” (Reggae) são religiosas. O rótulo vem do ranking do site e não mede relevância.
- **Consulta 1:** o BERT acertou os 5 (Pagode, Forró e Rap, todas sobre saudade ou fim de relacionamento). BoW e Word2Vec trouxeram “COMO É QUE FALA DE AMOR?”, cuja letra é sobre o sertanejo e contém “amor” 9 vezes e “acabou” 8 vezes, palavras da consulta; o BoW trouxe também “Seja Para Mim”, uma declaração de amor, e o TF-IDF a colocou em 1º. Isso sugere correspondência por palavras da consulta, não pelo sentido. O Word2Vec trouxe ainda “A Banda”, sem relação com perda.
- **Consulta 2 (festa, dança, amigos) foi a mais difícil:** só duas músicas foram totalmente relevantes em todos os resultados (“Voando Pro Pará” e “Laço na Fivela”); vários resultados são parciais (Funk em que festa ou balada aparece de passagem). O BoW (P@5 0,20) trouxe um hino religioso e uma canção de amor (“Ainda Bem”), e o Word2Vec repete o mesmo hino em 4º. A precisão@k não mede revocação, então não há como saber se o corpus tem mais músicas relevantes.
- **Limites:** 3 consultas, 39 músicas julgadas, um avaliador e notas parciais subjetivas. A precisão@k mede só o topo, e o top-5 do Word2Vec muda entre execuções (as notas valem para as tabelas acima).

**Utilizável em contexto real?** Só parcialmente. Com o BERT, o top-5 das consultas 1 e 3 foi todo relevante, mas na consulta 2 (abstrata) só metade (P@5 0,50), e BoW e Word2Vec ficaram em 0,20 nela. Como protótipo, funciona para temas explícitos e é instável para consultas abstratas. O corpus pequeno, a mistura de idiomas, o uso de letras inteiras (o BERT trunca em 256 tokens) e a ausência de limpeza de stopwords na consulta limitam o resultado.

## 4) Classificação

Foi possível treinar um classificador de **gênero** (10 classes) para cada representação: Regressão Logística (`class_weight="balanced"`, `max_iter=1500`), divisão estratificada 80/20 (373 treino / 94 teste), igual para as quatro. O acaso daria ~10%. Detalhes: [`metricas_classificacao.csv`](resultados_trabalho_unidade3/metricas_classificacao.csv).

<details>
<summary>Código da classificação (trecho do notebook)</summary>

```python
X_train_idx, X_test_idx = train_test_split(
    indices, test_size=0.2, random_state=RANDOM_STATE, stratify=df_clf["genero"]
)
for nome in representacoes:
    Xtr = matriz_subset(nome, X_train_idx)
    Xte = matriz_subset(nome, X_test_idx)
    modelo = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)
    modelo.fit(Xtr, y_train)
    pred = modelo.predict(Xte)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
```

</details>

| Representação | Acurácia | Precisão macro | Revocação macro | F1 macro |
| ------------- | -------: | -------------: | --------------: | -------: |
| BERT          |    0,479 |          0,470 |           0,481 |    0,468 |
| BoW           |    0,457 |          0,497 |           0,458 |    0,441 |
| Word2Vec      |    0,372 |          0,327 |           0,378 |    0,337 |
| TF-IDF        |    0,362 |          0,423 |           0,370 |    0,337 |

<details>
<summary>F1 por gênero no conjunto de teste (9 a 10 músicas por gênero)</summary>

| Gênero (F1) |  BoW | TF-IDF | Word2Vec | BERT |
| ----------- | ---: | -----: | -------: | ---: |
| Forró       | 0,38 |   0,25 |     0,30 | 0,44 |
| Funk        | 0,56 |   0,40 |     0,62 | 0,76 |
| Gospel      | 0,86 |   0,82 |     0,95 | 0,89 |
| MPB         | 0,35 |   0,00 |     0,13 | 0,35 |
| Pagode      | 0,38 |   0,30 |     0,36 | 0,38 |
| Pop         | 0,40 |   0,29 |     0,25 | 0,40 |
| Rap         | 0,60 |   0,18 |     0,27 | 0,53 |
| Reggae      | 0,36 |   0,46 |     0,00 | 0,15 |
| Rock        | 0,40 |   0,32 |     0,38 | 0,45 |
| Sertanejo   | 0,12 |   0,35 |     0,12 | 0,32 |

</details>

<table>
  <tr>
    <td align="center"><img src="imagens/fig06_confusao_bow.png" width="300"><br><sub>Fig. 6 — BoW</sub></td>
    <td align="center"><img src="imagens/fig07_confusao_tfidf.png" width="300"><br><sub>Fig. 7 — TF-IDF</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="imagens/fig08_confusao_word2vec.png" width="300"><br><sub>Fig. 8 — Word2Vec</sub></td>
    <td align="center"><img src="imagens/fig09_confusao_bert.png" width="300"><br><sub>Fig. 9 — BERT</sub></td>
  </tr>
</table>

- **Qual representação favorece mais:** o BERT teve a maior acurácia e F1, seguido do BoW; a diferença entre os dois é de cerca de 2 músicas em 94, então não dá para afirmar superioridade. Word2Vec (acurácia 0,372; F1 macro 0,337) e TF-IDF (0,362; 0,337) ficaram abaixo e praticamente empatados. A acurácia do Word2Vec variou entre as três execuções do notebook (0,372, 0,415 e 0,372; F1 macro 0,333, 0,375 e 0,337), então não dá para afirmar qual dos dois é melhor.
- **Por gênero:** Gospel é o mais fácil em todas as representações (F1 de 0,82 a 0,95), por causa do vocabulário religioso. Funk vai bem (até 0,76 no BERT). Sertanejo, MPB e Reggae são os mais difíceis (F1 de 0,00 a 0,46); no BoW, o Reggae teve precisão 1,00 e revocação 0,22 (quase não foi previsto, mas acertou quando foi).
- **Dados novos:** o BERT (melhor F1) classificou três frases curtas escritas no notebook:

  | Texto novo                                                | Gênero previsto (BERT) |
  | --------------------------------------------------------- | ---------------------- |
  | Deus me dá força e esperança, eu sigo com fé no coração.  | Rap                    |
  | Saudade da pessoa amada, lembro do beijo e do nosso amor. | Forró                  |
  | A festa começou, vamos dançar e cantar até amanhecer.     | Forró                  |

  A frase religiosa errou mesmo sendo Gospel a classe mais fácil no teste, o que indica que o modelo generaliza mal para textos curtos fora do formato de letras completas (só 3 exemplos).

- **Modelos usados:** o BERT (modelo de linguagem pré-treinado) foi usado só como extrator de características, sem ajuste fino. O classificador é sempre linear (Regressão Logística).
- **Limitações:** divisão única sem validação cruzada, teste com 94 músicas, vetorizadores ajustados no corpus inteiro antes da divisão (vazamento leve), gênero vindo do ranking do site e letras em inglês processadas por um BERT em português.

## 5) Sentimento, entidades nomeadas e relações

As três técnicas são aplicáveis e rodaram, mas com qualidade limitada, como mostram os resultados e a auditoria em amostra no fim desta seção.

**Sentimento** (`pysentimiento`, português, sem ajuste ao domínio; 467 músicas): [`analise_sentimentos.csv`](resultados_trabalho_unidade3/analise_sentimentos.csv).

|      NEG |         NEU |       POS |
| -------: | ----------: | --------: |
| 8 (1,7%) | 417 (89,3%) | 42 (9,0%) |

O resultado é dominado por “neutro”. Os positivos se concentram no Gospel (14 de 47), e os negativos aparecem só em Forró (3), Pagode (3) e Rap (2); Pop e Rock saíram quase todos neutros. Os de maior confiança positiva são “Plena Paz - 3” e “Grandioso És Tu - 526” (Harpa Cristã, Gospel); os de maior confiança negativa, “Vida Loka, Pt. 1” (Rap, 0,85) e “Assunto” (Forró, 0,77). “Evidências” saiu neutra (0,72) apesar de falar de amor e medo de se entregar. Cruzando com o idioma estimado (seção 2), 153 das 155 letras em inglês saíram neutras (2 positivas, 0 negativas), contra 264 neutras, 40 positivas e 8 negativas entre as 312 em português. Isso é compatível com um modelo para português que não responde a letras em inglês,

<details>
<summary>Sentimento por gênero (467 músicas)</summary>

| Gênero    | NEG | NEU | POS |
| --------- | --: | --: | --: |
| Forró     |   3 |  39 |   5 |
| Funk      |   0 |  42 |   4 |
| Gospel    |   0 |  33 |  14 |
| MPB       |   0 |  39 |   8 |
| Pagode    |   3 |  41 |   3 |
| Pop       |   0 |  46 |   0 |
| Rap       |   2 |  44 |   1 |
| Reggae    |   0 |  43 |   4 |
| Rock      |   0 |  46 |   1 |
| Sertanejo |   0 |  44 |   2 |

</details>

**NER** (spaCy `pt_core_news_sm`; 200 músicas, 199 com ao menos uma entidade; [`entidades_nomeadas.csv`](resultados_trabalho_unidade3/entidades_nomeadas.csv)): 6.149 ocorrências (MISC 3.694 / 60%, PER 1.419 / 23%, LOC 691 / 11%, ORG 345 / 6%; refrões repetem ocorrências). A maioria dos itens mais frequentes são falsos positivos: interjeições, pronomes e palavras comuns, em português ou inglês.

| Categoria | Mais frequentes (ocorrências)                          |
| --------- | ------------------------------------------------------ |
| PER       | Pra (37), Baby (29), Tô (25), Pai (23), Hootie (22)    |
| LOC       | Vou (26), go (18), uh (15), Quero (14), Lua (12)       |
| ORG       | Cause (24), Ah (15), ParaFAL (12), yeah (11), BYD (10) |
| MISC      | yeah (70), Yeah (51), I (50), Me (40), We (33)         |

Entre os itens aparecem possíveis nomes reais (“Gabriela”, “BYD”). O modelo é para português, e as letras misturam inglês, gírias, abreviações (“Tô”, “Cê”) e versos iniciados com maiúscula.

**Relações** (regras sujeito–verbo–objeto sobre a análise de dependência do spaCy, verso a verso, nas mesmas 200 músicas; [`relacoes_extraidas.csv`](resultados_trabalho_unidade3/relacoes_extraidas.csv)): 2.498 linhas de 192 músicas; 993 são duplicatas exatas (versos repetidos nos refrões), restando 1.505 distintas. Os sujeitos mais comuns entre as distintas são “eu” (198), “que” (78), “você” (48), “ela” (43) e “you” (43), e os verbos, fazer (47), ter (45) e querer (38). Funcionam bem em letras em português com estrutura de frase; em inglês geram verbos inválidos (“shitr”, “triedr”, “therir”), e palavras como “never” e “then” aparecem entre os verbos mais frequentes.

| Sujeito | Relação | Objeto           | Avaliação (por leitura)                |
| ------- | ------- | ---------------- | -------------------------------------- |
| eu      | querer  | te               | Correta                                |
| você    | cair    | na minha cantada | Correta                                |
| que     | ter     | saudade          | Erro (sujeito é pronome relativo)      |
| so that | shitr   | helps            | Erro (inglês analisado como português) |

**Auditoria em amostra.** Em cada tarefa foram sorteadas 30 linhas (`random_state=42`) e julgadas pela leitura do verso ou da letra de origem. O julgamento é de um único avaliador, é subjetivo (principalmente em sentimento) e, com n=30, a margem de erro é grande. Cada nota e comentário está em [`auditoria_sentimento.csv`](resultados_trabalho_unidade3/auditoria_sentimento.csv), [`auditoria_ner.csv`](resultados_trabalho_unidade3/auditoria_ner.csv) e [`auditoria_relacoes.csv`](resultados_trabalho_unidade3/auditoria_relacoes.csv).

Critérios: _sentimento_ — o rótulo é uma leitura defensável da polaridade predominante do trecho (ambíguo quando o próprio trecho é misto); _NER_ — o item é um nome próprio e a categoria está certa (parcial: entidade real com categoria errada); _relações_ — sujeito, verbo e objeto expressam o que o verso diz (parcial: verbo certo, mas sujeito pronominal ou objeto incompleto).

| Tarefa     | Amostra                     | Corretas | Parciais | Ambíguas | Incorretas |
| ---------- | --------------------------- | -------: | -------: | -------: | ---------: |
| Sentimento | 30 músicas                  |        7 |        – |       15 |          8 |
| NER        | 30 ocorrências de entidades |        0 |        2 |        1 |         27 |
| Relações   | 30 relações distintas       |        3 |        8 |        – |         19 |

- **Sentimento:** as 8 incorretas foram previstas como neutras apesar de polaridade evidente (positivas como “Perfect”, “Chuva de Arroz” e “Sonho de Amor”; negativas como “Love Yourself” e “Creep”). Foram avaliadas também as 8 músicas previstas como NEG: 7 estão corretas (términos e saudade) e 1 não (“Japão X França (Blue Lock)”, sobre uma partida de futebol). O modelo acerta quando aponta negativo, mas quase nunca o aponta.
- **NER:** nenhuma das 30 estava totalmente correta; 27 são trechos de verso ou palavras comuns (“Beijo”, “Wish”, “Every now and then I”). As 2 parciais são entidades reais com categoria errada (“Buchecha” como ORG e “BYD” como LOC). Como a amostra é por ocorrência, pesam as repetições dos refrões e as letras em inglês.
- **Relações:** nos versos em português, 3 de 10 corretas e 5 parciais; nos versos em inglês, 0 de 20 corretas e 17 incorretas (verbos inventados como “comir” e “curser”, advérbios como “never” tratados como verbo). O idioma de cada verso foi identificado na leitura.

<details>
<summary>Código das amostras da auditoria</summary>

```python
import pandas as pd
d = "resultados_trabalho_unidade3/"
s = pd.read_csv(d + "analise_sentimentos.csv").sample(30, random_state=42)   # CSV com as 467 músicas
e = pd.read_csv(d + "entidades_nomeadas.csv").sample(30, random_state=42)
r = pd.read_csv(d + "relacoes_extraidas.csv").drop_duplicates().sample(30, random_state=42)
```

</details>

<details>
<summary>Código de sentimento, NER e relações (trechos do notebook)</summary>

```python
# Sentimento: corpus inteiro, letra cortada em 1.500 caracteres
analisador_sentimento = create_analyzer(task="sentiment", lang="pt")
amostra_sent = df.copy()
for _, row in amostra_sent.iterrows():
    trecho = row["letra"][:1500]
    res = analisador_sentimento.predict(trecho)

# NER: spaCy, 200 primeiras músicas
nlp = spacy.load("pt_core_news_sm")
N_MUSICAS_NER = min(200, len(df))
docs_ner = list(nlp.pipe(df.iloc[:N_MUSICAS_NER]["letra"].tolist(), batch_size=16))
for i, doc in enumerate(docs_ner):
    for ent in doc.ents:
        ent_rows.append({"titulo": df.iloc[i]["titulo"], "gênero": df.iloc[i]["genero"],
                         "entidade": ent.text, "categoria": ent.label_})

# Relações (sujeito, verbo, objeto) por regras sobre a análise de dependência, verso a verso
def extrair_svo(texto):
    doc = nlp(texto)
    saida = []
    for verbo in doc:
        if verbo.pos_ not in ("VERB", "AUX"):
            continue
        sujeitos = [t for t in verbo.children if t.dep_ in ("nsubj", "nsubj:pass", "csubj")]
        objetos = [t for t in verbo.children if t.dep_ in ("obj", "dobj", "iobj", "obl", "attr", "oprd")]
        for s in sujeitos:
            for o in objetos:
                saida.append({"sujeito": sintagma(s), "relação": verbo.lemma_,
                              "objeto": sintagma(o), "verso": texto})
    return saida
```

</details>

**Resumo:** sentimento quase todo neutro (89%) e sem sinal nas letras em inglês; mais de 6 mil entidades, em sua maioria ruído; 1.505 relações distintas, parte aproveitável em português. Na auditoria em amostra, 7 de 30 sentimentos estavam corretos (15 ambíguos), 0 de 30 entidades estavam totalmente corretas e 3 de 30 relações estavam corretas (8 parciais).

## 6) Relação com o projeto da Unidade 1

O projeto propôs recomendar músicas pelo significado da letra (classificação por tema/sentimento, extração de informações, busca por descrição e chatbot).

| Tarefa da Unidade 1                  | O que os resultados mostram                                                                                                                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Busca por descrição                  | É o mais próximo de funcionar: com BERT, a consulta “fé em Deus e esperança” trouxe 5 de 5 Gospel; consultas abstratas são instáveis (seção 3).             |
| Classificação por tema/sentimento    | Só foi treinada por gênero (acurácia máxima 0,48), pois o corpus não tem rótulos de tema. Gênero não é tema, então o resultado não atende ao objetivo de classificar por tema. |
| Extração de informações e sentimento | NER e relações têm muito ruído (auditoria: 0 de 30 entidades totalmente corretas), e o sentimento deu “neutro” para 89% das letras. Têm pouca utilidade nesta forma.       |
| Chatbot e “significado”              | Fora do escopo desta unidade; o texto usado foi só a letra.                                                            |

As dificuldades previstas (linguagem informal e irônica) se confirmaram, e surgiu o **idioma** como problema novo: o corpus mistura português e inglês, o que domina a clusterização e prejudica NER, relações e sentimento. **Conclusão:** as técnicas ajudam, principalmente BERT na busca, mas não bastam sozinhas.

## Uso de IA generativa

Claude (Anthropic), modelo Claude Sonnet 5.5

1. Use como base os notebooks das aulas para reunir em um único codigo com as funcionalidades necessarias
2. Gere um resumo dos resultados
3. Com base no rascunho e resultados, gere um readme respondendo as perguntas

O Claude também foi usado para julgar a relevância dos resultados da busca (`avaliacao_busca.csv`) e para auditar amostras de sentimento, NER e relações (`auditoria_*.csv`); cada julgamento é de um único avaliador.
