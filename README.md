# PLN — Letras e Significados de Músicas

Trabalho de **Processamento de Linguagem Natural** (Ciência da Computação — FURB).
**Etapa Prática 1: Coleta de Dados e Pré-processamento.**

**Autor:** Bruno Cardoso
**Tema:** Letras e significados de músicas; recomendação de músicas com base no significado das letras.

---

## Sumário

1. [Tema e justificativa](#1-tema-e-justificativa)
2. [Coleta de dados (scraping)](#2-coleta-de-dados-scraping)
3. [Pré-processamento](#3-pré-processamento)
4. [Dicionário de dados](#4-dicionário-de-dados)
5. [Como reproduzir](#5-como-reproduzir)

---

## 1. Tema e justificativa

### Por que letras de música?

Letras de música são um tipo de texto curto, rico em vocabulário afetivo, figuras de linguagem e variação de registro (formal, coloquial, gírias). Isso as torna um bom desafio para PLN e permite várias tarefas sobre a mesma base:

| Tarefa                     | Aplicação no projeto                                                               |
| -------------------------- | ---------------------------------------------------------------------------------- |
| Classificação de texto     | Classificar letras por gênero, tema ou sentimento (amor, tristeza, superação etc.) |
| Extração de informações    | Extrair temas, palavras-chave, locais e personalidades citadas                     |
| Recuperação de informações | Buscar uma música a partir de uma descrição livre do usuário                       |
| Recomendação               | Sugerir músicas com significado semelhante (similaridade entre letras)             |
| Chatbot / agentes          | Identificar sentimento do usuário e recomendar músicas                             |
| Outros                     | Inferir o significado de uma música a partir da letra                              |

### Por que o letras.mus.br?

- Reúne **letra, artista, gênero e visualizações** em páginas com estrutura HTML regular, o que viabiliza o scraping.
- Possui páginas de **significado** (interpretação textual da letra) em várias músicas. Esse segundo texto permite comparar letra × interpretação e abre tarefas além da classificação simples.
- Tem rankings de **mais acessadas por gênero**, que fornecem um recorte objetivo e reproduzível das músicas.

### Por que esta base é adequada?

- **Volume:** 467 músicas, com 79.554 tokens após a limpeza. É suficiente para classificação, recuperação e clusterização em escala de disciplina.
- **Abrangência:** 10 gêneros (Sertanejo, Pop, Rock, MPB, Funk, Rap, Pagode, Forró, Gospel, Reggae), com 46–47 músicas cada, e **279 artistas** distintos. O balanceamento entre classes é útil para classificação.
- **Variedade:** vocabulário e estilo mudam bastante entre gêneros (ex.: coloquial do Funk × poético da MPB), e há letras em português, em inglês e mistas.
- **Estrutura rica:** cada registro traz metadados, letra bruta, significado e representações pré-processadas (ver [dicionário de dados](#4-dicionário-de-dados)).

---

## 2. Coleta de dados (scraping)

**Notebook:** `obtenção_de_dados_PLN.ipynb`
**Bibliotecas:** `requests`, `BeautifulSoup` (bs4), `json`, `time`

### Como funciona

1. **Configuração:** dicionário `GENEROS` (slug do site → nome do gênero), `User-Agent` de navegador, limite de 50 músicas por gênero e pausa de 0,5 s entre requisições.
2. **Listagem por gênero** (`obter_top_50_genero`): acessa `/mais-acessadas/{slug}/`, percorre os links da página e mantém apenas os de formato `/artista/musica/`. Descarta páginas institucionais (`estilos`, `blog`, `sobre`, `playlists` etc.) e links duplicados, e para ao chegar a 50.
3. **Remoção de duplicatas:** uma música que aparece em mais de um ranking é mantida uma única vez (chave: URL da letra).
4. **Extração por música** (`extrair_dados_musica`), em duas requisições:
   - **Página da letra:** título, artista, gêneros (breadcrumb), visualizações e a letra, montada a partir dos parágrafos `<p>` da div `lyric-original`.
   - **Página `/significado.html`:** parágrafos da div `lyric-meaning`, quando existem.
5. **Tratamento de falhas:** `fazer_request` usa timeout de 15 s, trata erros HTTP e de conexão e retorna `None`. Músicas que falham são ignoradas sem interromper a coleta.
6. **Saída:** `dataset_top10_generos_50_musicas.json` (UTF-8, `ensure_ascii=False`).

---

## 3. Pré-processamento

**Notebook:** `preprocessamento.ipynb`
**Bibliotecas:** `nltk`, `spaCy` (`pt_core_news_sm`), `re`, `json`

O pipeline é aplicado ao campo `letra` de cada música e gera quatro representações, guardadas em `preprocessamento`.

| Etapa                               | O que é feito                                                                               | Como                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Filtragem**                       | Descarta registros sem título ou letra                                                      | List comprehension sobre o JSON bruto                                                          |
| **Normalização e remoção de ruído** | Minúsculas; remoção de pontuação, números e símbolos; colapso de espaços e quebras de linha | `str.lower()` + regex `[^a-zà-ÿ\s]`. Os **acentos são mantidos**, pois são relevantes em PT-BR |
| **Tokenização**                     | Divide o texto normalizado em palavras                                                      | `nltk.word_tokenize(language='portuguese')`                                                    |
| **Remoção de stopwords**            | Remove palavras funcionais e tokens de 1 caractere                                          | Stopwords NLTK em **português + inglês** + lista própria de coloquialismos e interjeições      |
| **Stemming**                        | Reduz os tokens ao radical                                                                  | `RSLPStemmer` (NLTK)                                                                           |
| **Lematização**                     | Reduz as palavras ao lema                                                                   | spaCy `pt_core_news_sm`, aplicado ao texto normalizado                                         |

### Decisões de projeto

- **Stopwords em inglês:** muitas letras do dataset têm trechos ou são inteiramente em inglês, então só a lista em português deixaria muito ruído.
- **Stopwords coloquiais:** contrações do PT-BR e vocalizações comuns em letras (`pra`, `tá`, `tô`, `cê`, `né`, `aí`, `oh`, `yeah`, `uou`, `nana` etc.) não estão na lista padrão do NLTK e foram adicionadas manualmente.
- **Texto bruto preservado:** `letra` continua intacta no JSON final, para que outras etapas possam aplicar um pré-processamento diferente.
- **Stemming e lematização lado a lado:** as duas versões ficam disponíveis, para comparar o efeito de cada técnica nas etapas seguintes.

---

## 4. Dicionário de dados

Arquivo `dataset_musicas_preprocessado.json` (lista de objetos):

| Campo                                 | Tipo             | Descrição                                                                           |
| ------------------------------------- | ---------------- | ----------------------------------------------------------------------------------- |
| `titulo`                              | string           | Título da música                                                                    |
| `artista`                             | string           | Nome do artista ou banda                                                            |
| `metadados.generos`                   | lista de strings | Gêneros associados à música conforme o breadcrumb da página (pode haver mais de um) |
| `metadados.genero_ranking`            | string           | Gênero do ranking usado na coleta (um dos 10 gêneros-alvo)                          |
| `metadados.exibicoes`                 | string           | Número de visualizações da página no site (quando disponível)                       |
| `metadados.url_letra`                 | string           | URL de origem da letra                                                              |
| `metadados.url_significado`           | string           | URL de origem do significado                                                        |
| `letra`                               | string           | Letra completa, como extraída do site (texto bruto)                                 |
| `significados`                        | lista de strings | Parágrafos da interpretação da letra, quando disponível                             |
| `preprocessamento.texto_normalizado`  | string           | Letra em minúsculas, sem pontuação, números e símbolos, com espaços colapsados      |
| `preprocessamento.tokens`             | lista de strings | Tokens do texto normalizado, sem stopwords e sem tokens de 1 caractere              |
| `preprocessamento.tokens_stemizados`  | lista de strings | `tokens` após stemming (RSLP)                                                       |
| `preprocessamento.tokens_lematizados` | lista de strings | Tokens lematizados via spaCy, a partir do texto normalizado e sem stopwords         |

---

## 5. Como reproduzir

```bash
# 1. Dependências
pip install requests beautifulsoup4 nltk spacy
python -m spacy download pt_core_news_sm

# 2. Coleta (gera dataset_top10_generos_50_musicas.json)
#    Executar obtenção_de_dados_PLN.ipynb

# 3. Pré-processamento (gera dataset_musicas_preprocessado.json)
#    Executar preprocessamento.ipynb
```

Os dados `punkt`, `punkt_tab`, `stopwords` e `rslp` do NLTK são baixados pelo próprio notebook.

> O scraping depende da estrutura HTML do letras.mus.br. Se o site mudar, os seletores podem precisar de ajuste. O ranking também muda com o tempo, então uma nova coleta pode trazer músicas diferentes. Os JSONs do repositório são a versão usada no trabalho.
