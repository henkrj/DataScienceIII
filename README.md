# Classificação de Reviews da Steam com NLP e Deep Learning

Este projeto implementa um pipeline completo de Processamento de Linguagem Natural (NLP) e Deep Learning para classificar reviews da plataforma Steam como **Recommended (1)** ou **Not Recommended (0)**. O trabalho segue os critérios exigidos pela Entrega Final, incluindo pré-processamento de texto, análise exploratória e construção de dois modelos de rede neural (um simples e outro aprimorado).

---

## 1. Visão Geral do Projeto

O objetivo principal é aplicar técnicas de NLP para preparar e explorar um conjunto de dados de reviews de jogos, seguido da utilização de modelos de Deep Learning para prever a recomendação do usuário.

As etapas incluem:

- Download automatizado do dataset  
- Pré-processamento completo das reviews  
- Tokenização com NLTK  
- Análise exploratória do texto  
- Conversão para vetores TF-IDF  
- Treinamento de duas redes neurais  
- Comparação entre modelo simples e modelo aprimorado  

---

## 2. Dataset

O dataset utilizado foi o **Steam Reviews Dataset**, baixado automaticamente via `kagglehub`.  
Após a filtragem inicial, foram utilizadas as colunas:

- `review` – texto da review  
- `recommendation` – classe alvo (`Recommended` ou `Not Recommended`)

A distribuição original (433.375 reviews) foi:

- Recommended (1): 302.751  
- Not Recommended (0): 130.624  

Para treinar o modelo, foi utilizada uma amostra estratificada de **50.000 reviews**, devido a limitações de memória.

---

## 3. Pré-processamento de Texto (NLP)

As principais etapas de limpeza do texto foram:

1. **Normalização:**  
   Conversão para minúsculas e remoção de quebras de linha.

2. **Remoção de pontuação e números:**  
   Expressões regulares foram usadas para manter apenas letras.

3. **Stopwords:**  
   Stopwords da língua inglesa (NLTK) foram removidas.

4. **Tokenização:**  
   Implementada com `word_tokenize` (NLTK), atendendo ao requisito do projeto.

O resultado foi armazenado em:

- `clean_review` – texto limpo  
- `tokens` – lista de tokens  

---

## 4. Análise Exploratória de Texto

As análises incluíram:

- Distribuição de classes  
- Tamanho das reviews (em tokens)  
- Frequência das palavras mais comuns  
- Geração de WordClouds:
  - Geral  
  - Apenas reviews positivas  
  - Apenas reviews negativas  
  - Com remoção opcional de termos “game/games” (stopwords de domínio)

Essas etapas permitiram observar padrões linguísticos nas avaliações.

---

## 5. Representação dos Textos (TF-IDF)

As reviews foram vetorizadas usando **TF-IDF** (`TfidfVectorizer`), configurado com:

- `max_features=3000`  
- `ngram_range=(1,1)`  
- `stop_words='english'`  

Resultando em uma matriz de dimensão:

```
(50000, 3000)
```

---

## 6. Deep Learning

### 6.1 Modelo 1 — Rede Neural Simples (Baseline)

**Arquitetura:**
- Dense(32, relu)  
- Dense(1, sigmoid)

**Configuração:**
- Loss: Binary Crossentropy  
- Otimizador: Adam  
- Métrica: Acurácia  
- `class_weight` para lidar com desbalanceamento  

**Resultado:**
- Acurácia no teste: **~0.8253**

---

### 6.2 Modelo 2 — Rede Neural Aprimorada

**Arquitetura:**
- Dense(64, relu)  
- Dropout(0.3)  
- Dense(32, relu)  
- Dense(1, sigmoid)

**Configuração:**
- EarlyStopping (monitorando `val_loss`)  
- `class_weight`, epochs=10, batch_size=256  

**Resultado:**
- Acurácia no teste: **~0.8366**

O segundo modelo apresentou melhor desempenho e menor overfitting.

---

## 7. Conclusões

- Técnicas de pré-processamento e NLP foram aplicadas com sucesso.  
- WordClouds e análises de frequência reafirmaram padrões distintos entre reviews positivas e negativas.  
- O modelo simples já apresentou bom desempenho.  
- O modelo aprimorado demonstrou melhor capacidade de generalização.  
- TF-IDF mostrou-se eficaz para reviews curtos da Steam.

Este trabalho atende completamente aos requisitos da Entrega Final:  
✓ Pré-processamento de NLP  
✓ Construção de rede neural simples  
✓ Aprofundamento com arquitetura aprimorada  

---

## 8. Como Executar

1. Instalar dependências:

```bash
pip install -r requirements.txt
```

2. Executar o notebook principal:

```
Notebook2.ipynb
```

---

## 9. Dependências (requirements.txt)

As dependências estão listadas no arquivo `requirements.txt` incluído no projeto.

---

## 10. Autor

**Igor Renato da Fonseca Vasques**  
Entrega Final — NLP e Deep Learning  
Dataset: Steam Reviews Dataset

