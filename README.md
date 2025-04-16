# Emotion-Cause Pair Detection using GCN on RECCON Dataset

## 📌 Overview

This project addresses the task of detecting **emotion-cause clause pairs** in multi-turn conversations using the **RECCON dataset**. It involves segmenting utterances into meaningful **clauses**, annotating them as emotion or cause clauses, and constructing a **graph-based model (GCN)** to learn inter-clause relationships and detect emotion-cause pairs.

---

## ✅ Tasks

### Task 1: Dataset Preparation

- **Input:** RECCON conversations (`tr_xxxx` format), where each turn contains an utterance and its associated emotion/cause annotations.

- **Objective:** Break each utterance into **meaningful clauses**.

- **Clause Segmentation Approach:**
  - Split based on delimiters such as `?`, `!`, `.`, `,`.
  - Use coordinating conjunctions (`and`, `but`, `or`, etc.) and subordinating conjunctions (`because`, `although`, `if`, etc.) to extract semantically independent clauses.
  - Maintain clause integrity — each clause must convey a coherent thought or idea.

- **Note:** An automated clause segmentation approach is prioritized for scalability and reproducibility.

---

### Task 2: Clause Annotation

- **Objective:** Automatically label each clause as one or more of the following:
  - **Emotion Clause**: Expresses the emotion felt by the speaker.
  - **Cause Clause**: Describes the cause of that emotion.

- **Annotation Strategy:**
  - For each non-neutral utterance, all its clauses are marked as **emotion clauses**.
  - To label **cause clauses**, compare each clause with the "expanded emotion cause span" from the RECCON annotations using embedding similarity.
  - A clause is labeled as a cause if its embedding is similar to any span in the cause annotation of subsequent turns (including itself).

---

### Task 3: Model - Graph Construction & GCN

- **Clause Graph Construction:**
  - Each **clause becomes a node**.
  - Edges are added between nodes based on:
    - **Syntactic dependencies** (e.g., subject-verb, verb-object using dependency parsing).
    - **Temporal proximity** (e.g., same utterance or nearby turns).

- **Embedding Representation:**
  - Use pre-trained embeddings (e.g., BERT, RoBERTa) for clause representations.

- **Graph-based Model:**
  - Use a **Graph Convolutional Network (GCN)** to process the graph of clauses.
  - Each node is classified into one of three categories: **emotion**, **cause**, or **neither**.
  - Post-processing step pairs emotion and cause clauses based on learned embeddings or attention mechanisms.

---

## 📊 Metrics & Evaluation

- **Precision, Recall, F1-Score** for emotion-cause pair detection.
- **AUC-ROC** to measure discriminative ability of the model.
- Performance is interpreted in terms of the confusion matrix to assess true positives, false positives, etc.

---

## 🛠️ Tools & Libraries

- Python
- PyTorch
- spaCy (for clause segmentation and dependency parsing)
- PyTorch Geometric (for graph modeling)



