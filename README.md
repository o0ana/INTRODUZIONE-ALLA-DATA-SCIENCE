# Classificazione del valore delle case in California

Progetto finale del corso di **Introduzione alla Data Science** 

**Autrice:** Ana Maria Raducan — S5797859

## Obiettivo

Prevedere la fascia di prezzo (5 classi) delle case in California a partire da dati su posizione, popolazione, reddito e dimensioni delle abitazioni, seguendo l'intera pipeline di data science: pulizia dati, analisi esplorativa, model selection con cross-validation e valutazione finale.

## Dataset

10.320 campioni, 14 variabili (posizione geografica, età mediana delle case, numero di stanze, popolazione, reddito, distanze dalle principali città). Il target `median_house_value` è categoriale: 5 fasce di prezzo (da <$100k a >$300k, dati del 1999).

## Contenuto del repository
- notebook con l'intera analisi, in forma di report
- script eseguibile che allena il modello migliore e produce le predizioni sul test set

## Pipeline seguita

1. **Caricamento e pulizia dati** — rimozione di 98 valori mancanti (<1% del dataset) in `total_bedrooms`
2. **Analisi esplorativa** — correlazioni tra variabili e target; `median_income` (+0.63) e `distance_to_coast` (-0.53) risultano i predittori più forti
3. **Codifica del target** — one-vs-rest per la loss quadratica; gestione nativa multi-classe per gli altri algoritmi
4. **Cross-validation** — split 80/20 train/test, 5-fold CV per la selezione dei parametri

## Modelli confrontati

| Modello | Accuracy (test set) |
|---|---|
| ERM con loss quadratica (Ridge) | 0.496 |
| Regressione logistica (C=100) | 0.569 |
| k-NN (k=15) | 0.576 |
| Albero decisionale | vedi notebook |

Le classi centrali (fasce di prezzo intermedie) risultano più difficili da predire rispetto agli estremi (case molto economiche o molto costose), che hanno caratteristiche più distintive.



## Librerie usate

`pandas`, `numpy`, `scikit-learn`, `matplotlib`
