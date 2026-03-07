# Using Graph Neural Networks to Minimize Financial Contagion

**Master in Data Science and Advanced Analytics** — Specialization in Data Science

**Author:** Rúben Marques - 20240352
**Supervisor:** Professor/Prof. Flávio Luís Portas Pinheiro

---

## Context

Banks are closely connected through lending and financial exposure. When one bank fails, the shock can spread to others, potentially triggering a chain reaction (Krause & Giansante, 2012). This phenomenon, known as financial contagion or cascade effect (Valente, 2012a), can be studied using network models. Traditional approaches rely on centrality measures to identify which banks are most important, but these methods may overlook complex structural risks (Krause & Giansante, 2012). Graph Neural Networks (GNNs) are a promising alternative, as they can learn directly from the network and capture hidden patterns (Gupta et al., 2021).

## Research Gap and Objectives

Although centrality measures are common in network analysis, they are static and may miss important patterns (Valente, 2012b). Graph Neural Networks (GNNs) offer a more flexible way to assess node importance, but their use in financial contagion is still limited (Gupta et al., 2021; Krause & Giansante, 2012). The study will examine whether GNNs can better identify banks whose failure would have the greatest impact, by comparing them with traditional centrality methods in reducing contagion through targeted interventions.

## Methodological Approach

The study will use synthetic datasets representing interbank networks. Banks will be modelled as nodes, with links showing financial exposures. Different network types, such as random and scale-free, will be tested, and to complement these, empirical networks will also be considered to add robustness to the analysis. A simple contagion model will simulate how defaults spread. Centrality measures and GNNs will be implemented in Python, focusing on how they work. The goal is to compare how well they limit contagion when used for targeted interventions.

## Expected Results and Contributions

The project is expected to provide a comparison between classical and GNN-based intervention strategies. It should offer insight into how GNNs behave in different network structures and whether they provide practical benefits for identifying systemic risk. The final outcome will include a fully working simulation framework and an analysis of how well each approach performs under various conditions.
