# 📋 Master's Thesis Project Management System

## 🎯 Thesis Overview

**Title:** Using Graph Neural Networks to Minimize Financial Contagion in Dynamic Interbank Networks

**Dataset:** 32 quarters of interbank network data

**Methods:** GNN training, contagion simulations, systemic risk modeling, embeddings analysis

---

## 📊 Kanban Board — Task Management

### Phase 1 — Data Validation

- [x]  **Verify data loader** | Priority: High | Estimated: 2 days
- [x]  **Validate node and edge consistency** | Priority: High | Estimated: 2 days
- [x]  **Check temporal consistency across 32 quarters** | Priority: High | Estimated: 3 days
- [x]  **Produce summary statistics** | Priority: Medium | Estimated: 2 days
- [x]  **Perform descriptive network analysis** | Priority: Medium | Estimated: 3 days

### Phase 2 — Simulation Framework

- [x]  **Modularize contagion code** | Priority: High | Estimated: 4 days
- [x]  **Implement failure and distress models** | Priority: High | Estimated: 5 days
- [x]  **Create shock generator** | Priority: High | Estimated: 3 days
- [x]  **Define shock scenarios** | Priority: Medium | Estimated: 2 days

### Phase 3 — Simulation Dataset Generation

- [x]  **Build simulation loop** | Priority: High | Estimated: 4 days
- [x]  **Run simulations per quarter** | Priority: High | Estimated: 5 days
- [x]  **Store graph, features, and outcomes** | Priority: High | Estimated: 3 days
- [ ]  **Define labels (default, loss, systemic importance)** | Priority: Medium | Estimated: 2 days

### Phase 4 — GNN Dataset Construction

- [ ]  **Convert simulations into graph dataset** | Priority: High | Estimated: 4 days
- [ ]  **Implement PyTorch Geometric format** | Priority: High | Estimated: 3 days
- [ ]  **Train-test split using temporal structure** | Priority: High | Estimated: 2 days

### Phase 5 — GNN Modeling

- [ ]  **Train baseline GCN or GraphSAGE** | Priority: High | Estimated: 5 days
- [ ]  **Hyperparameter tuning** | Priority: High | Estimated: 4 days
- [ ]  **Evaluate model** | Priority: High | Estimated: 3 days

### Phase 6 — Embeddings and Machine Learning

- [ ]  **Extract node embeddings** | Priority: High | Estimated: 3 days
- [ ]  **Train classical ML models** | Priority: High | Estimated: 4 days
- [ ]  **Compare performance** | Priority: Medium | Estimated: 3 days

### Phase 7 — Benchmark and Financial Contribution

- [ ]  **Compare with centrality methods** | Priority: High | Estimated: 4 days
- [ ]  **Intervention strategies** | Priority: Medium | Estimated: 4 days
- [ ]  **Loss reduction experiments** | Priority: Medium | Estimated: 3 days

### Phase 8 — Robustness and Finalization

- [ ]  **Stress testing** | Priority: Medium | Estimated: 4 days
- [ ]  **Final experiments** | Priority: High | Estimated: 5 days
- [ ]  **Thesis writing** | Priority: High | Estimated: 15 days
- [ ]  **Preparation for defense** | Priority: High | Estimated: 5 days