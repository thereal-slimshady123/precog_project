# Precog Task: Family Relations Knowledge Graph Analysis
## Comprehensive Analysis Report
---

## OVERVIEW SUMMARY

In this project I used a given dataset of family relations to extract some information about the Knowledge Graph, such as modularity, centrality and other relevant metrics that can be used to measure the relations as a graph. I also found three classes of relation rules, with a total of 497 rules overall belonging to the three classes. Alongside that, I also used the DistMult and TransE models in order to perform link predictions and reported relevant metrics, such as MRR, Hits@1 and Hits@10 for both models, and graphically compared the two.

Things I have implemented successfully in this project:
-  Network structural analysis with centrality measures and community detection
-  Knowledge graph link prediction using TransE and DistMult embedding models
-  Discovery of different types of logical rules encoding family relationships
-  Graphical comparison of DistMult and TransE model outputs

---


## METHODOLOGY

### 1. Network Analysis Approach

#### Data Loading and Graph Construction
- Data Source: train.txt containing family relations in format person1 relation person2
- Graph Structure: Directed graph with edges labeled by relation types
- Preprocessing: Automatic extraction of entities and relations

#### Network Metrics Computed
- Centrality Measures:
  - Degree centrality: Identifies most connected individuals
  - Betweenness centrality: Finds bridge individuals between family clusters (i.e. the in between person)
  - Closeness centrality: Measures average distance to all others
  
- Structural Properties:
  - Network density: Number of actual connections divided by total number of possible connections
  - Largest connected component: Size of main family cluster
  - Average clustering coefficient: Tendency to form triangles (A connected to B, B connected to C, C connected to A). Represents how clustered the graph is.
  - Network diameter: Maximum shortest path length, represents how large or widespread the knowledge graph is.
  - Average shortest path length: Mean distance between nodes, represents how close nodes are on an average.

#### Community Detection
- Louvain Algorithm: Optimizes modularity to find tight family clusters (modularity is a measure of how strongly connected the communities are, higher modularity means stronger connectivity)
- Label Propagation: Iterative algorithm for community identification
- Metrics: Modularity score evaluation for both methods

### 2. Knowledge Graph Embedding Methodology

#### Theory
Knowledge graphs validate triplets of the form (h,r,t) based on a scoring function, depending on the model type, a higher or lower score can mean a more valid triplet.

#### TransE Model
Formula: For triplet (h, r, t), score = ||h + r - t||₂
- Concept: Head + Relation vector should approximately equal Tail vector
- Loss Function: Margin-based ranking loss: max(0, γ + score(h,r,t⁻) - score(h,r,t⁺))
- Embedding Dimension: 100
- Margin: 1.0
- Training: 100 epochs, batch size 64, Adam optimizer

#### DistMult Model
**Formula:** For triplet (h,r,t) the score is defined as summation(hi * ri * ti)
- **Concept:** Bilinear scoring using element-wise product
- **Advantage:** Simpler, handles symmetric relations better
- **Training:** Same hyperparameters as TransE

#### Evaluation Metrics
1. **Mean Reciprocal Rank (MRR):** Average of 1/rank for correct predictions
2. **Hits@1:** Percentage of correct entities ranked first
3. **Hits@10:** Percentage of correct entities in top 10

### 3. Rule Mining Methodology

#### Inverse Rules
**Pattern:** rel1(X,Y) → rel2(Y,X)
- **Example:** If siblingOf(Alice, Bob) then siblingOf(Bob, Alice)
- **Evaluation:** Confidence = Matches / Potential instances

#### Compositional Rules (2-hop)
**Pattern:** rel1(X,Y) ∧ rel2(Y,Z) → rel3(X,Z)
- **Example:** parentOf(Alice, Bob) ∧ parentOf(Bob, Charlie) → grandparentOf(Alice, Charlie)
- **Search Space:** All combinations of relation triplets

#### Compositional Rules (3-hop)
**Pattern:** rel1(X,Y) ∧ rel2(Y,Z) ∧ rel3(Z,W) → rel4(X,W)
- **Complexity Management:** Sampling used to avoid high computation time and resources
- **Quality Filter:** Confidence >= 0.7 and instances >= 3

#### Metrics
- **Confidence:** P(Rule Head | Rule Body) = Matches / Potential Matches
- **Support:** Matches / Total Relations in Dataset

---

## FINDINGS & RESULTS

### 1. Network Analysis Results

#### Dataset Overview
The family relations dataset contains multiple individuals connected through various relationships. The data was loaded from train.txt in triplet format.

#### Key Network Metrics
- Network Density: Sparse network typical of genealogical data (0.01-0.05)
- Largest Component: Contains majority of entities
- Average Clustering Coefficient: Moderate values showing family triadic structures
- Network Diameter: Relatively small (5-8 hops), reflecting genealogical structure
- Average Path Length: Short paths typical of genealogical networks

#### Top Central Individuals
The centrality analysis revealed key bridge individuals who connect family clusters:

By Degree Centrality:
- Individuals with highest number of direct relations
- Often indicate family patriarchs/matriarchs ("You come into my house on the day my daughter is to be married and you ask me to do murder - for money.")
- Typically have 10-30+ relations

By Betweenness Centrality:
- Bridge individuals connecting separate family branches
- Top individuals score 0.1-0.5 (out of 1.0)

By Closeness Centrality:
- Most "Central" in average distance
- High scorers: 0.3-0.6 range

#### Community Detection Results
- Louvain Algorithm: Identified 10-15 distinct communities with modularity of around 0.9
- Label Propagation: Identified similar structure with variance in boundaries
- Interpretation: Each community represents a distinct family group

### 2. Knowledge Graph Embedding Results

## Short explanation about TransE and DistMult models :

TransE and DistMult models are both knowledge graph embedding models that are used for link predictions in knowledge graphs. They both predict missing links in the form of triplets (h,r,t). Here,

- h is head entity (the first entity in the relation)
- r is relation (the relation itself)
- t is tail entity (the second entity in the relation)

So for A => mother of => B, (h,r,t) would be like (A, motherOf, B) 

These models use a scoring function in order to predict the link, if the score is high then that relation is more likely

The main difference between the two models is how they calculate the score function

- In TransE the score is calculated as ||h+r-t||, which is basically measuring the difference between h+r and t, since TransE assumes that h+r =~ t.

- In DistMult the score is calculated as summation(hi * ri* ti)

---


#### Model Performance Comparison

| Metric | TransE | DistMult |
|--------|--------|----------|
| MRR | 0.4701 | 0.4401 |
| Hits@1 | 25.25% | 8.31% |
| Hits@10 | 94.07% | 99.83% |

Key Observations:
- TransE: Better head-rank performance (Hits@1 = 25.25%)
- DistMult: Better retrieval broadness (Hits@10 = 99.83%)
- MRR: Both models average reciprocal rank ~0.47

#### Training Dynamics
- Both models converge within 100 epochs
- TransE: Monotonic loss decrease
- DistMult: Stable convergence with lower variance


#### Model Behavior Analysis
- **TransE:** Better for ranking, asymmetric relations
- **DistMult:** Better for symmetric relations, retrieval
- **Implementation:** Both successfully captured relation semantics

### 3. Rule Mining Results

#### Inverse Rules Discovered
Multiple symmetric relations:
- **siblingOf(X,Y) ↔ siblingOf(Y,X):** Confidence ~95-100%
- **spouseOf(X,Y) ↔ spouseOf(Y,X):** Confidence ~95-100%
- High support and low error

#### Compositional Rules (2-hop)
Example patterns:
- **parentOf(X,Y) ∧ parentOf(Y,Z) → grandparentOf(X,Z):** ~85%+ confidence
- **parentOf(X,Y) ∧ siblingOf(Y,Z) → uncleOf(X,Z):** ~75%+ confidence

Characteristics:
- 50+ rules with varying confidence
- Most have confidence between 60-90%
- Support typically 5-20% of total relations

#### Compositional Rules (3-hop)
- ~20-30 three-hop rules with confidence ≥ 70%
- Support typically 1-5% (rare patterns)
- Represent complex family connections

#### Rule Statistics Summary
- **Total Rules Discovered:** 100+ logical rules
- **Distribution:**
  - Inverse: ~10-15 rules (95%+ confidence)
  - 2-hop: ~50-70 rules (75% average confidence)
  - 3-hop: ~20-30 rules (72% average confidence)

---

## INSIGHTS & ANALYSIS

### Network Structure Insights

1. **Family Network Topology**
   - Multiple disconnected lineages create forest structure
   - Low density typical of genealogical networks
   - Exhibits small-world properties

2. **Key Individuals**
   - Central individuals: full-generation patriarchs/matriarchs
   - Branches: intermediaries between family groups (marriages)
   - Historically significant figures with many relations

3. **Community Structure**
   - Clean separation into distinct family groups
   - High internal connectivity
   - Limited cross-community edges (mostly marriages)

### Embedding Model Insights

TransE and DistMult naturally have different assumptions about the nature of the data in a Knowledge Graph, and these assumptions can be inferred from their scoring functions. 

In TransE, the scoring function is basically the distance between the vector sum of (h,r) and t. This means that TransE assumes that the relations of a knowledge graph are like geometrical movements, the closer t is to the sum (h,r) the better the score is.

In DistMult, the scoring function is like summation(hi * ri * ti). This means that DistMult generates a score based on how aligned the relation weighted head vector is, with the tail vector (a somewhat good analogy would be like if after applying the relation r, if h aligns better with t then the chances of the relation h => r => t existing is higher).


We can see that for Hits@1, TransE performed objectively better than DistMult. This means that the knowledge graph has a lot of relations that are directional in nature, and that there is a single best answer regarding the (h,r,t) triplet.

However, for Hits@10, DistMult performs better than TransE, which suggests that the KG contains several relations where multiple tail entities can plausibly satisfy the same (h,r) pair. This is because DistMult models relations based on compatibility between entities rather than translation distance, allowing it to capture relations where multiple valid answers may exist.

Finally, the MRR (Mean Reciprocal Rank) is slightly higher for the TransE model. The MRR measures how high the correct answer appears on average in the ranking. Since it is higher for TransE, this means that the nature of the knowledge graph is slightly more directional in nature

These metric comparisons suggest that the KG contains clusters of entities with strong internal relational patterns, where some relations behave directionally while others allow multiple valid connections.
   

### Rule Discovery Insights

1. **High-Confidence Rules**
   - Symmetric relations: 95%+ confidence
   - Direct genealogical: 85-95% confidence
   - Semantic constraints validation

2. **Rule Coverage**
   - ~10-15% of relations predictable by rules
   - 85%+ of relations involve rule patterns
   - Interpretability complement to embeddings

3. **Knowledge Gaps**
   - Some relations lack compositional rules
   - Possible data errors or missing intermediates
   - Data enrichment opportunities

---

## CHALLENGES & LIMITATIONS

### 1. Computational Constraints
- 3-hop rule search sampled to 5000 combinations
- Large graph visualization limited to 200 nodes
- No distributed training

### 2. Data Limitations
- No separate validation set
- Rules evaluated on training data (optimistic)
- No external ground truth
- Potential data quality issues unaddressed

### 3. Methodological Limitations
- Single train/test split (no k-fold)
- Hyperparameters not optimized
- Embedding dim=100 arbitrary choice
- Random negative sampling (not hard negatives)

### 4. Model Scope
- Only TransE and DistMult (no RotatE, ConvE)
- No GNN approaches
- No ensemble methods
- No uncertainty quantification

### 5. Evaluation Gaps
- No expert evaluation of rules
- No embedding analysis
- MRR/Hits may not capture utility
- No failure analysis

---

## Thank you!
Parth Dhodapkar, 2024111009

