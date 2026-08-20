# Extreme Growth Classifier for Research Clusters

Cluster features and code necessary to train (1) a classic Probit regression model and (2) a linear neural network to identify extreme growth research clusters in the Map of Science

## Cluster Features

The full list of 141 cluster features used as input for the classifier are provided in [Cluster Features.xlsx](Cluster_Features.).

There are 93 features that relate directly to time series information (articles, citations, references), and the remaining 48 are derived features included for a specific reason someone might think they’re relevant.

- **Life Cycle**: Inspired by stages of innovation, such as peak years, vitality, growth, etc.
  - 16 Features (L0 - L15)
- **Academic Importance**: Looks at top journals from the past couple of years
  - 2 Features (I0 - I1)
- **Document Counts**: Different document types (review, pre-prints, etc.)
  - 4 Features (D0 - D3)
- **Network**: Explores network features of the clusters, such as average weighted or unweighted degree
  - 7 Features (N0 - N6)
- **Field**: Fraction of articles in the 12 top level classification art from the Map of Science
  - 12 Features (F0 - F11)
- **Patents**: Number of citations from patent families
  - 1 Feature (P0)
- **Constructs**: Other metrics identified from papers that involve more elaborate construction (e.g. entropy)
  - 3 Features (C0 - C2)
- **Countries and Collaborations**: How international or collaborative a cluster is
  - 3 Features (CC0 - CC2)
- **Time series Information**: The share of articles from each year assigned to the cluster
  - 16 Features (S0 - S15) for growth share info
  - 15 Features (dS0 - dS14) for change in growth share info

## Funding-Critical Clusters

A full list of funding-critical clusters are provided in the document funding_critical_cluster_metadata.csv
