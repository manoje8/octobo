## What is K-Means Clustering?

K-Means is an **unsupervised machine learning algorithm** that partitions a dataset into **K distinct, non-overlapping clusters** based on similarity. It groups data points so that points in the same cluster are more similar to each other than to those in other clusters.

---

## How K-Means Works (The Algorithm)

### Step-by-Step Process:

1. **Choose K** (number of clusters)
2. **Initialize centroids** - Randomly select K data points as initial cluster centers
3. **Assign points** - Each data point is assigned to the nearest centroid (using distance metrics like Euclidean distance)
4. **Update centroids** - Recalculate centroids as the mean of all points in each cluster
5. **Repeat** - Steps 3-4 until convergence (centroids stabilize or max iterations reached)

### Visual Representation:


```
Initial Data    →    Assign to Closest    →    Update Centroids    →    Converge
	    ★                ★ ○ ●                  ★ ○ ●                 ★ ○ ●
	   ○ ● ★            ○ ● ★                  ○ ● ★                 ○ ● ★
	  ● ○ ○            ● ○ ○                  ● ○ ○                 ● ○ ○
```
---

## Mathematical Formulation

**Objective:** Minimize within-cluster sum of squares (WCSS)

$WCSS= \sum_{i=1}^{k} \sum_{x\epsilon C\iota }∣∣x−μi∣∣2$

Where:

- K = number of clusters
- Ci = cluster i
- μi = centroid of cluster i
- x = data point

**Distance metric (Euclidean):**  

$d(x,y)= \sqrt{\sum_{i=1}^{n}​(xi​−yi​)2}$

---

## Choosing the Right K

### Elbow Method

Plot WCSS vs. number of clusters and look for the "elbow point"

text

```
WCSS
  |
  |    *
  |   * *
  |  *   *
  | *     *
  |*       *  ← Elbow point (optimal K)
  |         * * *
  |_____________*_____→ K
        1  2  3  4  5
```

### Silhouette Score

Measures how similar a point is to its own cluster vs. other clusters

- **Range:** -1 to +1
    
- **+1:** Well-clustered
    
- **0:** Overlapping clusters
    
- **-1:** Misclassified
    

### Other Methods:

- **Gap Statistic** - Compares to null reference distribution
    
- **Davies-Bouldin Index** - Lower is better
    
- **Calinski-Harabasz Index** - Higher is better