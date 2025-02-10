# planarian-gene-expression

* `kmeans_clustering.ipynb`: Apply kmeans to the gene expression patterns from 2 months to 23 months and group the genes into 8 clusters.
* `Visualization.ipynb`: Randomly select five genes from each cluster and visualize their time trends.
* `Autocorrelation.ipynb`: Calculate autocorrelation for each gene at lag 1-9. Test if each autocorrelation is significantly different from zero.
* `feature_selection.ipynb`: Select potential marker genes for prediction based on the autocorrelation calculation.
* `PCA_ML.ipynb`: Apply PCA to the selected marker genes to reduce dimensions. Carry out leave-one-out analysis to see if dimensioned reduced gene expression can be predictive of ages.
