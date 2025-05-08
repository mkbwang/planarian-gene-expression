# planarian-gene-expression

* `split_samples.py` split the raw data into training data and test data. Training data contains 50 samples with known ages.
* `gene_variation_sort.py` sort the genes based on their ratio of between age variance over within age variance.
* `utils/` folder contains utility functions for plotting, calculating variance, fitting deconvolution and elastic net regressions.
* `deconvolution.ipynb` folder contains experiments of evaluating the best way of selecting a subset of useful genes for deconvolution prediction.
* `Elastic_Net.ipynb` and `Elastic_Net_2.ipynb` are two early notebooks of trying to fit elastic net regressions
* `Correlated_Genes.ipynb` is meant to answer a request by Long to report the genes with the highest correlation with SSMa053560.



