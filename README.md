# ExplodingHam
*Oddball scikit-learn style ML+ approaches.*

This library provides scikit-learn style solutions for ML problems. The goal of this library is to extend beyond what scikit-learn to include obscure ML approaches, as well as non-ML approaches that you can easily baseline your ML approachs against. 

## Models
A couple examples of currently available models include:
 - **Normalized Compression Distance KNN**, this is a nifty approach to building a classifier on text data.
 - **RegEx Classifier**, takes a provided RegEx and uses it to create a quick and dirty classifier. This is *not* machine learning, it doesn't learn, but having this as a scikit-learn classifier allows you to plug and play with your existing code to provide a baseline, or experiment with different versions of your RegEx.

***Future*** methods to include:
 - **Bumping Classifier**, an ensemble method that trains many models on subsets of the data, and keeps the model that performs the best.
 - **Rotation Forests**, a variation on a random forest that performs PCA before each tree is fit.
 - **InfoGain Trees**, classifiers that use ID3 and C4.5 on non-binary trees.
 - **Zero-Shot LLM Classifier**, a classifier that uses an LLM with a prompt to make predictions.
 - **Few-Shot LLM Classifier**, a classifier that uses an LLM with multiple examples to provide a classification prediction.
 - **Normalized Compression Distance Clustering**, hierarchical clustering, KMeans, DBScan using compression algorithms to represent the data.

 ## Integrations:
 - ExplodingHam uses the [narwhals](https://narwhals-dev.github.io/narwhals/) library for dataset processing, this means that it is plug and play with most DataFrames you might supply.
 - ExplodingHam uses base classes from scikit-learn, so these models should be relatively indistinguishable from 