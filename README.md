# Santander Product Recommendation

In the current system, Santander’s small number of customer see the recommendation but many others don’t see any. This result in uneven customer experience. The goal of this project is to predict which product their customer will use in the future based on their past behavior and that of similar customers. This was a kaggle challenge 

The data set provided by Santander does not include any real Santander Spain customer, and thus it is not representative of Spain's customer base.

# Approach



# Data
* The training data consists of nearly 10 million users with monthly historical user and product data between January 2015     and May 2016.
* Data consist of 24 features of the customer and another 24 features corresponding to the products of the bank.
* Product data consists of boolean flags for all 24 products and indicates whether the user owned the product in the           respective months.



# Prerequisites
* Python-3.6
* GCP (google cloud platform)

# Scripts


# Results

| Method   Used                                       | Mean average precision   |
|-----------------------------------------------------|--------------------------|
| xgboast                                             | 0.021                    |
| Singular Valued Decomposition                       | 0.019                    |
| Random Forest                                       | 0.0012                   |


# Authors
* Abhishek Chatrath
* Anant Tripathi
* Priyank Malviya

# Contribution
There are no specific guidlines for contibuting. If you see something that could be improved, send a pull request! If you think something should be done differently (or is just-plain-broken), please create an issue.

# Reference
[1] https://medium.com/@m_n_malaeb/singular-value-decomposition-svd-in-recommender-systems-for-non-math-statistics-     programming-4a622de653e9

[2] https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

[3] 

[4] 

# License

This Project is under the GNU General Public License v3.0. For more details visit License file here: https://github.com/chatrathabhishek/DSP-Final/blob/master/LICENSE
