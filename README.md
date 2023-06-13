# Map Matching for Gretel Algorithm
This repository contains code to produce the metadata needed by the Gretel Algorithm [1]. The algorithm used for the map matching procedure can be found in [2]. The datasets used is Geolife Trajectories 1.3 [3].   

## Reproductibility and Datasets
To reproduce the metadata in `MetaDataForGretell`, the script `run_feature_extraction.py` is needed. The execution of this file is described with comments inside. In addition, the dataset [3] has to be extracted in the folder `Datasets/Geolife`. The version of the packages used in the metadata extraction can be found in `environment.yml`.

# References 
[1] Cordonnier, J. B., & Loukas, A. (2019). Extrapolating paths with graph neural networks.
 arXiv preprint arXiv:1903.07518. Weblink: https://github.com/jbcdnr/gretel-path-extrapolation
 
[2] Yang, C., & Gidofalvi, G. (2018). Fast map matching, an algorithm integrating hidden Markov model 
with precomputation. International Journal of Geographical Information Science, 32(3), 547-570. 
Weblink: https://fmm-wiki.github.io/

[3] Yu Zheng, Xing Xie, Wei-Ying Ma, GeoLife: A Collaborative Social Networking Service among User, 
location and trajectory. Invited paper, in IEEE Data Engineering Bulletin. 33, 2, 2010, pp. 32-40.
weblink: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

# Acknowledgement

This research was carried out as part of the project “Optimal Path Recommendation with Multi Criteria” (Project code: KMP6-0078997) under the framework of the Action "Invest-ment Plans of Innovation" of the Operational Program "Central Macedonia 2014-2020", that is co-funded by the European Regional Development Fund and Greece.

We would like to thank Jean-Baptiste Cordonnier for his support to understand the structure of the metadata. 
