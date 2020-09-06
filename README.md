# PANDA_Challenge
This is the DeepLearning Approach of PANDA Challenge @Kaggle


Model: 
       - EfficientNet - B0
       - Pre-trained From ImageNet
       
Label & Loss: 
       - Classification Bin Label, Cross-Entropy Loss

Ensemble: 
       - 4 - fold Ensemble

Noisy Label Cleaning:
       - Direct Exclusion if abs(pred - true) >= 1.5

Final Performance & Ranking:
       - Weighted Quadratic Kappa:  0.93123 Private Score
       - Accuracy: ~ 0.66
       - Ranking: ~ 1%
