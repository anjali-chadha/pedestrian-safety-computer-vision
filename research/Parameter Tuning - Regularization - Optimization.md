
* We want to make a decision on:
  - #layers
  - #hidden units
  - learning rates
  - activation functions to choose
* ML is iterative process
* How efficiently you can iterate through multiple cycles?

* Hold out/ Dev/ Validation Set - all same
* Previous era- split data into 70-30 split of the available data.
* Modern big data era - say 1M image dataset. Keep just 10k, say, datasets in Validation dataset
* **Mismatched train/test distribution**
  - Training set - high resolution pictures from the webpages
  - TDev/testing set - cat pictures taken by users on the phone (blurry, low-res set)
  - Make sure dev and test set come from the same distribution.
  - If you follow above rule of thumb, even if training data is from above distribution, we will get good results
* In DL, generally used datasets - training and dev set (often called testing)

* Bias-Variance Tradeoff
  - High bias - Underfitting
  - High variance - Overfitting
  
 

  
