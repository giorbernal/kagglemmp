# Microsoft Malware Prediction
---
This is a solution for this kaggle [competition](https://www.kaggle.com/c/microsoft-malware-prediction) project.
the dataset can be downloaded by executing this commands (see kaggle documentation to configure kaggle command): 
```
>> mkdir datasets
>> cd datasets
>> kaggle competitions download -c microsoft-malware-prediction
```

## Procedure
Our procedure implements the ensembling/stacking technique, wich is based on divide the dataset in two parts, one for ensembling (9 folds in this case) and the rest for stack the variables obtained in the ensemble phase.
The whole procedure have 5 steps:
1. **Cleaning**
2. **Segmentation and train folds**
3. **Stack Phase setting**
4. **Stack phase training**
5. **Test Evaluation**

## Results
The final performance of the system is around **0.662**. Although, the top of the leaderboard was about 0.702, we are quite far from the medals positions ... .
The final test evaluation was imposible to compute due to the lack of optimization in the process

## Conslusions and takeaways
The main goal of this analysis was to check the ensembling/stacking procedure. However, we have had some perfomance issues and limitations.
After analyzing kernels from other competitors, we will consider next takeaways for future cases:
+ Take care of the analysis of data and cleaning: It is very important to **understand** the dataset.
+ Optimize memory defining proper dtypes. (i.e: int8 by int16, float32 by float64 ...)
+ Analyze train and test data together.
+ Analyze the best way to **clean** nulls (Median and Mode maybe ..., or even keep as a category ) after high-na columns and too skewed columns.
+ Evaluate correlation of variables continuous on one hand and discrete on the other, and remove duplicated if they correlate as well with target. Use *label encoders* for this discrete case
+ When there are a lot of discrete variable, like in this dataset, and word embeddings can not be used, consider certain things like the use of sparse matrixes or better, avoid one-hot encoding using other discrete encoding technics like target encoding or frecuency encoding (choose those in which the range is higher).

