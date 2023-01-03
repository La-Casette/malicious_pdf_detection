## ML classification techniques

### Best features
In our analysis, we identified the best features for a decision tree model by calculating a weighted average and classifying them. This allowed us to optimize the model's performance and achieve the best results.
```
/Colors          1.000000
/RichMedia       1.000000
/XFA             0.988604
/ObjStm          0.971428
/JBIG2Decode     0.969095
/OpenAction      0.848678
/AA              0.830323
/JavaScript      0.773956
endstream        0.768453
stream           0.767761
/JS              0.758707
endobj           0.742549
obj              0.741606
/Page            0.594955
/Encrypt         0.480315
/AcroForm        0.357867
startxref        0.288893
xref             0.188899
/EmbeddedFile    0.160116
trailer          0.133669
/Launch          0.108434
dtype: float64
```
We conducted an analysis to determine the best subset of features for a decision tree model. To do this, we added one of the top performing features at a time and plotted the results. The resulting graphics in Figure 1 showed us which combination of features resulted in the best model performance.
[Figure]

### Decision Tree
The decision tree is a popular classification technique in which predictions are made in a sequence of single-attribute tests. 
A decision tree consists of three types of nodes:
- decision nodes (squares)
- chance nodes (circles)
- end nodes (triangles)

Decision trees are used to evaluate the expected values or expected utility of competing alternatives. Each internal node in a decision tree represents a test on an attribute, and each branch represents the outcome of the test. The paths from the root to the leaf nodes represent classification rules, and the leaf nodes represent class labels.
In this test, we evaluated the performance of a decision tree using various parameters such as tree depth, minimum number of samples per leaf, and maximum number of features. We found that adjusting these parameters had a significant impact on the false negative rate and the accuracy of the model. For example, as shown in Figure 1, we found that the optimal tree depth parameter was 15, which provided the best balance between model performance and overfitting.
[Figure]

### Random Forest
Random Forest is a machine learning algorithm that is used for both classification and regression tasks. It is an ensemble method, which means that it combines the predictions of multiple decision tree models to make a more accurate and stable prediction.
To train a Random Forest model, multiple decision trees are trained on different samples of the data and their predictions are combined. This helps to reduce overfitting and improve the model's generalization ability. The algorithm is also robust to missing values, making it a versatile and widely used machine learning tool.
In our analysis we chose this model because of its high accuracy and stability. It is known for performing well on a wide range of datasets and tasks.
Also with Random Forest, we evaluated the performance varying parameters and in Figure we can see the results for maximum trees depth.
[Figure]

## Attacks
