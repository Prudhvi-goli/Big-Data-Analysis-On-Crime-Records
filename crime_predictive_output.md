## **Model Evaluation Outputs**

### **Random Forest Model**
- **Accuracy:** 0.6474  
- **Confusion Matrix:**
```plaintext
[[11054   329   907    31   392]
 [ 1477   840   124     7   168]
 [ 2438    33  2013    22   102]
 [  480    21   137    11    52]
 [ 1344   117   276    13  1631]]
```
- **Classification Report:**
```plaintext
                  precision    recall  f1-score   support

        Assault       0.66      0.87      0.75     12713
        Robbery       0.63      0.32      0.42      2616
Break and Enter       0.58      0.44      0.50      4608
     Theft Over       0.13      0.02      0.03       701
     Auto Theft       0.70      0.48      0.57      3381

       accuracy                           0.65     24019
      macro avg       0.54      0.43      0.45     24019
   weighted avg       0.63      0.65      0.62     24019
```

---

### **One-Hot Encoded Random Forest Model**
- **Accuracy:** 0.6616  
- **Confusion Matrix:**
```plaintext
[[11204   285   825    26   373]
 [ 1514   821   119     4   158]
 [ 2204    30  2225    24   125]
 [  478    15   139     9    60]
 [ 1389    73   273    14  1632]]
```
- **Classification Report:**
```plaintext
                  precision    recall  f1-score   support

        Assault       0.67      0.88      0.76     12713
        Robbery       0.67      0.31      0.43      2616
Break and Enter       0.62      0.48      0.54      4608
     Theft Over       0.12      0.01      0.02       701
     Auto Theft       0.70      0.48      0.57      3381

       accuracy                           0.66     24019
      macro avg       0.55      0.43      0.46     24019
   weighted avg       0.65      0.66      0.63     24019
```

---

### **Balanced Random Forest Model**
- **Accuracy:** 0.6454  
- **Confusion Matrix:**
```plaintext
[[11072   356   877    36   372]
 [ 1481   856   115     7   157]
 [ 2518    30  1936    21   103]
 [  488    16   132    10    55]
 [ 1375   101   265    13  1627]]
```
- **Classification Report:**
```plaintext
                  precision    recall  f1-score   support

        Assault       0.65      0.87      0.75     12713
        Robbery       0.63      0.33      0.43      2616
Break and Enter       0.58      0.42      0.49      4608
     Theft Over       0.11      0.01      0.03       701
     Auto Theft       0.70      0.48      0.57      3381

       accuracy                           0.65     24019
      macro avg       0.54      0.42      0.45     24019
   weighted avg       0.63      0.65      0.62     24019
```

---

### **Gradient Boosting Model**
- **Accuracy:** 0.5588  
- **Confusion Matrix:**
```plaintext
[[12537     0     0     1   175]
 [ 2540     0     0     1    75]
 [ 4532     0     0     1    75]
 [  678     0     0     1    22]
 [ 2497     0     0     0   884]]
```
- **Classification Report:**
```plaintext
                  precision    recall  f1-score   support

        Assault       0.55      0.99      0.71     12713
        Robbery       0.00      0.00      0.00      2616
Break and Enter       0.00      0.00      0.00      4608
     Theft Over       0.25      0.00      0.00       701
     Auto Theft       0.72      0.26      0.38      3381

       accuracy                           0.56     24019
      macro avg       0.30      0.25      0.22     24019
   weighted avg       0.40      0.56      0.43     24019
```
- **One-Hot Encoded Random Forest Accuracy:** 0.6616  
- **Balanced Random Forest Accuracy:** 0.6454  
- **Gradient Boosting Accuracy:** 0.5588  

---
