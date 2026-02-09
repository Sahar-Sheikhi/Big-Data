**# Outlier Detection**
related to big data problems using apache spark and python(pyspark lib)




**Introduction:**
This project implements and compares exact and approximate algorithms for detecting outliers in large datasets using Apache Spark. Outlier detection is crucial in data analysis but can be computationally expensive. The goal is to efficiently identify outliers based on a given set of parameters.


**Project Overview:**
The project consists of the following components:

    **ExactOutliers (Sequential Algorithm):**
    Implementation of the exact algorithm using standard sequential code without RDDs.
    Computes the number of outliers and lists the top outliers in non-decreasing order of their distances from other points.
    
    
    **MRApproxOutliers (Spark Implementation):**
    Implementation of the approximate algorithm using Apache Spark.
    Divides the input data into suitable partitions and processes them using Spark RDDs.
    Computes the number of sure outliers, uncertain points, and lists non-empty cells in non-decreasing order of their sizes.
    
    
    
