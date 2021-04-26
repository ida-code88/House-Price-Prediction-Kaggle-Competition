# House-Price-Prediction-Kaggle-Competition

I am a beginner at Kaggle and this is my first competition ever in Data Science. As I understand, this project aims to predict the house prices in the test dataset based on given variables on the training dataset. Overall, the steps I use can be seen below. I am trying my best to make it as clear and simple as possible:

### Exploratory Data Analysis

* Gathering information for the independent and target variables: Before jumping into building a machine learning algorithm, the understanding of the features' properties such as the type of data, their description, etc. will give us the benefit in our data analysis process. We can select the appropriate descriptive statistics, choose the correct type of plots, and prepare it better for further analysis.

* Multivariate Study: By understanding the relationship between variables on our data, we will be able to get an insight into our data. We then have a choice to only select the variables that we find important in building the algorithm and ignore the others or use them as a whole. It will be easier for us to build code for any project.

Steps:

1. Load the data and do feature selection (drop columns with excessive amount of null value)
2. Investigate terget variable: here I found that the target variable does not follow normal distribution, so it need to be converted to normal distribution

![alt text](https://www.kaggleusercontent.com/kf/58073919/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..5KYBejDgzOh62AX44sO4Gg.MysbzFAhMzUcGpT8P04aJZqCYdlP2FrmAH8pUKuuVlZcFTqb6eYKEteIWoQUxZrJd_6eWIwOrJ0-2H34bsculhMHsy7kx9ne2I1MKqttHlXqtttExvi3Eahu-XbhVo9OJKLLf_2Hp7Ihp0Au4HDmbnj94c7pj3__zdoEMnniofgo8Z0O2FdRKS6BZl8qKnidua1_twBCwrQzAaeL_2vU-bWgdzLOUdwciZo1lNwlf3zB1LmvqYXFPucMTGTNUYle_A5nxkzgAg9KGT85v6ODB1H3bqgyE_nbaH5BlvXpT80gNbxipyEx2Zy1ZX4rDk16wKUcdxLYzuYlLfMgeDa7abY9abML_8tSIMpGjMO5KlaizbvRDnHoaocDypBPZ2GMXQkwDhJUGKuvxKevEa3EnUskgSykv-nQrYnEpmNB9xU2wUXqs8NqNHmpuRwkXdxGMUTGDRUDUmStONJek2nTWffGZ9SbgxkQ4L6SX3DotMSeduasoNL8otHJz-7P79j1pwNndpyIlghhonikYsAk4i-efqgv2zloVQnq7J5iF1foxVef8nBUXJcb5_O--Qh6ANYB2sVBAURdAi52ZVfy0p4bwWleUcr2RGf3ubM_Bv-Te0es9D4DhY8dhyHqrp5ROI8y-RwLRfYNOeUuH7g7VwOJz66vkiA8gQvRQ5yTDrA.XgKGxRDfzrvcZe0Ij5Bjcw/__results___files/__results___14_0.png)

4. Investigate numeric variables: Using heatmap, it is revealed that OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd are features that have the strongest correaltion with target feature which is SalePrice
5. Investigate categorical variables:


### Data Cleaning: 

I think it is universally accepted why data cleaning is a must-do step before ml analysis. So I am not going to explain it here.

### Apply Machine Learning

As we know, the dependent or target variable is numeric. For this case, regression comes to action. I will use the regression models that I am more familiar with.


