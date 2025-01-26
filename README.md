# Diamond Price Predictor

## Description

The **Diamond Price Predictor** is a Flask application designed to estimate the price of a diamond based on its attributes. The application uses machine learning models trained on a dataset of diamond characteristics and pricing. Users can input details like carat, cut, color, and clarity to get price predictions from multiple models.

---

## Features

- **Model Training and Evaluation**: Implements multiple regression models including Linear Regression, Random Forest, XGBoost, SVR, and KNN.
- **Feature Engineering**: Enhances the dataset with features like diamond volume and price per carat.
- **Feature Selection**: Automatically selects the most relevant features for prediction.
- **Interactive Web Interface**: Built with Streamlit for easy user interaction.
- **Visualization**: Exploratory data analysis includes histograms, correlation matrices, and box plots.
- **Model Comparison**: Compares models on metrics like R², MSE, RMSE, and MAE.

---

## Dataset

- **Source**: The dataset should be a CSV file containing diamond attributes such as:
  - Carat Weight
  - Cut
  - Color
  - Clarity
  - Price
  - Dimensions (Length, Width, Height)

---

## Models Used

  1. Linear Regression
  2. Random Forest
  3. XGBoost
  4. Support Vector Regressor (SVR)
  5. K-Nearest Neighbors (KNN)
     
Each model is trained with hyperparameter tuning via GridSearchCV.

---

## Evaluation Metrics

 - **R² Score**: Measures how well the model explains variance in the data.
 - **Mean Squared Error (MSE)**: The average squared difference between predictions and actual values.
 - **Root Mean Squared Error (RMSE)**: The square root of MSE, providing a more interpretable metric.
 - **Mean Absolute Error (MAE)**: The average absolute difference between predictions and actual values.

---

## Visualizations

1. **Exploratory Data Analysis (EDA):**
   - Distribution plots for prices
   - Correlation matrix heatmaps
   - Box plots for categorical features
2. **Model Performance:**
   - Bar charts comparing R², MSE, RMSE, and MAE across models.

---

## Future Enhancements

- **Add more advanced models like Gradient Boosting or Neural Networks.**
- **Implement a RESTful API to serve predictions.**
- **Deploy the application using Docker, AWS, or Heroku.**
- **Expand the dataset for better generalization.**

---

## License

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for details.

## Developers

- [Pramodya Maddekanda](https://github.com/PramoW22)
- [Vidmini Minipuma](https://github.com/VidminiMinupama)
- [Mariyam Muad](https://github.com/mariyammuad)
- [Zimra Mohamed](https://github.com/ZimraMohamed)
- [Sakeena Navavi](https://github.com/SakeenaNavavi)
