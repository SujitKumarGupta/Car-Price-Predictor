# Car Price Prediction Application

This application predicts car prices using a Random Forest model with a Streamlit interface. The program takes the features like  year, milage, model, brand of car from user as input through the sidebar on the frontend. It nomalized the input data and predict the value on realtime. It also keeps track of the average predicted value of all the predictions.it stores the features, predicted price, date as a document in mongoDB database. It also plot scatter in to graph for data visualization.

## Setup Instructions

1. First, run the training script to generate the model and dataset:
   ```bash
   python model/train.py
2. Run the below command to run the project on the server port
   ```bash
   streamlit run app.py
   
## 📸 Screenshots

### Dashboard View
![Dashboard](screenshots/Screenshot%202025-07-19%20140425.png)

### Prediction Result
![Prediction](screenshots/Screenshot%202025-07-19%20134926.png)
