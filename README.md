# BA_Final_Report
Final report for the Business Analytics programme - MSc Business Analytics

This project was the final submission for the MSc Business Analytics programme, the only requirement was that the main report topic would cover the possibilities of implementing machine learning in some industry. The focus in this report was machine learning implementation in the construction industry along with some proof of concept work for time estimation using taxi data. 

The report could not be included here because of classified data used for the project.

Majority of the code is written in Jupyter notebooks because of the convenience for project submission.

# Explanation of Jupyter notebooks and Python scripts
* **Construction_Time_Estimation_Models.ipynb**: Takes in the summary construction data. Calculates some rough ratio-based time estimation and experiments with a few regression based algorithms.
* **Geocoding.ipynb:** Takes in the taxi data and predefined coordinate polygons, these include
borough and airport borders. The pickup and drop of coordinates are then segmented into boroughs based on the polygons. Furthermore, a binary column is created that identifies if the pickup or drop of was near an airport.
* **OSRM_Route_Scraper.ipynb:** Takes in the taxi data and uses the pickup and drop of coordinates to identify the best route between the two points. The route is supplied by a python client for the open source routing machine, OSRM, project API.
* **Data_Wrangling_NYC_Taxi.ipynb:** Takes in the taxi, geocoding and OSRM routing data. Does all required cleaning and feature engineering for the final time estimation models.
* **Taxi_Time_Estimation_Models.ipynb:** Takes in the cleaned data and runs multiple models to identify the best solution. When the best model had been determined, it trains and tests a fine tuned version of that model.
* **XGB_Tuning.py:** A separate Python script that takes care of hyperparameter tuning for the XGBoost model. This had to be in a separate file so the tuning could be done with Amazon Web Services.
* **NLP_Analysis.ipynb:** Takes in accident and injury reports and does some experimental natural language processing on that data. Both by checking accuracy based on predefined keywords and by extracting the most common words to identify repeating causes.
