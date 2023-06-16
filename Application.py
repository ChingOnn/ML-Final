import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel


st.header("""
# Loan Eligibility Prediction App

This app automate the loan eligibility process based on customer detail provided!
""")

#import dataset
df =  pd.read_csv('C:/Users/Limpek/Downloads/loan_sanction_train.csv')
initial_df = pd.read_csv('C:/Users/Limpek/Downloads/loan_sanction_train.csv')

missing_value = df.isnull()

df.dropna(inplace=True)

###preprocessing###
# Encode categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Married'] = label_encoder.fit_transform(df['Married'])
df['Education'] = label_encoder.fit_transform(df['Education'])
df['Self_Employed'] = label_encoder.fit_transform(df['Self_Employed'])
df['Property_Area'] = label_encoder.fit_transform(df['Property_Area'])
df['Loan_Status'] = label_encoder.fit_transform(df['Loan_Status'])

# Convert 'Dependents' column
df['Dependents'] = df['Dependents'].replace('3+', 3)

X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
# y = df.iloc[:,4].values
y = df['Loan_Status'].values

# Sidebar navigation
st.sidebar.title("MENU")
options = ["HOME","EDA", "Algorithm", "Logistic Regression Prediction","K-Nearest Neighbors Prediction"]
selected_option = st.sidebar.radio("Go to", options)

# EDA (Exploratory Data Analysis)
if selected_option == "HOME":
    st.subheader("HOME")
    st.write("Dataset of this research: https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval")
    st.write("CD20139 NG  CHING ONN")
    # Display algorithm content
    st.subheader("Project Idea")
    st.write("The project aims to automate the loan eligibility process by developing a machine learning model that predicts whether a customer is eligible for a loan based on their demographic and financial information provided in the online application form. This real-time automation will help the company streamline the loan approval process and target specific customer segments for loan offers. The dataset dislay details such as Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and etc. Total of 12 attributes and 1 predictive attribute.")

# EDA (Exploratory Data Analysis)
elif selected_option == "EDA":
    st.subheader("Exploratory Data Analysis")
    # Display EDA content
    # Add your EDA code here
    st.subheader("Original Dataset")
    st.write("There is",initial_df.shape[0],"number of rows and", initial_df.shape[1],"numbers of columns in the original dataset retrieve from kaggle.")
    st.dataframe(initial_df)

    # Display the value range of each column
    st.subheader("Value range for each column before preprocessing")
    st.write("Below shows the range of numerical data within each features:")
    range_info = initial_df.describe().loc[['min', 'max']]
    st.write(range_info)
    
    st.write("Below shows the numbers of categorical data within each features:")
    # Select the categorical columns
    categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Dependents', 'Property_Area', 'Loan_Status']

    # Display value counts for each categorical column
    for column in categorical_columns:
        value_counts = initial_df[column].value_counts()
        st.write(f"Value counts for column '{column}':")
        st.write(value_counts)

    st.subheader("Missing Value")
    st.write("All null value will be dropped from the dataset to ensure more accurate training set.")
    st.dataframe(missing_value)

    st.subheader("Numbers of Null values or NaN within each columns of dataset")
    st.write("Below shows total number of null values within each features")
    st.write(missing_value.sum())

    st.subheader("Dataset after row of missing value are dropped and categorical data is handled")
    st.write("There is",df.shape[0],"number of rows and", df.shape[1],"numbers of columns in the dataset after missing values is handled.")
    st.dataframe(df)

    # Display the value range of each column
    st.subheader("Value range for each column after preprocessing")
    range_info = df.describe().loc[['min', 'max']]
    st.write(range_info)

    loan_status = df['Loan_Status']
    loan_status_counts = loan_status.value_counts()
    # Data for the pie chart
    labels = ['Approved' if status == 1 else 'Not Approved' for status in loan_status_counts.index]
    sizes = loan_status_counts.values.tolist()
    colors = ['blue',  'red']
    explode = (0, 0.1)  # To highlight a slice, set its explode value > 0

    # Create the pie chart
    st.subheader("Pie Chart Visualization on Loan Status")
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_aspect('equal')
    ax.set_title('Loan Approval Status')

    # Display the pie chart in Streamlit
    st.pyplot(fig)

    applicant_income = df['ApplicantIncome']
    total_loan_amount = df['LoanAmount'].values*df['Loan_Amount_Term'].values


    # Create the scatter plot
    st.subheader("Scatter Plot Visualization")
    fig1, ax1 = plt.subplots()
    ax1.scatter(total_loan_amount, applicant_income)

    # Set labels and title
    ax1.set_xlabel('Total Loan (RM)')
    ax1.set_ylabel('Applicant Income (RM)')
    ax1.set_title('Scatter Plot: Applicant Income based on Total Loan Amount')

    # Display the scatter plot in Streamlit
    st.pyplot(fig1)



# Algorithm
elif selected_option == "Algorithm":
    st.subheader("Algorithm")
    # Display algorithm content
    st.write("The project aims to automate the loan eligibility process by developing a machine learning model that predicts whether a customer is eligible for a loan based on their demographic and financial information provided in the online application form. This real-time automation will help the company streamline the loan approval process and target specific customer segments for loan offers. The dataset dislay details such as Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and etc. Total of 12 attributes and 1 predictive attribute.")

    st.subheader("Logistic Regression")
    st.write("It is a simple yet effective algorithm for binary classification problems. Logistic regression can handle categorical and numerical features, making it suitable for this loan eligibility prediction task.")

    st.subheader("K-Nearest Neighbors")
    st.write("KNN is a non-parametric algorithm that classifies new data points based on their similarity to the training data. It can handle both numerical and categorical features and is relatively easy to understand and implement.")
    

# Logistic Prediction
elif selected_option == "Logistic Regression Prediction":
    st.subheader("Logistic Regression Prediction")
    # Display prediction content
    # Add your prediction code here
    #split training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader('Please complete the form below')

    def user_input_features():
        gender = st.selectbox("Gender", ['Male', 'Female'])
        married = st.selectbox("Married", ['Yes', 'No'])
        dependents = st.number_input("Dependents", min_value=0, step=1)
        education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
        self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
        applicant_income = st.slider('Applicant Income', 0, 99999, 3000)
        coapplicant_income = st.slider('Coapplicant Income', 0, 99999, 3000)
        loan_amount = st.slider('Loan Amount', 1, 999, 200)
        loan_amount_term = st.slider('Loan Term', 0, 360, 120)
        credit_history = st.selectbox("Credit History", [0, 1])
        property_area = st.selectbox("Property Area", ['Rural', 'Urban', 'Semiurban'])

        new_data = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': property_area
                }
        
        # Encode categorical variables in new data
        new_data['Gender'] =  1 if gender == 'Male' else 0
        new_data['Married'] = 1 if married == 'Yes' else 0
        new_data['Education'] = 1 if education == 'Not Graduate' else 0
        new_data['Self_Employed'] = 1 if self_employed == 'Yes' else 0
        
        # Property_Area encoding using if-else
        if property_area == 'Rural':
            new_data['Property_Area'] = 0
        elif property_area == 'Semiurban':
            new_data['Property_Area'] = 1
        else:  # Urban
            new_data['Property_Area'] = 2

        features = pd.DataFrame(new_data, index=[0])
        return features

    new_input = user_input_features()

    # Make predictions on the new input
    new_predict = model.predict(new_input)
    st.write("Prediction:", 'Approved' if new_predict == 1 else 'Not Approved')

    # Evaluate the model based on test data
    accuracy = accuracy_score(y_test, y_pred)
    

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("Confusion Matrix")
    st.write("True Positive:", tp)
    st.write("False Positive:", fp)
    st.write("True Negative:", tn)
    st.write("False Negative:", fn)
    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1-score:", f1)

    st.subheader("Features Selection")

    # Perform feature selection based on logistic regression coefficients
    selector = SelectFromModel(LogisticRegression())
    selector.fit(X_train, y_train)

    # Transform the training and test sets to include only the selected features
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Train the logistic regression model with the selected features
    model_selected = LogisticRegression()
    model_selected.fit(X_train_selected, y_train)

    # Get the coefficients of the selected features
    feature_importances = model_selected.coef_[0]

    # Normalize the feature importances to represent percentages
    total_importance = sum(abs(feature_importances))
    feature_importances = [(abs(importance) / total_importance) * 100 for importance in feature_importances]

    # Create a DataFrame to display the features and their importance percentages
    features_df = pd.DataFrame({"Feature": X.columns[selector.get_support()], "Percentage": feature_importances})
    st.dataframe(features_df)

    # Get the selected feature names and importances
    selected_features = X.columns[selector.get_support()]
    feature_importances = feature_importances

    # Create a bar plot of feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(selected_features, feature_importances)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Percentage')
    ax.set_title('Feature Selection - Logistic Regression')
    ax.set_xticklabels(selected_features)
    st.pyplot(fig)

    # Make predictions on the new input with the selected features
    new_input_selected = selector.transform(new_input)
    new_predict_selected = model_selected.predict(new_input_selected)
    st.write("Prediction (after feature selection):", 'Approved' if new_predict_selected == 1 else 'Not Approved')


    # Evaluate the model based on test data with selected features
    y_pred_selected = model_selected.predict(X_test_selected)

    accuracy_selected = accuracy_score(y_test, y_pred_selected)
    tn_selected, fp_selected, fn_selected, tp_selected = confusion_matrix(y_test, y_pred_selected).ravel()
    precision_selected = precision_score(y_test, y_pred_selected)
    recall_selected = recall_score(y_test, y_pred_selected)
    f1_selected = f1_score(y_test, y_pred_selected)

    st.subheader("Confusion Matrix (after feature selection)")
    st.write("True Positive:", tp_selected)
    st.write("False Positive:", fp_selected)
    st.write("True Negative:", tn_selected)
    st.write("False Negative:", fn_selected)
    st.write("Accuracy:", accuracy_selected)
    st.write("Precision:", precision_selected)
    st.write("Recall:", recall_selected)
    st.write("F1-score:", f1_selected)

# KNN Prediction
elif selected_option == "K-Nearest Neighbors Prediction":
    st.subheader("K-Nearest Neighbors Prediction")
    # Display prediction content
    # Add your prediction code here

    #split training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the K-Nearest Neighbors model
    sc_x=StandardScaler()
    X_train=sc_x.fit_transform(X_train)
    X_test=sc_x.fit_transform(X_test)

    #fitting KNN classifier to the training set
    classifier= KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    st.subheader('Please complete the form below')

    def user_input_features():
        gender = st.selectbox("Gender", ['Male', 'Female'])
        married = st.selectbox("Married", ['Yes', 'No'])
        dependents = st.number_input("Dependents", min_value=0, step=1)
        education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
        self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
        applicant_income = st.slider('Applicant Income', 0, 99999, 3000)
        coapplicant_income = st.slider('Coapplicant Income', 0, 99999, 3000)
        loan_amount = st.slider('Loan Amount', 1, 999, 200)
        loan_amount_term = st.slider('Loan Term', 0, 360, 120)
        credit_history = st.selectbox("Credit History", [0, 1])
        property_area = st.selectbox("Property Area", ['Rural', 'Urban', 'Semiurban'])

        new_data = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': property_area
                }
        
        # Encode categorical variables in new data
        new_data['Gender'] =  1 if gender == 'Male' else 0
        new_data['Married'] = 1 if married == 'Yes' else 0
        new_data['Education'] = 1 if education == 'Not Graduate' else 0
        new_data['Self_Employed'] = 1 if self_employed == 'Yes' else 0
        
        # Property_Area encoding using if-else
        if property_area == 'Rural':
            new_data['Property_Area'] = 0
        elif property_area == 'Semiurban':
            new_data['Property_Area'] = 1
        else:  # Urban
            new_data['Property_Area'] = 2

        features = pd.DataFrame(new_data, index=[0])
        return features

    new_input = user_input_features()

    # Make predictions on the new input
    new_predict = classifier.predict(new_input)
    st.write("Prediction:", 'Approved' if new_predict == 1 else 'Not Approved')

    # Evaluate the model based on test data
    accuracy = accuracy_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("Confusion Matrix")
    st.write("True Positive:", tp)
    st.write("False Positive:", fp)
    st.write("True Negative:", tn)
    st.write("False Negative:", fn)
    st.write("Accuracy:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1-score:", f1)

    st.subheader("Features Selection")

    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.preprocessing import MinMaxScaler

    # Apply feature scaling to make the input data non-negative
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Select the top k features using chi-square test
    k = 5  # Specify the number of top features to select
    selector = SelectKBest(score_func=chi2, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Train the K-Nearest Neighbors model with the selected features
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train_selected, y_train)

    # Get the selected feature scores
    feature_scores = selector.scores_

    # Calculate the importance percentage for each feature
    total_score = sum(feature_scores)
    feature_importances = [(score / total_score) * 100 for score in feature_scores]

    # Create a DataFrame to display the features and their importance percentages
    features_df = pd.DataFrame({"Feature": X.columns, "Percentage": feature_importances})
    st.dataframe(features_df)

    # Create a bar plot of feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(X.columns, feature_importances)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance Percentage')
    ax.set_title('Feature Selection - KNN')
    ax.set_xticklabels(X.columns, rotation=90)
    st.pyplot(fig)

    # Initialize and train the K-Nearest Neighbors model with selected features
    sc_x = StandardScaler()
    X_train_scaled = sc_x.fit_transform(X_train_selected)
    X_test_scaled = sc_x.transform(X_test_selected)

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train_scaled, y_train)

    # Evaluate the model based on test data with selected features
    y_pred_selected = classifier.predict(X_test_scaled)

    accuracy_selected = accuracy_score(y_test, y_pred_selected)
    tn_selected, fp_selected, fn_selected, tp_selected = confusion_matrix(y_test, y_pred_selected).ravel()
    precision_selected = precision_score(y_test, y_pred_selected)
    recall_selected = recall_score(y_test, y_pred_selected)
    f1_selected = f1_score(y_test, y_pred_selected)

    st.subheader("Confusion Matrix (after feature selection)")
    st.write("True Positive:", tp_selected)
    st.write("False Positive:", fp_selected)
    st.write("True Negative:", tn_selected)
    st.write("False Negative:", fn_selected)
    st.write("Accuracy:", accuracy_selected)
    st.write("Precision:", precision_selected)
    st.write("Recall:", recall_selected)
    st.write("F1-score:", f1_selected)
