import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler
MODEL_PATH = 'Credit Artifacts/model_data.joblib'

#load the model and its components
model_data =  load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']



def prepare_df(age, loan_tenure_months, number_of_open_accounts, credit_utilization_ratio,
            deliquency_ratio, average_dpd, residence_type,
            loan_purpose, loan_type, loan_amount, monthly_income):


    # Create a dictionary to pass to your model
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': number_of_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / monthly_income if monthly_income > 0 else 0,
        'deliquency_ratio': deliquency_ratio,
        'average_dpd': average_dpd,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented':1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education':1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        #add additional features assign dummy value dummy value 1
        'number_of_dependants': 0,
        'years_at_current_address': 0,
        'zipcode': 0,
        'sanction_amount': 0,
        'processing_fee': 0,
        'gst' : 0,
        'net_disbursement': 0,
        'principal_outstanding': 0,
        'bank_balance_at_application': 0,
        'number_of_closed_accounts': 0,
        'enquiry_count': 0
    }

    df = pd.DataFrame([input_data])

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    print("Scaled input:")
    print(df[features])

    df = df[features]

    return df

def calculate_credit_score(input_df, base_score=300, scale_length = 600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    print(input_df.values)
    print(np.dot(input_df.values, model.coef_.T))

    # DEBUG: See the raw number
    print(f"Raw Model Output (x): {x}")

    default_probability = 1 / ( 1 + np.exp(-x))
    non_default_probability = 1 - default_probability

    credit_score = base_score + non_default_probability.flatten() * scale_length
    print(f"Credit Score: {credit_score}")

    #determine the risk score from the credit score

    def get_rating(score):
        if 300 <= score < 500:
            return "Poor"
        elif 500 <= score < 650:
            return "Average"
        elif 650 <= score < 750:
            return "Good"
        elif 750 <= score <= 900:
            return "Excellent"
        else:
            return "Undefined"

    rating = get_rating(credit_score.item())

    return default_probability.flatten()[0], int(credit_score.item()), rating

def predict(age, loan_tenure_months, number_of_open_accounts, credit_utilization_ratio,
            deliquency_ratio, average_dpd, residence_type,
            loan_purpose, loan_type, loan_amount, monthly_income):

    input_df = prepare_df(age, loan_tenure_months, number_of_open_accounts, credit_utilization_ratio,
            deliquency_ratio, average_dpd, residence_type,
            loan_purpose, loan_type, loan_amount, monthly_income)


    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating
