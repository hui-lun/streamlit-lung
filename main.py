import streamlit as st
import joblib
import pandas as pd
from padelpy import from_smiles
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

lung_data = pd.read_csv("web_logPapp.csv")
lung_X = lung_data.drop("logPapp", axis=1)
lung_y = lung_data["logPapp"]

st.title("Prediction of in vitro permeability")

task_name = st.sidebar.selectbox("Select Task", ("Calu-3 permeability", 
                                                 "Caco-2 permeability", 
                                                 "Multi-tasks"))
st.write(task_name)

# model_name = st.sidebar.selectbox("Select Model", ("Linear regression", 
#                                                    "Random forest", 
#                                                    "Voting regression(Recommend)"))
# st.write(model_name)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


SMILES = st.text_input("Enter SMILES")

# 選擇特定任務的模型
def add_model(task_name):
    if task_name=="Calu-3 permeability":
        model_name = st.sidebar.selectbox("Select Model", ("Linear regression", 
                                                           "Random forest", 
                                                           "Voting regression"))
        st.write(model_name)
    else:
        return st.write("No tasks")

    return model_name
model_name = add_model(task_name)

def get_model(model_name):
    if model_name == "Voting regression":
        clf = joblib.load("VotingRegression.pkl")
    else:
        return st.write("No model")

    return clf 

def variance_threshold_selector(data, threshold=0.8*(1-0.8)):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]



if st.button("Submit"): 
    clf = get_model(model_name)
    descriptors = from_smiles(SMILES)
    df_features = pd.DataFrame.from_dict(descriptors, orient='index').T

    lung_x = variance_threshold_selector(lung_X)
    lung_x_all = lung_x[["AATS7i","ATSC5v","AATSC4m","AATSC0v","CrippenLogP",
                         "minHBint3","minHBint8","maxaasC","ZMIC4","MDEO-11","VE3_D"]]

    
    stdsc = StandardScaler().fit(lung_x_all) 

    df_features_lowv = df_features[lung_x_all.columns]
    df_std_feature = stdsc.transform(df_features_lowv)

    y_pred = float(clf.predict(df_std_feature))

    st.write(df_features_lowv)

    st.write(f'Predicted logPapp = {y_pred}')
    papp = 10**y_pred 
    st.write(f'Predicted Papp = {papp} * 10^-6 cm/s')
