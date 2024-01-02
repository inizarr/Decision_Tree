import streamlit as st
from web_functions import load_data
from sklearn.tree import DecisionTreeClassifier
from Tabs import home, predict, visualise
from sklearn.tree import export_graphviz


Tabs = {
    "Home": home,
    "Prediction": predict,
    "Visualisation": visualise
}

# membuat sidebar
st.sidebar.title("Navigasi")

# membuat radio option
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# load dataset
df, x, y = load_data()

# kondisi call app function
if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(df, x, y)
else:
    Tabs[page].app()
