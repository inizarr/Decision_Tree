import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import confusion_matrix
from web_functions import train_model
import streamlit as st
from sklearn.tree import export_graphviz


def plot_confusion_matrix_heatmap(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot()


def app(df, x, y):

    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Halaman Visualisasi Prediksi IRIS")

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x, y)
        y_pred = model.predict(x)
        plot_confusion_matrix_heatmap(y, y_pred)
        st.pyplot()

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x, y)
        dot_data = tree.export_graphviz(decision_tree=model, max_depth=4, out_file=None, filled=True, rounded=True,
                                        feature_names=x.columns,
                                        class_names=['1', '2', '3', '4'],
                                        special_characters=True
                                        )

        st.graphviz_chart(dot_data)
