import streamlit as st

from web_functions import predict
from sklearn.tree import export_graphviz


def app(df, x, y):

    st.title("Halaman Prediksi")

    col1, col2 = st.columns(2)
    with col1:
        SepalLengthCm = st.number_input("input nilai SepalLengthCm")
    with col1:
        SepalWidthCm = st.number_input("input nilai SepalWidthCm")
    with col2:
        PetalLengthCm = st.number_input("input nilai thalachh")
    with col2:
        PetalWidthCm = st.number_input("input nilai exng")

    features = [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]

    # tombol prediksi
    if st.button("Prediksi"):
        prediction, score = predict(x, y, features)
        score = score
        st.info("Prediksi Sukses...")

        if (prediction == 1):
            st.warning("Accuracy Sebelumnya")
        else:
            st.success("Accuracy Sukses")

        st.write("Model yang digunakan memiliki tingkat akurasi",
                 (score*100), "%")
