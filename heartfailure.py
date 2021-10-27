import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from sklearn.preprocessing import LabelEncoder


labelencoder = LabelEncoder()

#data pre-processing
data_import = pd.read_csv("heart.csv")
data_import["Sex"] = labelencoder.fit_transform(data_import["Sex"])

unhealthy_data = data_import[data_import["HeartDisease"]==1]
unhealthy_data = unhealthy_data[unhealthy_data["Cholesterol"] > 0]
male_data = unhealthy_data[unhealthy_data["Sex"]==1]
female_data = unhealthy_data[unhealthy_data["Sex"]==0]


#streamlit code
st.title("Heart Failure Prediction")


#sidebar menu
option = st.sidebar.selectbox(
    "Menu:",
    ["Homepage", "Predict by Resting Blood Pressure", "Predict by Cholesterol Serum", "Predict by Chest Pain Type", "Download Data"]
)


if option == "Homepage":
    st.write("Cardiovascular diseases (CVDs) are the number 1 cause of death globally,"
             " taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. ")
    st.write("People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of "
             "one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) "
             "need early detection and management wherein a machine learning model can be of great help.")


    st.header("Overview")
    age_male = male_data["Age"]
    age_female = female_data["Age"]

    fig1, ax1 = plt.subplots()
    colors = ["#94D2F8", "#FE93BA"]
    labels = ["Male", "Female"]
    ax1.hist([age_male, age_female], color=colors, label=labels)

    ax1.legend()
    ax1.set_ylabel("Number of patients")
    ax1.set_xlabel("Age")
    ax1.set_title("Number of patients diagnosed with heart disease")

    st.write(fig1)


elif option == "Predict by Resting Blood Pressure":

    st.header("Resting Blood Pressure")
    age_restingBP = pd.read_csv("heart.csv")
    age_restingBP = age_restingBP[["Age", "RestingBP", "Sex"]]
    c = alt.Chart(age_restingBP).mark_circle(size=60).encode(
        x="Age",
        y="RestingBP",
        color="Sex",
        tooltip=["Age", "RestingBP", "Sex"]
    ).interactive()

    st.altair_chart(c, use_container_width=True)


elif option == "Predict by Cholesterol Serum":

    st.header("Cholesterol Serum")

    #selection of data
    age_data = unhealthy_data["Age"]
    cholesterol_data = unhealthy_data["Cholesterol"]

    #create graphic
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.scatter(age_data, cholesterol_data, color="#FEA293")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Cholesterol")
    ax2.set_title("Cholesterol serum and age")

    st.write(fig2)


elif option == "Predict by Chest Pain Type":

    st.header("Chest Pain Type")

    chest_pain = unhealthy_data["ChestPainType"]
    ASY_size = chest_pain[chest_pain == "ASY"].size
    ATA_size = chest_pain[chest_pain == "ATA"].size
    NAP_size = chest_pain[chest_pain == "NAP"].size
    TA_size = chest_pain[chest_pain == "TA"].size

    labels = ["Aysmptomatic", "Typical Angina", "Atypical Angina", "Non-Anginal Pain"]
    sizes = [ASY_size, TA_size, ATA_size, NAP_size]
    explode = (0, 0.1, 0.1, 0.1)

    fig3, ax3 = plt.subplots()
    ax3.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False)
    ax3.axis('equal')
    ax3.set_title("Percentage of patients with reported chest pain")

    st.write(fig3)


else:

    st.header("Dowload")
    st.write("You can download the data used in this analysis.")

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(data_import)

    st.download_button(
        label="Download data as CSV",
        data = csv,
        file_name="heartdiseasedata.csv",
        mime='text/csv'
    )
