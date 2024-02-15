import streamlit as st
#import pickle
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import time

# Charger le modèle pré-entraîné
model_filename = "classifier_model-0.1.0.pkl"
# with open(model_filename, "rb") as f:
#     model = pickle.load(f)

model = joblib.load(model_filename)

# Définir les catégories pour le diagramme
categories = ['Pas de défaut de paiement', 'Défaut de paiement']

# Fonction pour faire des prédictions
def make_prediction(features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    probability = np.round(probability * 100, 2)
    return prediction, probability

st.set_page_config(page_title="ML | Prédiction de Défaut de Paiement")

# Interface utilisateur de l'application
st.title("Application de Prédiction de Défaut de Paiement")

st.write("Cette application utilise un modèle de Machine Learning pour prédire si un client sera en \
         défaut de paiement ou non en fonction des caractéristiques fournies.")

st.warning("Renseignez les informations du client puis appuyez sur :green[Prédire] pour voir le résultat de la prédiction.")

st.sidebar.header("Informations sur le client")

# Saisie des caractéristiques du client
limit_bal = st.sidebar.number_input("Montant du crédit", min_value=0, value=50000)
sex = st.sidebar.selectbox("Sexe du client", ['Female', 'Male'])
education = st.sidebar.selectbox("Niveau d'éducation du client", ['Graduate school', 'University', 'High school', 'Others'])
marriage = st.sidebar.selectbox("Statut matrimonial du client", ['Single', 'Married', 'Others'])
age = st.sidebar.number_input("Âge du client", min_value=18, max_value=100, value=30)

# Saisie des statuts de paiement
payment_status_sep = st.sidebar.selectbox("Statut de paiement en septembre", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_aug = st.sidebar.selectbox("Statut de paiement en août", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_jul = st.sidebar.selectbox("Statut de paiement en juillet", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_jun = st.sidebar.selectbox("Statut de paiement en juin", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_may = st.sidebar.selectbox("Statut de paiement en mai", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)
payment_status_apr = st.sidebar.selectbox("Statut de paiement en avril", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=2)

# Saisie des relevés de facturation
bill_statement_sep = st.sidebar.number_input("Relevé de facturation en septembre", min_value=0, value=0)
bill_statement_aug = st.sidebar.number_input("Relevé de facturation en août", min_value=0, value=0)
bill_statement_jul = st.sidebar.number_input("Relevé de facturation en juillet", min_value=0, value=0)
bill_statement_jun = st.sidebar.number_input("Relevé de facturation en juin", min_value=0, value=0)
bill_statement_may = st.sidebar.number_input("Relevé de facturation en mai", min_value=0, value=0)
bill_statement_apr = st.sidebar.number_input("Relevé de facturation en avril", min_value=0, value=0)

# Saisie des paiements précédents
previous_payment_sep = st.sidebar.number_input("Paiement précédent en septembre", min_value=0, value=0)
previous_payment_aug = st.sidebar.number_input("Paiement précédent en août", min_value=0, value=0)
previous_payment_jul = st.sidebar.number_input("Paiement précédent en juillet", min_value=0, value=0)
previous_payment_jun = st.sidebar.number_input("Paiement précédent en juin", min_value=0, value=0)
previous_payment_may = st.sidebar.number_input("Paiement précédent en mai", min_value=0, value=0)
previous_payment_apr = st.sidebar.number_input("Paiement précédent en avril", min_value=0, value=0)

# Créer un DataFrame à partir des caractéristiques
input_data = pd.DataFrame({
    'limit_bal': [limit_bal],
    'sex': [sex],
    'education': [education],
    'marriage': [marriage],
    'age': [age],
    'payment_status_sep': [payment_status_sep],
    'payment_status_aug': [payment_status_aug],
    'payment_status_jul': [payment_status_jul],
    'payment_status_jun': [payment_status_jun],
    'payment_status_may': [payment_status_may],
    'payment_status_apr': [payment_status_apr],
    'bill_statement_sep': [bill_statement_sep],
    'bill_statement_aug': [bill_statement_aug],
    'bill_statement_jul': [bill_statement_jul],
    'bill_statement_jun': [bill_statement_jun],
    'bill_statement_may': [bill_statement_may],
    'bill_statement_apr': [bill_statement_apr],
    'previous_payment_sep': [previous_payment_sep],
    'previous_payment_aug': [previous_payment_aug],
    'previous_payment_jul': [previous_payment_jul],
    'previous_payment_jun': [previous_payment_jun],
    'previous_payment_may': [previous_payment_may],
    'previous_payment_apr': [previous_payment_apr]
})

# Prédiction
if st.sidebar.button("Prédire"):
    progress_bar = st.progress(0)
    prediction, probability = make_prediction(input_data)

    # bar de progression
    for pct_progress in range(100):
        time.sleep(0.01)
        progress_bar.progress(pct_progress+1)

    if pct_progress+1 == 100 :
        st.success("Prédiction effectuée avec succes !!!")

    st.subheader("Probabilités :")
    prob_df = pd.DataFrame({'Catégories': categories, 'Probabilité': probability[0]})
    fig = px.bar(prob_df, x='Catégories', y='Probabilité', text='Probabilité', labels={'Probabilité': 'Probabilité (%)'})
    st.plotly_chart(fig)

    st.subheader("Résultat de la prédiction :")
    if prediction[0] == 1:
        st.error("Le client sera en défaut de paiement.")
    else:
        st.success("Le client ne sera pas en défaut de paiement.")