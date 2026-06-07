import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Loading models
label_encoder = pickle.load(open(r"le2_clothing_model.pkl", 'rb'))
classifier = pickle.load(open(r"random_forest_classifier_model.pkl", 'rb'))
regressor = pickle.load(open(r"best_regression_model.pkl",'rb'))
regressor_scaler = pickle.load(open(r"regression_standard_scaler.pkl", 'rb'))
classifier_scaler = pickle.load(open(r"classification_standard_scaler.pkl", 'rb'))

# Set page configuration and add title
st.set_page_config(layout="wide")
st.title("🛒  E-Commerce Clickstream Prediction App 🛍️ ")
# Add custom CSS for background color
st.sidebar.header("Upload Data or Enter Manually")
# File upload and model type
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV File", type=['csv'])
model_type = st.sidebar.radio("Select Model Type", ("Regression", "Classification"))
# Feature lists
regression_features = ['page1_main_category', 'page2_clothing_model', 'colour']
classify_features = ['page1_main_category', 'page2_clothing_model', 'colour', 'order', 'price','page', 'location', 'model_photography']

# Main category and clothing models
main_dict = {'Trousers': 1, 'Skirts': 2, 'Blouses': 3, 'Shirts': 4}
Trousers = ['A15', 'A2', 'A39', 'A1', 'A9', 'A7', 'A29', 'A11', 'A12', 'A17', 'A5', 'A4', 'A22', 'A18', 'A25', 'A27', 'A8', 'A30', 'A6', 'A35', 'A10', 'A3', 'A33', 'A14', 'A34', 'A20', 'A32', 'A13', 'A21', 'A28', 'A41', 'A31', 'A24', 'A40', 'A42', 'A37', 'A23', 'A16', 'A26', 'A43', 'A36', 'A38', 'A19']
Skirts = ['B24', 'B32', 'B3', 'B10', 'B13', 'B19', 'B12', 'B16', 'B4', 'B11', 'B8', 'B2', 'B15', 'B9', 'B21', 'B31', 'B1', 'B7', 'B14', 'B26', 'B23', 'B17', 'B20', 'B27', 'B28', 'B25', 'B22', 'B30', 'B6', 'B33', 'B34', 'B29', 'B5']
Blouses = ['C11', 'C35', 'C14', 'C15', 'C57', 'C40', 'C33', 'C44', 'C9', 'C50', 'C55', 'C51', 'C21', 'C17', 'C34', 'C43', 'C22', 'C54', 'C5', 'C12', 'C6', 'C2', 'C46', 'C53', 'C25', 'C29', 'C26', 'C36', 'C48', 'C10', 'C20', 'C13', 'C30', 'C18', 'C23', 'C1', 'C19', 'C56', 'C39', 'C59', 'C7', 'C45', 'C37', 'C28', 'C4', 'C27', 'C42', 'C32', 'C41', 'C16', 'C52', 'C31', 'C3', 'C38', 'C8', 'C24', 'C49', 'C58', 'C47']
Shirts = ['P48', 'P23', 'P51', 'P33', 'P11', 'P7', 'P82', 'P1', 'P56', 'P62', 'P12', 'P43', 'P16', 'P4', 'P49', 'P3', 'P6', 'P47', 'P61', 'P17', 'P15', 'P57', 'P64', 'P20', 'P65', 'P37', 'P19', 'P5', 'P63', 'P74', 'P69', 'P39', 'P60', 'P40', 'P44', 'P21', 'P46', 'P38', 'P14', 'P52', 'P26', 'P41', 'P18', 'P68', 'P29', 'P2', 'P34', 'P36', 'P70', 'P72', 'P77', 'P50', 'P32', 'P78', 'P45', 'P53', 'P42', 'P58', 'P35', 'P30', 'P8', 'P10', 'P67', 'P59', 'P25', 'P9', 'P80', 'P75', 'P13', 'P71', 'P73', 'P76', 'P27', 'P55', 'P24', 'P81', 'P31', 'P66', 'P28']

# Color mapping
colors = {
    "Beige": "#F5F5DC", "Black": "#000000", "Blue": "#0000FF", "Brown": "#A52A2A", 
    "Burgundy": "#800020", "Gray": "#808080", "Green": "#008000", "Navy": "#000080", 
    "Purple": "#800080", "Olive": "#808000", "Pink": "#FFC0CB", "Red": "#FF0000", 
    "Violet": "#8A2BE2", "White": "#FFFFFF"
}

colour_dict = {'Beige': 1, 'Black': 2, 'Blue': 3, 'Brown': 4, 'Burgundy': 5, 'Gray': 6, 'Green': 7, 'Navy': 8, 'Purple': 9, 'Olive': 10, 'Pink': 11, 'Red': 12, 'Violet': 13, 'White': 14}
location_dict = {
    'top left': 1, 'top in the middle': 2, 'top right': 3,
    'bottom left': 4, 'bottom in the middle': 5, 'bottom right': 6,
    'Top Left': 1, 'Top in the Middle': 2, 'Top Right': 3,
    'Bottom Left': 4, 'Bottom in the Middle': 5, 'Bottom Right': 6
}
model_photography_dict = {
    'en-face': 1, 'profile': 2, 'en face': 1,
    'En Face': 1, 'Profile': 2
}

# Normalize uploaded CSV feature values to model input encodings
def normalize_uploaded_features(df):
    df = df.copy()
    if 'page1_main_category' in df.columns:
        df['page1_main_category'] = df['page1_main_category'].map(main_dict).fillna(df['page1_main_category'])
        df['page1_main_category'] = pd.to_numeric(df['page1_main_category'], errors='ignore')

    if 'colour' in df.columns:
        df['colour'] = df['colour'].map(colour_dict).fillna(df['colour'])
        df['colour'] = pd.to_numeric(df['colour'], errors='ignore')

    if 'location' in df.columns:
        df['location'] = df['location'].astype(str).map(location_dict).fillna(df['location'])
        df['location'] = pd.to_numeric(df['location'], errors='ignore')

    if 'model_photography' in df.columns:
        df['model_photography'] = df['model_photography'].astype(str).map(model_photography_dict).fillna(df['model_photography'])
        df['model_photography'] = pd.to_numeric(df['model_photography'], errors='ignore')

    return df

# Function for selecting clothing options
def first3():
    page1_main_category = st.selectbox("Main Category", ['Trousers', 'Skirts', 'Blouses', 'Shirts'])
    page1_main_category = main_dict[page1_main_category]
        
    if page1_main_category == 1:
        page2_clothing_model = st.selectbox("Model", Trousers)
    elif page1_main_category == 2:
        page2_clothing_model = st.selectbox("Model", Skirts)    
    elif page1_main_category == 3:
        page2_clothing_model = st.selectbox("Model", Blouses)
    else:
        page2_clothing_model = st.selectbox("Model", Shirts)
        
    page2_clothing_model = label_encoder.transform([page2_clothing_model])
    selected_color = st.selectbox("Color", list(colors.keys()))
    hex_code = colors[selected_color]
    st.markdown(f"<div style='width:100px; height:50px; background-color:{hex_code}; border:1px solid #000'></div>", unsafe_allow_html=True)
    colour_dict = {'Beige': 1, 'Black': 2, 'Blue': 3, 'Brown': 4, 'Burgundy': 5, 'Gray': 6, 'Green': 7, 'Navy': 8, 'Purple': 9, 'Olive': 10, 'Pink': 11, 'Red': 12, 'Violet': 13, 'White': 14}
    colour = colour_dict[selected_color]

    return page1_main_category, page2_clothing_model[0], colour

# Model predictions for regression and classification
if model_type == "Regression":
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.title("Regression - Price Prediction")
        st.markdown("This section uses a regression model to predict the price of a clothing item.")
        st.dataframe(df[regression_features + ['price']], width=1000, height=400)   
        row_index = st.sidebar.selectbox("Select a Row for Prediction", df.index)
        selected_row = df.loc[[row_index], regression_features].copy()
        selected_row = normalize_uploaded_features(selected_row)
        selected_row['page2_clothing_model'] = label_encoder.transform(selected_row['page2_clothing_model'])
        selected_row = regressor_scaler.transform(selected_row)
    else:
        st.title("Regression - Price Prediction")
        st.header("✨ Enter Data Manually Below ✨")
        st.markdown("This section uses a regression model to predict the price of a clothing item.")
        page1_main_category, page2_clothing_model, colour = first3()
        selected_row = [page1_main_category, page2_clothing_model, colour]
        selected_row = np.array(selected_row).reshape(1, -1)
        selected_row = regressor_scaler.transform(selected_row)
        
    if st.button("✨ Predict Price ✨"):
        with st.spinner('Processing...'):
            predicted = regressor.predict(selected_row)
            st.subheader("Predicted Price")
            st.markdown(f"<h3 style='color:green;'>${predicted[0]:.2f}</h3>", unsafe_allow_html=True)
            st.balloons() # Celebration animation
        
elif model_type == "Classification":
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.title("Classification - Purchase Prediction")
        st.markdown("""
    This section uses a classification model to predict the probability of a user purchasing an item from a clothing category.
    """)
        st.dataframe(df[classify_features + ['price_2']], width=1000, height=400)        
        row_index = st.sidebar.selectbox("Select a Row for Prediction", df.index)
        selected_row = df.loc[[row_index], classify_features].copy()
        selected_row = normalize_uploaded_features(selected_row)
        selected_row['page2_clothing_model'] = label_encoder.transform(selected_row['page2_clothing_model'])
        selected_row = classifier_scaler.transform(selected_row)
    else:
        st.title("Classification - Purchase Prediction")
        st.markdown("""
    This section uses a classification model to predict the probability of a user purchasing an item from a clothing category.
    """)
        page1_main_category, page2_clothing_model, colour = first3()

        order = st.slider("Order", 1, 100)
        price = st.slider("Price", 0, 100)
        page = st.number_input("Page",min_value=1,max_value=5)
        location = st.selectbox("Product Location",['Top Left', 'Top in the Middle', 'Top Right', 'Bottom Left', 'Bottom in the Middle', 'Bottom Right'])
        location_dict = {'Top Left': 1, 'Top in the Middle': 2, 'Top Right': 3, 'Bottom Left': 4, 'Bottom in the Middle': 5, 'Bottom Right': 6}
        location = location_dict[location]
        model_photography = st.selectbox("Model Photography",['En Face', 'Profile'])
        model_photography_dict = {'En Face': 1, 'Profile': 2}
        model_photography = model_photography_dict[model_photography]
        
        selected_row = [page1_main_category, page2_clothing_model, colour, order, price, page , location, model_photography]
        selected_row = np.array(selected_row).reshape(1, -1)
        selected_row = classifier_scaler.transform(selected_row)

    if st.button("🚀 Classify Purchase 🚀"):
        with st.spinner('Classifying...'):
            predicted = classifier.predict(selected_row)
            st.subheader("Purchase Classification")
            if predicted[0] == 2:
                st.markdown("<h3 style='color:green;'>Going To Buy</h3>", unsafe_allow_html=True)
                st.balloons() # Celebration animation
            else:
                st.markdown("<h3 style='color:red;'>Not Going To Buy</h3>", unsafe_allow_html=True)
                st.snow() # Celebration animation
