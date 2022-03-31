import streamlit as st
import pandas as pd
import numpy as np
import pickle


from sklearn.ensemble import RandomForestClassifier

#Css class for styling the app

st.markdown("""
<style>
.app-header {
    font-size:50px;
    color: #F63366;
    font-weight: 700;
}
.sidebar-header{
    font-family: "Lucida Sans Unicode", "Lucida Grande", sans-serif;
    font-size: 28px;
    letter-spacing: -1.2px;
    word-spacing: 2px;
    color: #FFFFFF;
    font-weight: 700;
    text-decoration: none;
    font-style: normal;
    font-variant: normal;
    text-transform: capitalize;
}
.positive {
    color: #F63366;
    font-size:30px;
    font-weight: 700;  
}
.negative {
    color: #008000;
    font-size:30px;
    font-weight: 700;  
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="app-header">Dementia Prediction app</p>', unsafe_allow_html=True)

expander = st.expander("Insight to the Project")
expander.write("""
    Original source of the data we used for the development of this website
""")

expander.write("""


This app predicts the **Dementia** of patients!

Data obtained from [Kaggle Website](https://www.kaggle.com/sid321axn/eda-for-predicting-dementia/data)
""")


# st.set_page_config(layout="wide")

#Css Styling for the app



st.sidebar.markdown('<p class="sidebar-header">Input the patient details</p>', unsafe_allow_html=True)

# st.sidebar.markdown("""
# [Example CSV input file](https://gist.githubusercontent.com/PUUDI/861771ffca8462507b487b6f75f2386d/raw/44e4760f1f6ee628c9674fe1c87e63bd4fbcf19d/gistfile1.txt)
# """)

# Collects user input features into dataframe
# uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if 1>2:
    a=5
    # input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        
        sex = st.sidebar.selectbox('Sex',('Male','Female'))
        Age = st.sidebar.slider('Age', min_value =0,max_value =100,step =1)
        EDUC = st.sidebar.slider('Educational Years', min_value = 6,max_value = 23,step = 1)
        SES = st.sidebar.slider('SES',min_value =1,max_value =5,step =1)
        MMSE = st.sidebar.slider('MMSE', min_value =4,max_value =30,step =1)
        CDR = st.sidebar.slider('Time since last visit', min_value =0,max_value =30,step =1)
        eTIV = st.sidebar.slider('eTIV', min_value =1100,max_value =2010,step =1)
        nWBV = st.sidebar.slider('MMSE', min_value =0.5,max_value =0.85,step =0.01)
        ASF = st.sidebar.slider('ASF', min_value =0.875,max_value =1.6,step =0.01)
        data = {'M/F': sex,
                'Age': Age,
                'EDUC': EDUC,
                'SES': SES,
                'MMSE': MMSE,
                'eTIV':eTIV,
                'nWBV':nWBV,
                'ASF':ASF}
        features = pd.DataFrame(data, index=[0])
        return (features,sex,Age,EDUC,SES , MMSE, eTIV, nWBV, ASF, CDR)
    input_df, sex,Age,EDUC,SES , MMSE, eTIV, nWBV, ASF, CDR = user_input_features()
 

input_df['M/F'] = input_df['M/F'].apply(lambda x: ['Male', 'Female'].index(x))


# Displays the user input features
st.subheader('Predictor variables')


# if uploaded_file is not None:
#     st.write(input_df)

    # st.write('Predictor variables')
    # st.write(input_df)
col1, col2, col3, col4, col5 = st.columns(5)
    
col1.metric("Sex", sex)
col2.metric("Age", Age)
col3.metric("Edcation", EDUC)
col4.metric("Social Status", SES)
col5.metric("Time since last visit", CDR)

col6, col7, col8, col9, col10= st.columns(5)

col6.metric("MMSE", MMSE)
col7.metric("eTIV", eTIV)
col8.metric("Brain volume", nWBV)
col9.metric("ASF", ASF)


st.markdown("""---""")

# Reads in saved classification model
load_clf = pickle.load(open('dementia_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

#Creating the columns
col1,col2 = st.columns(2)

c1 = col1.container()
c2 = col2.container()


url = "https://nimh.health.gov.lk/en/"
# st.write("National Institute of Mental Health [link](%s)" % url)

# st.markdown("check out this [link](%s)" % url)

c1.subheader('Prediction')
penguins_species = np.array(['Negative','Positive'])
if prediction[0] == 0:
    
    c1.markdown('<p class="negative">The Patient is Negative</p>', unsafe_allow_html=True)
else:
   
    c1.markdown('<p class="positive">The Patient is Positive with Dementia</p>' , unsafe_allow_html=True)
    c1.write("[National Institute of Mental Health](%s)" % url)




c2.subheader('Prediction Probability')
# c2.write(prediction_proba)

c2.metric(label="Risk of Dementia", value=round(prediction_proba[0][1],2))


# go.Indicator(
#             mode="gauge+number+delta",
#             value=metrics["test_accuracy"],
#             title={"text": f"Accuracy (test)"},
#             domain={"x": [0, 1], "y": [0, 1]},
#             gauge={"axis": {"range": [0, 1]}},
#             delta={"reference": metrics["train_accuracy"]},
#         )
