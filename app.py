import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load saved models
pha_model = joblib.load('models/pha_model.pkl')
diameter_model = joblib.load('models/diameter_model.pkl')
kmeans = joblib.load('models/kmeans_model.pkl')
cluster_scaler = joblib.load('models/cluster_scaler.pkl')
cluster_names = joblib.load('models/cluster_names.pkl')
cluster_sample = pd.read_csv('models/cluster_sample.csv')

# Test RMSE of the diameter model on log(diameter), from 4.Regression.ipynb
DIAMETER_RMSE = 0.1058

st.set_page_config(page_title='Planetary Defense Analytics', layout='wide')
st.title('Planetary Defense Analytics')
st.write('Enter the orbital and physical values of an asteroid. The range in brackets is what is possible in nature.')

# Default input values
DEFAULTS = {'a': 1.5, 'e': 0.4, 'i': 8.0, 'H': 20.0, 'albedo': 0.15, 'moid': 0.03}

# Put the default values in memory the first time the app opens
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Function used by the reset button
def reset_inputs():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


# Inputs from the user (natural min - max in brackets)
st.sidebar.header('Asteroid inputs')
st.sidebar.button('Reset to default values', on_click=reset_inputs)

a = st.sidebar.number_input('Semi-major axis a in AU (0.1 - 1000)', min_value=0.1, max_value=1000.0, key='a')
e = st.sidebar.number_input('Eccentricity e (0 - 0.99)', min_value=0.0, max_value=0.99, key='e')
i = st.sidebar.number_input('Inclination i in degrees (0 - 180)', min_value=0.0, max_value=180.0, key='i')
H = st.sidebar.number_input('Absolute magnitude H (-2 - 35)', min_value=-2.0, max_value=35.0, key='H')
albedo = st.sidebar.number_input('Albedo (0.01 - 1.0)', min_value=0.01, max_value=1.0, key='albedo')
moid = st.sidebar.number_input('MOID in AU (0 - 1000), used only for clustering', min_value=0.0, max_value=1000.0, key='moid')

# Values calculated from inputs
q = a * (1 - e)
per = 365.25 * a ** 1.5
moid_ld = moid * 389.17

st.sidebar.write('Perihelion q =', round(q, 3), 'AU')
st.sidebar.write('Period =', round(per, 1), 'days')


# Function to get SHAP values for one row
def get_shap(pipe, row, display_row):
    scaled = pd.DataFrame(pipe.named_steps['scaler'].transform(row), columns=row.columns)
    model = pipe.named_steps['model']
    if hasattr(model, 'coef_'):
        explainer = shap.LinearExplainer(model, np.zeros((1, row.shape[1])))
    else:
        explainer = shap.TreeExplainer(model)
    values = explainer(scaled)
    if len(values.values.shape) == 3:
        values = values[:, :, 1]
    values.data = display_row.values
    return values[0]


# Function to draw the SHAP waterfall
def draw_waterfall(sv):
    fig = plt.figure()
    shap.plots.waterfall(sv, show=False)
    st.pyplot(fig, clear_figure=True)


tab1, tab2, tab3 = st.tabs(['PHA Classification', 'Diameter Prediction', 'Cluster'])

# Tab 1 - PHA
with tab1:
    pha_row = pd.DataFrame({'H': [H], 'e': [e], 'a': [np.log1p(a)], 'i': [i]})
    pha_show = pd.DataFrame({'H': [H], 'e': [e], 'a': [a], 'i': [i]})

    prob = pha_model.predict_proba(pha_row)[0][1]
    pred = pha_model.predict(pha_row)[0]

    if pred == 1:
        st.error('Prediction: Potentially Hazardous Asteroid (PHA)')
    else:
        st.success('Prediction: Not a Potentially Hazardous Asteroid')

    st.metric('PHA probability', str(round(prob * 100, 2)) + ' %')
    st.caption('The model says PHA when this is 50% or more. It is the model\'s score, not real-world odds, '
               'because the model was trained on balanced (SMOTE) data.')

    if a > 100 or H < 3 or H > 33:
        st.warning('Some inputs are outside the range the model was trained on, so this result is less reliable.')

    st.subheader('Why? (SHAP)')
    sv = get_shap(pha_model, pha_row, pha_show)
    draw_waterfall(sv)

    st.subheader('Inference')
    order = np.argsort(-np.abs(sv.values))
    for n in order:
        name = pha_show.columns[n]
        value = pha_show.iloc[0, n]
        effect = sv.values[n]
        if effect > 0:
            st.write('- **' + name + ' = ' + str(round(value, 3)) + '** pushed the result **towards PHA** (SHAP +' + str(round(effect, 2)) + ')')
        else:
            st.write('- **' + name + ' = ' + str(round(value, 3)) + '** pushed the result **away from PHA** (SHAP ' + str(round(effect, 2)) + ')')
    top = pha_show.columns[order[0]]
    st.info('The biggest reason for this prediction is **' + top + '**. Perihelion q = ' + str(round(q, 3)) +
            ' AU (PHAs need q below about 1.05 AU) and H = ' + str(H) + ' (PHAs need H of 22 or less).')

# Tab 2 - Diameter
with tab2:
    dia_row = pd.DataFrame({'H': [H], 'albedo': [np.log(albedo)], 'a': [np.log1p(a)], 'i': [i]})
    dia_show = pd.DataFrame({'H': [H], 'albedo': [albedo], 'a': [a], 'i': [i]})

    log_dia = diameter_model.predict(dia_row)[0]
    diameter = np.exp(log_dia)

    # 95% range: prediction times / divided by exp(1.96 x RMSE)
    factor = np.exp(1.96 * DIAMETER_RMSE)
    low = diameter / factor
    high = diameter * factor

    col1, col2 = st.columns(2)
    col1.metric('Predicted diameter', str(round(diameter, 3)) + ' km')
    col2.metric('95% confidence range', str(round(low, 3)) + ' - ' + str(round(high, 3)) + ' km')
    st.write('The real diameter is expected to be within about +/- ' + str(round((factor - 1) * 100)) + '% of the prediction.')

    # Physics formula as a check: D = 1329 / sqrt(albedo) x 10^(-H/5)
    physics = 1329 / np.sqrt(albedo) * 10 ** (-H / 5)
    st.write('Physics formula check: D = 1329 / sqrt(albedo) x 10^(-H/5) = **' + str(round(physics, 3)) + ' km**')

    if H < 3 or H > 27.5 or a > 100:
        st.warning('H or a is outside the range of asteroids with measured diameters (H 3 - 27.5), so this result is less reliable.')
    if physics / diameter > 1.5:
        st.warning('The physics formula gives a much bigger size than the model. The model cannot predict sizes larger than the '
                   'asteroids it was trained on (very large asteroids were removed as outliers), so trust the physics formula here.')
    elif diameter / physics > 1.5:
        st.warning('The model gives a much bigger size than the physics formula. Small, faint asteroids (H above about 18) are rare '
                   'in the training data, so the model is less accurate for them.')

    st.subheader('Why? (SHAP, in log diameter)')
    sv = get_shap(diameter_model, dia_row, dia_show)
    draw_waterfall(sv)

    st.subheader('Inference')
    order = np.argsort(-np.abs(sv.values))
    for n in order:
        name = dia_show.columns[n]
        value = dia_show.iloc[0, n]
        times = np.exp(sv.values[n])
        if times >= 1:
            st.write('- **' + name + ' = ' + str(round(value, 3)) + '** made the diameter **' + str(round(times, 2)) + ' times bigger** than average')
        else:
            st.write('- **' + name + ' = ' + str(round(value, 3)) + '** made the diameter **' + str(round(1 / times, 2)) + ' times smaller** than average')
    st.info('A low H (bright) means a big asteroid. For the same H, a low albedo (dark surface) also means a big asteroid, '
            'because it must be large to reflect that much light.')

# Tab 3 - Cluster
with tab3:
    cl_row = pd.DataFrame({
        'a': [np.log1p(a)],
        'e': [e],
        'i': [i],
        'per': [np.log1p(per)],
        'moid_ld': [np.log1p(moid_ld)]
    })
    scaled_row = cluster_scaler.transform(cl_row)
    cluster = kmeans.predict(scaled_row)[0]

    # Confidence: how much closer the asteroid is to its own centre than to the next nearest centre
    distances = np.sort(kmeans.transform(scaled_row)[0])
    confidence = 1 - distances[0] / distances[1]

    col1, col2 = st.columns(2)
    col1.metric('Cluster', str(cluster) + ' - ' + cluster_names[cluster])
    col2.metric('Confidence', str(round(confidence * 100, 2)) + ' %')
    st.write('Confidence compares the distance to this group\'s centre with the distance to the next nearest group. '
             'Close to 100% means clearly inside the group; close to 0% means on the border between two groups.')

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in sorted(cluster_sample['cluster'].unique()):
        part = cluster_sample[cluster_sample['cluster'] == c]
        ax.scatter(np.log10(part['a']), part['e'], s=3, label=cluster_names[c])
    ax.scatter(np.log10(a), e, color='red', marker='*', s=400, label='Your asteroid')
    ax.set_xlabel('log10(a)')
    ax.set_ylabel('e')
    ax.legend()
    st.pyplot(fig)

    st.subheader('Inference')
    if cluster_names[cluster] == 'Near-Earth group':
        st.info('This orbit is in the Near-Earth group: small a and high e bring it close to Earth\'s orbit. All known PHAs are in this group.')
    elif cluster_names[cluster] == 'Main belt group':
        st.info('This orbit is in the Main belt group: between Mars and Jupiter with a low e, so it stays far from Earth.')
    else:
        st.info('This orbit is in the Outer solar system group: far beyond Jupiter (TNOs and Centaurs), no risk to Earth.')
