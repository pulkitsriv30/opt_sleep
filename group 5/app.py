import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
import io
from datetime import datetime
import base64
import os
import seaborn as sns
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Sleep Health Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
# Custom CSS for improved UI with better contrast and visibility
st.markdown("""
<style>
    /* Base styles and dark mode compatibility */
    .main-header {
        font-size: 2.5rem;
        color: #3B82F6;
        text-align: center;
        padding: 1.5rem 0;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        padding: 0.8rem 0;
        font-weight: 600;
    }
    
    /* Card improvements */
    .card {
        border-radius: 10px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        background: #1E293B;
        color: #E5E7EB;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .metric-card {
        text-align: center;
        padding: 1.2rem;
        border-radius: 10px;
        background: #1E293B;
        color: #E5E7EB;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.2);
        margin-bottom: 1rem;
    }
    
    /* Better metric visibility */
    .metric-value {
        font-size: 2.3rem;
        font-weight: bold;
        color: #60A5FA;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #D1D5DB;
        margin-top: 0.5rem;
    }
    
    /* Status indicators with higher contrast */
    .status-optimal {
        color: #10B981;
        font-weight: bold;
        text-shadow: 0 0 1px rgba(16, 185, 129, 0.2);
    }
    
    .status-below {
        color: #F59E0B;
        font-weight: bold;
        text-shadow: 0 0 1px rgba(245, 158, 11, 0.2);
    }
    
    .status-above {
        color: #EF4444;
        font-weight: bold;
        text-shadow: 0 0 1px rgba(239, 68, 68, 0.2);
    }
    
    /* Improved recommendation items */
    .recommendation-item {
        padding: 1rem;
        margin-bottom: 0.8rem;
        background: rgba(59, 130, 246, 0.15);
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        color: #E5E7EB;
    }
    
    /* Interactive button styling */
    .report-btn {
        background-color: #3B82F6;
        color: white;
        border-radius: 6px;
        padding: 0.7rem 1.2rem;
        font-weight: 600;
        text-align: center;
        cursor: pointer;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        display: inline-block;
    }
    
    .report-btn:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
    }
    
    /* Tab styling improvements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1E293B;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        background-color: #1E293B;
        color: #D1D5DB;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
    }
    
    /* Make text in cards more readable */
    .card h3 {
        color: #93C5FD;
        margin-bottom: 1rem;
    }
    
    .card p, .card li {
        color: #E5E7EB;
    }
    
    /* Make charts readable with better backgrounds */
    div.stPlotlyChart {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    /* Fix white boxes in the sidebar */
    div.css-1kyxreq.e115fcil2, div.css-7oyrr6.e1akgbir11 {
        background-color: #1E293B;
        color: #E5E7EB;
    }
    
    /* Make dropdowns and inputs consistent with theme */
    .stSelectbox > div, .stNumberInput > div, .stSlider > div {
        background-color: #1E293B;
        color: #E5E7EB;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and components
@st.cache_resource
def load_model():
    try:
        with open('models/sleep_prediction_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please make sure the model file exists.")
        return None

model_components = load_model()

# -------------------------------------------
# Helper Functions
# -------------------------------------------
def predict_sleep_duration_advanced(user, pipeline, my_cols, use_hybrid=True):
    """Advanced sleep duration prediction using ML model with domain knowledge adjustments"""
    # Convert user dictionary to DataFrame format for ML prediction
    user_df = pd.DataFrame([user])
    
    # Extract necessary features for ML prediction
    required_features = [col for col in my_cols if col in user_df.columns]
    missing_features = [col for col in my_cols if col not in user_df.columns]
    
    # For any missing features, use reasonable defaults
    for feature in missing_features:
        if feature == 'Sleep Duration':
            user_df[feature] = 7.0  # Use population average
        elif feature in ['Centroid_0', 'Centroid_1', 'Centroid_2', 'Centroid_3', 
                        'Centroid_4', 'Centroid_5', 'Centroid_6', 'Centroid_7', 
                        'Centroid_8', 'Centroid_9']:
            user_df[feature] = 0.5
    
    # Make initial ML prediction using trained pipeline
    try:
        ml_prediction = pipeline.predict(user_df[my_cols])[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 7.0  # Fallback to average sleep duration
    
    # If not using hybrid approach, return ML prediction directly
    if not use_hybrid:
        return round(ml_prediction, 1)
    
    # Domain knowledge adjustments based on individual factors
    
    # Age-based adjustment
    if user['Age'] < 18:
        age_factor = 0.5
    elif user['Age'] < 25:
        age_factor = 0.4
    elif user['Age'] < 35:
        age_factor = 0.2
    elif user['Age'] < 50:
        age_factor = 0.0
    elif user['Age'] < 65:
        age_factor = -0.1
    else:
        age_factor = -0.2
    
    # Physical activity adjustment
    activity_level = user['Physical Activity Level']
    occupation = user['Occupation']
    
    # High physical demand jobs get extra adjustment
    physical_jobs = ['Nurse', 'Doctor', 'Teacher', 'Sales Representative']
    job_activity_bonus = 0.1 if occupation in physical_jobs else 0.0
    
    if activity_level >= 75:
        activity_factor = 0.5 + job_activity_bonus
    elif activity_level >= 50:
        activity_factor = 0.4 + job_activity_bonus
    elif activity_level >= 30:
        activity_factor = 0.2 + job_activity_bonus
    elif activity_level >= 15:
        activity_factor = 0.0
    else:
        activity_factor = -0.2
    
    # Stress level adjustment
    stress_level = user['Stress Level']
    if stress_level >= 8:
        stress_factor = 0.5
    else:
        stress_factor = ((stress_level - 5) / 10) * 0.6
    
    # BMI category effect
    if user['BMI Category'] == 'Normal':
        bmi_factor = 0
    elif user['BMI Category'] == 'Overweight':
        bmi_factor = 0.2
    else:  # Obese
        bmi_factor = 0.5
    
    # Sleep disorder adjustment
    disorder = user.get('Sleep Disorder')
    if disorder == 'Sleep Apnea':
        disorder_factor = 0.8
    elif disorder == 'Insomnia':
        disorder_factor = 0.6
    else:
        disorder_factor = 0.0
    
    # Heart rate adjustment
    hr = user['Heart Rate']
    if 60 <= hr <= 70:
        hr_factor = 0.0
    elif 55 <= hr <= 75:
        hr_factor = 0.1
    else:
        hr_factor = 0.3
    
    # Daily steps adjustment
    steps = user['Daily Steps']
    if steps >= 12000:
        steps_factor = 0.3
    elif steps >= 10000:
        steps_factor = 0.2
    elif steps >= 7500:
        steps_factor = 0.1
    elif steps >= 4000:
        steps_factor = 0.0
    else:
        steps_factor = -0.1
    
    # Gender differences with age interaction
    if user['Gender'] == 'Female':
        if user['Age'] > 50:
            gender_factor = 0.3
        else:
            gender_factor = 0.2
    else:
        gender_factor = 0.0
    
    # Occupation effect
    high_stress_jobs = ['Sales Representative', 'Doctor', 'Lawyer']
    shift_work_jobs = ['Nurse', 'Doctor']
    mental_demand_jobs = ['Software Engineer', 'Accountant', 'Scientist']
    
    if occupation in high_stress_jobs:
        occupation_factor = 0.2
    elif occupation in shift_work_jobs:
        occupation_factor = 0.3
    elif occupation in mental_demand_jobs:
        occupation_factor = 0.15
    else:
        occupation_factor = 0.0
    
    # Blood pressure effect
    bp_factor = 0.0
    if 'Blood Pressure' in user:
        try:
            sys, dias = map(int, user['Blood Pressure'].split('/'))
            if sys >= 160 or dias >= 100:
                bp_factor = 0.3
            elif sys >= 140 or dias >= 90:
                bp_factor = 0.2
            elif sys >= 130 or dias >= 80:
                bp_factor = 0.1
        except:
            pass
    
    # Calculate total adjustment based on domain knowledge
    domain_adjustment = (age_factor + activity_factor + stress_factor + bmi_factor + 
                        disorder_factor + hr_factor + steps_factor + gender_factor + 
                        occupation_factor + bp_factor)
    
    # Combine ML prediction with domain knowledge
    if activity_level >= 50 or steps >= 10000:
        hybrid_prediction = (0.6 * ml_prediction) + (0.4 * (ml_prediction + domain_adjustment))
    else:
        hybrid_prediction = (0.7 * ml_prediction) + (0.3 * (ml_prediction + domain_adjustment))
    
    # Ensure prediction is within reasonable bounds (6.0-10.0 hours)
    final_prediction = max(6.0, min(10.0, hybrid_prediction))
    
    return round(final_prediction, 1)

def get_personalized_recommendations(user, predicted_duration):
    """Generate personalized sleep recommendations based on user's health profile"""
    recommendations = []
    
    # Activity-based recommendations
    activity_level = user['Physical Activity Level']
    if activity_level < 15:
        recommendations.append("Increase physical activity to at least 30 minutes daily - even light walking can improve sleep quality")
    elif activity_level < 30:
        recommendations.append("Gradually increase your physical activity to 30-45 minutes daily for optimal sleep benefits")
    elif activity_level >= 75:
        recommendations.append("Maintain your excellent activity level, but ensure workouts end at least 2-3 hours before bedtime")
    
    # Stress-based recommendations
    stress_level = user['Stress Level']
    if stress_level >= 8:
        recommendations.append("Your high stress levels are significantly affecting sleep - prioritize stress reduction techniques like meditation or deep breathing")
    elif stress_level >= 6:
        recommendations.append("Moderate stress may be affecting your sleep quality - try a 10-minute mindfulness practice before bed")
    
    # Steps-based recommendations
    steps = user['Daily Steps']
    if steps < 5000:
        recommendations.append(f"Your daily step count ({steps}) is below recommended levels - aim to gradually increase to 7,500+ steps")
    elif steps < 7500:
        recommendations.append(f"Your step count ({steps}) is moderate - try to increase to 7,500-10,000 steps for better sleep quality")
    
    # Heart rate recommendations
    hr = user['Heart Rate']
    if hr > 80:
        recommendations.append(f"Your elevated resting heart rate ({hr} bpm) may affect sleep quality - cardiovascular exercise can help lower it over time")
    
    # BMI-specific recommendations
    if user['BMI Category'] == 'Overweight':
        recommendations.append("Being overweight can impact sleep quality - focus on balanced nutrition and consistent physical activity")
    elif user['BMI Category'] == 'Obese':
        recommendations.append("Obesity significantly increases risk of sleep disorders - consider being evaluated for sleep apnea, especially if you snore heavily")
    
    # Sleep disorder specific recommendations
    disorder = user.get('Sleep Disorder')
    if disorder == 'Sleep Apnea':
        recommendations.append("With sleep apnea, position therapy may help - try sleeping on your side rather than back")
    elif disorder == 'Insomnia':
        recommendations.append("For insomnia, establish a consistent bedtime routine and sleep schedule every day (even weekends)")
    
    # Age-specific recommendations
    if user['Age'] < 30:
        recommendations.append("Young adults often need more sleep than they get - prioritize your sleep schedule even on weekends")
    elif user['Age'] >= 50:
        recommendations.append("As we age, sleep patterns naturally change - focus on sleep quality and consistent routines rather than just duration")
    
    # Blood pressure recommendations
    if 'Blood Pressure' in user:
        try:
            sys, dias = map(int, user['Blood Pressure'].split('/'))
            if sys >= 140 or dias >= 90:
                recommendations.append(f"Your blood pressure ({user['Blood Pressure']}) is elevated - managing it can improve sleep quality")
        except:
            pass
    
    # Select the top 5 most relevant recommendations
    if len(recommendations) > 5:
        # Prioritize the most important factors for this specific user
        priority_recs = []
        
        # Always include stress recommendations for high stress users
        if stress_level >= 8:
            stress_recs = [r for r in recommendations if "stress" in r.lower()]
            priority_recs.extend(stress_recs)
            
        # Always include disorder-specific recommendations
        if disorder in ['Sleep Apnea', 'Insomnia']:
            disorder_recs = [r for r in recommendations if disorder.lower() in r.lower()]
            priority_recs.extend(disorder_recs)
        
        # Always include activity recommendations for sedentary users
        if activity_level < 30:
            activity_recs = [r for r in recommendations if "activity" in r.lower() or "exercise" in r.lower()]
            priority_recs.extend(activity_recs)
        
        # Make final selection - get unique recommendations up to 5
        final_recs = list(set(priority_recs))
        
        # If we don't have enough priority recs, add others until we have 5
        other_recs = [r for r in recommendations if r not in final_recs]
        final_recs.extend(other_recs[:max(0, 5-len(final_recs))])
        
        return final_recs[:5]  # Return top 5 recommendations
    
    return recommendations

def analyze_sleep_prediction(user, pipeline, cols):
    """Analysis of sleep needs using ML model with domain expertise"""
    # Use our prediction algorithm
    predicted_duration = predict_sleep_duration_advanced(user, pipeline, cols)
    
    # Determine status and color based on prediction
    if predicted_duration < 7:
        status = "BELOW RECOMMENDED"
        color = "#FFA500"  # orange
        concern = "insufficient sleep"
        health_impacts = "reduced cognitive performance, weakened immunity, and increased stress"
    elif predicted_duration <= 9:
        status = "OPTIMAL"
        color = "#059669"  # green
        concern = None
        health_impacts = "optimal cognitive performance, immune function, and stress management"
    else:
        status = "ABOVE RECOMMENDED"
        color = "#DC2626"  # red
        concern = "excessive sleep"
        health_impacts = "daytime drowsiness, potential underlying health issues"
    
    # Generate basic analysis text
    analysis = f"Analysis for {user['Gender']}, {user['Age']} years ({user['Occupation']})\n"
    analysis += f"Predicted optimal sleep duration: {predicted_duration:.1f} hours ({status})\n\n"
    
    # Add activity level assessment
    if user['Physical Activity Level'] < 30:
        analysis += "• Low physical activity detected - may affect sleep quality\n"
    elif user['Physical Activity Level'] >= 60:
        analysis += "• High physical activity detected - may require more recovery sleep\n"
    
    # Add stress level assessment
    if user['Stress Level'] >= 7:
        analysis += "• Elevated stress levels detected - may disrupt sleep patterns\n"
    
    # Add specific recommendations using our detailed function
    recommendations = get_personalized_recommendations(user, predicted_duration)
    
    # Return everything needed for the UI
    return {
        "duration": predicted_duration,
        "status": status,
        "color": color,
        "analysis": analysis,
        "recommendations": recommendations,
        "health_impacts": health_impacts,
        "concern": concern
    }

def generate_radar_chart_data(user):
    """Generate normalized data for the radar chart"""
    # Normalize metrics to 0-1 scale for radar chart
    activity_score = min(user['Physical Activity Level'] / 60, 1.0)
    stress_score = 1 - (user['Stress Level'] / 10)  # Invert so lower stress is better
    steps_score = min(user['Daily Steps'] / 10000, 1.0)
    
    # Heart rate optimality (closest to 70 bpm is best)
    hr_optimality = 1 - min(abs(user['Heart Rate'] - 70) / 30, 1.0)
    
    # BMI score
    if user['BMI Category'] == 'Normal':
        bmi_score = 0.9
    elif user['BMI Category'] == 'Overweight':
        bmi_score = 0.5
    else:
        bmi_score = 0.3
    
    categories = ['Physical Activity', 'Stress Management', 'Daily Steps', 'Heart Rate', 'BMI Status']
    values = [activity_score, stress_score, steps_score, hr_optimality, bmi_score]
    
    return categories, values

def generate_pdf_report(user_data, results):
    """Generate a PDF report for the user"""
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Title page
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.9, 'Sleep Health Analysis Report', 
                fontsize=24, ha='center', fontweight='bold')
        plt.text(0.5, 0.85, f'Generated on {datetime.now().strftime("%Y-%m-%d")}', 
                fontsize=14, ha='center')
        plt.text(0.5, 0.8, f'For: {user_data["Gender"]}, {user_data["Age"]} years old ({user_data["Occupation"]})', 
                fontsize=14, ha='center')
        plt.text(0.5, 0.5, 'This report provides personalized sleep duration predictions\n'
                        'and recommendations based on your lifestyle factors including\n'
                        'physical activity, stress levels, and physiological metrics.',
                fontsize=12, ha='center')
        pdf.savefig()
        plt.close()
        
        # Results page
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={'height_ratios': [1, 1.5]})
        
        # Bar chart for sleep duration
        categories = ['Your Predicted\nSleep Need', 'Minimum\nRecommended', 'Maximum\nRecommended']
        values = [results['duration'], 7.0, 9.0]
        colors = [results['color'], '#90CAF9', '#90CAF9']
        bars = ax1.bar(categories, values, color=colors, width=0.4)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f} hours',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12)
        
        # Recommended range
        ax1.axhspan(7, 9, color='#E0F7FA', alpha=0.3, zorder=0)
        ax1.text(1, 8, "Recommended Range", ha='center', va='center', color='#006064', fontsize=10)
        
        ax1.set_ylim(0, 10.5)
        ax1.set_title(f"Your Optimal Sleep Duration: {results['duration']} hours ({results['status']})", 
                    fontsize=16, fontweight='bold')
        ax1.set_ylabel("Hours of Sleep")
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Key health metrics section
        ax2.axis('off')
        
        # Create a table with key metrics
        table_data = [
            ["Physical Activity", f"{user_data['Physical Activity Level']} min/day", "✓" if user_data['Physical Activity Level'] >= 30 else "✗"],
            ["Stress Level", f"{user_data['Stress Level']}/10", "✗" if user_data['Stress Level'] > 7 else "✓"],
            ["Daily Steps", f"{user_data['Daily Steps']}", "✓" if user_data['Daily Steps'] >= 7500 else "✗"],
            ["Heart Rate", f"{user_data['Heart Rate']} bpm", "✓" if 60 <= user_data['Heart Rate'] <= 80 else "✗"],
            ["BMI Category", f"{user_data['BMI Category']}", "✓" if user_data['BMI Category'] == 'Normal' else "✗"],
            ["Sleep Disorder", f"{user_data.get('Sleep Disorder', 'None')}", "✓" if user_data.get('Sleep Disorder') is None else "✗"]
        ]
        
        col_labels = ["Health Factor", "Your Value", "Status"]
        table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        # Style the table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#1E3A8A')
            elif j == 2:  # Status column
                if cell.get_text().get_text() == "✓":
                    cell.set_facecolor('#D1FAE5')
                elif cell.get_text().get_text() == "✗":
                    cell.set_facecolor('#FEE2E2')
        
        ax2.set_title("Your Health Metrics", fontsize=16, pad=20, fontweight='bold')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Recommendations page
        plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        
        plt.text(0.5, 0.95, 'Your Personalized Sleep Recommendations', 
                fontsize=20, ha='center', fontweight='bold')
        
        # Add recommendations as bullet points
        y_pos = 0.85
        for i, rec in enumerate(results['recommendations']):
            plt.text(0.1, y_pos, f"• {rec}", fontsize=14, va='center', wrap=True)
            y_pos -= 0.08
        
        # Add lifestyle impact section
        plt.text(0.5, 0.4, 'How Your Lifestyle Affects Your Sleep', 
                fontsize=16, ha='center', fontweight='bold')
        
        lifestyle_text = (
            f"Your predicted optimal sleep need of {results['duration']} hours is based on your unique profile.\n\n"
            f"With your current activity level of {user_data['Physical Activity Level']} minutes per day and "
            f"stress level of {user_data['Stress Level']}/10, your body may require specific sleep patterns.\n\n"
        )
        
        if results['status'] == "BELOW RECOMMENDED":
            lifestyle_text += ("Your below-recommended sleep need may be influenced by factors like age, high physical activity, "
                            "or good overall health. However, ensure you're still getting quality rest.")
        elif results['status'] == "OPTIMAL":
            lifestyle_text += ("Your sleep need falls within the optimal range, suggesting a good balance of daily activity, "
                            "stress management, and physical health indicators.")
        else:
            lifestyle_text += ("Your above-recommended sleep need may be influenced by higher stress levels, "
                            "recovery needs, or potential sleep inefficiency. Focus on sleep quality.")
        
        plt.text(0.1, 0.35, lifestyle_text, fontsize=12, va='top', wrap=True)
        
        # Add footer with date
        plt.text(0.5, 0.05, f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                fontsize=10, ha='center', color='#666666')
        
        pdf.savefig()
        plt.close()
        
        # Radar chart of health metrics
        plt.figure(figsize=(11, 8.5))
        
        plt.subplot(111, polar=True)
        
        categories, values = generate_radar_chart_data(user_data)
        
        # Number of variables
        N = len(categories)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Values for the plot, plus close the plot
        values = values + values[:1]
        
        # Draw the plot
        plt.polar(angles, values, 'o-', linewidth=2)
        plt.fill(angles, values, alpha=0.25)
        
        # Fix axis to go in the right order and start at 12 o'clock
        plt.xticks(angles[:-1], categories)
        
        # Draw ylabels
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["Poor", "Fair", "Good", "Excellent"], 
                color="grey", size=10)
        plt.ylim(0, 1)
        
        plt.title('Your Health Metrics Profile', size=20, y=1.1)
        plt.tight_layout()
        
        pdf.savefig()
        plt.close()
    
    buffer.seek(0)
    return buffer

# -------------------------------------------
# Main UI
# -------------------------------------------

def main():
    """Main function for Sleep Health Predictor App"""
    
    # Header
    st.markdown('<h1 class="main-header">Sleep Health Predictor</h1>', unsafe_allow_html=True)
    
    # Custom sidebar
    with st.sidebar:
        # st.image("https://cdn.prod.website-files.com/65d5520ef65eaba833ec52bb/66f690d521bc1fbd02cae531_Blog%20Images%20(23).png", width=80)
        st.title("Sleep Predictor")
        
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Predict", "Health Tips", "About"],
            icons=["house", "calculator", "book", "info-circle"],
            menu_icon="list",
            default_index=0,
        )
    
    # Home Page
    if selected == "Home":
        st.markdown('<h2 class="sub-header">Welcome to the Sleep Health Predictor</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            <div class="card">
                <h3>Predict Your Optimal Sleep Duration</h3>
                <p>This tool uses advanced machine learning and personalized health factors to predict how much sleep is optimal for your body based on:</p>
                <ul>
                    <li>Age, gender and occupation</li>
                    <li>Physical activity level and daily steps</li>
                    <li>Stress levels and physiological indicators</li>
                    <li>BMI category and sleep disorders</li>
                </ul>
                <p>Get personalized recommendations to improve your sleep quality and download a comprehensive report.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h3>Why Sleep Matters</h3>
                <p>Quality sleep is essential for:</p>
                <ul>
                    <li>Cognitive performance and mood regulation</li>
                    <li>Immune system function and disease prevention</li>
                    <li>Hormonal balance and weight management</li>
                    <li>Memory consolidation and learning</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.image("https://assets.kive.ai/sizes%2F6ixyBB9AcuoZb3cP4vls%2FniF08BgfiHnz9sHpQwaa_1600.jpg", use_column_width=True)
            
            st.markdown("""
            <div class="metric-card">
                <p class="metric-value">7-9</p>
                <p class="metric-label">Recommended sleep hours for adults</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="metric-card">
                <p class="metric-value">1/3</p>
                <p class="metric-label">Fraction of life spent sleeping</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.button("Go to Prediction Tool", on_click=lambda: st.session_state.update({"selected": "Predict"}))
    
    # Predict Page
    elif selected == "Predict":
        st.markdown('<h2 class="sub-header">Predict Your Optimal Sleep Duration</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Personal Information
            st.subheader("Personal Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 90, 35)
            
            occupation_options = [
                "Doctor", "Engineer", "Lawyer", "Teacher", "Nurse", 
                "Accountant", "Sales Representative", "Software Engineer", 
                "Manager", "Office Worker", "Student", "Other"
            ]
            occupation = st.selectbox("Occupation", occupation_options)
            
            # Health Metrics
            st.subheader("Health Metrics")
            physical_activity = st.slider("Physical Activity (minutes/day)", 0, 120, 30)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            daily_steps = st.number_input("Daily Steps", 1000, 20000, 5000, step=500)
            heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 120, 70)
            
            bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
            
            # Blood Pressure
            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                systolic = st.number_input("Systolic BP", 80, 200, 120)
            with col_bp2:
                diastolic = st.number_input("Diastolic BP", 40, 120, 80)
            
            blood_pressure = f"{systolic}/{diastolic}"
            
            # Sleep Disorder
            sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Insomnia", "Sleep Apnea"])
            sleep_disorder = None if sleep_disorder == "None" else sleep_disorder
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Create user data dictionary
            user_data = {
                'Person ID': 999,
                'Gender': gender,
                'Age': age,
                'Occupation': occupation,
                'Sleep Duration': None,  # This is what we're predicting
                'Quality of Sleep': 5,  # Default value
                'Physical Activity Level': physical_activity,
                'Stress Level': stress_level,
                'BMI Category': bmi_category,
                'Blood Pressure': blood_pressure,
                'Heart Rate': heart_rate,
                'Daily Steps': daily_steps,
                'Sleep Disorder': sleep_disorder
            }
            
            # Add a predict button
            predict_btn = st.button("Predict Sleep Duration")
            
            # Create empty container for results
            results_container = st.container()
            
            if predict_btn:
                with st.spinner("Analyzing your health data and calculating optimal sleep duration..."):
                    if model_components:
                        pipeline = model_components['pipeline']
                        my_cols = model_components['feature_columns']
                        
                        # Get prediction results
                        results = analyze_sleep_prediction(user_data, pipeline, my_cols)
                        
                        # Store in session state
                        st.session_state.user_data = user_data
                        st.session_state.results = results
            
            # Display results if available
            if hasattr(st.session_state, 'results') and hasattr(st.session_state, 'user_data'):
                results = st.session_state.results
                
                status_class = ""
                if results['status'] == "OPTIMAL":
                    status_class = "status-optimal"
                elif results['status'] == "BELOW RECOMMENDED":
                    status_class = "status-below"
                else:
                    status_class = "status-above"
                
                with results_container:
                    st.markdown(f"""
                    <div class="card" style="border-left: 5px solid {results['color']};">
                        <h3>Your Results</h3>
                        <h2>Optimal Sleep Duration: 
                            <span class="{status_class}">{results['duration']} hours</span>
                        </h2>
                        <p>Status: <span class="{status_class}">{results['status']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Recommendations"])
                    
                    with tab1:
                        # Create a radar chart of user metrics
                        categories, values = generate_radar_chart_data(user_data)
                        
                        # Convert to plotly for better interactive experience
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Your Metrics'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title="Your Health Metrics Profile",
                            showlegend=False,
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Simple bar chart comparing prediction to recommended range
                        fig = go.Figure()
                        
                        # Add recommended range as a shaded area
                        fig.add_shape(
                            type="rect",
                            x0=0, x1=3,
                            y0=7, y1=9,
                            fillcolor="lightgreen",
                            opacity=0.3,
                            line_width=0,
                            layer="below"
                        )
                        
                        fig.add_annotation(
                            x=1.5, y=8,
                            text="Recommended Range (7-9 hours)",
                            showarrow=False,
                            font_size=12,
                            bgcolor="white",
                            opacity=0.8
                        )
                        
                        fig.add_trace(go.Bar(
                            x=["Your Optimal Sleep"],
                            y=[results['duration']],
                            marker_color=results['color'],
                            text=[f"{results['duration']} hours"],
                            textposition='auto',
                            width=[0.4],
                            name="Your Sleep Need"
                        ))
                        
                        fig.update_layout(
                            title="Your Sleep Prediction vs. Recommended Range",
                            yaxis=dict(
                                title="Hours of Sleep",
                                range=[0, 10]
                            ),
                            height=350,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        st.subheader("Health Factors Analysis")
                        
                        # Create a more detailed analysis
                        col_a1, col_a2 = st.columns([1, 1])
                        
                        with col_a1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <p class="metric-label">Physical Activity</p>
                                <p class="metric-value" style="color: {'green' if physical_activity >= 30 else 'orange'};">
                                    {physical_activity} min/day
                                </p>
                                <p class="metric-label">{'Adequate' if physical_activity >= 30 else 'Below recommended'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <p class="metric-label">Heart Rate</p>
                                <p class="metric-value" style="color: {'green' if 60 <= heart_rate <= 80 else 'orange'};">
                                    {heart_rate} bpm
                                </p>
                                <p class="metric-label">
                                    {
                                      'Optimal' if 60 <= heart_rate <= 70 else 
                                      'Normal' if 55 <= heart_rate <= 80 else 
                                      'Outside optimal range'
                                    }
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_a2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <p class="metric-label">Stress Level</p>
                                <p class="metric-value" style="color: {'green' if stress_level <= 6 else 'orange' if stress_level <= 7 else 'red'};">
                                    {stress_level}/10
                                </p>
                                <p class="metric-label">
                                    {
                                      'Low' if stress_level <= 3 else 
                                      'Moderate' if stress_level <= 6 else 
                                      'High' if stress_level <= 8 else 
                                      'Very high'
                                    }
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <p class="metric-label">Daily Steps</p>
                                <p class="metric-value" style="color: {'green' if daily_steps >= 7500 else 'orange' if daily_steps >= 5000 else 'red'};">
                                    {daily_steps}
                                </p>
                                <p class="metric-label">
                                    {
                                      'Excellent' if daily_steps >= 10000 else 
                                      'Good' if daily_steps >= 7500 else 
                                      'Moderate' if daily_steps >= 5000 else 
                                      'Low'
                                    }
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("""<hr>""", unsafe_allow_html=True)
                        
                        st.subheader("Impact on Sleep Health")
                        
                        if results['status'] == "OPTIMAL":
                            st.success("Your predicted sleep need falls within the optimal range for adults.")
                            st.markdown(f"A sleep duration of {results['duration']} hours is associated with {results['health_impacts']}.")
                        elif results['status'] == "BELOW RECOMMENDED":
                            st.warning(f"Your predicted sleep need of {results['duration']} hours is below the general recommendation.")
                            st.markdown(f"Insufficient sleep can lead to {results['health_impacts']}.")
                            st.markdown("Consider implementing the recommendations to optimize your sleep quality.")
                        else:
                            st.error(f"Your predicted sleep need of {results['duration']} hours is above typical recommendations.")
                            st.markdown(f"Excessive sleep can sometimes indicate {results['health_impacts']}.")
                            st.markdown("Focus on improving sleep quality rather than quantity.")
                    
                    with tab3:
                        st.subheader("Your Personalized Recommendations")
                        
                        for rec in results['recommendations']:
                            st.markdown(f"""
                            <div class="recommendation-item">
                                {rec}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add general tips
                        st.markdown("### General Sleep Hygiene Tips")
                        
                        col_t1, col_t2 = st.columns(2)
                        
                        with col_t1:
                            st.markdown("""
                            - Maintain a consistent sleep schedule
                            - Create a restful environment (dark, quiet, cool)
                            - Limit exposure to screens before bedtime
                            - Avoid caffeine and alcohol near bedtime
                            """)
                        
                        with col_t2:
                            st.markdown("""
                            - Exercise regularly, but not too close to bedtime
                            - Manage stress with relaxation techniques
                            - Use comfortable mattress and pillows
                            - Don't eat heavy meals before sleep
                            """)
                    
                    # Add download button for PDF report
                    pdf_buffer = generate_pdf_report(st.session_state.user_data, results)
                    b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
                    
                    st.markdown(f"""
                        <div style="text-align: center; margin-top: 30px;">
                            <a href="data:application/pdf;base64,{b64_pdf}" download="Sleep_Analysis_Report.pdf" 
                               class="report-btn" style="text-decoration: none; padding: 10px 20px;">
                                📄 Download PDF Report
                            </a>
                        </div>
                    """, unsafe_allow_html=True)
    
    # Health Tips Page
    elif selected == "Health Tips":
        st.markdown('<h2 class="sub-header">Sleep Health Tips</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["General Tips", "Specific Conditions", "Sleep Science"])
        
        with tab1:
            st.markdown("""
            <div class="card">
                <h3>Establishing a Sleep Routine</h3>
                <ul>
                    <li><strong>Consistent Schedule:</strong> Go to bed and wake up at the same time daily, even on weekends.</li>
                    <li><strong>Bedtime Ritual:</strong> Create a relaxing pre-sleep routine like reading or taking a warm bath.</li>
                    <li><strong>Limit Naps:</strong> Keep daytime naps under 30 minutes and before 3 PM.</li>
                    <li><strong>Digital Curfew:</strong> Stop using electronic devices 1 hour before bedtime.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h3>Optimizing Your Sleep Environment</h3>
                <ul>
                    <li><strong>Darkness:</strong> Use blackout curtains or an eye mask to eliminate light.</li>
                    <li><strong>Temperature:</strong> Keep your bedroom between 60-67°F (15-19°C).</li>
                    <li><strong>Noise Control:</strong> Use earplugs or white noise machines if needed.</li>
                    <li><strong>Comfortable Bedding:</strong> Invest in a quality mattress and pillows.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h3>Daytime Habits for Better Sleep</h3>
                <ul>
                    <li><strong>Morning Sunlight:</strong> Get exposure to natural light soon after waking.</li>
                    <li><strong>Regular Exercise:</strong> Aim for 30 minutes daily, but not within 2-3 hours of bedtime.</li>
                    <li><strong>Mindful Eating:</strong> Avoid heavy meals, alcohol, and caffeine close to bedtime.</li>
                    <li><strong>Stress Management:</strong> Practice meditation, deep breathing, or journaling.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="card">
                <h3>Tips for Insomnia</h3>
                <ul>
                    <li><strong>Cognitive Behavioral Therapy:</strong> CBT-I is the first-line treatment for chronic insomnia.</li>
                    <li><strong>Stimulus Control:</strong> Only use your bed for sleep and intimacy.</li>
                    <li><strong>Sleep Restriction:</strong> Temporarily limit time in bed to consolidate sleep.</li>
                    <li><strong>Get Up Rule:</strong> If you can't fall asleep within 20 minutes, get up and do something relaxing until you feel sleepy.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h3>Tips for Sleep Apnea</h3>
                <ul>
                    <li><strong>Position Therapy:</strong> Sleep on your side rather than your back.</li>
                    <li><strong>CPAP Compliance:</strong> Use your prescribed device consistently.</li>
                    <li><strong>Weight Management:</strong> Even modest weight loss can improve symptoms.</li>
                    <li><strong>Avoid Alcohol:</strong> Especially in the evening as it relaxes airway muscles.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h3>Tips for Shift Workers</h3>
                <ul>
                    <li><strong>Consistent Schedule:</strong> Keep the same sleep schedule on workdays and days off.</li>
                    <li><strong>Light Management:</strong> Use blackout curtains during day sleep and bright light during night shifts.</li>
                    <li><strong>Strategic Napping:</strong> Take a short nap before night shifts.</li>
                    <li><strong>Social Support:</strong> Communicate your sleep needs to family and friends.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("""
            <div class="card">
                <h3>Understanding Sleep Cycles</h3>
                <p>Sleep progresses through multiple 90-minute cycles of REM and non-REM sleep:</p>
                <ul>
                    <li><strong>Stage 1:</strong> Light sleep, easily awakened</li>
                    <li><strong>Stage 2:</strong> Body temperature drops, breathing and heart rate regular</li>
                    <li><strong>Stage 3:</strong> Deep sleep, difficult to wake up, body repairs tissues and builds bone/muscle</li>
                    <li><strong>REM Sleep:</strong> Brain activity increases, dreaming occurs, crucial for cognitive function</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Display sleep cycle chart
            fig = go.Figure()
            
            # Time points for x-axis (hours of sleep)
            times = list(range(9))
            
            # Sleep stages (0=awake, 1=light, 2=medium, 3=deep, 4=REM)
            stages = [0, 1, 2, 3, 2, 4, 2, 3, 2, 1, 4, 2, 3, 2, 4, 2, 1, 4]
            times_expanded = []
            stages_expanded = []
            
            # Expand data to show a continuous curve
            for i in range(len(stages)-1):
                times_expanded.append(i*0.5)
                stages_expanded.append(stages[i])
            
            # Add trace
            fig.add_trace(go.Scatter(
                x=times_expanded,
                y=stages_expanded,
                mode='lines',
                line=dict(color='#3B82F6', width=3),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.2)'
            ))
            
            # Update layout
            fig.update_layout(
                title="Typical Sleep Cycle Pattern",
                xaxis_title="Hours of Sleep",
                yaxis_title="Sleep Stage",
                yaxis=dict(
                    ticktext=["Awake", "Light", "Medium", "Deep", "REM"],
                    tickvals=[0, 1, 2, 3, 4],
                    range=[-0.5, 4.5]
                ),
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="card">
                <h3>Sleep Across the Lifespan</h3>
                <p>Sleep needs change with age:</p>
                <ul>
                    <li><strong>Newborns (0-3 months):</strong> 14-17 hours</li>
                    <li><strong>Infants (4-11 months):</strong> 12-15 hours</li>
                    <li><strong>Toddlers (1-2 years):</strong> 11-14 hours</li>
                    <li><strong>Preschoolers (3-5 years):</strong> 10-13 hours</li>
                    <li><strong>School-age (6-13 years):</strong> 9-11 hours</li>
                    <li><strong>Teenagers (14-17 years):</strong> 8-10 hours</li>
                    <li><strong>Young Adults (18-25 years):</strong> 7-9 hours</li>
                    <li><strong>Adults (26-64 years):</strong> 7-9 hours</li>
                    <li><strong>Older Adults (65+ years):</strong> 7-8 hours</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # About Page
    elif selected == "About":
        st.markdown('<h2 class="sub-header">About the Sleep Health Predictor</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>How It Works</h3>
            <p>The Sleep Health Predictor uses a sophisticated machine learning model combined with domain expertise to predict optimal sleep duration based on individual factors.</p>
            <p>Our model considers:</p>
            <ul>
                <li>Demographic data (age, gender, occupation)</li>
                <li>Activity metrics (physical activity level, daily steps)</li>
                <li>Physiological indicators (heart rate, blood pressure, BMI)</li>
                <li>Stress levels and existing sleep disorders</li>
            </ul>
            <p>The prediction combines machine learning with medical knowledge to provide personalized recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Data Privacy</h3>
            <p>Your data is processed locally in your browser and is not stored on any server. We do not collect, share, or sell any personal information.</p>
            <p>The PDF reports generated are created on your device and are not transmitted to us.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>Limitations</h3>
            <p>This tool is for informational purposes only and does not substitute professional medical advice. Consult with a healthcare provider for sleep disorders or health concerns.</p>
            <p>The predictions are based on general population data and may not account for all individual variations or medical conditions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3>References</h3>
            <ul>
                <li>National Sleep Foundation Recommendations</li>
                <li>American Academy of Sleep Medicine Guidelines</li>
                <li>Sleep Health and Lifestyle Dataset Analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()