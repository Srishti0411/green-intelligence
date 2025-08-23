import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Green Intelligence", layout="wide", page_icon="🤖")

# ---------- CSS for professional styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

body, .stApp {
    font-family: 'Poppins', sans-serif;
}

.navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background: linear-gradient(90deg, #6C63FF, #9D7CFC);
    padding: 15px 30px;
    color: white;
    font-weight: 600;
    font-size: 20px;
    z-index: 9999;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    border-radius: 0 0 15px 15px;
}
.navbar span {
    margin-right: 40px;
    color: white;
    cursor: pointer;
}

.section {
    padding-top: 100px;
    padding-bottom: 40px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    margin-bottom: 30px;
}

.card h3 {
    color: #6C63FF;
    font-weight: 600;
}

.card p {
    font-size: 16px;
    color: #333;
}

h1, h2, h3 {
    font-family: 'Poppins', sans-serif;
}

</style>
""", unsafe_allow_html=True)

# ---------- NAVBAR ----------
st.markdown("""
<div class="navbar">
    <span>Green Intelligence</span>
    <span>Predictions</span>
    <span>Emissions</span>
    <span>Facts</span>
    <span>Graphs</span>
    <span>Recommendations</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='section'></div>", unsafe_allow_html=True)

# ---------- HOMEPAGE ----------
st.markdown("""
<div style="background: linear-gradient(to right, #EDE7F6, #D1C4E9); padding: 50px; border-radius: 20px;">
    <h1 style="color: #6C63FF; font-size:48px;">Green Intelligence</h1>
    <p style="color: #4A148C; font-size:20px;">Track energy usage, visualize GPU & CPU loads, and optimize AI model performance responsibly.</p>
</div>
""", unsafe_allow_html=True)

# ---------- DATA ----------
# Load your clean_gpus CSV
gpus_df = pd.read_csv("data/clean_gpus.csv")  # adjust path if needed

# Separate GPU and CPU names dynamically
all_gpus = gpus_df[gpus_df['type'] == 'gpu']['name'].tolist()
all_cpus = gpus_df[gpus_df['type'] == 'cpu']['name'].tolist()

# Instead of hardcoding 4 models, dynamically create options
# Let's just assume each "model" = one GPU/CPU combo
all_models = all_gpus + all_cpus

models = { "GPT-3": {"GPUs": ["A100", "V100", "T4"], "CPUs": ["Intel Xeon", "AMD EPYC"]}, "BERT": {"GPUs": ["V100", "T4", "RTX3090"], "CPUs": ["Intel Xeon", "AMD EPYC"]}, "RandomNet": {"GPUs": ["RTX3090", "T4"], "CPUs": ["Intel i9", "AMD Ryzen"]}, "VisionNet": {"GPUs": ["A100", "V100"], "CPUs": ["Intel i9", "AMD Ryzen"]} }
# ---------- PREDICTIONS ----------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("Predictions")

selected_model_pred = st.selectbox("Select a model (GPU/CPU):", all_models)
pred_hours = st.slider("Estimated run hours for this model:", 1, 50, 10)
energy_per_hour = np.random.randint(50, 120)  # mock kWh per hour
predicted_energy = pred_hours * energy_per_hour
st.metric("Predicted Energy (kWh)", f"{predicted_energy}")

# ---------- EMISSIONS ----------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("Emissions")
emission_factor = 0.5  # kg CO2 per kWh
predicted_emissions = predicted_energy * emission_factor
st.metric("Estimated CO2 Emissions (kg)", f"{predicted_emissions:.2f}")

# ---------- GRAPHS ----------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("GPU & CPU Usage Graphs")

# ALL available GPUs & CPUs
all_gpus = []
all_cpus = []
for m in models.values():
    all_gpus.extend(m["GPUs"])
    all_cpus.extend(m["CPUs"])
all_gpus = list(set(all_gpus))
all_cpus = list(set(all_cpus))

selected_gpus = st.multiselect("Select GPUs to visualize:", all_gpus, default=all_gpus[:2])
selected_cpus = st.multiselect("Select CPUs to visualize:", all_cpus, default=all_cpus[:2])

# Mock usage data
time_points = list(range(1,11))
gpu_data = pd.DataFrame({
    "Device": np.repeat(selected_gpus, 10),
    "Usage (%)": np.random.randint(30, 100, size=10*len(selected_gpus)),
    "Time (hrs)": time_points*len(selected_gpus)
})

cpu_data = pd.DataFrame({
    "Device": np.repeat(selected_cpus, 10),
    "Usage (%)": np.random.randint(10, 80, size=10*len(selected_cpus)),
    "Time (hrs)": time_points*len(selected_cpus)
})

fig_gpu = px.line(gpu_data, x="Time (hrs)", y="Usage (%)", color="Device", title="GPU Usage Over Time")
fig_cpu = px.line(cpu_data, x="Time (hrs)", y="Usage (%)", color="Device", title="CPU Usage Over Time")

st.plotly_chart(fig_gpu, use_container_width=True)
st.plotly_chart(fig_cpu, use_container_width=True)

# ---------- RECOMMENDATIONS ----------
# ---------- RECOMMENDATIONS ----------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("Recommendations")

rec_list = []

if predicted_emissions > 500:
    rec_list.append(
        "⚠️ High emissions detected: Your current run is projected to exceed 500 kg of CO₂. "
        "This is equivalent to the monthly carbon footprint of an average household. "
        "Consider reducing the number of run hours, breaking training into shorter cycles, "
        "or migrating to newer, energy-efficient GPUs such as NVIDIA A100 or T4. "
        "Using renewable-energy backed cloud providers can also offset a large portion of these emissions."
    )
else:
    rec_list.append(
        "✅ Moderate emissions level: Your configuration shows a balanced energy footprint. "
        "Still, you can optimize by monitoring GPU utilization — often GPUs are underutilized when batch sizes are not tuned properly. "
        "Try adjusting hyperparameters or consolidating runs to maximize compute efficiency."
    )

if len(selected_gpus) > 3:
    rec_list.append(
        "⚡ Multiple GPU usage detected: Running on more than three GPUs can lead to unnecessary power draw, "
        "especially if model scaling efficiency plateaus. Before scaling horizontally, verify that each GPU is consistently above 70–80% utilization. "
        "If not, you may be wasting power on idle resources. Use profiling tools like NVIDIA Nsight or PyTorch Profiler to measure this."
    )

# Always include at least one generic sustainability tip
rec_list.append(
    "🌍 General Tip for Sustainable AI: Document your experiments with both accuracy and energy metrics. "
    "Tracking emissions per epoch or per training run helps build awareness and creates accountability. "
    "Over time, this data can guide smarter choices, like moving to smaller distilled models or quantized versions without sacrificing performance."
)

for rec in rec_list:
    st.markdown(f"<div class='card'><p>{rec}</p></div>", unsafe_allow_html=True)


# ---------- FOOTER ----------
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<hr><p style='text-align:center;color:#6C63FF;'>Made with 💜 for sustainable 🌍</p>", unsafe_allow_html=True)
