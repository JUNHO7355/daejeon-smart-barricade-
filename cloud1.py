import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta

# Basic Configuration
st.set_page_config(page_title="AI-Based Ho Chi Minh Smart Barricade", layout="wide")

# AI Prediction Function
def ai_predict_pm25_advanced(past_data, weather_factor, traffic_factor, construction_nearby):
    X = np.arange(len(past_data)).reshape(-1, 1)
    y = np.array(past_data)
    lr_model = LinearRegression().fit(X, y)
    future_X = np.array([[len(past_data) + 1]])
    base_prediction = lr_model.predict(future_X)[0]
    
    recent_change = (past_data[-1] - past_data[-3]) / 3
    trend_factor = 1.0 + (recent_change / 100)
    
    final_prediction = base_prediction * trend_factor * weather_factor * traffic_factor
    
    if construction_nearby:
        final_prediction *= 1.4
    
    final_prediction = max(20, min(int(final_prediction), 250))
    return int(final_prediction), int(base_prediction), trend_factor

def calculate_prediction_confidence(past_data):
    if len(past_data) < 3:
        return 50
    std_dev = np.std(past_data[-5:])
    mean_val = np.mean(past_data[-5:])
    if mean_val == 0:
        return 50
    variation_coef = (std_dev / mean_val) * 100
    confidence = max(60, min(95, 100 - variation_coef))
    return int(confidence)

# QR Code Generation
def make_qr(url):
    try:
        import qrcode
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_b64}"
    except:
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

# Data Initialization
if "devices" not in st.session_state:
    st.session_state.scenario_time = "November 25, 2024 14:00"
    st.session_state.scenario_weather = "Sunny, SW Wind 3m/s, Humidity 75%"
    
    pm_scenarios = {
        "Ben Thanh Market": [85, 88, 92, 95, 98, 102, 105, 108, 112, 115],
        "District 1 Center": [115, 122, 128, 135, 140, 145, 148, 150, 152, 155],
        "Tan Son Nhat Airport": [65, 68, 70, 72, 75, 73, 71, 69, 67, 65],
        "Thu Duc City": [88, 88, 88, 88, 88, 88, 88, 88, 88, 88],
        "Saigon River Park": [52, 55, 58, 62, 65, 68, 71, 73, 75, 78],
    }
    
    device_scenarios = {
        "Ben Thanh Market": {
            "lat": 10.7720, "lng": 106.6981, "battery": 85, "rain": 35,
            "weather_factor": 1.15, "traffic_factor": 1.25,
            "construction_nearby": False, "sensor_stable": True,
            "priority": 2, "reason": "SW wind spreading construction dust from District 4"
        },
        "District 1 Center": {
            "lat": 10.7770, "lng": 106.7010, "battery": 72, "rain": 15,
            "weather_factor": 1.1, "traffic_factor": 1.35,
            "construction_nearby": True, "sensor_stable": True,
            "priority": 1, "reason": "Metro construction within 200m, top priority activation"
        },
        "Tan Son Nhat Airport": {
            "lat": 10.8184, "lng": 106.6595, "battery": 92, "rain": 80,
            "weather_factor": 0.9, "traffic_factor": 1.0,
            "construction_nearby": False, "sensor_stable": True,
            "priority": 5, "reason": "Declining trend, normal monitoring"
        },
        "Thu Duc City": {
            "lat": 10.8503, "lng": 106.7717, "battery": 45, "rain": 60,
            "weather_factor": 1.0, "traffic_factor": 1.4,
            "construction_nearby": False, "sensor_stable": False,
            "priority": 4, "reason": "Sensor anomaly detected (same reading for 3 hours)"
        },
        "Saigon River Park": {
            "lat": 10.7877, "lng": 106.7051, "battery": 88, "rain": 45,
            "weather_factor": 0.95, "traffic_factor": 0.9,
            "construction_nearby": False, "sensor_stable": True,
            "priority": 3, "reason": "Gradual increase trend, preventive monitoring"
        }
    }
    
    st.session_state.devices = []
    for name, scenario in device_scenarios.items():
        d = {"name": name}
        d.update(scenario)
        d["pm_data"] = pm_scenarios[name]
        d["pm_now"] = d["pm_data"][-1]
        
        prediction, base_pred, trend = ai_predict_pm25_advanced(
            d["pm_data"], d["weather_factor"], d["traffic_factor"], d["construction_nearby"]
        )
        d["pm_predict"] = prediction
        d["pm_base_predict"] = base_pred
        d["trend_factor"] = trend
        d["confidence"] = calculate_prediction_confidence(d["pm_data"])
        
        if d["pm_predict"] >= 120 and d["rain"] > 10:
            d["status"] = "💧 Washing Mode Predicted"
            d["color"] = "blue"
        elif d["pm_predict"] >= 80:
            d["status"] = "🌀 Fan Activation Predicted"
            d["color"] = "red"
        else:
            d["status"] = "🟢 Normal Predicted"
            d["color"] = "green"
        
        d["qr"] = make_qr("https://atf9h7g3asnzqz4xgapwdj.streamlit.app/")
        st.session_state.devices.append(d)
    
    st.session_state.construction_sites = [
        {"name": "District 1 Metro Line Construction", "lat": 10.7780, "lng": 106.6995, "radius": 200, "pm_increase": "+60%"},
        {"name": "Vo Van Kiet Boulevard Expansion", "lat": 10.7590, "lng": 106.6830, "radius": 300, "pm_increase": "+45%"},
    ]
    
    st.session_state.vulnerable_facilities = [
        {"name": "Sunshine Kindergarten", "lat": 10.7745, "lng": 106.6970, "type": "Kindergarten", "hours": "Pickup time 15:00"},
        {"name": "Golden Age Senior Center", "lat": 10.7795, "lng": 106.7025, "type": "Senior Center", "hours": "Operating 14:00-17:00"},
        {"name": "HCMC Central Hospital", "lat": 10.8485, "lng": 106.7695, "type": "Hospital", "hours": "24 hours"},
    ]
    
    st.session_state.cost_savings = {"power": 18400, "filter": 45000, "maintenance": 12000}

devices = st.session_state.devices
construction_sites = st.session_state.construction_sites
vulnerable_facilities = st.session_state.vulnerable_facilities
cost_savings = st.session_state.cost_savings

def generate_ai_decision():
    return [
        {"icon": "🏗️", "text": "District 1 Metro construction detected (200m radius) → 60% PM2.5 increase expected → District 1 Center barricade top priority (Rank #1)"},
        {"icon": "💨", "text": "Current wind SW→NE 3m/s → Construction dust spreading toward Ben Thanh Market → Preemptive response (Rank #2)"},
        {"icon": "🚸", "text": "Sunshine Kindergarten pickup in 1 hour (15:00) → Air quality priority purification mode activated within 200m"},
        {"icon": "📊", "text": "Thu Duc City sensor anomaly detected (fixed at 88 for 3 hours) → Maintenance team auto-dispatched → Enhanced monitoring"}
    ]

def generate_alerts():
    return [
        {"type": "warning", "icon": "⚠️", "text": "Thu Duc City sensor anomaly (same reading 88 for 3 hours) → Maintenance required"},
        {"type": "battery", "icon": "🔋", "text": "Thu Duc City battery 45% → Charging needed within 24 hours (based on current consumption)"},
        {"type": "pollution", "icon": "🚨", "text": "District 1 Center PM2.5 forecast 202 in 2 hours (89% confidence) → Maximum operation recommended"}
    ]

# Header
st.title("🤖 AI-Based Ho Chi Minh Smart Barricade Control System")
st.caption(f"📅 Scenario Time: {st.session_state.scenario_time} | 🌤️ {st.session_state.scenario_weather}")

total_savings = sum(cost_savings.values())
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 Today's AI Cost Savings", f"₫{total_savings * 1000:,}", delta="↑ +12% vs yesterday")
with col2:
    st.metric("⚡ Power Cost Saved", f"₫{cost_savings['power'] * 1000:,}", delta="5 operations prevented")
with col3:
    st.metric("🔧 Filter Replacement Delayed", f"₫{cost_savings['filter'] * 1000:,}", delta="1 cycle extended")
with col4:
    st.metric("🛠️ Early Failure Detection", f"₫{cost_savings['maintenance'] * 1000:,}", delta="1 issue prevented")

st.markdown("---")

st.markdown("### 🧠 AI Real-time Decision Making Status")
for decision in generate_ai_decision():
    st.info(f"{decision['icon']} {decision['text']}")

st.markdown("---")

alerts = generate_alerts()
if alerts:
    st.markdown("### 🔔 AI Anomaly Detection Alerts")
    for alert in alerts:
        if alert['type'] == 'warning':
            st.warning(f"{alert['icon']} {alert['text']}")
        elif alert['type'] == 'battery':
            st.error(f"{alert['icon']} {alert['text']}")
        else:
            st.warning(f"{alert['icon']} {alert['text']}")
    st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🏙 Integrated Control Map", "📊 AI Prediction Analysis", "🏗️ Public Data Integration", "📱 Citizen View"])

with tab1:
    col1, col2 = st.columns([1.4, 1])
    
    with col1:
        st.markdown("### 🗺️ Real-time Control Map")
        m = folium.Map(location=[10.7769, 106.6981], zoom_start=13, tiles="OpenStreetMap")
        
        for d in devices:
            popup_html = f"""
            <div style="width:250px">
            <b style="font-size:14px">{d['name']}</b><br><br>
            <b>Current PM2.5:</b> {d['pm_now']} μg/m³<br>
            <b>Predicted PM2.5:</b> {d['pm_predict']} μg/m³<br>
            <b>Confidence:</b> {d['confidence']}%<br>
            <b>Priority:</b> Rank {d['priority']}<br>
            <b>Status:</b> {d['status']}<br><br>
            <img src="{d['qr']}" width="150" height="150" style="display:block; margin:10px auto;">
            </div>
            """
            
            icon_color = 'red' if d['color'] == 'red' else 'green' if d['color'] == 'green' else 'blue'
            
            folium.Marker(
                location=[d["lat"], d["lng"]],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=icon_color, icon='cloud', prefix='fa'),
                tooltip=f"{d['name']}: PM2.5 {d['pm_now']}"
            ).add_to(m)
            
            folium.CircleMarker(
                location=[d["lat"], d["lng"]],
                radius=15,
                color=d["color"],
                fill=True,
                fillColor=d["color"],
                fill_opacity=0.4,
                weight=3
            ).add_to(m)
        
        for site in construction_sites:
            folium.Circle(
                location=[site["lat"], site["lng"]],
                radius=site["radius"],
                color="red",
                fill=True,
                fill_opacity=0.2,
                popup=f"<b>🏗️ {site['name']}</b><br>PM2.5 Impact: {site['pm_increase']}",
            ).add_to(m)
        
        for fac in vulnerable_facilities:
            folium.Marker(
                location=[fac["lat"], fac["lng"]],
                icon=folium.Icon(color="orange", icon="info-sign"),
                popup=f"<b>🚸 {fac['name']}</b><br>{fac['type']}<br>{fac['hours']}",
            ).add_to(m)
        
        folium_static(m, height=500, width=750)
    
    with col2:
        st.markdown("### 📋 AI Priority Assessment Results")
        sorted_devices = sorted(devices, key=lambda x: x["priority"])
        df = pd.DataFrame([
            [d["priority"], d["name"], d["pm_now"], f"{d['pm_predict']} ({d['confidence']}%)",
             "✅" if d["sensor_stable"] else "⚠️", d["status"]]
            for d in sorted_devices
        ], columns=["Rank", "Location", "Current", "Predicted (Conf.)", "Sensor", "AI Decision"])
        st.dataframe(df, use_container_width=True, height=250)
        
        st.markdown("### 🎯 AI Deployment Strategy Rationale")
        st.write("**Rank #1: District 1 Center**")
        st.write("- Within 200m of construction site (PM2.5 +60%)")
        st.write("- Current 155 → Predicted 202 (rapid increase)")
        st.write("")
        st.write("**Rank #2: Ben Thanh Market**")
        st.write("- SW wind construction dust dispersion path")
        st.write("- Gradual increase trend (preemptive response)")
        st.write("")
        st.write("**Vulnerable Group Special Care**")
        st.write("- Kindergarten activation 1 hour before pickup")
        st.write("- Senior center intensive purification during hours")

with tab2:
    st.markdown("### 🔮 AI Multi-Variable Prediction Model Detailed Analysis")
    names = [d["name"] for d in devices]
    selected_name = st.selectbox("Select device to analyze", names)
    selected = next(d for d in devices if d["name"] == selected_name)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"#### 📍 {selected['name']}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Current PM2.5", f"{selected['pm_now']} μg/m³")
            st.metric("Base Prediction", f"{selected['pm_base_predict']} μg/m³")
        with col_b:
            st.metric("AI Final Prediction", f"{selected['pm_predict']} μg/m³", 
                     delta=f"+{selected['pm_predict'] - selected['pm_now']}", delta_color="inverse")
            st.metric("Prediction Confidence", f"{selected['confidence']}%")
        
        st.write("")
        sensor_status = "✅ Normal" if selected["sensor_stable"] else "⚠️ Anomaly Detected"
        st.write(f"**Sensor Status:** {sensor_status}")
        st.write(f"**AI Decision:** {selected['status']}")
        st.write(f"**Priority:** Rank {selected['priority']}")
        st.write(f"**Decision Rationale:** {selected['reason']}")
        
        st.write("---")
        st.markdown("#### 🧮 AI Prediction Variable Analysis")
        st.write(f"**Weather Impact Factor:** {selected['weather_factor']:.2f}x")
        if selected['weather_factor'] > 1.0:
            st.caption("↑ Low humidity, PM2.5 increase expected")
        else:
            st.caption("↓ High humidity, PM2.5 decrease expected")
        
        st.write(f"**Traffic Impact Factor:** {selected['traffic_factor']:.2f}x")
        st.caption(f"Current traffic {int((selected['traffic_factor']-1)*100)}% above normal")
        st.write(f"**Near Construction:** {'Yes (+40%)' if selected['construction_nearby'] else 'No'}")
        st.write(f"**Trend Coefficient:** {selected['trend_factor']:.2f}x")
    
    with c2:
        st.markdown("#### 📈 PM2.5 Changes Over Time")
        chart_data = pd.DataFrame({
            "Actual": selected["pm_data"] + [None],
            "AI Prediction": [None] * len(selected["pm_data"]) + [selected["pm_predict"]],
            "Base Prediction": [None] * len(selected["pm_data"]) + [selected["pm_base_predict"]],
        }, index=[f"-{10-i}h" for i in range(10)] + ["2h later"])
        st.line_chart(chart_data, height=280)
        
        st.markdown("#### 🔍 Prediction Analysis")
        change = selected["pm_predict"] - selected["pm_now"]
        change_percent = (change / selected["pm_now"]) * 100
        
        if change > 20:
            st.error(f"⚠️ **Rapid Increase Expected**: +{change} μg/m³ ({change_percent:+.1f}%)")
            st.write("→ Immediate maximum operation recommended")
        elif change > 10:
            st.warning(f"⚡ **Increase Expected**: +{change} μg/m³ ({change_percent:+.1f}%)")
            st.write("→ Preemptive operation recommended")
        elif change < -10:
            st.success(f"✅ **Improvement Expected**: {change} μg/m³ ({change_percent:+.1f}%)")
            st.write("→ Normal monitoring")
        else:
            st.info(f"📊 **Stable Expected**: {change:+} μg/m³ ({change_percent:+.1f}%)")
            st.write("→ Maintain current status")
        
        st.markdown("#### 💡 AI Recommended Actions")
        st.write(f"- Expected operation time: {max(1, abs(change) // 10)} hours")
        st.write(f"- Recommended fan intensity: {min(100, 50 + abs(change))}%")
        st.write(f"- Expected power consumption: {max(1, abs(change) * 15)}Wh")

with tab3:
    st.markdown("### 🔗 Public Data-Based AI Comprehensive Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏗️ Construction Site Impact Analysis")
        for site in construction_sites:
            st.write(f"**{site['name']}**")
            st.write(f"- Impact radius: {site['radius']}m")
            st.write(f"- PM2.5 increase rate: {site['pm_increase']}")
            st.write(f"- AI Decision: Priority operation of nearby barricades")
            st.write("")
        
        st.markdown("#### 🚗 Traffic Data (14:00 Current)")
        traffic_data = pd.DataFrame({
            "Road": ["Nguyen Hue St.", "Le Loi Blvd.", "Vo Van Kiet Blvd."],
            "Vehicles/hr": [1240, 890, 1050],
            "vs Normal": ["+20%", "+5%", "+15%"],
            "AI Impact": ["1.2x", "1.05x", "1.15x"]
        })
        st.dataframe(traffic_data, use_container_width=True)
    
    with col2:
        st.markdown("#### 🚸 Vulnerable Facility Protection")
        for fac in vulnerable_facilities:
            st.write(f"**{fac['name']} ({fac['type']})**")
            st.write(f"- {fac['hours']}")
            st.write(f"- Protection status: Under priority care")
            st.write("")
        
        st.markdown("#### 🌤️ Weather Data (Real-time)")
        weather_data = {
            "Wind Dir.": "SW → NE", "Wind Speed": "3 m/s",
            "Humidity": "75% (High)", "Temperature": "32°C",
            "AI Impact": "1.15x (PM increase due to traffic)"
        }
        for key, value in weather_data.items():
            st.write(f"**{key}:** {value}")
    
    st.markdown("---")
    st.markdown("### 🧠 AI Comprehensive Decision Results (14:00 Baseline)")
    st.success("""
    **Current Situation Summary:**
    - 🏗️ District 1 Metro construction in progress → Active PM2.5 source
    - 💨 SW wind 3m/s → Eastward (Ben Thanh) dispersion expected
    - 🚗 Traffic volume 20% above normal
    - 🌤️ 75% humidity (tropical) → PM retention increased
    - 🚸 1 hour before kindergarten pickup → Special care needed
    
    **AI Final Decision:**
    ✅ Rank #1: District 1 Center (Current 155 → Predicted 202, 89% confidence)
    ✅ Rank #2: Ben Thanh Market (Current 115 → Predicted 141, 85% confidence)
    ✅ Rank #3: Saigon River Park (Current 78 → Predicted 97, 91% confidence)
    ⚠️ Special Action: Thu Duc City sensor anomaly → Maintenance team dispatch
    """)

with tab4:
    st.markdown("### 📱 Citizen View via QR Code")
    citizen_device = devices[0]
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"## Air Quality near {citizen_device['name']}")
        st.markdown(f"### {citizen_device['status']}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Current PM2.5", f"{citizen_device['pm_now']} μg/m³")
        with c2:
            st.metric("2 Hours Prediction", f"{citizen_device['pm_predict']} μg/m³", 
                     delta=f"{citizen_device['pm_predict'] - citizen_device['pm_now']}", delta_color="inverse")
        with c3:
            if citizen_device['pm_now'] < 50:
                air_quality, quality_color = "Good", "🟢"
            elif citizen_device['pm_now'] < 80:
                air_quality, quality_color = "Moderate", "🟡"
            elif citizen_device['pm_now'] < 150:
                air_quality, quality_color = "Unhealthy", "🟠"
            else:
                air_quality, quality_color = "Very Unhealthy", "🔴"
            st.metric("Air Quality Level", f"{quality_color} {air_quality}")
        
        st.markdown("---")
        st.markdown("### 💡 AI Health Guidance")
        
        if citizen_device['pm_now'] >= 80:
            st.warning(f"""
            ⚠️ **Current air quality is unhealthy** (PM2.5: {citizen_device['pm_now']})
            
            **Health Protection Guidelines:**
            - 👶 Children, elderly, and respiratory patients should limit outdoor activities
            - 😷 N95 masks recommended when going outside
            - 🏃 Avoid strenuous outdoor exercise
            - 🪟 Postpone indoor ventilation
            
            **AI Response Status:**
            - ✅ Local barricades operating to improve air quality
            - 📊 AI Forecast: {citizen_device['pm_predict']} expected in 2 hours ({citizen_device['confidence']}% confidence)
            """)
        else:
            st.success("""
            ✅ **Current air quality is good**
            
            - 😊 Outdoor activities are safe
            - 🌳 Enjoy walking, exercise and outdoor activities
            - 🤖 AI continuously monitoring air quality
            """)
        
        st.markdown("### 📊 Real-time Trend (Last 5 Hours)")
        chart_df = pd.DataFrame({
            "PM2.5": citizen_device["pm_data"][-5:] + [citizen_device["pm_predict"]],
        }, index=[f"-{5-i}h" for i in range(5)] + ["2h forecast"])
        st.line_chart(chart_df, height=200)
        
        st.caption(f"※ AI prediction confidence: {citizen_device['confidence']}% | Last update: {st.session_state.scenario_time}")
    
    with col2:
        st.markdown("### 📱 QR Code")
        st.markdown(f'<img src="{citizen_device["qr"]}" width="200" style="border: 2px solid #ccc; padding: 10px; background: white;">', unsafe_allow_html=True)
        st.caption("Scan QR Code")
        st.markdown("---")
        
        st.markdown("### 📍 Nearby Facilities")
        st.write("**Within 200m:**")
        st.write("- 🏫 District 1 People's Committee")
        st.write("- 🏪 3 Convenience Stores")
        st.write("- 🚇 Ben Thanh Station 500m")
        st.write("")
        
        st.markdown("### ℹ️ Service Information")
        st.write("- 📱 Real-time air quality check")
        st.write("- 🔮 AI prediction information")
        st.write("- 💊 Health action guidelines")
        st.write("- 📢 Citizen feedback reception")
        st.write("")
        
        if st.button("😷 Having trouble breathing now", key="citizen_feedback"):
            st.warning("""
            Citizen feedback delivered to AI!
            
            When 10+ reports received:
            - Auto-increase barricade intensity
            - Control center emergency inspection
            - Additional nearby devices activated
            """)
        
        st.caption("🤖 HCMC Smart City AI Air Quality Management Service")
        st.caption("Contact: +84-28-XXX-XXXX")

st.markdown("---")
st.markdown("### 📊 System Performance Summary (Today)")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("AI Predictions", "120 times", delta="+15%")
with col2:
    st.metric("Avg. Confidence", "87%", delta="+3%")
with col3:
    st.metric("Anomalies Detected", "1 case", delta="Sensor failure")
with col4:
    st.metric("Citizen Access", "342 people", delta="+28%")
with col5:
    st.metric("Total Cost Saved", f"₫{total_savings * 1000:,}", delta="↑ 12%")

st.caption("🤖 AI-Based Ho Chi Minh Smart Barricade | Real-time Data Integration System | v2.0 Enhanced AI")
