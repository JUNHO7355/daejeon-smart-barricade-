import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta

# 기본 설정
st.set_page_config(page_title="AI 기반 대전 스마트 바리케이드", layout="wide")

# AI 예측 함수
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

# QR코드 생성
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

# 데이터 초기화
if "devices" not in st.session_state:
    st.session_state.scenario_time = "2024년 11월 11일 14:00"
    st.session_state.scenario_weather = "맑음, 서풍 3m/s, 습도 45%"
    
    pm_scenarios = {
        "대전시청 앞": [65, 68, 72, 75, 78, 82, 85, 88, 92, 95],
        "유성온천역": [95, 102, 108, 115, 120, 125, 128, 130, 132, 135],
        "정부청사역": [55, 58, 60, 62, 65, 63, 61, 59, 57, 55],
        "중앙로역": [78, 78, 78, 78, 78, 78, 78, 78, 78, 78],
        "대덕연구단지": [42, 45, 48, 52, 55, 58, 61, 63, 65, 68],
    }
    
    device_scenarios = {
        "대전시청 앞": {
            "lat": 36.3504, "lng": 127.3845, "battery": 85, "rain": 35,
            "weather_factor": 1.1, "traffic_factor": 1.2,
            "construction_nearby": False, "sensor_stable": True,
            "priority": 2, "reason": "서풍으로 공사장 미세먼지 확산 예상"
        },
        "유성온천역": {
            "lat": 36.3553, "lng": 127.3449, "battery": 72, "rain": 15,
            "weather_factor": 1.0, "traffic_factor": 1.3,
            "construction_nearby": True, "sensor_stable": True,
            "priority": 1, "reason": "도로공사 현장 200m 이내, 최우선 가동"
        },
        "정부청사역": {
            "lat": 36.3626, "lng": 127.3829, "battery": 92, "rain": 80,
            "weather_factor": 0.9, "traffic_factor": 1.0,
            "construction_nearby": False, "sensor_stable": True,
            "priority": 5, "reason": "하락 추세, 정상 모니터링"
        },
        "중앙로역": {
            "lat": 36.3286, "lng": 127.4276, "battery": 45, "rain": 60,
            "weather_factor": 1.0, "traffic_factor": 1.4,
            "construction_nearby": False, "sensor_stable": False,
            "priority": 4, "reason": "센서 이상 감지 (3시간째 동일 수치)"
        },
        "대덕연구단지": {
            "lat": 36.3830, "lng": 127.3775, "battery": 88, "rain": 45,
            "weather_factor": 0.95, "traffic_factor": 0.9,
            "construction_nearby": False, "sensor_stable": True,
            "priority": 3, "reason": "완만한 증가 추세, 예방적 모니터링"
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
            d["status"] = "💧 세척모드 예측"
            d["color"] = "blue"
        elif d["pm_predict"] >= 80:
            d["status"] = "🌀 팬 작동 예측"
            d["color"] = "red"
        else:
            d["status"] = "🟢 정상 예측"
            d["color"] = "green"
        
        d["qr"] = make_qr("https://google.com")
        st.session_state.devices.append(d)
    
    st.session_state.construction_sites = [
        {"name": "유성구 도로공사 (현재 진행 중)", "lat": 36.3560, "lng": 127.3400, "radius": 200, "pm_increase": "+60%"},
        {"name": "둔산대로 지하철 공사", "lat": 36.3500, "lng": 127.3800, "radius": 300, "pm_increase": "+45%"},
    ]
    
    st.session_state.vulnerable_facilities = [
        {"name": "해님어린이집", "lat": 36.3520, "lng": 127.3460, "type": "어린이집", "hours": "하원 15:00"},
        {"name": "행복경로당", "lat": 36.3600, "lng": 127.3800, "type": "경로당", "hours": "이용시간 14:00-17:00"},
        {"name": "대전중앙병원", "lat": 36.3300, "lng": 127.4250, "type": "병원", "hours": "24시간"},
    ]
    
    st.session_state.cost_savings = {"power": 18400, "filter": 45000, "maintenance": 12000}

devices = st.session_state.devices
construction_sites = st.session_state.construction_sites
vulnerable_facilities = st.session_state.vulnerable_facilities
cost_savings = st.session_state.cost_savings

def generate_ai_decision():
    return [
        {"icon": "🏗️", "text": "유성구 도로공사 감지 (반경 200m) → PM2.5 60% 증가 예상 → 유성온천역 바리케이드 최우선 가동 (우선순위 1위)"},
        {"icon": "💨", "text": "현재 풍향 서→동 3m/s → 공사장 미세먼지가 대전시청 방향 확산 → 대전시청 앞 선제 대응 (우선순위 2위)"},
        {"icon": "🚸", "text": "해님어린이집 하원 시간 1시간 전 (15:00) → 주변 200m 이내 공기질 우선 정화 모드 활성화"},
        {"icon": "📊", "text": "중앙로역 센서 이상 감지 (3시간째 78 고정) → 유지보수팀 자동 출동 요청 → 임시 모니터링 강화"}
    ]

def generate_alerts():
    return [
        {"type": "warning", "icon": "⚠️", "text": "중앙로역 센서 이상 감지 (3시간째 동일 수치 78) → 유지보수 필요"},
        {"type": "battery", "icon": "🔋", "text": "중앙로역 배터리 45% → 24시간 내 충전 필요 (현재 소모율 기준)"},
        {"type": "pollution", "icon": "🚨", "text": "유성온천역 2시간 후 PM2.5 162 예상 (신뢰도 89%) → 선제 최대 가동 권장"}
    ]

# 헤더
st.title("🤖 AI 기반 대전형 스마트 바리케이드 관제 시스템")
st.caption(f"📅 시나리오 시간: {st.session_state.scenario_time} | 🌤️ {st.session_state.scenario_weather}")

total_savings = sum(cost_savings.values())
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 오늘 AI 절감 비용", f"₩{total_savings:,}", delta="↑ 전일 대비 12%")
with col2:
    st.metric("⚡ 전력비 절감", f"₩{cost_savings['power']:,}", delta="5회 가동 방지")
with col3:
    st.metric("🔧 필터 교체 연기", f"₩{cost_savings['filter']:,}", delta="1회 연장")
with col4:
    st.metric("🛠️ 조기 고장 감지", f"₩{cost_savings['maintenance']:,}", delta="1건 예방")

st.markdown("---")

st.markdown("### 🧠 AI 실시간 의사결정 현황")
for decision in generate_ai_decision():
    st.info(f"{decision['icon']} {decision['text']}")

st.markdown("---")

alerts = generate_alerts()
if alerts:
    st.markdown("### 🔔 AI 이상 탐지 알림")
    for alert in alerts:
        if alert['type'] == 'warning':
            st.warning(f"{alert['icon']} {alert['text']}")
        elif alert['type'] == 'battery':
            st.error(f"{alert['icon']} {alert['text']}")
        else:
            st.warning(f"{alert['icon']} {alert['text']}")
    st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🏙 통합 관제 지도", "📊 AI 예측 상세 분석", "🏗️ 공공데이터 연계", "📱 시민용 화면"])

with tab1:
    col1, col2 = st.columns([1.4, 1])
    
    with col1:
        st.markdown("### 🗺️ 실시간 관제 맵")
        m = folium.Map(location=[36.35, 127.38], zoom_start=12, tiles="OpenStreetMap")
        
        for d in devices:
            popup_html = f"""
            <div style="width:250px">
            <b style="font-size:14px">{d['name']}</b><br><br>
            <b>현재 PM2.5:</b> {d['pm_now']} μg/m³<br>
            <b>예측 PM2.5:</b> {d['pm_predict']} μg/m³<br>
            <b>신뢰도:</b> {d['confidence']}%<br>
            <b>우선순위:</b> {d['priority']}위<br>
            <b>상태:</b> {d['status']}<br><br>
            <img src="{d['qr']}" width="150" height="150" style="display:block; margin:10px auto;">
            </div>
            """
            folium.CircleMarker(
                location=[d["lat"], d["lng"]],
                radius=12,
                color=d["color"],
                fill=True,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(m)
        
        for site in construction_sites:
            folium.Circle(
                location=[site["lat"], site["lng"]],
                radius=site["radius"],
                color="red",
                fill=True,
                fill_opacity=0.2,
                popup=f"<b>🏗️ {site['name']}</b><br>PM2.5 영향: {site['pm_increase']}",
            ).add_to(m)
        
        for fac in vulnerable_facilities:
            folium.Marker(
                location=[fac["lat"], fac["lng"]],
                icon=folium.Icon(color="orange", icon="info-sign"),
                popup=f"<b>🚸 {fac['name']}</b><br>{fac['type']}<br>{fac['hours']}",
            ).add_to(m)
        
        folium_static(m, height=500, width=750)
    
    with col2:
        st.markdown("### 📋 AI 우선순위 판단 결과")
        sorted_devices = sorted(devices, key=lambda x: x["priority"])
        df = pd.DataFrame([
            [d["priority"], d["name"], d["pm_now"], f"{d['pm_predict']} ({d['confidence']}%)",
             "✅" if d["sensor_stable"] else "⚠️", d["status"]]
            for d in sorted_devices
        ], columns=["순위", "위치", "현재", "예측(신뢰도)", "센서", "AI 판단"])
        st.dataframe(df, use_container_width=True, height=250)
        
        st.markdown("### 🎯 AI 배치 전략 근거")
        st.write("**1순위: 유성온천역**")
        st.write("- 공사장 200m 이내 (PM2.5 +60%)")
        st.write("- 현재 135 → 예측 162 (급증 추세)")
        st.write("")
        st.write("**2순위: 대전시청 앞**")
        st.write("- 서풍으로 공사장 확산 경로")
        st.write("- 완만한 증가 추세 (선제 대응)")
        st.write("")
        st.write("**취약계층 특별 관리**")
        st.write("- 어린이집 하원 1시간 전 가동")
        st.write("- 경로당 이용 시간대 집중 정화")

with tab2:
    st.markdown("### 🔮 AI 다변수 예측 모델 상세 분석")
    names = [d["name"] for d in devices]
    selected_name = st.selectbox("분석할 장치를 선택하세요.", names)
    selected = next(d for d in devices if d["name"] == selected_name)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"#### 📍 {selected['name']}")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("현재 PM2.5", f"{selected['pm_now']} μg/m³")
            st.metric("기본 예측", f"{selected['pm_base_predict']} μg/m³")
        with col_b:
            st.metric("AI 최종 예측", f"{selected['pm_predict']} μg/m³", 
                     delta=f"+{selected['pm_predict'] - selected['pm_now']}", delta_color="inverse")
            st.metric("예측 신뢰도", f"{selected['confidence']}%")
        
        st.write("")
        sensor_status = "✅ 정상" if selected["sensor_stable"] else "⚠️ 이상 감지"
        st.write(f"**센서 상태:** {sensor_status}")
        st.write(f"**AI 판단:** {selected['status']}")
        st.write(f"**우선순위:** {selected['priority']}위")
        st.write(f"**판단 근거:** {selected['reason']}")
        
        st.write("---")
        st.markdown("#### 🧮 AI 예측 변수 분석")
        st.write(f"**날씨 영향도:** {selected['weather_factor']:.2f}x")
        if selected['weather_factor'] > 1.0:
            st.caption("↑ 건조한 날씨로 미세먼지 증가 예상")
        else:
            st.caption("↓ 습도 높아 미세먼지 감소 예상")
        
        st.write(f"**교통량 영향도:** {selected['traffic_factor']:.2f}x")
        st.caption(f"현재 교통량 평소 대비 {int((selected['traffic_factor']-1)*100)}% 수준")
        st.write(f"**공사장 인접:** {'예 (+40%)' if selected['construction_nearby'] else '아니오'}")
        st.write(f"**추세 계수:** {selected['trend_factor']:.2f}x")
    
    with c2:
        st.markdown("#### 📈 시간대별 PM2.5 변화")
        chart_data = pd.DataFrame({
            "실측값": selected["pm_data"] + [None],
            "AI 예측": [None] * len(selected["pm_data"]) + [selected["pm_predict"]],
            "기본 예측": [None] * len(selected["pm_data"]) + [selected["pm_base_predict"]],
        }, index=[f"-{10-i}h" for i in range(10)] + ["2h후"])
        st.line_chart(chart_data, height=280)
        
        st.markdown("#### 🔍 예측 분석")
        change = selected["pm_predict"] - selected["pm_now"]
        change_percent = (change / selected["pm_now"]) * 100
        
        if change > 20:
            st.error(f"⚠️ **급증 예상**: +{change} μg/m³ ({change_percent:+.1f}%)")
            st.write("→ 즉시 최대 강도 가동 권장")
        elif change > 10:
            st.warning(f"⚡ **증가 예상**: +{change} μg/m³ ({change_percent:+.1f}%)")
            st.write("→ 선제적 가동 권장")
        elif change < -10:
            st.success(f"✅ **개선 예상**: {change} μg/m³ ({change_percent:+.1f}%)")
            st.write("→ 정상 모니터링")
        else:
            st.info(f"📊 **안정 예상**: {change:+} μg/m³ ({change_percent:+.1f}%)")
            st.write("→ 현상 유지")
        
        st.markdown("#### 💡 AI 권장 조치")
        st.write(f"- 예상 가동 시간: {max(1, abs(change) // 10)}시간")
        st.write(f"- 권장 팬 강도: {min(100, 50 + abs(change))}%")
        st.write(f"- 예상 전력 소모: {max(1, abs(change) * 15)}Wh")

with tab3:
    st.markdown("### 🔗 공공데이터 기반 AI 종합 분석")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏗️ 공사현장 영향 분석")
        for site in construction_sites:
            st.write(f"**{site['name']}**")
            st.write(f"- 영향 반경: {site['radius']}m")
            st.write(f"- PM2.5 증가율: {site['pm_increase']}")
            st.write(f"- AI 판단: 인근 바리케이드 우선 가동")
            st.write("")
        
        st.markdown("#### 🚗 교통량 데이터 (14시 현재)")
        traffic_data = pd.DataFrame({
            "도로": ["둔산대로", "대덕대로", "유성대로"],
            "차량/시": [1240, 890, 1050],
            "평소 대비": ["+20%", "+5%", "+15%"],
            "AI 영향도": ["1.2x", "1.05x", "1.15x"]
        })
        st.dataframe(traffic_data, use_container_width=True)
    
    with col2:
        st.markdown("#### 🚸 취약계층 시설 보호")
        for fac in vulnerable_facilities:
            st.write(f"**{fac['name']} ({fac['type']})**")
            st.write(f"- {fac['hours']}")
            st.write(f"- 보호 상태: 우선 관리 중")
            st.write("")
        
        st.markdown("#### 🌤️ 기상 데이터 (실시간)")
        weather_data = {
            "풍향": "서풍 → 동풍", "풍속": "3 m/s",
            "습도": "45% (건조)", "온도": "18°C",
            "AI 영향도": "1.1x (미세먼지 증가)"
        }
        for key, value in weather_data.items():
            st.write(f"**{key}:** {value}")
    
    st.markdown("---")
    st.markdown("### 🧠 AI 종합 판단 결과 (14:00 기준)")
    st.success("""
    **현재 상황 종합:**
    - 🏗️ 유성구 도로공사 진행 중 → PM2.5 발생원 활성
    - 💨 서풍 3m/s → 동쪽(대전시청 방향) 확산 예상
    - 🚗 교통량 평소 대비 20% 증가
    - 🌤️ 습도 45% (건조) → 미세먼지 체류 증가
    - 🚸 어린이집 하원 1시간 전 → 특별 관리 필요
    
    **AI 최종 판단:**
    ✅ 1순위: 유성온천역 (현재 135 → 예측 162, 신뢰도 89%)
    ✅ 2순위: 대전시청 앞 (현재 95 → 예측 121, 신뢰도 85%)
    ✅ 3순위: 대덕연구단지 (현재 68 → 예측 87, 신뢰도 91%)
    ⚠️ 특별 조치: 중앙로역 센서 이상 → 유지보수팀 출동
    """)

with tab4:
    st.markdown("### 📱 QR 접속 시 시민이 보는 화면")
    citizen_device = devices[0]
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"## {citizen_device['name']} 주변 공기질")
        st.markdown(f"### {citizen_device['status']}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("현재 PM2.5", f"{citizen_device['pm_now']} μg/m³")
        with c2:
            st.metric("2시간 후 예측", f"{citizen_device['pm_predict']} μg/m³", 
                     delta=f"{citizen_device['pm_predict'] - citizen_device['pm_now']}", delta_color="inverse")
        with c3:
            if citizen_device['pm_now'] < 50:
                air_quality, quality_color = "좋음", "🟢"
            elif citizen_device['pm_now'] < 80:
                air_quality, quality_color = "보통", "🟡"
            elif citizen_device['pm_now'] < 150:
                air_quality, quality_color = "나쁨", "🟠"
            else:
                air_quality, quality_color = "매우 나쁨", "🔴"
            st.metric("공기질 등급", f"{quality_color} {air_quality}")
        
        st.markdown("---")
        st.markdown("### 💡 AI가 드리는 건강 안내")
        
        if citizen_device['pm_now'] >= 80:
            st.warning(f"""
            ⚠️ **현재 공기질이 나쁩니다** (PM2.5: {citizen_device['pm_now']})
            
            **건강 보호 행동 지침:**
            - 👶 어린이, 노약자, 호흡기 질환자는 실외 활동을 자제해 주세요
            - 😷 외출 시 KF94 마스크 착용을 권장합니다
            - 🏃 격렬한 실외 운동은 피해주세요
            - 🪟 실내 환기는 잠시 미뤄주세요
            
            **AI 대응 현황:**
            - ✅ 이 구역 바리케이드가 공기질 개선을 위해 작동 중입니다
            - 📊 AI 예측: 2시간 후 {citizen_device['pm_predict']} 예상 (신뢰도 {citizen_device['confidence']}%)
            """)
        else:
            st.success("""
            ✅ **현재 공기질이 양호합니다**
            
            - 😊 실외 활동이 가능합니다
            - 🌳 산책, 운동 등 야외 활동을 즐기세요
            - 🤖 AI가 지속적으로 공기질을 모니터링 중입니다
            """)
        
        st.markdown("### 📊 실시간 변화 추이 (최근 5시간)")
        chart_df = pd.DataFrame({
            "PM2.5": citizen_device["pm_data"][-5:] + [citizen_device["pm_predict"]],
        }, index=[f"-{5-i}h" for i in range(5)] + ["2h후 예측"])
        st.line_chart(chart_df, height=200)
        
        st.caption(f"※ AI 예측 신뢰도: {citizen_device['confidence']}% | 마지막 업데이트: {st.session_state.scenario_time}")
    
    with col2:
        st.markdown("### 📱 QR 코드")
        st.markdown(f'<img src="{citizen_device["qr"]}" width="200" style="border: 2px solid #ccc; padding: 10px; background: white;">', unsafe_allow_html=True)
        st.caption("QR 코드 스캔")
        st.markdown("---")
        
        st.markdown("### 📍 주변 시설 정보")
        st.write("**200m 이내 시설:**")
        st.write("- 🏫 대전시청")
        st.write("- 🏪 편의점 3곳")
        st.write("- 🚇 정부청사역 500m")
        st.write("")
        
        st.markdown("### ℹ️ 이용 안내")
        st.write("- 📱 실시간 공기질 확인")
        st.write("- 🔮 AI 예측 정보 제공")
        st.write("- 💊 건강 행동 지침 안내")
        st.write("- 📢 시민 의견 접수")
        st.write("")
        
        if st.button("😷 지금 숨쉬기 힘들어요", key="citizen_feedback"):
            st.warning("""
            시민 의견이 AI에 전달되었습니다!
            
            10명 이상 신고 시:
            - 바리케이드 강도 자동 증가
            - 관제센터 긴급 점검
            - 인근 장치 추가 가동
            """)
        
        st.caption("🤖 대전시 스마트시티 AI 공기질 관리 서비스")
        st.caption("문의: 042-XXX-XXXX")

st.markdown("---")
st.markdown("### 📊 시스템 성능 요약 (오늘 기준)")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("AI 예측 횟수", "120회", delta="+15%")
with col2:
    st.metric("평균 신뢰도", "87%", delta="+3%")
with col3:
    st.metric("이상 감지", "1건", delta="센서 고장")
with col4:
    st.metric("시민 접속", "342명", delta="+28%")
with col5:
    st.metric("총 절감 비용", f"₩{total_savings:,}", delta="↑ 12%")

st.caption("🤖 AI 기반 대전형 스마트 바리케이드 | 실시간 데이터 연동 시스템 | v2.0 Enhanced AI")