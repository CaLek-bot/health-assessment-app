import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 设置页面配置
st.set_page_config(page_title="健康状况评估系统", layout="centered")

# 页面标题
st.title("🏥 健康状况综合评估系统")
st.markdown("欢迎使用本系统，请填写以下信息以评估您的基本健康状态。")

# 分栏输入
with st.form("health_form"):
    st.header("🧍 个人基本信息")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("姓名", value="张三")
        age = st.number_input("年龄", min_value=1, max_value=120, value=25)
        gender = st.selectbox("性别", ["男", "女"])
    with col2:
        height = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0, value=170.0)
        weight = st.number_input("体重 (kg)", min_value=10.0, max_value=300.0, value=65.0)

    st.header("📊 可选健康指标")
    col3, col4 = st.columns(2)
    with col3:
        sbp = st.number_input("收缩压 SBP (高压 mmHg)", min_value=50, max_value=250, value=120)
        hr = st.number_input("静息心率 (bpm)", min_value=30, max_value=200, value=70)
    with col4:
        dbp = st.number_input("舒张压 DBP (低压 mmHg)", min_value=30, max_value=150, value=80)
        waist = st.number_input("腰围 (cm)", min_value=30.0, max_value=200.0, value=75.0)

    submit = st.form_submit_button("🚀 开始评估")

# 👉 评估逻辑
if submit:
    st.success("✅ 正在生成评估报告...")

    # BMI 计算
    height_m = height / 100
    bmi = weight / (height_m ** 2)

    if bmi < 18.5:
        bmi_status = "偏瘦"
        bmi_color = "blue"
    elif 18.5 <= bmi < 24:
        bmi_status = "正常"
        bmi_color = "green"
    elif 24 <= bmi < 28:
        bmi_status = "超重"
        bmi_color = "orange"
    else:
        bmi_status = "肥胖"
        bmi_color = "red"

    # 血压分析
    if sbp >= 140 or dbp >= 90:
        bp_status = "高血压"
        bp_color = "red"
    elif sbp < 90 or dbp < 60:
        bp_status = "低血压"
        bp_color = "orange"
    else:
        bp_status = "正常"
        bp_color = "green"

    # 心率评估
    if hr < 60:
        hr_status = "偏慢"
    elif hr > 100:
        hr_status = "偏快"
    else:
        hr_status = "正常"

    # 腰臀比建议（近似，只用腰围）
    waist_flag = "正常"
    if (gender == "男" and waist >= 90) or (gender == "女" and waist >= 85):
        waist_flag = "腹型肥胖"

    # ===================== 输出部分 =====================
    st.header("📄 健康评估报告")

    st.markdown(f"**姓名**：{name}  \n**性别**：{gender}  \n**年龄**：{age} 岁")

    col5, col6 = st.columns(2)
    with col5:
        st.markdown(f"**💪 BMI 指数：{bmi:.2f}**")
        st.markdown(f"体重分类：<span style='color:{bmi_color}'><strong>{bmi_status}</strong></span>",
                    unsafe_allow_html=True)

    with col6:
        st.markdown(f"**🩸 血压状态：{sbp}/{dbp} mmHg**")
        st.markdown(f"血压分类：<span style='color:{bp_color}'><strong>{bp_status}</strong></span>",
                    unsafe_allow_html=True)

    st.markdown(f"**❤️ 心率：{hr} bpm**（{hr_status}）")
    st.markdown(f"**腰围**：{waist} cm（{waist_flag}）")

    # 建议输出
    st.header("🧠 AI 健康建议")
    if bmi_status == "正常" and bp_status == "正常" and hr_status == "正常" and waist_flag == "正常":
        st.success("身体状况良好，请继续保持良好的生活习惯和锻炼习惯！")
    else:
        st.info("建议保持合理饮食、控制血压、规律运动，并定期体检。")

    # ===================== 图表部分 =====================
    st.header("📈 健康指标图示")

    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    # BMI 区间图改进版
    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.set_xlim(10, 40)
    ax.set_ylim(0, 1)

    # 区间色块
    bmi_ranges = [(10, 18.5, 'lightblue', '偏瘦'),
                  (18.5, 24, 'lightgreen', '正常'),
                  (24, 28, 'orange', '超重'),
                  (28, 40, 'red', '肥胖')]

    for start, end, color, label in bmi_ranges:
        ax.axvspan(start, end, color=color, alpha=0.4)
        ax.text((start + end) / 2, 0.5, label, ha='center', va='center', fontsize=10)

    # 当前 BMI 指示线
    ax.axvline(bmi, color='black', linewidth=2)
    ax.text(bmi, 0.9, f"BMI={bmi:.1f}", rotation=90, va='bottom', ha='center', fontsize=9, color='black')

    # 清除不必要元素
    ax.set_yticks([])
    ax.set_xlabel("BMI 值")
    ax.set_title("BMI 区间分布图")

    st.pyplot(fig)

    # 血压图（柱状）
    df_bp = pd.DataFrame({
        '项目': ['收缩压', '舒张压'],
        '值': [sbp, dbp]
    })
    st.bar_chart(df_bp.set_index("项目"))

    # ========== 健康问答助手模块 ==========

# 注意：问答输入放在表单外，避免和表单冲突导致刷新丢失
st.markdown("---")
st.markdown("## 🧠 健康问答助手")

# 使用 st.session_state 保持输入内容
if 'user_question' not in st.session_state:
    st.session_state['user_question'] = ''

user_question = st.text_area(
    "你可以问我一些关于健康的问题（如：我想减肥、怎么改善睡眠等）",
    value=st.session_state['user_question'],
    key='user_question'
)

def simple_health_bot(question):
    question = question.lower()

    if "减肥" in question:
        return """
        ✅ 减肥建议：
        - 合理控制热量摄入，避免高油高糖食物；
        - 每周保持3~5次有氧运动（如快走、游泳、骑车）；
        - 保证每天7-8小时高质量睡眠；
        - 避免久坐，每小时活动一次；
        - 晚餐尽量清淡，晚上8点后不进食；
        - 每天饮水 1500~2000ml。
        """
    elif "睡眠" in question:
        return """
        🛌 改善睡眠建议：
        - 每天固定上床与起床时间；
        - 睡前1小时不玩手机不看屏幕；
        - 睡前不喝浓茶咖啡或酒精饮料；
        - 保持安静、昏暗的睡眠环境；
        - 如持续失眠建议就医检查。
        """
    elif "增肌" in question or "健身" in question:
        return """
        💪 增肌健身建议：
        - 每日保证 1.5g/公斤体重以上蛋白质摄入；
        - 每周进行 3~5 次力量训练；
        - 合理补充碳水，训练后补餐；
        - 睡眠不少于7小时以利肌肉恢复；
        - 记录训练进展，每月复盘一次。
        """
    else:
        return "🤔 抱歉，我目前只支持关于减肥、睡眠、健身等主题的建议，后续会扩展更多内容！"

if user_question.strip():
    st.markdown("#### 🤖 健康建议：")
    st.write(simple_health_bot(user_question))

# 可选保存记录
with st.expander("💾 保存记录为 CSV（可选）"):
    save = st.checkbox("保存评估记录")
    if save and submit:  # 只有提交后才保存
        df = pd.DataFrame([{
            "姓名": name, "年龄": age, "性别": gender,
            "身高": height, "体重": weight, "BMI": round(bmi, 2), "体重状态": bmi_status,
            "SBP": sbp, "DBP": dbp, "血压状态": bp_status,
            "心率": hr, "心率状态": hr_status,
            "腰围": waist, "腹型肥胖": waist_flag
        }])
        df.to_csv("health_records.csv", mode='a', header=False, index=False)
        st.success("✅ 已保存到 health_records.csv")
