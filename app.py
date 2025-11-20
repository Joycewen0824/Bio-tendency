import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ===== 页面基础设置 =====
st.set_page_config(page_title="实验趋势与流式示意图小助手", layout="wide")

st.title("🧪 实验趋势 & 流式示意图生成器")
st.write("根据常见实验场景（qPCR / WB / 肿瘤曲线 / 细胞表型 / 流式点图），快速生成示意图用于方案。")

# ===== 选择模式：一维趋势图 vs 流式点图 =====
mode = st.radio(
    "选择图形模式",
    ["方案趋势图（柱状/折线）", "流式点图示意（FACS 样式）"],
    horizontal=True,
)

# ===== 通用趋势模板定义（用于一维趋势图） =====
trend_options = {
    "持续上升": "linear_up",
    "持续下降": "linear_down",
    "先升后降（钟形）": "bell",
    "对照稳定，处理组升高": "control_flat_treated_up",
    "对照稳定，处理组降低": "control_flat_treated_down",
    "剂量依赖性上升": "dose_response_up",
    "剂量依赖性下降": "dose_response_down",
}

# 场景模板：预设常用实验的习惯命名
scene_configs = {
    "qPCR 相对表达（2^-ΔΔCt）": {
        "y_label": "相对表达量 (2^-ΔΔCt，Fold Change)",
        "title": "qPCR 相对表达量示意图",
        "chart_type": "柱状图",
        "x_label_type": "自定义文本",
        "trend_label": "对照稳定，处理组升高",
    },
    "Western Blot 灰度定量": {
        "y_label": "条带灰度（相对对照）",
        "title": "Western Blot 定量示意图",
        "chart_type": "柱状图",
        "x_label_type": "自定义文本",
        "trend_label": "对照稳定，处理组升高",
    },
    "肿瘤生长曲线": {
        "y_label": "肿瘤体积（相对初始）",
        "title": "肿瘤生长曲线示意图",
        "chart_type": "折线图",
        "x_label_type": "时间点 (Day 0,1,2...)",
        "trend_label": "持续上升",
    },
    "细胞表型变化（散点/折线）": {
        "y_label": "表型水平（相对对照）",
        "title": "细胞表型变化示意图",
        "chart_type": "折线图",
        "x_label_type": "时间点 (Day 0,1,2...)",
        "trend_label": "先升后降（钟形）",
    },
    "自定义通用趋势": {
        "y_label": "相对数值（任意单位）",
        "title": "自定义趋势示意图",
        "chart_type": "柱状图",
        "x_label_type": "自定义文本",
        "trend_label": "持续上升",
    },
}

# ===== 工具函数（一维趋势） =====
def generate_base_trend(trend: str, n: int, max_fold_change: float) -> np.ndarray:
    """根据趋势类型生成 1 组基础曲线（不含噪声）"""
    x = np.linspace(0, 1, n)

    if trend == "linear_up":
        return 1 + (max_fold_change - 1) * x
    elif trend == "linear_down":
        return max_fold_change - (max_fold_change - 1) * x
    elif trend == "bell":
        center = 0.5
        width = 0.2
        y = 1 + (max_fold_change - 1) * np.exp(-((x - center) ** 2) / (2 * width ** 2))
        return y
    elif trend == "dose_response_up":
        steepness = 10
        midpoint = 0.4
        y = 1 + (max_fold_change - 1) / (1 + np.exp(-steepness * (x - midpoint)))
        return y
    elif trend == "dose_response_down":
        steepness = 10
        midpoint = 0.4
        y = max_fold_change - (max_fold_change - 1) / (1 + np.exp(-steepness * (x - midpoint)))
        return y
    else:
        return np.ones(n)


def add_noise(y: np.ndarray, noise_percent: float) -> np.ndarray:
    """加一点随机波动，模拟实验误差"""
    if noise_percent <= 0:
        return y
    noise = np.random.normal(loc=0.0, scale=noise_percent / 100.0, size=y.shape)
    return y * (1 + noise)


def generate_all_groups(
    trend: str,
    n_points: int,
    n_groups: int,
    max_fold_change: float,
    noise_percent: float,
    x_labels,
    group_names,
) -> pd.DataFrame:
    """
    生成所有组的数据，返回长格式 DataFrame：
    columns: ["X", "Group", "Value"]
    """
    control = np.ones(n_points)
    control = add_noise(control, noise_percent)

    records = []

    # 对照组
    for i in range(n_points):
        records.append(
            {
                "X": x_labels[i],
                "Group": group_names[0],
                "Value": control[i],
            }
        )

    # 处理组
    for g in range(1, n_groups):
        if trend in ["control_flat_treated_up", "control_flat_treated_down"]:
            if trend == "control_flat_treated_up":
                base = generate_base_trend("linear_up", n_points, max_fold_change)
            else:
                base = generate_base_trend("linear_down", n_points, max_fold_change)
        else:
            base = generate_base_trend(trend, n_points, max_fold_change)

        group_scale = 1 + 0.15 * g
        y = base * group_scale
        y = add_noise(y, noise_percent)

        for i in range(n_points):
            records.append(
                {
                    "X": x_labels[i],
                    "Group": group_names[g],
                    "Value": y[i],
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


# ===== 模式 1：方案趋势图（柱状/折线） =====
if mode == "方案趋势图（柱状/折线）":
    # ---- 侧边栏：基础设置 ----
    with st.sidebar:
        st.header("基本设置（通用）")
        exp_type = st.selectbox("实验类型", ["细胞实验", "动物实验", "其他"], key="exp_type")
        n_groups = st.number_input("组别数量", min_value=1, max_value=6, value=2, step=1, key="n_groups")
        n_points = st.slider("X 轴点数（时间点 / 剂量点 / 条件数）", 2, 10, 4, key="n_points")

        max_fold = st.slider("自动模式：最大变化倍数", 1.0, 10.0, 3.0, 0.5, key="max_fold")
        noise_level = st.slider("自动模式：随机波动（%）", 0, 30, 5, step=5, key="noise_level")

        st.markdown("---")
        st.caption("提示：现在支持两种方式：自动趋势 或 手动指定每个数值。")

    # ---- 主区域：场景模板 + 详细设置 ----
    st.subheader("1️⃣ 选择实验场景模板 & 数据方式")

    scene = st.selectbox(
        "场景模板",
        list(scene_configs.keys()),
        key="scene_template",
    )
    config = scene_configs[scene]

    # 是否用自动趋势，还是手动输入
    data_mode = st.radio(
        "数据生成方式",
        ["根据趋势自动生成（示意用）", "手动输入每个时间点/组的数值"],
        horizontal=True,
        key="data_mode",
    )

    # 场景切换时，设置默认值
    if "prev_scene" not in st.session_state or st.session_state["prev_scene"] != scene:
        st.session_state["y_label"] = config["y_label"]
        st.session_state["title"] = f"{exp_type} - {config['title']}"
        st.session_state["chart_type"] = config["chart_type"]
        st.session_state["x_label_type"] = config["x_label_type"]
        st.session_state["trend_label"] = config["trend_label"]
        st.session_state["prev_scene"] = scene

    col_scene1, col_scene2 = st.columns(2)

    with col_scene1:
        chart_type = st.radio(
            "图形类型",
            ["柱状图", "折线图"],
            key="chart_type",
        )

        x_label_type = st.selectbox(
            "X 轴类型",
            ["时间点 (Day 0,1,2...)", "剂量 (0,1,10,100...)", "自定义文本"],
            key="x_label_type",
        )

    with col_scene2:
        trend_labels = list(trend_options.keys())
        trend_label = st.selectbox(
            "（自动模式）趋势方向",
            trend_labels,
            key="trend_label",
        )
    trend_key = trend_options[trend_label]

    st.subheader("2️⃣ 坐标轴与组别信息")

    # X 轴标签
    if x_label_type.startswith("时间点"):
        x_values = [f"Day {i}" for i in range(st.session_state["n_points"])]
    elif x_label_type.startswith("剂量"):
        x_values = [str(int(10 ** (i))) for i in range(st.session_state["n_points"])]
    else:
        x_values = [f"P{i+1}" for i in range(st.session_state["n_points"])]

    # 自定义组名
    st.markdown("**组别名称设置**（按顺序对应图例）")
    group_names = []
    for i in range(st.session_state["n_groups"]):
        default_name = "Control" if i == 0 else f"Treatment {i}"
        name = st.text_input(
            f"组 {i+1} 名称",
            value=default_name,
            key=f"group_name_{i}",
        )
        group_names.append(name or default_name)

    col_axis1, col_axis2 = st.columns(2)
    with col_axis1:
        y_label = st.text_input("Y 轴名称", key="y_label")
    with col_axis2:
        title = st.text_input("图标题", key="title")

    # ===== 手动模式：提供可编辑表格 =====
    manual_wide_df = None
    if data_mode == "手动输入每个时间点/组的数值":
        st.subheader("3️⃣ 手动输入数据（类似 Excel）")

        # 生成一个默认的表格：行是 X，列是组
        idx = pd.Index(x_values, name="X")
        cols = group_names
        default_df = pd.DataFrame(1.0, index=idx, columns=cols)

        # 如果之前没有保存过，或形状/标签变化了，就重建
        if "manual_wide_df" not in st.session_state:
            st.session_state["manual_wide_df"] = default_df.copy()
        else:
            old = st.session_state["manual_wide_df"]
            if list(old.index) != list(idx) or list(old.columns) != list(cols):
                st.session_state["manual_wide_df"] = default_df.copy()

        manual_wide_df = st.data_editor(
            st.session_state["manual_wide_df"],
            key="manual_wide_df_editor",
            use_container_width=True,
            num_rows="fixed",
        )
        # 同步回 session_state，方便下次保留你填的数据
        st.session_state["manual_wide_df"] = manual_wide_df

        st.caption("提示：双击单元格即可修改数值，回车确认。")

    st.markdown("—— 下面点击按钮生成图形和数据 ——")

    # ---- 生成数据并画图 ----
    if st.button("📈 生成趋势图", use_container_width=True):
        if data_mode == "根据趋势自动生成（示意用）":
            # 自动模式：用趋势函数生成
            df = generate_all_groups(
                trend_key,
                n_points=st.session_state["n_points"],
                n_groups=st.session_state["n_groups"],
                max_fold_change=st.session_state["max_fold"],
                noise_percent=st.session_state["noise_level"],
                x_labels=x_values,
                group_names=group_names,
            )
        else:
            # 手动模式：使用表格中的数值
            manual_wide_df = st.session_state.get("manual_wide_df", None)
            if manual_wide_df is None:
                st.error("没有找到手动数据表，请先在上面编辑表格。")
                st.stop()
            # 宽表 → 长表
            tmp = manual_wide_df.copy()
            tmp.index = tmp.index.astype(str)
            df = tmp.reset_index(names="X").melt(
                id_vars="X", var_name="Group", value_name="Value"
            )

        st.subheader("图形预览")

        if chart_type == "柱状图":
            # 群组柱状图：同一个 X 下面，不同组并排显示
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("X:N", title=None),               # X 轴：时间点/剂量
                    xOffset="Group:N",                        # 关键：不同组在同一 X 下并排
                    y=alt.Y("Value:Q", title=y_label),        # Y 轴：数值
                    color=alt.Color("Group:N", title="组别"), # 颜色区分不同组
                )
                .properties(title=title)
            )
        else:
            chart = (
                alt.Chart(df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("X:N", title=None),
                    y=alt.Y("Value:Q", title=y_label),
                    color=alt.Color("Group:N", title="组别"),
                )
                .properties(title=title)
            )

        st.altair_chart(chart, use_container_width=True)

        st.subheader("数据表（可复制到 Excel / GraphPad）")
        wide_df_show = df.pivot(index="X", columns="Group", values="Value")
        st.dataframe(wide_df_show)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="💾 下载原始数据 CSV",
            data=csv,
            file_name="trend_data.csv",
            mime="text/csv",
        )

    else:
        if data_mode == "根据趋势自动生成（示意用）":
            st.info("选择好场景和趋势后，点击上面的 **📈 生成趋势图** 按钮。")
        else:
            st.info("在表格中填好每个数值，然后点击 **📈 生成趋势图**。")


# ===== 模式 2：流式点图示意（FACS-like） =====
else:
    # ---- 侧边栏：流式设置 ----
    with st.sidebar:
        st.header("流式点图设置")
        n_groups_flow = st.number_input("组别数量", min_value=1, max_value=5, value=2, step=1, key="n_groups_flow")
        cells_per_group = st.slider("每组点数（细胞数示意）", 50, 2000, 500, step=50, key="cells_per_group")

        flow_pattern = st.selectbox(
            "趋势类型（群体变化）",
            [
                "阳性细胞比例升高（右上象限增多）",
                "阳性细胞比例降低（右上象限减少）",
                "整体右移（Marker1 表达增强）",
                "整体上移（Marker2 表达增强）",
            ],
            key="flow_pattern",
        )

        spread = st.slider("云团散布程度（标准差）", 0.02, 0.3, 0.08, 0.01, key="flow_spread")

        st.markdown("---")
        st.caption("说明：这里生成的是 2D 高斯分布点，用于说明流式结果趋势。")

    # ---- 主区域：流式点图 ----
    st.subheader("1️⃣ 设置 Marker 与标题和组别名称")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        marker_x = st.text_input("X 轴 Marker", value="Marker 1（如 CD4）", key="marker_x")
    with col_f2:
        marker_y = st.text_input("Y 轴 Marker", value="Marker 2（如 CD8）", key="marker_y")

    flow_title = st.text_input("流式图标题", value="流式点图示意图", key="flow_title")

    # 自定义流式每组名字
    st.markdown("**流式组别名称设置**")
    flow_group_labels = []
    for i in range(n_groups_flow):
        default_name = "Control" if i == 0 else f"Treatment {i}"
        name = st.text_input(
            f"流式组 {i+1} 名称",
            value=default_name,
            key=f"flow_group_name_{i}",
        )
        flow_group_labels.append(name or default_name)

    # 工具函数：生成流式数据（使用自定义组名）
    def generate_flow_data(
        n_groups: int,
        cells_per_group: int,
        pattern: str,
        spread: float,
        group_labels,
    ) -> pd.DataFrame:
        records = []

        base_means = (0.2, 0.2)

        for g in range(n_groups):
            if g == 0:
                mean_x, mean_y = base_means
            else:
                if pattern == "阳性细胞比例升高（右上象限增多）":
                    mean_x = base_means[0] + 0.2 * g
                    mean_y = base_means[1] + 0.2 * g
                elif pattern == "阳性细胞比例降低（右上象限减少）":
                    mean_x = max(0.05, base_means[0] - 0.15 * g)
                    mean_y = max(0.05, base_means[1] - 0.15 * g)
                elif pattern == "整体右移（Marker1 表达增强）":
                    mean_x = base_means[0] + 0.25 * g
                    mean_y = base_means[1]
                else:
                    mean_x = base_means[0]
                    mean_y = base_means[1] + 0.25 * g

            xs = np.random.normal(loc=mean_x, scale=spread, size=cells_per_group)
            ys = np.random.normal(loc=mean_y, scale=spread, size=cells_per_group)

            for x, y in zip(xs, ys):
                records.append(
                    {
                        "X": float(np.clip(x, 0, 1)),
                        "Y": float(np.clip(y, 0, 1)),
                        "Group": group_labels[g],
                    }
                )

        df_flow = pd.DataFrame.from_records(records)
        return df_flow

    st.markdown("—— 下面点击按钮生成流式点图示意 ——")

    if st.button("🔬 生成流式点图", use_container_width=True):
        df_flow = generate_flow_data(
            n_groups=n_groups_flow,
            cells_per_group=cells_per_group,
            pattern=flow_pattern,
            spread=st.session_state["flow_spread"],
            group_labels=flow_group_labels,
        )

        st.subheader("2️⃣ 点图预览（0~1 归一化坐标）")

        flow_chart = (
            alt.Chart(df_flow)
            .mark_circle(size=20, opacity=0.4)
            .encode(
                x=alt.X("X:Q", title=marker_x, scale=alt.Scale(domain=(0, 1))),
                y=alt.Y("Y:Q", title=marker_y, scale=alt.Scale(domain=(0, 1))),
                color=alt.Color("Group:N"),
            )
            .properties(title=flow_title, width=500, height=500)
        )

        st.altair_chart(flow_chart, use_container_width=False)

        st.subheader("3️⃣ 原始点数据（可导出）")
        st.dataframe(df_flow.head(100))

        csv_flow = df_flow.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="💾 下载流式点数据 CSV",
            data=csv_flow,
            file_name="flow_scatter_data.csv",
            mime="text/csv",
        )

    else:
        st.info("设置好 Marker 与组名后，点击 **🔬 生成流式点图**。")