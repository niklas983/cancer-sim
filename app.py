import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# -----------------------------
# Age-group presets
# -----------------------------
AGE_PRESETS = {
    "Custom": None,
    "40-49 (low risk)": 0.005,
    "50-69 (moderate risk)": 0.02,
    "70+ (high risk)": 0.05
}

# -----------------------------
# Simulation with CI
# -----------------------------
def run_simulation(n_patients, n_simulations, prevalence, sensitivity, specificity):

    TP_list, FP_list, TN_list, FN_list = [], [], [], []

    for _ in range(n_simulations):
        disease = np.random.rand(n_patients) < prevalence

        test_positive = np.where(
            disease,
            np.random.rand(n_patients) < sensitivity,
            np.random.rand(n_patients) < (1 - specificity)
        )

        TP = np.sum((disease == 1) & (test_positive == 1))
        FP = np.sum((disease == 0) & (test_positive == 1))
        TN = np.sum((disease == 0) & (test_positive == 0))
        FN = np.sum((disease == 1) & (test_positive == 0))

        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)

    TP_arr, FP_arr, TN_arr, FN_arr = map(np.array, [TP_list, FP_list, TN_list, FN_list])

    def ci(x):
        return np.percentile(x, 2.5), np.percentile(x, 97.5)

    TP, FP, TN, FN = TP_arr.mean(), FP_arr.mean(), TN_arr.mean(), FN_arr.mean()

    sens = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    acc = (TP + TN) / (TP + FP + TN + FN)

    bal_acc = (sens + spec) / 2
    F1 = 2 * (PPV * sens) / (PPV + sens) if (PPV + sens) > 0 else 0

    LR_pos = sens / (1 - spec) if (1 - spec) > 0 else np.inf
    LR_neg = (1 - sens) / spec if spec > 0 else np.inf

    DOR = (TP * TN) / (FP * FN) if (FP * FN) > 0 else np.inf
    NNS = (TP + FP + TN + FN) / TP if TP > 0 else np.inf

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "TP_low": ci(TP_arr)[0], "TP_high": ci(TP_arr)[1],
        "FP_low": ci(FP_arr)[0], "FP_high": ci(FP_arr)[1],
        "Sensitivity": sens, "Specificity": spec,
        "PPV": PPV, "NPV": NPV,
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "F1 Score": F1,
        "LR+": LR_pos, "LR-": LR_neg,
        "DOR": DOR,
        "NNS": NNS
    }

# -----------------------------
# Tornado analysis
# -----------------------------
def tornado_analysis(base_scenario, n_patients, n_simulations, variation=0.2):

    params = ["prevalence", "sensitivity", "specificity"]

    base_result = run_simulation(
        n_patients,
        n_simulations,
        base_scenario["prevalence"],
        base_scenario["sensitivity"],
        base_scenario["specificity"]
    )

    base_tp = base_result["TP"]
    results = []

    for param in params:
        low = base_scenario[param] * (1 - variation)
        high = base_scenario[param] * (1 + variation)

        sc_low = base_scenario.copy()
        sc_high = base_scenario.copy()

        sc_low[param] = max(0, low)
        sc_high[param] = min(1, high)

        res_low = run_simulation(n_patients, n_simulations,
                                sc_low["prevalence"],
                                sc_low["sensitivity"],
                                sc_low["specificity"])

        res_high = run_simulation(n_patients, n_simulations,
                                 sc_high["prevalence"],
                                 sc_high["sensitivity"],
                                 sc_high["specificity"])

        results.append({
            "Parameter": param,
            "Low Impact": res_low["TP"] - base_tp,
            "High Impact": res_high["TP"] - base_tp
        })

    return pd.DataFrame(results)

# -----------------------------
# Excel export
# -----------------------------
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# -----------------------------
# UI
# -----------------------------
st.title("🧪 Cancer Screening Simulator (Full Version)")

st.sidebar.header("Global Parameters")
n_patients = st.sidebar.number_input("Patients", 1000, 200000, 10000)
n_simulations = st.sidebar.number_input("Simulations", 100, 20000, 10000)

if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

st.sidebar.header("Scenario Input")

name = st.sidebar.text_input("Scenario name")

age_group = st.sidebar.selectbox("Age group", list(AGE_PRESETS.keys()))
prevalence = AGE_PRESETS[age_group] if AGE_PRESETS[age_group] else st.sidebar.slider("Prevalence", 0.001, 0.2, 0.01)

sensitivity = st.sidebar.slider("Sensitivity", 0.5, 1.0, 0.85)
specificity = st.sidebar.slider("Specificity", 0.5, 1.0, 0.90)

st.sidebar.subheader("Costs")
cost_test = st.sidebar.number_input("Cost/test", 1.0, 500.0, 20.0)
cost_fp = st.sidebar.number_input("Cost/FP", 0.0, 5000.0, 200.0)
cost_fn = st.sidebar.number_input("Cost/FN", 0.0, 20000.0, 5000.0)

if st.sidebar.button("Add Scenario"):
    st.session_state.scenarios.append({
        "name": name,
        "prevalence": prevalence,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "cost_test": cost_test,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn
    })

# -----------------------------
# Run simulation
# -----------------------------
if "results_df" not in st.session_state:
    st.session_state.results_df = None

if st.button("Run Simulation"):

    results = []

    for sc in st.session_state.scenarios:
        res = run_simulation(
            n_patients,
            n_simulations,
            sc["prevalence"],
            sc["sensitivity"],
            sc["specificity"]
        )

        total_cost = (
            n_patients * sc["cost_test"] +
            res["FP"] * sc["cost_fp"] +
            res["FN"] * sc["cost_fn"]
        )

        res["Total Cost"] = total_cost
        res["Cost per TP"] = total_cost / res["TP"] if res["TP"] > 0 else np.inf
        res["Scenario"] = sc["name"]

        results.append(res)

    df = pd.DataFrame(results)

    base = df.iloc[0]
    df["ICER"] = [
        0 if i == 0 else
        (df.iloc[i]["Total Cost"] - base["Total Cost"]) /
        (df.iloc[i]["TP"] - base["TP"]) if (df.iloc[i]["TP"] - base["TP"]) != 0 else np.inf
        for i in range(len(df))
    ]

    st.session_state.results_df = df

# -----------------------------
# Display results
# -----------------------------
if st.session_state.results_df is not None:

    df = st.session_state.results_df

    st.subheader("📊 Results")
    st.dataframe(df)

    # Download
    st.download_button(
        "📥 Download Excel",
        data=to_excel(df),
        file_name="results.xlsx"
    )

    # Metric comparison
    metric = st.selectbox("Metric", ["PPV", "NPV", "Balanced Accuracy", "F1 Score", "Cost per TP"])

    fig, ax = plt.subplots()
    ax.bar(df["Scenario"], df[metric])
    ax.set_title(metric)
    st.pyplot(fig)

    # CI plot
    st.subheader("📉 Confidence Intervals (TP)")
    fig_ci, ax_ci = plt.subplots()
    y = df["TP"]
    yerr = [df["TP"] - df["TP_low"], df["TP_high"] - df["TP"]]
    ax_ci.bar(df["Scenario"], y)
    ax_ci.errorbar(df["Scenario"], y, yerr=yerr, fmt='none', capsize=5)
    st.pyplot(fig_ci)

    # ROC plot
    st.subheader("📈 ROC Space")
    fig2, ax2 = plt.subplots()
    ax2.scatter(1 - df["Specificity"], df["Sensitivity"])
    for i, txt in enumerate(df["Scenario"]):
        ax2.annotate(txt, (1 - df["Specificity"][i], df["Sensitivity"][i]))
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("Sensitivity")
    st.pyplot(fig2)

    # Tornado
    st.subheader("🌪️ Tornado Analysis")

    base_scenario = st.session_state.scenarios[0]
    tornado_df = tornado_analysis(base_scenario, n_patients, int(n_simulations/5))

    fig_tor, ax_tor = plt.subplots()
    ax_tor.barh(tornado_df["Parameter"], tornado_df["Low Impact"])
    ax_tor.barh(tornado_df["Parameter"], tornado_df["High Impact"])
    ax_tor.set_title("Impact on True Positives")
    st.pyplot(fig_tor)
