"""Streamlit analyst console for generated benchmark investigations."""

# ruff: noqa: E501 -- analyst-facing prose and inline HTML remain readable as literals.

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from fraud_agent.dashboard import DashboardView, build_dashboard_view

CASES_DIR = Path("cases")


@st.cache_data(show_spinner=False)
def load_view() -> DashboardView:
    return build_dashboard_view(CASES_DIR)


def relation_figure(case_id: str, view: DashboardView) -> go.Figure:
    edges = view.case_graph.get(case_id, [])
    labels = [case_id, *[edge[1] for edge in edges]]
    colors = ["#F4B942", *["#48CAE4" if edge[2] == "device" else "#FB7185" for edge in edges]]
    x = [0.0, *([1.0] * len(edges))]
    if edges:
        y = [0.5, *[(index + 1) / (len(edges) + 1) for index in range(len(edges))]]
    else:
        y = [0.5]
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    for index in range(len(edges)):
        line_x.extend([0.0, 1.0, None])
        line_y.extend([0.5, y[index + 1], None])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=line_x, y=line_y, mode="lines", line={"color": "#334155", "width": 1})
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker={"size": [28, *([18] * len(edges))], "color": colors},
            text=labels,
            textposition="top center",
            hovertext=["case", *[edge[2] for edge in edges]],
            hoverinfo="text",
        )
    )
    figure.update_layout(
        height=310,
        margin={"l": 15, "r": 15, "t": 25, "b": 15},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis={"visible": False, "range": [-0.2, 1.2]},
        yaxis={"visible": False, "range": [0, 1]},
    )
    return figure


def render() -> None:
    st.set_page_config(page_title="Sentinel Graph", page_icon="◈", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #07111f; color: #e2e8f0; }
        [data-testid="stSidebar"] { background: #0b1729; border-right: 1px solid #1e293b; }
        div[data-testid="stMetric"] { background: #0f1d31; border: 1px solid #22324a;
          border-radius: 10px; padding: 14px 18px; }
        div[data-testid="stMetric"] label { color: #94a3b8; }
        .eyebrow { color: #48cae4; text-transform: uppercase; letter-spacing: .16em;
          font-size: .72rem; font-weight: 700; }
        .verdict { display:inline-block; padding:.25rem .7rem; border-radius:999px;
          background:#172554; color:#bfdbfe; font-weight:700; text-transform:uppercase;
          letter-spacing:.08em; font-size:.72rem; }
        .evidence { border-left: 2px solid #48cae4; padding:.3rem 0 .3rem 1rem;
          margin:.65rem 0; color:#cbd5e1; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    view = load_view()
    st.markdown('<p class="eyebrow">TigerGraph × Hacker House Goa</p>', unsafe_allow_html=True)
    st.title("Sentinel Graph")
    st.caption(
        "Evidence-first fraud investigations · policy-routed actions · persistent case memory"
    )
    if not view.answers:
        st.warning("No case answers found. Run `python -m fraud_agent batch --no-llm` first.")
        return

    with st.sidebar:
        st.markdown("### Investigation queue")
        case_id = st.selectbox("Case", list(view.answers), label_visibility="collapsed")
        answer = view.answers[case_id]
        st.caption(f"{answer.case.status.replace('_', ' ').title()}")
        st.progress(answer.case.fraud_probability, text=f"Risk {answer.case.fraud_probability:.0%}")
        st.markdown("---")
        st.caption(f"{answer.tool_calls} graph calls · {answer.latency_s:.2f}s")

    metrics = st.columns(5)
    metrics[0].metric("Cases", view.total_cases)
    metrics[1].metric("Fraud", view.verdict_counts["fraud"])
    metrics[2].metric("Legitimate", view.verdict_counts["legitimate"])
    metrics[3].metric("Exposure", f"${view.total_exposure:,.2f}")
    metrics[4].metric("SARs", view.sar_count)

    answer = view.answers[case_id]
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown("### Analyst finding")
        st.markdown(f'<span class="verdict">{answer.case.verdict}</span>', unsafe_allow_html=True)
        st.markdown(f"#### {answer.case.pattern.replace('_', ' ').title()}")
        st.write(answer.case.summary)
        st.markdown("##### Evidence trail")
        for item in answer.case.evidence:
            st.markdown(
                f'<div class="evidence">{item.claim}<br><small>{item.source} · {item.ref}</small></div>',
                unsafe_allow_html=True,
            )
        st.markdown("##### Stop condition")
        st.write(answer.stop_reason)
    with right:
        st.markdown("### Graph neighborhood")
        st.plotly_chart(relation_figure(case_id, view), use_container_width=True)
        st.markdown("### Final actions")
        for action in answer.next_best_actions.final:
            st.markdown(f"**{action.action}** · `{action.route}`")
            st.caption(action.reason)

    tab_case, tab_sar, tab_json = st.tabs(["Case memory", "SAR", "Raw answer"])
    with tab_case:
        st.json(answer.case.model_dump(mode="json"), expanded=False)
    with tab_sar:
        if answer.sar.file:
            st.write(answer.sar.narrative)
            st.caption(answer.sar.reason)
        else:
            st.info(answer.sar.reason)
    with tab_json:
        st.json(answer.model_dump(mode="json"), expanded=False)


if __name__ == "__main__":
    render()
