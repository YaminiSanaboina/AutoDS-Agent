import streamlit as st
import pandas as pd
import html

from agents.cleaning_agent import clean_dataset
from ui.components import ai_assistant_panel, dataset_banner, glass_panel, glass_panel_small, end_glass_panel, issue_badge, require_dataset
from utils.health_score import compute_health_score, detect_data_issues
from utils.llm_insights import generate_cleaning_advice_llm
from utils.session_manager import SessionKeys, get_autonomous_result, get_dataframe, set_dataframe
from utils.styles import render_hero


def render():
    render_hero("Data Quality Lab", "AI-powered data cleaning assistant")

    if not require_dataset():
        return

    df = get_dataframe()
    dataset_banner()

    # Check if autonomous pipeline has been run
    autonomous_result = get_autonomous_result()
    if autonomous_result is not None and isinstance(autonomous_result, dict):
        glass_panel("Autonomous Pipeline Results", "Read-only results from your latest pipeline run.")
        
        dataset_report = autonomous_result.get("dataset_report") or autonomous_result.get("dataset_analysis") or {}
        cleaning_results = autonomous_result.get("cleaning_results", {})
        risk_analysis = dataset_report.get("risk_analysis", {})
        detailed_scores = risk_analysis.get("detailed_scores", {}) if isinstance(risk_analysis, dict) else {}

        st.markdown("<div style='margin-bottom:1rem;color:#64748B;'>The AI pipeline has analyzed your dataset and generated data quality insights. Review the top findings below.</div>", unsafe_allow_html=True)
        st.subheader("AI-Detected Data Quality Issues")
        issue_lines = []
        if "missing_values" in detailed_scores:
            issue_lines.append(f"Missing values impact score: {detailed_scores.get('missing_values')}")
        if "duplicates" in detailed_scores:
            issue_lines.append(f"Duplicate records impact score: {detailed_scores.get('duplicates')}")
        if "data_leakage" in detailed_scores:
            issue_lines.append(f"Data leakage risk score: {detailed_scores.get('data_leakage')}")
        if not issue_lines:
            issue_lines.append("No critical data quality issues were identified by the autonomous pipeline.")
        for line in issue_lines:
            st.write(f"- {line}")

        st.subheader("Recommended Cleaning Actions")
        actions = cleaning_results.get("report") if isinstance(cleaning_results.get("report"), list) else []
        if actions:
            for action in actions:
                issue_badge(action, "ok")
        else:
            st.info("No autonomous cleaning actions were generated. Manual cleaning tools are available below.")

        st.subheader("Cleaning Summary")
        summary_lines = cleaning_results.get("report") if isinstance(cleaning_results.get("report"), list) else []
        if summary_lines:
            for line in summary_lines:
                st.write(f"- {line}")
        else:
            st.write("Cleaning analysis completed by autonomous pipeline.")

        if cleaning_results.get("shape") and isinstance(cleaning_results.get("shape"), (list, tuple)) and len(cleaning_results.get("shape")) == 2:
            before_rows, before_cols = df.shape
            after_rows, after_cols = cleaning_results.get("shape")
            c1, c2 = st.columns(2)
            c1.metric("Before", f"{before_rows:,} rows × {before_cols} cols")
            c2.metric("After", f"{after_rows:,} rows × {after_cols} cols")
        
        if dataset_report.get("intelligence_score"):
            score = dataset_report["intelligence_score"].get("score")
            if score is not None:
                st.metric("Data Health Score", f"{score}/100")
        
        return
    else:
        # Manual cleaning workflow (existing logic)
        if st.session_state.get(SessionKeys.CLEANING_ISSUES) is None:
            st.session_state[SessionKeys.CLEANING_ISSUES] = detect_data_issues(df)

        issues = st.session_state[SessionKeys.CLEANING_ISSUES]
        health_before = st.session_state.get(SessionKeys.HEALTH_BEFORE) or compute_health_score(df)
        st.session_state[SessionKeys.HEALTH_BEFORE] = health_before

        missing_pct = df.isnull().sum().sum() / max(df.size, 1) * 100
        duplicates = int(df.duplicated().sum())

        st.subheader("AI Detected Problems")
        if missing_pct > 0:
            issue_badge(f"Missing values found ({missing_pct:.1f}%)", "error")
        else:
            issue_badge("No missing values", "ok")
        if duplicates > 0:
            issue_badge(f"Duplicate records found ({duplicates})", "error")
        else:
            issue_badge("No duplicates", "ok")

        cat_cols = df.select_dtypes(include="object").columns
        if len(cat_cols) > 0:
            issue_badge(f"Categorical columns need encoding ({len(cat_cols)})", "warn")

        advice = generate_cleaning_advice_llm(missing_pct, duplicates, issues)

        # Build a single, beginner-friendly conversational message
        parts = []
        parts.append("Hello — I reviewed your dataset's quality and summarized the key cleaning suggestions below.")

        # What data quality means
        parts.append(
            "Data quality means your data is accurate, complete, consistent, and formatted correctly. "
            "High-quality data leads to better, more reliable models."
        )

        # Missing values
        if missing_pct > 0:
            parts.append(
                f"I found about {missing_pct:.1f}% missing cells. Missing values matter because they can bias results or "
                "prevent some algorithms from working. We'll usually either fill them in (imputation) or remove affected rows/columns."
            )
            # show top missing columns from issues
            top_missing = [i for i in issues if i['title'].startswith('Missing Values')][:3]
            if top_missing:
                examples = ", ".join(
                    f'{html.escape(it["title"].split("'",2)[1])}: {it["description"].split()[0]}'
                    for it in top_missing
                )
                parts.append(f"Top missing columns: {examples}.")
        else:
            parts.append("No missing values were detected — that's great.")

        # Duplicates
        if duplicates > 0:
            parts.append(
                f"I detected {duplicates} duplicate row(s). Duplicates can give a false sense of importance to repeated records, "
                "so removing them usually improves model fairness and accuracy."
            )
        else:
            parts.append("No duplicate rows detected.")

        # Incorrect data types (simple heuristic)
        mis_typed = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            sample = df[col].dropna().head(100)
            if sample.empty:
                continue
            coerced = pd.to_numeric(sample, errors="coerce")
            numeric_pct = coerced.notna().mean()
            if numeric_pct >= 0.9:
                mis_typed.append(col)

        if mis_typed:
            parts.append(
                "Some columns look like numbers but are stored as text: "
                f"{html.escape(', '.join(mis_typed))}. Converting these to numeric types helps calculations and modeling."
            )
        else:
            parts.append("Column types look appropriate at a glance.")

        # Outliers (IQR rule summary)
        num_outlier_cols = 0
        outlier_examples = []
        for col in df.select_dtypes(include="number").columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((df[col] < lower) | (df[col] > upper)).sum())
            if count > 0:
                num_outlier_cols += 1
                if len(outlier_examples) < 3:
                    outlier_examples.append(f"{col} ({count})")

        if num_outlier_cols > 0:
            parts.append(
                f"I found outliers in {num_outlier_cols} numerical column(s) (examples: {', '.join(outlier_examples)}). "
                "Outliers can skew averages and hurt some models; consider winsorizing, transforming, or inspecting those rows."
            )
        else:
            parts.append("No widespread outlier problems detected, though individual values may still be worth checking.")

        # Recommended actions
        recs = recommendations = analysis = None
        try:
            recs = generate_cleaning_advice_llm(missing_pct, duplicates, issues)
        except Exception:
            recs = advice

        parts.append("Recommended next steps: " + html.escape(str(recs)))

        # Short actionable list
        parts.append("Quick actions: 1) Handle missing values in Data Quality Lab, 2) Remove duplicates, 3) Convert mistyped columns, 4) Re-run EDA.")

        # Use standardized AI assistant panel: short summary + collapsible details + recommendation
        summary = parts[0] if parts else "Cleaning analysis completed"
        # Last two items are recommendation and quick actions — present them as the main recommendation
        recommendation_text = ""
        if len(parts) >= 2:
            recommendation_text = parts[-2] + " " + parts[-1] if len(parts) >= 2 else parts[-1]
        details = parts[1:-2] if len(parts) > 3 else parts[1:-1]

        ai_assistant_panel(
                title="Cleaning Assistant",
                summary=summary,
                details=details,
                recommendation=recommendation_text,
            )

        st.subheader("AI Recommendations")
        issue_badge("Fill missing values", "ok")
        issue_badge("Encode categorical data", "ok")
        issue_badge("Remove duplicates", "ok")


        st.markdown("---")
        if st.button("Run AI Cleaning Pipeline", type="primary"):
            cleaned_df, report = clean_dataset(df)
            set_dataframe(cleaned_df, st.session_state.get(SessionKeys.DATASET_NAME))
            health_after = compute_health_score(cleaned_df)

            history = st.session_state.get(SessionKeys.CLEANING_HISTORY) or []
            history.append({"actions": report, "health": health_after["score"]})
            st.session_state[SessionKeys.CLEANING_HISTORY] = history
            st.session_state[SessionKeys.CLEANING_REPORT] = report
            st.session_state[SessionKeys.CLEANING_ISSUES] = detect_data_issues(cleaned_df)
            st.session_state[SessionKeys.HEALTH_AFTER] = health_after
            st.rerun()

        if st.session_state.get(SessionKeys.CLEANING_REPORT):
            health_after = st.session_state.get(SessionKeys.HEALTH_AFTER) or compute_health_score(get_dataframe())
            improvement = health_after["score"] - health_before["score"]

            st.success("Cleaning completed!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Before", f"{health_before['score']}/100")
            c2.metric("After", f"{health_after['score']}/100", delta=f"{improvement:+.1f}")
            c3.metric("Improvement", f"{improvement:+.1f}%")

            st.subheader("Cleaning History")
            for idx, entry in enumerate(st.session_state.get(SessionKeys.CLEANING_HISTORY) or [], 1):
                with st.expander(f"Run #{idx} — Health: {entry['health']}/100"):
                    for action in entry["actions"]:
                        st.write(f"✅ {action}")

            st.subheader("Cleaned Preview")
            st.dataframe(get_dataframe().head(10), width="stretch", hide_index=True)
