import streamlit as st

st.title("📊 Interview Report Card")
st.markdown("<hr>", unsafe_allow_html=True)

# 1. Ensure we have something to show
if "scoring_results" not in st.session_state:
    st.error("No scoring results found. Please complete the interview first.")
    if st.button("↩️ Back to Interview"):
        st.switch_page("interview_app.py")
    st.stop()

results = st.session_state.scoring_results

# 2. Handle empty list
if not isinstance(results, list) or len(results) == 0:
    st.error("Scoring results are empty or invalid.")
    if st.button("↩️ Back to Interview"):
        st.switch_page("interview_app.py")
    st.stop()

try:
    # 3. Compute overall average safely
    avg_list = []
    for idx, r in enumerate(results):
        avg = r.get("average_score")
        if avg is None:
            st.warning(f"Result #{idx+1} missing 'average_score'; defaulting to 0.")
            avg = 0.0
        avg_list.append(float(avg))
    overall_avg = sum(avg_list) / len(avg_list)

    # 4. Clamp progress value to [0.0, 1.0]
    progress_value = max(0.0, min(1.0, overall_avg / 10))

    # Display
    st.subheader("Overall Performance")
    st.progress(progress_value)
    st.metric("Final Score", f"{overall_avg:.2f}/10")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Detailed question scores
    st.subheader("Detailed Feedback")
    for idx, result in enumerate(results):
        q = result.get("question", "Unknown question")
        ans = result.get("answer", "No answer provided")
        avg_score = float(result.get("average_score", 0.0))

        with st.expander(f"Question {idx + 1}: {q}", expanded=False):
            st.markdown(f"**Your Answer:** {ans}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("GPT-4 Score", f"{result.get('gpt-4-turbo_score', 'N/A')}/10")
                st.metric("Gemini Score", f"{result.get('gemini-pro_score', 'N/A')}/10")
            with col2:
                st.metric("Claude Score", f"{result.get('claude-opus_score', 'N/A')}/10")
                st.metric("Llama Score", f"{result.get('llama-3.3_score', 'N/A')}/10")

            st.markdown(f"**Average Score: `{avg_score:.2f}/10`**")

            # Visual feedback
            if avg_score >= 8:
                st.success("Excellent response! 👍")
            elif avg_score >= 6:
                st.info("Good response! 👌")
            elif avg_score >= 4:
                st.warning("Fair response. Needs improvement. 💡")
            else:
                st.error("Weak response. Needs significant improvement. 📚")

        st.markdown("---")

except Exception as e:
    # 5. Catch anything unexpected
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Final footer
st.success("✅ Evaluation Complete")
st.markdown("""
**Thank you for your patience.**  
**Best wishes for your future!** 🚀
""")

if st.button("🔙 Go Back to Interview Home"):
    # Clear session state and go back
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page("interview_app.py")
