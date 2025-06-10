import streamlit as st

st.title("📊 Interview Report Card")
st.markdown("<hr>", unsafe_allow_html=True)

if "scoring_results" in st.session_state:
    results = st.session_state.scoring_results

    # Calculate overall average
    overall_avg = sum(r['average_score'] for r in results) / len(results)

    st.subheader("Overall Performance")
    st.progress(min(1.0, overall_avg / 10))
    st.metric("Final Score", f"{overall_avg:.2f}/10")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Detailed question scores
    st.subheader("Detailed Feedback")
    for idx, result in enumerate(results):
        with st.expander(f"Question {idx + 1}: {result['question']}", expanded=False):
            st.markdown(f"**Your Answer:** {result['answer']}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("GPT-4 Score", f"{result.get('gpt-4-turbo_score', 'N/A')}/10")
                st.metric("Gemini Score", f"{result.get('gemini-pro_score', 'N/A')}/10")
            with col2:
                st.metric("Claude Score", f"{result.get('claude-opus_score', 'N/A')}/10")
                st.metric("Llama Score", f"{result.get('llama-3.3_score', 'N/A')}/10")

            st.markdown(f"**Average Score: `{result['average_score']}/10`**")

            # Add visual feedback
            if result['average_score'] >= 8:
                st.success("Excellent response! 👍")
            elif result['average_score'] >= 6:
                st.info("Good response! 👌")
            elif result['average_score'] >= 4:
                st.warning("Fair response. Needs improvement. 💡")
            else:
                st.error("Weak response. Needs significant improvement. 📚")

        st.markdown("---")
else:
    st.error("No scoring results found. Please complete the interview first.")
    if st.button("↩️ Back to Interview"):
        st.switch_page("interview_app.py")
    st.stop()

st.success("✅ Evaluation Complete")
st.markdown("""
**Thank you for your patience.**  
**Best wishes for your future!** 🚀
""")

if st.button("🔙 Go Back to Interview Home"):
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.switch_page("interview_app.py")