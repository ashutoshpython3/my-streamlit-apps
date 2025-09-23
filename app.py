import streamlit as st

st.set_page_config(page_title="My Streamlit Apps", layout="wide")

st.title("My Streamlit Apps")

st.markdown("""
## Select an Application:
1. **Portfolio Manager** - Track and manage your investment portfolio
2. **Stock Checker** - Check real-time stock prices and information
3. **Intra Signal** - Generate intraday trading signals
4. **Ticket System** - Automated ticketing system
""")

app_selection = st.selectbox("Choose an app:", ["", "Portfolio Manager", "Stock Checker", "Intra Signal", "Ticket System"])

if app_selection == "Portfolio Manager":
    st.switch_page("app1/Ashutosh_portfolio_manager.py")
elif app_selection == "Stock Checker":
    st.switch_page("app2/Ashutosh_stock_checker.py")
elif app_selection == "Intra Signal":
    st.switch_page("app3/Ashutosh_intra_Signal.py")
elif app_selection == "Ticket System":
    st.switch_page("app4/Ticket.py")