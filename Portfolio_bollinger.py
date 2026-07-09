import streamlit as st
import yfinance as yf
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from supabase import create_client, Client

# --- CONFIGURATION & DATABASE ENGINE ---
st.set_page_config(page_title="Alpha Bollinger Portfolio", layout="wide", page_icon="📊")

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# --- FREE GMAIL ALERT FUNCTION ---
def send_gmail_alert(ticker, current_price, upper_bb):
    """Sends a real-time portfolio alert completely free via Gmail SMTP"""
    try:
        sender_email = st.secrets["SENDER_EMAIL"]
        sender_password = st.secrets["SENDER_PASSWORD"]
        receiver_email = st.secrets["RECEIVER_EMAIL"]
        
        # Setup email structural headers
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"🚨 3:00 PM Portfolio Exit Signal Alert: {ticker} 🚨"
        
        body = (
            f"Hello Ashutosh,\n\n"
            f"This is an automated structural risk alert from your Portfolio Tracker.\n"
            f"The following asset has triggered an explicit exit signal:\n\n"
            f"• Asset Ticker: {ticker}\n"
            f"• 3:00 PM Active Price: ₹{current_price:.2f}\n"
            f"• Upper Bollinger Band (20, 0.6) Cap: ₹{upper_bb:.2f}\n\n"
            f"Technical Status: Price structurally trading below Upper BB (0.6).\n"
            f"Action Recommended: Evaluate target liquidation rules before market close.\n\n"
            f"Best Regards,\n"
            f"Alpha Bollinger Automation Suite"
        )
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect securely to Gmail's TLS server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to dispatch Email Alert: {e}")
        return False

# --- DATABASE FUNCTIONS ---
def load_active_positions():
    try:
        response = supabase.table("active_positions").select("*").order("created_at", desc=True).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error loading positions: {e}")
        return []

def load_closed_positions():
    try:
        response = supabase.table("closed_positions").select("*").order("created_at", desc=True).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return []

def add_active_position(trade_data):
    try:
        supabase.table("active_positions").insert(trade_data).execute()
        return True
    except Exception as e:
        st.error(f"Error adding position: {e}")
        return False

def close_position(trade_id, sell_price, sell_date):
    try:
        response = supabase.table("active_positions").select("*").eq("id", trade_id).execute()
        if not response.data:
            return False
        
        position = response.data[0]
        closed_data = {
            "ticker": position["ticker"],
            "qty": position["qty"],
            "buy_price": position["buy_price"],
            "buy_date": position["buy_date"],
            "sell_price": sell_price,
            "sell_date": sell_date
        }
        supabase.table("closed_positions").insert(closed_data).execute()
        supabase.table("active_positions").delete().eq("id", trade_id).execute()
        return True
    except Exception as e:
        st.error(f"Error closing position: {e}")
        return False

def update_position(trade_id, qty, buy_price):
    try:
        supabase.table("active_positions").update({"qty": qty, "buy_price": buy_price}).eq("id", trade_id).execute()
        return True
    except Exception as e:
        st.error(f"Error updating position: {e}")
        return False

# --- INITIALIZE STATE & DATA SYNC ---
st.session_state.active = load_active_positions()
st.session_state.history = load_closed_positions()

if "sent_alerts" not in st.session_state:
    st.session_state.sent_alerts = set()

def resolve_ticker(user_input):
    if not user_input:
        return None
    try:
        search_results = yf.Search(user_input, max_results=3).quotes
        if search_results:
            resolved_symbol = search_results[0]['symbol']
            return resolved_symbol.upper().strip()
    except Exception:
        pass
    return user_input.upper().strip()

# --- APP HEADER ---
st.title("📊 Ashutosh Bollinger Portfolio & Performance Analytics")
st.markdown("Track exit signals below the **Upper Bollinger Band (20, 0.6)**. Notifications are automated via Gmail for 3:00 PM weekdays.")
st.markdown("---")

# --- SIDEBAR: TRANSACTION MANAGEMENT ---
st.sidebar.header("🛒 Log New Position")
raw_ticker_input = st.sidebar.text_input("1. Stock Name / Ticker (e.g., RELIANCE, INFIBEAM, AAPL)").strip()

fetched_price = 0.0
resolved_ticker_symbol = ""

if raw_ticker_input:
    try:
        with st.spinner(f"Searching for '{raw_ticker_input}'..."):
            resolved_ticker_symbol = resolve_ticker(raw_ticker_input)
            if resolved_ticker_symbol:
                st.sidebar.caption(f"Resolved to system symbol: **{resolved_ticker_symbol}**")
                ticker_obj = yf.Ticker(resolved_ticker_symbol)
                ticker_df = ticker_obj.history(period="5d")
                if not ticker_df.empty:
                    last_close = ticker_df['Close'].squeeze().iloc[-1]
                    fetched_price = float(last_close.item()) if hasattr(last_close, 'item') else float(last_close)
                    st.sidebar.success(f"Live Price: ₹{fetched_price:.2f}")
    except Exception as e:
        st.sidebar.error(f"Error fetching: {e}")

with st.sidebar.form("buy_form", clear_on_submit=True):
    st.subheader("2. Entry Configuration")
    buy_price = st.number_input("Purchase Price per Share (₹)", min_value=0.01, step=0.1, value=fetched_price if fetched_price > 0 else 100.0)
    qty = st.number_input("Quantity", min_value=1, step=1, value=10)
    buy_date = st.date_input("Purchase Date", max_value=datetime.today())
    
    submitted = st.form_submit_button("Deploy Position")
    if submitted:
        if resolved_ticker_symbol:
            new_trade = {"ticker": resolved_ticker_symbol, "qty": int(qty), "buy_price": float(buy_price), "buy_date": str(buy_date)}
            if add_active_position(new_trade):
                st.sidebar.success(f"Deployed {resolved_ticker_symbol} position successfully!")
                st.rerun()

# --- TABS FOR ORGANIZED LAYOUT ---
tab1, tab2, tab3 = st.tabs(["⚡ Live Positions & Tracking", "📈 Performance Analytics", "⚙️ Position Modification"])

# ==========================================
# TAB 1: LIVE POSITIONS & EXITS
# ==========================================
active_df_shared = pd.DataFrame()

with tab1:
    st.header("Active Monitor Dashboard")
    active_positions = st.session_state.active
    
    if not active_positions:
        st.info("No active holdings found. Log a transaction via the sidebar.")
    else:
        unique_tickers = list(set([trade["ticker"] for trade in active_positions]))
        live_market_data = {}
        
        with st.spinner("Processing technical indicators..."):
            for t in unique_tickers:
                try:
                    ticker_obj = yf.Ticker(t)
                    df = ticker_obj.history(period="3mo", interval="1d")
                    if not df.empty and len(df) >= 20:
                        close_series = df['Close'].squeeze()
                        df['SMA20'] = close_series.rolling(window=20).mean()
                        df['STD20'] = close_series.rolling(window=20).std()
                        df['Upper_BB_06'] = df['SMA20'] + (0.6 * df['STD20'])
                        
                        last_close = df['Close'].squeeze().iloc[-1]
                        last_bb = df['Upper_BB_06'].squeeze().iloc[-1]
                        
                        live_market_data[t] = {
                            "current_price": float(last_close.item()) if hasattr(last_close, 'item') else float(last_close),
                            "upper_bb": float(last_bb.item()) if hasattr(last_bb, 'item') else float(last_bb)
                        }
                except Exception as e:
                    st.error(f"Indicator calculation error for {t}: {e}")

        active_rows = []
        total_value, total_cost = 0.0, 0.0
        
        # Time-check validation framework
        now = datetime.now()
        is_weekday = now.weekday() < 5  # Mon-Fri are 0-4
        is_three_pm_window = now.hour == 15  # 15 matches 3:00 PM - 3:59 PM
        
        for trade in active_positions:
            t_symbol = trade["ticker"]
            m_data = live_market_data.get(t_symbol)
            
            if m_data:
                current_price = m_data["current_price"]
                upper_bb = m_data["upper_bb"]
                cost = trade["qty"] * trade["buy_price"]
                value = trade["qty"] * current_price
                pnl = value - cost
                pnl_pct = (pnl / cost * 100) if cost > 0 else 0
                
                total_value += value
                total_cost += cost
                
                status = "🟢 HOLD"
                reason = "Price structural trend remains strong above Upper BB (0.6)."
                
                if current_price <= upper_bb:
                    status = "🔴 SELL"
                    reason = "Trend Weakness: Closed below Upper BB (20, 0.6)."
                    
                    # Target 3 PM Email Dispatch Rule check
                    if is_weekday and is_three_pm_window:
                        trade_unique_key = f"{trade['id']}_{t_symbol}_{now.strftime('%Y-%m-%d')}"
                        if trade_unique_key not in st.session_state.sent_alerts:
                            if send_gmail_alert(t_symbol, current_price, upper_bb):
                                st.session_state.sent_alerts.add(trade_unique_key)

                active_rows.append({
                    "Signal Status": status, "Ticker": t_symbol, "Guidance": reason,
                    "Date Bought": trade["buy_date"], "Qty": trade["qty"],
                    "Buy Price (₹)": round(trade["buy_price"], 2), "Current Price (₹)": round(current_price, 2),
                    "Upper BB (20, 0.6)": round(upper_bb, 2), "Total Cost (₹)": round(cost, 2),
                    "Current Value (₹)": round(value, 2), "PnL (₹)": round(pnl, 2), "Return (%)": round(pnl_pct, 2),
                    "ID": trade["id"]
                })

        if active_rows:
            active_df_shared = pd.DataFrame(active_rows)
            tot_pnl = total_value - total_cost
            tot_pnl_pct = (tot_pnl / total_cost * 100) if total_cost > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Holdings Valuation", f"₹{total_value:,.2f}")
            c2.metric("Total Money Capitalized", f"₹{total_cost:,.2f}")
            c3.metric("Floating Net Profit/Loss", f"₹{tot_pnl:,.2f}", f"{tot_pnl_pct:+.2f}%")
            
            sell_alerts = active_df_shared[active_df_shared["Signal Status"] == "🔴 SELL"]
            if not sell_alerts.empty:
                if is_weekday and is_three_pm_window:
                    st.error(f"🚨 **Action Required:** {len(sell_alerts)} exit criteria triggered! Gmail notifications dispatched.")
                else:
                    st.warning(f"⚠️ {len(sell_alerts)} stock(s) showing exit criteria. Gmail alert queues at 3:00 PM on weekdays.")

            st.subheader("Live Portfolio Tracker Ledger")
            def color_signal(val):
                if val == "🔴 SELL": return 'background-color: #fce8e6; color: #c5221f; font-weight: bold;'
                if val == "🟢 HOLD": return 'background-color: #e6f4ea; color: #137333; font-weight: bold;'
                return ''
                
            st.dataframe(
                active_df_shared.drop(columns=["ID"]).style.map(color_signal, subset=["Signal Status"]),
                use_container_width=True, hide_index=True
            )
            
            st.markdown("---")
            st.subheader("⚡ Close Out Tracked Position")
            sell_opts = {f"{r['Ticker']} ({r['Qty']} shares bought on {r['Date Bought']})": r['ID'] for i, r in active_df_shared.iterrows()}
            sel_sell_label = st.selectbox("Select position to realize exit transaction:", options=list(sell_opts.keys()))
            target_sell_id = sell_opts[sel_sell_label]
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                realized_sell_price = st.number_input("Actual Sell Price per Share (₹)", min_value=0.01, step=0.1, value=active_df_shared.loc[active_df_shared['ID'] == target_sell_id, 'Current Price (₹)'].values[0])
            with col_s2:
                realized_sell_date = st.date_input("Exit Settlement Execution Date", max_value=datetime.today())
                
            if st.button("Execute Sale Settlement", type="primary"):
                if close_position(target_sell_id, float(realized_sell_price), str(realized_sell_date)):
                    st.success("Asset position moved into historical ledger.")
                    st.rerun()

# ==========================================
# TAB 2 & 3 PERFORMANCE & EDITS
# ==========================================
with tab2:
    st.header("📈 Calendar Performance Analytics")
    if not st.session_state.history:
        st.info("Log closed trades from active tracking to generate historical performance calculations.")
    else:
        hist_rows = []
        for h in st.session_state.history:
            cost = h["qty"] * h["buy_price"]
            rev = h["qty"] * h["sell_price"]
            net = rev - cost
            try:
                date_obj = datetime.strptime(h["sell_date"], "%Y-%m-%d")
                month_key = date_obj.strftime("%Y-%m (%B)")
            except:
                month_key = "Unknown Calendar Axis"
            hist_rows.append({"MonthKey": month_key, "Ticker": h["ticker"], "Cost": cost, "Revenue": rev, "NetPnL": net})
            
        hdf = pd.DataFrame(hist_rows)
        monthly_summary = hdf.groupby("MonthKey").agg(Total_Invested_Capital=("Cost", "sum"), Total_Net_Profit=("NetPnL", "sum")).reset_index()
        monthly_summary["Monthly Return (%)"] = (monthly_summary["Total_Net_Profit"] / monthly_summary["Total_Invested_Capital"]) * 100
        
        st.subheader("Monthly Performance Scorecard")
        st.dataframe(monthly_summary.rename(columns={"MonthKey": "Calendar Month", "Total_Invested_Capital": "Invested Capital (₹)", "Total_Net_Profit": "Realized Profits (₹)"}).style.map(lambda v: f"color: {'#137333' if v >= 0 else '#c5221f'}; font-weight: bold;", subset=["Realized Profits (₹)", "Monthly Return (%)"]), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("📜 Complete Historical Transaction Logs")
        display_hist_df = pd.DataFrame([{"Ticker": x["ticker"], "Qty": x["qty"], "Date Bought": x["buy_date"], "Buy Price (₹)": x["buy_price"], "Date Sold": x["sell_date"], "Sell Price (₹)": x["sell_price"], "Net Profit (₹)": (x["qty"] * x["sell_price"]) - (x["qty"] * x["buy_price"])} for x in st.session_state.history])
        st.dataframe(display_hist_df, use_container_width=True, hide_index=True)

with tab3:
    st.header("🔧 Correct Existing Entry Data")
    active_positions = st.session_state.active
    if not active_positions:
        st.info("No active entries are available for modification.")
    else:
        if active_df_shared.empty:
            edit_opts = {f"{r['ticker']} ({r['qty']} shares bought on {r['buy_date']})": r['id'] for r in active_positions}
        else:
            edit_opts = {f"{r['Ticker']} ({r['Qty']} shares bought on {r['Date Bought']})": r['ID'] for i, r in active_df_shared.iterrows()}
        
        sel_edit_label = st.selectbox("Select entity position to adjust fields:", options=list(edit_opts.keys()))
        target_edit_id = edit_opts[sel_edit_label]
        curr_edit = next((x for x in active_positions if str(x["id"]) == str(target_edit_id)), None)
        
        if curr_edit:
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                m_qty = st.number_input("Update Quantity", min_value=1, step=1, value=int(curr_edit["qty"]))
            with col_e2:
                m_prc = st.number_input("Update Purchase Price (₹)", min_value=0.01, step=0.1, value=float(curr_edit["buy_price"]))
            if st.button("Save Strategic Overwrites", type="secondary"):
                if update_position(target_edit_id, int(m_qty), float(m_prc)):
                    st.success("Entry details successfully updated.")
                    st.rerun()
