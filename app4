import streamlit as st
import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime

# --- CONFIGURATION & DATABASE ENGINE ---
st.set_page_config(page_title="Alpha Bollinger Portfolio", layout="wide", page_icon="📊")
DB_FILE = "portfolio_advanced_db.json"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"active": [], "history": []}
    return {"active": [], "history": []}

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

if "db" not in st.session_state:
    st.session_state.db = load_db()

db = st.session_state.db

# --- APP HEADER ---
st.title("📊 Ashutosh Bollinger Portfolio & Performance Analytics")
st.markdown("Track exit signals below the **Upper Bollinger Band (20, 0.6)** and evaluate your historical monthly returns.")
st.markdown("---")

# --- SIDEBAR: TRANSACTION MANAGEMENT (CLEANED) ---
st.sidebar.header("🛒 Log New Position")

ticker_input = st.sidebar.text_input("1. Stock Ticker (e.g., RELIANCE.NS, INFIBEAM.NS)").upper().strip()

fetched_price = 0.0
if ticker_input:
    try:
        # FIX: Changed from st.sidebar.spinner to st.spinner
        with st.spinner(f"Fetching quotes for {ticker_input}..."):
            ticker_df = yf.download(ticker_input, period="5d", progress=False)
            if not ticker_df.empty:
                fetched_price = float(ticker_df['Close'].iloc[-1])
                st.sidebar.success(f"Live Price: ₹{fetched_price:.2f}")
            else:
                st.sidebar.warning("Failed to resolve asset price.")
    except Exception as e:
        st.sidebar.error(f"Error fetching: {e}")

with st.sidebar.form("buy_form", clear_on_submit=True):
    st.subheader("2. Entry Configuration")
    buy_price = st.number_input("Purchase Price per Share (₹)", min_value=0.01, step=0.1, value=fetched_price if fetched_price > 0 else 100.0)
    qty = st.number_input("Quantity", min_value=1, step=1, value=10)
    buy_date = st.date_input("Purchase Date", max_value=datetime.today())
    
    # FIX: Combined with st.form_submit_button to attach correctly to form context
    submitted = st.form_submit_button("Deploy Position")
    if submitted:
        if ticker_input:
            new_trade = {
                "id": str(int(datetime.now().timestamp())),
                "ticker": ticker_input,
                "qty": int(qty),
                "buy_price": float(buy_price),
                "buy_date": str(buy_date)
            }
            db["active"].append(new_trade)
            save_db(db)
            st.sidebar.success(f"Deployed {ticker_input} position successfully!")
            st.rerun()
        else:
            st.sidebar.error("Valid ticker symbol required.")

# --- TABS FOR ORGANIZED LAYOUT ---
tab1, tab2, tab3 = st.tabs(["⚡ Live Positions & Tracking", "📈 Performance Analytics", "⚙️ Position Modification"])

# ==========================================
# TAB 1: LIVE POSITIONS & EXITS
# ==========================================
with tab1:
    st.header("Active Monitor Dashboard")
    
    if not db["active"]:
        st.info("No active holdings found. Log a transaction via the sidebar.")
    else:
        unique_tickers = list(set([trade["ticker"] for trade in db["active"]]))
        live_market_data = {}
        
        with st.spinner("Processing technical indicators..."):
            for t in unique_tickers:
                try:
                    df = yf.download(t, period="3mo", interval="1d", progress=False)
                    if not df.empty and len(df) >= 20:
                        df['SMA20'] = df['Close'].rolling(window=20).mean()
                        df['STD20'] = df['Close'].rolling(window=20).std()
                        # Upper Bollinger Band formula using 0.6 standard deviations
                        df['Upper_BB_06'] = df['SMA20'] + (0.6 * df['STD20'])
                        
                        live_market_data[t] = {
                            "current_price": float(df['Close'].iloc[-1]),
                            "upper_bb": float(df['Upper_BB_06'].iloc[-1])
                        }
                except Exception as e:
                    st.error(f"Indicator calculation error for {t}: {e}")

        active_rows = []
        total_value, total_cost = 0.0, 0.0
        
        for trade in db["active"]:
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
                
                # Exit Rule Logic: Close < Upper Bollinger Band (20, 0.6)
                status = "🟢 HOLD"
                reason = "Price structural trend remains strong above Upper BB (0.6)."
                
                if current_price <= upper_bb:
                    status = "🔴 SELL"
                    reason = "Trend Weakness: Closed below Upper BB (20, 0.6)."

                active_rows.append({
                    "ID": trade["id"],
                    "Ticker": t_symbol,
                    "Date Bought": trade["buy_date"],
                    "Qty": trade["qty"],
                    "Buy Price (₹)": round(trade["buy_price"], 2),
                    "Current Price (₹)": round(current_price, 2),
                    "Upper BB (20, 0.6)": round(upper_bb, 2),
                    "Total Cost (₹)": round(cost, 2),
                    "Current Value (₹)": round(value, 2),
                    "PnL (₹)": round(pnl, 2),
                    "Return (%)": round(pnl_pct, 2),
                    "Signal Status": status,
                    "Guidance": reason
                })

        if active_rows:
            active_df = pd.DataFrame(active_rows)
            tot_pnl = total_value - total_cost
            tot_pnl_pct = (tot_pnl / total_cost * 100) if total_cost > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Holdings Valuation", f"₹{total_value:,.2f}")
            c2.metric("Total Money Capitalized", f"₹{total_cost:,.2f}")
            c3.metric("Floating Net Profit/Loss", f"₹{tot_pnl:,.2f}", f"{tot_pnl_pct:+.2f}%")
            
            sell_alerts = active_df[active_df["Signal Status"] == "🔴 SELL"]
            if not sell_alerts.empty:
                st.error(f"🚨 **Action Required:** {len(sell_alerts)} position(s) triggered an exit criteria!")

            st.subheader("Live Portfolio Tracker Ledger")
            
            def color_signal(val):
                if val == "🔴 SELL": return 'background-color: #fce8e6; color: #c5221f; font-weight: bold;'
                if val == "🟢 HOLD": return 'background-color: #e6f4ea; color: #137333; font-weight: bold;'
                return ''
                
            st.dataframe(
                active_df.drop(columns=["ID"]).style.map(color_signal, subset=["Signal Status"]),
                use_container_width=True, hide_index=True
            )
            
            # --- LIQUIDATION INTERFACE ---
            st.markdown("---")
            st.subheader("⚡ Close Out Tracked Position")
            sell_opts = {f"{r['Ticker']} ({r['Qty']} shares bought on {r['Date Bought']})": r['ID'] for i, r in active_df.iterrows()}
            sel_sell_label = st.selectbox("Select position to realize exit transaction:", options=list(sell_opts.keys()))
            target_sell_id = sell_opts[sel_sell_label]
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                realized_sell_price = st.number_input("Actual Sell Price per Share (₹)", min_value=0.01, step=0.1, value=active_df.loc[active_df['ID'] == target_sell_id, 'Current Price (₹)'].values[0])
            with col_s2:
                realized_sell_date = st.date_input("Exit Settlement Execution Date", max_value=datetime.today())
                
            if st.button("Execute Sale Settlement", type="primary"):
                idx = next(i for i, item in enumerate(db["active"]) if item["id"] == target_sell_id)
                closed_trade = db["active"].pop(idx)
                closed_trade["sell_price"] = float(realized_sell_price)
                closed_trade["sell_date"] = str(realized_sell_date)
                
                db["history"].append(closed_trade)
                save_db(db)
                st.success("Asset position moved into historical ledger.")
                st.rerun()

# ==========================================
# TAB 2: HISTORICAL RETURN ANALYTICS
# ==========================================
with tab2:
    st.header("📈 Calendar Performance Analytics")
    
    if not db["history"]:
        st.info("Log closed trades from active tracking to generate historical performance calculations.")
    else:
        hist_rows = []
        for h in db["history"]:
            cost = h["qty"] * h["buy_price"]
            rev = h["qty"] * h["sell_price"]
            net = rev - cost
            
            try:
                date_obj = datetime.strptime(h["sell_date"], "%Y-%m-%d")
                month_key = date_obj.strftime("%Y-%m (%B)")
            except:
                month_key = "Unknown Calendar Axis"
                
            hist_rows.append({
                "MonthKey": month_key,
                "Ticker": h["ticker"],
                "Cost": cost,
                "Revenue": rev,
                "NetPnL": net
            })
            
        hdf = pd.DataFrame(hist_rows)
        
        # Monthly Performance Calculations
        monthly_summary = hdf.groupby("MonthKey").agg(
            Total_Invested_Capital=("Cost", "sum"),
            Total_Net_Profit=("NetPnL", "sum")
        ).reset_index()
        
        monthly_summary["Monthly Return (%)"] = (monthly_summary["Total_Net_Profit"] / monthly_summary["Total_Invested_Capital"]) * 100
        
        st.subheader("Monthly Performance Scorecard")
        
        st.dataframe(
            monthly_summary.rename(columns={
                "MonthKey": "Calendar Month",
                "Total_Invested_Capital": "Invested Capital (₹)",
                "Total_Net_Profit": "Realized Profits (₹)"
            }).style.map(lambda v: f"color: {'#137333' if v >= 0 else '#c5221f'}; font-weight: bold;", subset=["Realized Profits (₹)", "Monthly Return (%)"]),
            use_container_width=True, hide_index=True
        )
        
        st.markdown("---")
        st.subheader("📜 Complete Historical Transaction Logs")
        
        display_hist_df = pd.DataFrame([{
            "Ticker": x["ticker"], "Qty": x["qty"], 
            "Date Bought": x["buy_date"], "Buy Price (₹)": x["buy_price"],
            "Date Sold": x["sell_date"], "Sell Price (₹)": x["sell_price"],
            "Net Profit (₹)": (x["qty"] * x["sell_price"]) - (x["qty"] * x["buy_price"])
        } for x in db["history"]])
        
        st.dataframe(display_hist_df, use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: ERROR CORRECTIONS / POOL EDITING
# ==========================================
with tab3:
    st.header("🔧 Correct Existing Entry Data")
    if not db["active"]:
        st.info("No active entries are available for modification.")
    else:
        edit_opts = {f"{r['Ticker']} ({r['Qty']} shares bought on {r['Date Bought']})": r['ID'] for i, r in active_df.iterrows()}
        sel_edit_label = st.selectbox("Select entity position to adjust fields:", options=list(edit_opts.keys()))
        target_edit_id = edit_opts[sel_edit_label]
        
        edit_idx = next(i for i, x in enumerate(db["active"]) if x["id"] == target_edit_id)
        curr_edit = db["active"][edit_idx]
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            m_qty = st.number_input("Update Quantity", min_value=1, step=1, value=int(curr_edit["qty"]))
        with col_e2:
            m_prc = st.number_input("Update Purchase Price (₹)", min_value=0.01, step=0.1, value=float(curr_edit["buy_price"]))
            
        if st.button("Save Strategic Overwrites", type="secondary"):
            db["active"][edit_idx]["qty"] = int(m_qty)
            db["active"][edit_idx]["buy_price"] = float(m_prc)
            save_db(db)
            st.success("Entry details successfully updated.")
            st.rerun()
