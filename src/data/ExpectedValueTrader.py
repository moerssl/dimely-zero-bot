import pandas as pd
import numpy as np
import math
from scipy.stats import norm


class ExpectedValueTrader():
    def __init__(self, df: pd.DataFrame, logFunc=None):
        self.options_data = df
        self.logFunc = logFunc if logFunc else lambda x: None

    def find_best_ev_credit_spreads(self, symbol: str, max_wingspan: float, tp: float = 0.5,
                              sl: float = 2.0, alpha: float = 0.25,
                              min_credit: float = 0.2) -> dict:
        """
        Finds the iron condor with highest combined EV, enforcing long_put < short_put < short_call < long_call.
        Does not alter DataFrame index; uses ConId for row lookup.
        """
        try:
            df = self.options_data[
                (self.options_data['Symbol'] == symbol) &
                (self.options_data['delta'] <= 0.5) &
                (self.options_data['delta'] >= -0.5) &
                (self.options_data['bid'] > 0) &
                (self.options_data['ask'] > 0)
            ]
            if df.empty:
                return None

            def _spread_df(side_df, is_call: bool):
                pairs = side_df.merge(side_df, how='cross', suffixes=('_short','_long'))
                # proper strike order
                if is_call:
                    pairs = pairs[pairs['Strike_long'] > pairs['Strike_short']]
                else:
                    pairs = pairs[pairs['Strike_short'] > pairs['Strike_long']]
                pairs['wingspan'] = (pairs['Strike_long'] - pairs['Strike_short']).abs()
                pairs = pairs[pairs['wingspan'] <= max_wingspan]
                pairs['net_credit'] = pairs['bid_short'] - pairs['ask_long']
                pairs = pairs[pairs['net_credit'] >= min_credit]
                if pairs.empty:
                    return None
                # EV simple_partial
                d_s = pairs['delta_short'].abs()
                d_l = pairs['delta_long'].abs()
                p_win = 1 - d_s
                p_full = d_l
                p_part = (d_s - d_l).abs()
                mp = pairs['net_credit']
                ml = mp - pairs['wingspan']
                mid = (mp + ml) / 2
                pairs['EV'] = p_win * mp + p_full * ml + p_part * mid
                return pairs[['ConId_short','ConId_long','Strike_short','Strike_long','EV']]

            calls = df[df['Type'] == 'C']
            puts = df[df['Type'] == 'P']
            call_spreads = _spread_df(calls, True)
            put_spreads = _spread_df(puts, False)

            # both sides: best iron condor
            if call_spreads is not None and put_spreads is not None:
                combos = call_spreads.merge(put_spreads, how='cross', suffixes=('_call','_put'))
                valid = combos[
                    (combos['Strike_long_put'] < combos['Strike_short_put']) &
                    (combos['Strike_short_put'] < combos['Strike_short_call']) &
                    (combos['Strike_short_call'] < combos['Strike_long_call'])
                ]
                if valid.empty:
                    return None
                valid = valid.copy()
                valid['EV_total'] = valid['EV_call'] + valid['EV_put']
                best = valid.loc[valid['EV_total'].idxmax()]

                # lookup rows by ConId using boolean indexing and iloc
                lc = self.options_data[self.options_data['ConId'] == best['ConId_long_call']].iloc[0]
                sc = self.options_data[self.options_data['ConId'] == best['ConId_short_call']].iloc[0]
                sp = self.options_data[self.options_data['ConId'] == best['ConId_short_put']].iloc[0]
                lp = self.options_data[self.options_data['ConId'] == best['ConId_long_put']].iloc[0]
                return {'long_call': lc, 'short_call': sc,
                        'short_put': sp, 'long_put': lp}

            # fallback: best single side
            out = {}
            if call_spreads is not None:
                best_c = call_spreads.loc[call_spreads['EV'].idxmax()]
                out['long_call'] = self.options_data[self.options_data['ConId'] == best_c['ConId_long']].iloc[0]
                out['short_call'] = self.options_data[self.options_data['ConId'] == best_c['ConId_short']].iloc[0]
            if put_spreads is not None:
                best_p = put_spreads.loc[put_spreads['EV'].idxmax()]
                out['short_put'] = self.options_data[self.options_data['ConId'] == best_p['ConId_short']].iloc[0]
                out['long_put'] = self.options_data[self.options_data['ConId'] == best_p['ConId_long']].iloc[0]
            return out or None
        except Exception as e:
            self.logFunc(f"Error in find_best_ev_credit_spreads: {e}")
            return None

    def find_best_ev_credit_spreads_old(self, symbol, max_wingspan, tp=1.0, alpha=0.25, minTypeCredit = 0.2):
        try:
            """
            Sucht in self.options_data nach Credit-Spreads für das angegebene Symbol und einem
            maximalen Wingspan. Verwendet einen vectorisierten Ansatz (Cross Join) und berechnet den EV
            für jeden möglichen Spread mit:
            
                net_credit = short.bid - long.ask
                wingspan   = (long.Strike - short.Strike)  (bei Calls)
                        oder = (short.Strike - long.Strike)  (bei Puts)
                EV         = (net_credit * tp) - (abs(long.delta) * (wingspan - net_credit))
            
            Gibt ein Dictionary zurück mit den originalen Optionsdaten der besten
            Spreads. Wenn z.B. keine Put-Spreads gefunden werden, enthält das Ergebnis nur die Call-Spreads.
            Zurückgegebene Keys: "short_call", "long_call", "short_put", "long_put".
            """

            df = self.options_data[
                (self.options_data["Symbol"] == symbol) & 
                (self.options_data["delta"].abs() <= 0.50) &
                (self.options_data["bid"] > 0.0) &
                (self.options_data["ask"] > 0.0)
            ].copy()
            if df.empty:
                return None

            # Filtere Calls und Puts
            calls = df[df["Type"] == "C"].copy()
            puts = df[df["Type"] == "P"].copy()

            best_call = None
            best_put = None

            # --- Vectorized Suche für Call-Spreads ---
            if len(calls) > 1:
                # Zurückerhaltung des Originalindex, um später Originalzeilen zu holen.
                calls_reset = calls.reset_index(drop=False)  # Spalte 'index' wird hinzugefügt.
                # Erzeuge alle möglichen Pairings (Cross Join) zwischen Call-Optionen.
                call_pairs = calls_reset.merge(calls_reset, how="cross", suffixes=("_short", "_long"))
                # Nur Paare, bei denen long.Strike > short.Strike (richtiges Spread).
                call_pairs = call_pairs[call_pairs["Strike_long"] > call_pairs["Strike_short"]]
                
                # Wingspan berechnen
                call_pairs["wingspan"] = call_pairs["Strike_long"] - call_pairs["Strike_short"]
                call_pairs = call_pairs[call_pairs["wingspan"] <= max_wingspan]
                
                # Net Credit: bid_short minus ask_long
                call_pairs["net_credit"] = call_pairs["bid_short"] - call_pairs["ask_long"]
                # Nur gültige Spreads, bei denen ein positiver Net Credit erzielt wird.
                call_pairs = call_pairs[call_pairs["net_credit"] > minTypeCredit]

                call_pairs["weighted_delta"] = (alpha * call_pairs["delta_short"].abs()
                                  + (1 - alpha) * call_pairs["delta_long"].abs())

                
                # EV berechnen, wobei abs(long.delta) als Wahrscheinlichkeit für den max Verlust dient.
                call_pairs["EV"] = (call_pairs["net_credit"] * tp) - (
                    call_pairs["weighted_delta"].abs() * (call_pairs["wingspan"] - call_pairs["net_credit"])
                )
                if not call_pairs.empty:
                    best_call = call_pairs.loc[call_pairs["EV"].idxmax()]
            
            # --- Vectorized Suche für Put-Spreads ---
            if len(puts) > 1:
                puts_reset = puts.reset_index(drop=False)
                put_pairs = puts_reset.merge(puts_reset, how="cross", suffixes=("_short", "_long"))
                # Für Put-Spreads muss short.Strike > long.Strike gelten.
                put_pairs = put_pairs[put_pairs["Strike_short"] > put_pairs["Strike_long"]]
                
                put_pairs["wingspan"] = put_pairs["Strike_short"] - put_pairs["Strike_long"]
                put_pairs = put_pairs[put_pairs["wingspan"] <= max_wingspan]
                
                # Net Credit: bid_short minus ask_long
                put_pairs["net_credit"] = put_pairs["bid_short"] - put_pairs["ask_long"]
                put_pairs = put_pairs[put_pairs["net_credit"] > minTypeCredit]

                put_pairs["weighted_delta"] = (alpha * put_pairs["delta_short"].abs()
                                  + (1 - alpha) * put_pairs["delta_long"].abs())

                
                put_pairs["EV"] = (put_pairs["net_credit"] * tp) - (
                    put_pairs["weighted_delta"].abs() * (put_pairs["wingspan"] - put_pairs["net_credit"])
                )
                if not put_pairs.empty:
                    best_put = put_pairs.loc[put_pairs["EV"].idxmax()]

            result = {}
            # Retrieve the best call spread by using the original ConId values.
            if best_call is not None:
                conid_short = best_call["ConId_short"]
                conid_long  = best_call["ConId_long"]
                best_call_short = self.options_data[self.options_data["ConId"] == conid_short].iloc[0]
                best_call_long  = self.options_data[self.options_data["ConId"] == conid_long].iloc[0]
                result["long_call"]  = best_call_long
                result["short_call"] = best_call_short

            # Retrieve the best put spread by looking up by ConId.
            if best_put is not None:
                conid_short = best_put["ConId_short"]
                conid_long  = best_put["ConId_long"]
                best_put_short = self.options_data[self.options_data["ConId"] == conid_short].iloc[0]
                best_put_long  = self.options_data[self.options_data["ConId"] == conid_long].iloc[0]
                result["short_put"] = best_put_short
                result["long_put"]  = best_put_long

            # Return whatever spread(s) were found.
            return result if result else None

        except Exception as e:
            return None
        
    def calcSpreadExpectedValue(contract_rows, tp=0.5, sl=2, method="simple_partial"):
        """
        Calculate the expected value (EV) of a vertical credit spread,
        with optional TP/SL early‑exit levels.

        Parameters
        ----------
        contract_rows : dict
            Two entries (keys contain "short" and "long"), each a dict with:
            - 'bid' or 'ask' (float): short leg bid, long leg ask
            - 'Strike' (float): strike price
            - 'delta' (float): option delta (use absolute for ITM probability)
        tp : float
            Take‑profit level, as a fraction of the initial credit.
            (e.g. 0.5 → cap profit at 50% of the credit.)
        sl : float
            Stop‑loss multiple, relative to initial credit.
            (e.g. 2 → cap loss at 2× the credit.)
        method : {"simple", "simple_partial", "barrier"}
            - "simple":         EV = P(win)*max_profit + P(loss)*max_loss  
            - "simple_partial": includes linear partial‑payoff region  
            - "barrier":        EV = P(win)*TP_reward + P(loss)*(-SL_loss)

        Returns
        -------
        float
            EV in dollars per one spread contract.
        """
        
        # --- extract legs ---
        short_leg = next(v for k,v in contract_rows.items() if "short" in k.lower())
        long_leg  = next(v for k,v in contract_rows.items() if "long"  in k.lower())

        if short_leg is None or long_leg is None:
            return None
        
        Δ_SP        = short_leg["delta"]   # P(short ITM)
        Δ_LP        = long_leg["delta"]    # P(long  ITM)

        if Δ_SP is None or Δ_LP is None:
            return None
        


        short_bid   = short_leg["bid"]
        long_ask    = long_leg["ask"]
        K_short     = short_leg["Strike"]
        K_long      = long_leg["Strike"]
        Δ_SP        = abs(Δ_SP)   # P(short ITM)
        Δ_LP        = abs(Δ_LP)    # P(long  ITM)

        # --- basic checks & credit/loss geometry ---
        net_credit    = short_bid - long_ask
        if net_credit <= 0:
            return None
        width         = abs(K_long - K_short)
        max_profit    =  net_credit
        max_loss      =  net_credit - width   # negative

        # --- probability building blocks ---
        p_win         = 1.0 - Δ_SP            # prob spread expires worthless
        p_loss        = Δ_SP                  # prob SP in‑the‑money at expiry
        p_full_loss   = Δ_LP                  # prob you actually hit full loss
        p_partial     = abs(Δ_SP - Δ_LP)      # remainder for simple_partial

        # --- barrier thresholds (in $) ---
        TP_reward     = tp * net_credit       # capped profit
        SL_loss       = sl * net_credit       # capped loss (positive number)
        # note: actual payoff if SL hit is -SL_loss

        # --- EV calculations ---
        method = method.lower()
        if method == "simple":
            ev = p_win * max_profit + p_loss * max_loss

        elif method == "simple_partial":
            mid_payoff = (max_profit + max_loss) / 2.0
            ev = (
                p_win       * max_profit
            + p_full_loss * max_loss
            + p_partial  * mid_payoff
            )

        elif method == "barrier":
            ev = p_win * TP_reward + p_loss * (-SL_loss)

        else:
            raise ValueError(f"Unknown method: {method}")

        return round(ev,2)

    def calcSpreadExpectedValueSimple(contract_rows, tp=0.5, alpha=0.25):
        """
        Berechnet den Expected Value (EV) eines vertikalen Optionsspreads
        mithilfe eines gewichteten Delta-Ansatzes, sodass sowohl Verluste ab
        dem Short-Strike als auch volle Verluste ab dem Long-Strike berücksichtigt werden.
        
        Annahmen:
        - Es gibt genau zwei Legs: einen Short-Leg und einen Long-Leg.
        - Jeder Leg wird als Dictionary dargestellt mit den Schlüsseln:
                • "bid": Verkaufspreis (Short)
                • "ask": Kaufpreis (Long)
                • "delta": Delta der Option (bei Calls typischerweise positiv, bei Puts negativ)
                • "Strike": Ausübungspreis
        - tp (Target-Profit-Faktor) misst, welcher Anteil des Net Credits realisiert wird.
        - alpha bestimmt, wie stark der short delta versus der long delta gewichtet wird.
            Mit alpha=0.5 wird einfach der Durchschnitt der beiden absoluten Deltas verwendet.
        
        Vorgehen:
        1. Berechne net_credit:
                net_credit = (bid Short) - (ask Long)
        2. Berechne strike_diff:
                - Bei Call Spreads (short.delta > 0): strike_diff = long_Strike - short_Strike
                - Bei Put Spreads (short.delta < 0):  strike_diff = short_Strike - long_Strike
        3. max_loss = strike_diff - net_credit
        4. profit = net_credit * tp
        5. Berechne weighted_delta:
                weighted_delta = alpha * |short.delta| + (1 - alpha) * |long.delta|
        6. EV = profit - (weighted_delta * max_loss)
        
        :param contract_rows: Dictionary mit zwei Einträgen, beispielsweise {"short": {...}, "long": {...}}
        :param tp: Target-Profit-Faktor (zum Beispiel 0.8 für 80% des Kredits)
        :param alpha: Gewichtungsfaktor zwischen 0 und 1, mit Standardwert 0.5
        :return: Expected Value (EV) oder None, falls nötige Daten fehlen.
        """
        if not contract_rows:
            return None

        net_credit = 0.0
        short_strike = None
        long_strike = None
        short_delta = None
        long_delta = None

        # Extrahiere Werte aus den Legs (erwarte Keys, die "short" und "long" enthalten)
        for key, option in contract_rows.items():
            if option is not None:
                if "short" in key.lower():
                    net_credit += option.get("bid", 0.0)  # Erhaltene Prämie
                    short_strike = option.get("Strike")
                    short_delta = option.get("delta")
                elif "long" in key.lower():
                    net_credit -= option.get("ask", 0.0)    # Gezahlt wird der Ask-Preis
                    long_strike = option.get("Strike")
                    long_delta = option.get("delta")
        
        if (short_strike is None or long_strike is None or 
            short_delta is None or long_delta is None):
            return None

        # Bestimme den Spread-Typ anhand des short delta.
        if short_delta > 0:
            # Call Spread: Short Delta > 0, long strike muss größer sein als short strike.
            strike_diff = long_strike - short_strike
        else:
            # Put Spread: Short Delta < 0, short strike muss größer sein als long strike.
            strike_diff = short_strike - long_strike

        strike_diff = abs(strike_diff)
        net_credit = abs(net_credit)
        max_loss = strike_diff - net_credit
        profit = net_credit * tp

        # Berechne die gewichtete Delta: Bei alpha=0.5 = (|short_delta|+|long_delta|)/2
        weighted_delta = (alpha * abs(short_delta)) + ((1 - alpha) * abs(long_delta))

        EV = profit - (weighted_delta * max_loss)
        return round(EV,2) if EV else EV


    def calcExpectedValue(contract_rows, tp=0.5, alpha=0.25, simple=False):
        """
        Berechnet den Expected Value (EV) eines Iron Condors, indem er
        die EVs des Bear Call Spreads (obere Schene) und des Bull Put Spreads (untere Schene)
        summiert. Dabei wird calcSpreadExpectedValue wiederverwendet.
        
        Falls nicht alle vier notwendigen Legs ("short_call", "long_call", "short_put", "long_put")
        vorhanden sind, erfolgt ein Fallback zur Spread-Berechnung.
        
        :param contract_rows: Dictionary mit den vier Optionsdaten.
        :param tp: Target-Profit-Faktor.
        :param alpha: Gewichtungsfaktor, der für die einzelne Spread-Berechnung an calcSpreadExpectedValue übergeben wird.
        :return: Gesamter EV des Iron Condors oder None.
        """
        if contract_rows is None:
            return None

        required_keys = ["short_call", "long_call", "short_put", "long_put"]
        for key in required_keys:
            if key not in contract_rows or contract_rows[key] is None:
                # Fallback: Falls nicht alle Legs vorhanden, benutze die Spread-Berechnung.
                if simple:

                    return ExpectedValueTrader.calcSpreadExpectedValueSimple(contract_rows, tp, alpha)
                else:
                    return ExpectedValueTrader.calcSpreadExpectedValue(contract_rows, tp)

        # Rufe calcSpreadExpectedValue sowohl für den Call- als auch den Put-Spread auf.
        call_spread = {"short": contract_rows["short_call"], "long": contract_rows["long_call"]}
        if simple:
            ev_call =  ExpectedValueTrader.calcSpreadExpectedValueSimple(call_spread, tp, alpha)
        else:
            ev_call =  ExpectedValueTrader.calcSpreadExpectedValue(call_spread, tp)

        put_spread = {"short": contract_rows["short_put"], "long": contract_rows["long_put"]}
        if simple:
            ev_put = ExpectedValueTrader.calcSpreadExpectedValueSimple(put_spread, tp, alpha)
        else:
            ev_put = ExpectedValueTrader.calcSpreadExpectedValue(put_spread, tp)
        
        total_EV = 0
        if ev_put is not None and math.isnan(ev_put) == False:
            total_EV += ev_put

        if ev_call is not None and math.isnan(ev_call) == False:
            total_EV += ev_call
        
        return round(total_EV,2) if total_EV is not None else total_EV