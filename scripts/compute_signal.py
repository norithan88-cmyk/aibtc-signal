#!/usr/bin/env python3
"""
AI FX研究所 - 本日のAIシグナル 自動計算スクリプト

やっていること（概要）:
  1. Yahoo Finance の公開チャートAPI（無料・キー不要）から
     USD/JPY の価格（5分足・15分足・1時間足）を取得する。毎回実行時に取得。
  2. Alpha Vantage（無料枠）から米10年債利回り・WTI原油（いずれも日足）を取得する。
     ただし元データが日足のため、当日分をすでに取得済みなら再取得せず使い回す
    （無料枠が1日25回までのため、頻繁な実行でも枠を消費しないようにするため）。
  3. 各時間足について「線形回帰チャネル」を計算し、直近の価格が
     チャネルのどこに位置するかで SELL / BUY / WAIT / GATE を判定する。
  4. 4つの時間足の判定を集計して、総合バイアス・信頼度・相場モードを決める。
  5. Entry / Take Profit / Stop Loss を、直近のチャネル（1時間足）から算出する。
  6. 結果を signal.json として書き出す（GitHub Actionsがコミットし、
     jsDelivr経由でWordPress側から読み込む）。

設計方針:
  - ブラックボックスなAI予測ではなく、「なぜその判定になったか」を
    誰でも追える単純な統計ルール（回帰チャネル）にしている。
  - 実際のトレード成績を保証するものではない。あくまで
    「参考情報を自動更新する」ためのツール。
  - 為替の分足・時間足データは、Alpha Vantageの無料枠が2026年時点で
    「historical intraday」を有料化してしまったため、Yahoo Financeの
    非公式だが広く使われているチャートAPI（yfinance等でも使われているもの）
    を利用している。公式サポートのAPIではないため、将来URLの仕様が
    変わって取得できなくなる可能性はゼロではない（その場合はエラーとして
    検知され、既存の静的表示のまま維持される）。
  - 米10年債・WTIはAlpha Vantageの無料枠（日足データ）で取得。1日1回だけ
    実際にAPIを呼び、それ以外の実行では前回のsignal.jsonの値を使い回す。
    JP10Y・DXY・GOLDはAlpha Vantage無料枠では取得できないため、
    このスクリプトの計算には使っていない
    （サイト上ではTradingViewのライブティッカーで別途表示のみ）。
  - 経済指標カレンダー（今夜の重要指標）は無料で信頼できる自動取得先が
    見つからなかったため、今回は自動化していない（手動更新のまま）。
"""

import json
import os
import statistics
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "").strip()
BASE_URL = "https://www.alphavantage.co/query"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/JPY=X"

# 回帰チャネル計算に使う直近バーの本数
LOOKBACK = 50

# チャネル内での位置（sigma単位）による状態判定のしきい値
GATE_THRESHOLD = 2.2   # これを超えたら「GATE」（チャネルを突破。継続か反転か見極め）
EDGE_THRESHOLD = 1.3   # これを超えたら「SELL」または「BUY」（バンド際、逆張り優勢）
# それ未満は「WAIT」（中央付近、方向感なし）


def http_get_json(params, retries=3, wait_sec=15):
    """Alpha Vantage APIを呼び出してJSONを返す。レート制限時は少し待って再試行する。"""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE_URL}?{query}&apikey={ALPHA_VANTAGE_KEY}"
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as res:
                data = json.loads(res.read().decode("utf-8"))
            if "Note" in data or "Information" in data:
                # レート制限メッセージ。少し待って再試行。
                last_err = RuntimeError(data.get("Note") or data.get("Information"))
                time.sleep(wait_sec)
                continue
            return data
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(wait_sec)
    raise RuntimeError(f"API呼び出しに失敗しました: {params.get('function')} ({last_err})")


def fetch_fx_intraday(interval, range_):
    """
    Yahoo Financeの公開チャートAPIからUSD/JPYの分足・時間足データ（ローソク足）を取得し、
    [{"t":timestamp,"o":始値,"h":高値,"l":安値,"c":終値}, ...] を古い順（null値を除く）で返す。
    interval例: "5m" "15m" "60m" / range例: "5d" "60d"
    """
    url = f"{YAHOO_CHART_URL}?interval={interval}&range={range_}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                data = json.loads(res.read().decode("utf-8"))
            result = data.get("chart", {}).get("result")
            if not result:
                raise RuntimeError(f"USD/JPY {interval} のデータが取得できませんでした: {data.get('chart', {}).get('error')}")
            r = result[0]
            timestamps = r.get("timestamp") or []
            quote = r["indicators"]["quote"][0]
            opens = quote.get("open") or []
            highs = quote.get("high") or []
            lows = quote.get("low") or []
            closes = quote.get("close") or []
            bars = []
            for i, ts in enumerate(timestamps):
                c = closes[i] if i < len(closes) else None
                if c is None:
                    continue
                o = opens[i] if i < len(opens) and opens[i] is not None else c
                h = highs[i] if i < len(highs) and highs[i] is not None else max(o, c)
                low = lows[i] if i < len(lows) and lows[i] is not None else min(o, c)
                bars.append({"t": ts, "o": o, "h": h, "l": low, "c": c})
            if not bars:
                raise RuntimeError(f"USD/JPY {interval} のデータが空でした")
            return bars
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            last_err = e
            time.sleep(5)
    raise RuntimeError(f"Yahoo Financeからの取得に失敗しました: {interval} ({last_err})")


def fetch_treasury_yield_10y():
    data = http_get_json({
        "function": "TREASURY_YIELD",
        "interval": "daily",
        "maturity": "10year",
    })
    series = data.get("data")
    if not series:
        raise RuntimeError(f"米10年債利回りが取得できませんでした: {data}")
    # 新しい順で返ってくる想定。値が "." のことがあるので除外。
    values = [(row["date"], row["value"]) for row in series if row.get("value") not in (None, ".")]
    values = [(d, float(v)) for d, v in values]
    values.sort()  # 古い順に
    return values


def fetch_wti_daily():
    data = http_get_json({
        "function": "WTI",
        "interval": "daily",
    })
    series = data.get("data")
    if not series:
        raise RuntimeError(f"WTI原油が取得できませんでした: {data}")
    values = [(row["date"], row["value"]) for row in series if row.get("value") not in (None, ".")]
    values = [(d, float(v)) for d, v in values]
    values.sort()
    return values


def aggregate_to_4h(hourly_bars):
    """1時間足のローソク足リストから、4本ごとにまとめた4時間足のローソク足リストを作る。"""
    grouped = []
    for i in range(0, len(hourly_bars), 4):
        chunk = hourly_bars[i:i + 4]
        if chunk:
            grouped.append({
                "t": chunk[0]["t"],
                "o": chunk[0]["o"],
                "h": max(b["h"] for b in chunk),
                "l": min(b["l"] for b in chunk),
                "c": chunk[-1]["c"],
            })
    return grouped


def linear_regression_channel(closes, lookback=LOOKBACK):
    """
    直近 lookback 本の終値から線形回帰チャネルを計算する。
    戻り値: dict(mid, upper, lower, sigma, position, slope)
      position = 直近終値がチャネル中心から何sigma離れているか（+が上、-が下）
      slope    = 回帰直線の傾き（1本あたりの価格変化）
    """
    series = closes[-lookback:] if len(closes) > lookback else closes[:]
    n = len(series)
    if n < 5:
        raise RuntimeError("回帰チャネル計算に必要なデータ本数が不足しています")

    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(series) / n

    num = sum((xs[i] - x_mean) * (series[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    intercept = y_mean - slope * x_mean

    fitted = [intercept + slope * x for x in xs]
    residuals = [series[i] - fitted[i] for i in range(n)]
    sigma = statistics.pstdev(residuals) if n > 1 else 0.0
    sigma = sigma if sigma > 1e-6 else 1e-6  # ゼロ割回避

    mid = fitted[-1]
    upper = mid + 2 * sigma
    lower = mid - 2 * sigma
    latest = series[-1]
    position = (latest - mid) / sigma

    return {
        "mid": mid, "upper": upper, "lower": lower, "sigma": sigma,
        "position": position, "slope": slope, "intercept": intercept, "n": n,
        "latest": latest,
    }


def classify_state(position):
    if position >= GATE_THRESHOLD:
        return "GATE"
    if position >= EDGE_THRESHOLD:
        return "SELL"
    if position <= -GATE_THRESHOLD:
        return "GATE"
    if position <= -EDGE_THRESHOLD:
        return "BUY"
    return "WAIT"


def trend_direction(values, days=5):
    """直近days件の値から単純な向き（up/down/flat）を判定する。"""
    if len(values) < 2:
        return "flat"
    recent = [v for _, v in values[-days:]]
    if len(recent) < 2:
        return "flat"
    change = recent[-1] - recent[0]
    span = max(abs(v) for v in recent) or 1.0
    if abs(change) / span < 0.01:
        return "flat"
    return "up" if change > 0 else "down"


TREND_JA = {"up": "上昇", "down": "低下", "flat": "横ばい"}


def build_market_context(bias, sell_count, buy_count, gate_count, latest_price,
                          day_change_pct, us10y_trend, wti_trend):
    """
    「直近の指標・報道まとめ」欄用の文章を、その時点の実データから自動生成する。
    固定文ではなく、価格・トレンドという生きた数値を毎回埋め込むため、
    時間が経っても内容が古びない（＝手動更新が要らない）設計にしている。
    ここでは経済ニュースの見出しそのものは扱わず、あくまで「今の数値が
    何を示しているか」の解説にとどめている（ニュース自体の自動取得は
    別途 経済指標カレンダーAPIの導入が必要で、現状は未対応）。
    """
    change_txt = f"{day_change_pct:+.2f}%"
    y = TREND_JA.get(us10y_trend, "横ばい")
    w = TREND_JA.get(wti_trend, "横ばい")

    if bias == "SELL":
        stance = f"{sell_count}個の時間足が上値の重さを示しており、戻り売りが優勢な地合い"
        outlook = "目先は上値の重い展開が想定され、高値を追わず戻りを待つスタンスが機能しやすい局面。"
    elif bias == "BUY":
        stance = f"{buy_count}個の時間足が下値の堅さを示しており、押し目買いが優勢な地合い"
        outlook = "目先は下値の堅い展開が想定され、押し目を焦らず拾うスタンスが機能しやすい局面。"
    else:
        stance = f"時間足ごとの判定が割れており（SELL {sell_count}／BUY {buy_count}／GATE {gate_count}）、方向感に乏しいレンジ地合い"
        outlook = "明確なブレイクが出るまでは、無理に取りにいかず様子見が無難な局面。"

    return (
        f"USD/JPYは現在{latest_price:.2f}円付近で推移（直近1時間比{change_txt}）。{stance}。"
        f"米10年債利回りは{y}基調、WTI原油は{w}基調で推移している。{outlook}"
        "※このまとめは実データから自動生成された定型解説です。個別の経済指標発表や"
        "ニュース速報の内容までは反映していません。"
    )


def load_previous_signal(out_path):
    """前回書き出したsignal.jsonを読む（無ければNone）。"""
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def is_same_utc_date(iso_ts, now):
    """iso_ts（ISO形式の日時文字列）がnowと同じUTC日付かどうか。"""
    if not iso_ts:
        return False
    try:
        dt = datetime.fromisoformat(iso_ts)
    except ValueError:
        return False
    return dt.date() == now.date()


def build_signal(out_path=None):
    if not ALPHA_VANTAGE_KEY:
        raise RuntimeError("環境変数 ALPHA_VANTAGE_KEY が設定されていません")

    now = datetime.now(timezone.utc)

    # --- 為替データ: Yahoo Finance（無料・キー不要・毎回取得） ---
    m5 = fetch_fx_intraday("5m", "5d")
    m15 = fetch_fx_intraday("15m", "5d")
    h1 = fetch_fx_intraday("60m", "60d")

    # --- 米10年債・WTI: Alpha Vantage（無料枠 25回/日のため、1日1回だけ取得して使い回す） ---
    # そもそもTREASURY_YIELD/WTIは日足データなので、1日に何度呼んでも値は変わらない。
    prev = load_previous_signal(out_path) if out_path else None
    prev_macro = (prev or {}).get("macro", {})
    reuse_macro = prev and is_same_utc_date(prev.get("generated_at_utc"), now) and prev_macro.get("us10y_latest") is not None

    if reuse_macro:
        yield_trend = prev_macro.get("us10y_trend", "flat")
        us10y_latest = prev_macro.get("us10y_latest")
        wti_trend = prev_macro.get("wti_trend", "flat")
    else:
        us10y = fetch_treasury_yield_10y()
        time.sleep(13)
        wti = fetch_wti_daily()
        yield_trend = trend_direction(us10y)
        us10y_latest = us10y[-1][1] if us10y else None
        wti_trend = trend_direction(wti)

    bars_4h = aggregate_to_4h(h1)

    ch_5m = linear_regression_channel([b["c"] for b in m5])
    ch_15m = linear_regression_channel([b["c"] for b in m15])
    ch_1h = linear_regression_channel([b["c"] for b in h1])
    ch_4h = linear_regression_channel([b["c"] for b in bars_4h], lookback=30)

    timeframes = [
        {"label": "5分足", "key": "m5", "channel": ch_5m, "bars": m5, "lookback": LOOKBACK},
        {"label": "15分足", "key": "m15", "channel": ch_15m, "bars": m15, "lookback": LOOKBACK},
        {"label": "1時間足", "key": "h1", "channel": ch_1h, "bars": h1, "lookback": LOOKBACK},
        {"label": "4時間足", "key": "h4", "channel": ch_4h, "bars": bars_4h, "lookback": 30},
    ]
    for tf in timeframes:
        tf["state"] = classify_state(tf["channel"]["position"])

    sell_count = sum(1 for tf in timeframes if tf["state"] == "SELL")
    buy_count = sum(1 for tf in timeframes if tf["state"] == "BUY")
    gate_count = sum(1 for tf in timeframes if tf["state"] == "GATE")

    if sell_count >= 2 and sell_count >= buy_count:
        bias = "SELL"
    elif buy_count >= 2 and buy_count > sell_count:
        bias = "BUY"
    else:
        bias = "WAIT"

    agreeing = [tf for tf in timeframes if tf["state"] == bias] if bias in ("SELL", "BUY") else []
    if agreeing:
        avg_abs_pos = sum(abs(tf["channel"]["position"]) for tf in agreeing) / len(agreeing)
        agreement_ratio = len(agreeing) / len(timeframes)
        confidence = 50 + agreement_ratio * 30 + min(avg_abs_pos, 3.0) * 5
        confidence = max(50, min(95, round(confidence)))
        stars = max(1, min(5, round(confidence / 20)))
    else:
        confidence = 50
        stars = 2

    directional_tfs = sell_count + buy_count
    if directional_tfs >= 3:
        market_mode = "TREND"
        market_mode_note = "複数の時間足でチャネル際まで到達しており、方向感のある地合い。"
    elif gate_count >= 1:
        market_mode = "EVENT DRIVEN"
        market_mode_note = "回帰チャネルの突破が見られ、材料次第で振れやすい局面。"
    else:
        market_mode = "RANGE"
        market_mode_note = "多くの時間足が中央付近で推移しており、方向感に乏しいレンジ地合い。"

    latest_price = m5[-1]["c"] if m5 else ch_1h["latest"]
    day_change_pct = 0.0
    if len(h1) >= 24:
        base = h1[-24]["c"]
        if base:
            day_change_pct = (latest_price - base) / base * 100

    intervention_risk = "HIGH" if (latest_price >= 158.5 and day_change_pct >= 0.7) else (
        "MID" if latest_price >= 155.0 else "LOW"
    )

    ref_channel = ch_1h
    if bias == "SELL":
        entry = latest_price
        tp = ref_channel["mid"]
        sl = ref_channel["upper"] + 0.5 * ref_channel["sigma"]
        trade_lead = "戻り売り ― ただし押し目を深追いしない"
    elif bias == "BUY":
        entry = latest_price
        tp = ref_channel["mid"]
        sl = ref_channel["lower"] - 0.5 * ref_channel["sigma"]
        trade_lead = "押し目買い ― ただし高値を深追いしない"
    else:
        entry = tp = sl = None
        trade_lead = "様子見 ― チャネル中央で方向感なし"

    comments = {
        "SELL": [
            "強い相場ほど、飛び乗らない。戻りを丁寧に売る一日に。",
            "上値は重い。高値づかみを避け、戻り待ちに徹する。",
        ],
        "BUY": [
            "押し目は焦らず拾う。飛び乗りより、待つ勇気を。",
            "下値は堅い。押し目待ちで、無理な高値追いはしない。",
        ],
        "WAIT": [
            "方向感のない日は、休むも相場。無理に取りにいかない。",
            "チャネルの中央は様子見。ブレイクを待つのが賢明。",
        ],
    }
    commentary = comments.get(bias, comments["WAIT"])[0]
    market_context = build_market_context(
        bias, sell_count, buy_count, gate_count, latest_price, day_change_pct,
        yield_trend, wti_trend,
    )

    return {
        "generated_at_utc": now.isoformat(),
        "pair": "USD/JPY",
        "latest_price": round(latest_price, 3),
        "day_change_pct": round(day_change_pct, 2),
        "signal": {
            "bias": bias,
            "bias_label": {"SELL": "戻り売り優勢", "BUY": "押し目買い優勢", "WAIT": "方向感なし"}[bias],
            "stars": stars,
            "confidence": confidence,
        },
        "intervention_risk": intervention_risk,
        "market_mode": market_mode,
        "market_mode_note": market_mode_note,
        "priority_trade": {
            "lead": trade_lead,
            "entry": round(entry, 3) if entry is not None else None,
            "take_profit": round(tp, 3) if tp is not None else None,
            "stop_loss": round(sl, 3) if sl is not None else None,
        },
        "regression_channels": [
            {
                "label": tf["label"],
                "state": tf["state"],
                "position_sigma": round(tf["channel"]["position"], 2),
            }
            for tf in timeframes
        ],
        "charts": [
            {
                "label": tf["label"],
                "state": tf["state"],
                "bars": [
                    {
                        "o": round(b["o"], 3), "h": round(b["h"], 3),
                        "l": round(b["l"], 3), "c": round(b["c"], 3),
                    }
                    for b in tf["bars"][-tf["lookback"]:]
                ],
                "channel": {
                    "intercept": round(tf["channel"]["intercept"], 4),
                    "slope": round(tf["channel"]["slope"], 6),
                    "sigma": round(tf["channel"]["sigma"], 4),
                },
            }
            for tf in timeframes
        ],
        "macro": {
            "us10y_trend": yield_trend,
            "us10y_latest": round(us10y_latest, 2) if us10y_latest is not None else None,
            "wti_trend": wti_trend,
        },
        "commentary": commentary,
        "market_context": market_context,
        "disclaimer": "本データはルールベースの参考情報であり、投資成果を保証するものではありません。",
    }


def main():
    out_path = os.path.join(os.path.dirname(__file__), "..", "signal.json")
    out_path = os.path.abspath(out_path)

    try:
        signal = build_signal(out_path=out_path)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] シグナル計算に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)
    print(f"書き出し完了: {out_path}")
    print(json.dumps(signal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
