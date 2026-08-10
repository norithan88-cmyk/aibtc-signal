#!/usr/bin/env python3
"""
AI BTC研究所 - 本日のAIシグナル 自動計算スクリプト

やっていること（概要）:
  1. Binance の公開API（無料・キー不要・公式）から
     BTC/USDT の価格（5分足・15分足・1時間足・4時間足）を取得する。毎回実行時に取得。
  2. Binance Futuresの資金調達率（レバレッジ過熱度の指標）、
     Fear & Greed Index（市場心理指数）、CoinGeckoのBTCドミナンス（市場内シェア）を
     補助データとして取得する（いずれも無料・キー不要）。
  3. 各時間足について「線形回帰チャネル」を計算し、直近の価格が
     チャネルのどこに位置するかで SELL / BUY / WAIT / GATE を判定する。
  4. 4つの時間足の判定を集計して、総合バイアス・信頼度・相場モードを決める。
  5. Entry / Take Profit / Stop Loss を、直近のチャネル（1時間足）から算出する。
  6. 結果を signal.json として書き出す（GitHub Actionsがコミットし、
     WordPress側からraw.githubusercontent.com経由で読み込む）。

設計方針:
  - ブラックボックスなAI予測ではなく、「なぜその判定になったか」を
    誰でも追える単純な統計ルール（回帰チャネル）にしている。
  - 実際のトレード成績を保証するものではない。あくまで
    「参考情報を自動更新する」ためのツール。
  - 使用しているAPI（Binance公式API・CoinGecko・alternative.me）はいずれも無料枠に
    キー登録不要で使えるため、AI FX研究所（USD/JPY版）と違ってAPIキー管理が不要。
  - 仮想通貨は24時間365日取引されるため、FX版にあった「週末は市場が閉まっていて
    データが止まる」という事象は基本的に発生しない。
  - ローソク足データは api.binance.com ではなく data-api.binance.vision を使っている。
    api.binance.com はアメリカ等一部地域からのアクセスを地域制限（451エラー）で
    拒否することがあり、GitHub Actionsの実行環境（海外クラウド）から失敗する
    ことがあったため。data-api.binance.vision はBinanceが市場データの自動取得・
    Bot用途向けに公式に用意している地域制限を受けにくいミラーで、認証不要・
    レスポンス形式もapi.binance.comと同一。
"""

import json
import os
import statistics
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

SYMBOL = "BTCUSDT"

# 回帰チャネル計算に使う直近バーの本数
LOOKBACK = 50

# チャネル内での位置（sigma単位）による状態判定のしきい値
GATE_THRESHOLD = 2.2   # これを超えたら「GATE」（チャネルを突破。継続か反転か見極め）
EDGE_THRESHOLD = 1.3   # これを超えたら「SELL」または「BUY」（バンド際、逆張り優勢）
# それ未満は「WAIT」（中央付近、方向感なし）


def http_get_json(url, retries=3, wait_sec=5, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                return json.loads(res.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            import time
            time.sleep(wait_sec)
    raise RuntimeError(f"取得に失敗しました: {url} ({last_err})")


def fetch_klines(interval, limit):
    """
    Binance公式APIからBTC/USDTのローソク足を取得し、
    [{"t":timestamp,"o":始値,"h":高値,"l":安値,"c":終値}, ...] を古い順で返す。
    interval例: "5m" "15m" "1h" "4h"
    """
    url = f"{BINANCE_KLINES_URL}?symbol={SYMBOL}&interval={interval}&limit={limit}"
    data = http_get_json(url)
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"BTC/USDT {interval} のデータが取得できませんでした: {data}")
    bars = []
    for row in data:
        bars.append({
            "t": row[0],
            "o": float(row[1]),
            "h": float(row[2]),
            "l": float(row[3]),
            "c": float(row[4]),
        })
    return bars


def fetch_funding_rate():
    """直近の資金調達率（%換算前の小数値）。正値=ロング優勢（過熱）、負値=ショート優勢。"""
    data = http_get_json(f"{BINANCE_FUNDING_URL}?symbol={SYMBOL}&limit=1")
    if not data:
        raise RuntimeError("資金調達率が取得できませんでした")
    return float(data[0]["fundingRate"])


def fetch_fear_greed():
    """Fear & Greed Index（0-100、0に近いほど恐怖、100に近いほど強欲）を返す。"""
    data = http_get_json(FEAR_GREED_URL + "?limit=1")
    series = data.get("data")
    if not series:
        raise RuntimeError(f"Fear & Greed Indexが取得できませんでした: {data}")
    row = series[0]
    return int(row["value"]), row.get("value_classification", "")


def fetch_btc_dominance():
    """暗号資産市場全体に占めるBTCの時価総額シェア（%）。"""
    data = http_get_json(COINGECKO_GLOBAL_URL)
    pct = data.get("data", {}).get("market_cap_percentage", {})
    if "btc" not in pct:
        raise RuntimeError(f"BTCドミナンスが取得できませんでした: {data}")
    return float(pct["btc"])


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


def build_market_context(bias, sell_count, buy_count, gate_count, latest_price,
                          day_change_pct, fear_greed_value, fear_greed_label,
                          funding_rate_pct, btc_dominance_pct):
    """
    「今の相場環境」欄用の文章を、その時点の実データから自動生成する。
    固定文ではなく、価格・トレンドという生きた数値を毎回埋め込むため、
    時間が経っても内容が古びない（＝手動更新が要らない）設計にしている。
    """
    change_txt = f"{day_change_pct:+.2f}%"

    if bias == "SELL":
        stance = f"{sell_count}個の時間足が上値の重さを示しており、戻り売りが優勢な地合い"
        outlook = "目先は上値の重い展開が想定され、高値を追わず戻りを待つスタンスが機能しやすい局面。"
    elif bias == "BUY":
        stance = f"{buy_count}個の時間足が下値の堅さを示しており、押し目買いが優勢な地合い"
        outlook = "目先は下値の堅い展開が想定され、押し目を焦らず拾うスタンスが機能しやすい局面。"
    else:
        stance = f"時間足ごとの判定が割れており（SELL {sell_count}／BUY {buy_count}／GATE {gate_count}）、方向感に乏しいレンジ地合い"
        outlook = "明確なブレイクが出るまでは、無理に取りにいかず様子見が無難な局面。"

    macro_parts = []
    if fear_greed_value is not None:
        macro_parts.append(f"市場心理は「{fear_greed_label}」（{fear_greed_value}/100）")
    if funding_rate_pct is not None:
        funding_note = "レバレッジロングがやや過熱気味" if funding_rate_pct >= 0.01 else (
            "レバレッジショートがやや優勢" if funding_rate_pct <= -0.01 else "レバレッジは中立圏"
        )
        macro_parts.append(f"資金調達率は{funding_rate_pct:+.3f}%で{funding_note}")
    if btc_dominance_pct is not None:
        macro_parts.append(f"BTCドミナンスは{btc_dominance_pct:.1f}%")
    macro_txt = "、".join(macro_parts) + "。" if macro_parts else ""

    return (
        f"BTC/USDTは現在${latest_price:,.0f}付近で推移（直近1時間比{change_txt}）。{stance}。"
        f"{macro_txt}{outlook}"
        "※このまとめは実データから自動生成された定型解説です。個別のニュース速報の内容までは反映していません。"
    )


def build_signal():
    now = datetime.now(timezone.utc)

    # --- BTC/USDT価格データ: Binance公式API（無料・キー不要・毎回取得） ---
    m5 = fetch_klines("5m", LOOKBACK)
    m15 = fetch_klines("15m", LOOKBACK)
    h1 = fetch_klines("1h", LOOKBACK)
    h4 = fetch_klines("4h", 30)

    # --- 補助データ: すべて無料・キー不要なので毎回取得する。
    #     ただし価格・チャート本体（上記）の更新を止めないよう、
    #     いずれか1つが取得失敗してもここではエラーにせずNoneのまま続行する。
    try:
        funding_rate_pct = fetch_funding_rate() * 100
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 資金調達率の取得に失敗しました（続行します）: {e}", file=sys.stderr)
        funding_rate_pct = None
    try:
        fear_greed_value, fear_greed_label = fetch_fear_greed()
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Fear & Greed Indexの取得に失敗しました（続行します）: {e}", file=sys.stderr)
        fear_greed_value, fear_greed_label = None, None
    try:
        btc_dominance_pct = fetch_btc_dominance()
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] BTCドミナンスの取得に失敗しました（続行します）: {e}", file=sys.stderr)
        btc_dominance_pct = None

    ch_5m = linear_regression_channel([b["c"] for b in m5])
    ch_15m = linear_regression_channel([b["c"] for b in m15])
    ch_1h = linear_regression_channel([b["c"] for b in h1])
    ch_4h = linear_regression_channel([b["c"] for b in h4], lookback=30)

    timeframes = [
        {"label": "5分足", "key": "m5", "channel": ch_5m, "bars": m5, "lookback": LOOKBACK},
        {"label": "15分足", "key": "m15", "channel": ch_15m, "bars": m15, "lookback": LOOKBACK},
        {"label": "1時間足", "key": "h1", "channel": ch_1h, "bars": h1, "lookback": LOOKBACK},
        {"label": "4時間足", "key": "h4", "channel": ch_4h, "bars": h4, "lookback": 30},
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

    # 仮想通貨はFXよりも値動きが大きいため、しきい値はFX版より広めに設定
    # （急変動リスク＝短時間で大きく動いてロスカットが連鎖しやすい状況の目安）
    abs_change = abs(day_change_pct)
    volatility_risk = "HIGH" if abs_change >= 5.0 else ("MID" if abs_change >= 2.5 else "LOW")

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
        fear_greed_value, fear_greed_label, funding_rate_pct, btc_dominance_pct,
    )

    return {
        "generated_at_utc": now.isoformat(),
        "pair": "BTC/USDT",
        "latest_price": round(latest_price, 2),
        "day_change_pct": round(day_change_pct, 2),
        "signal": {
            "bias": bias,
            "bias_label": {"SELL": "戻り売り優勢", "BUY": "押し目買い優勢", "WAIT": "方向感なし"}[bias],
            "stars": stars,
            "confidence": confidence,
        },
        "volatility_risk": volatility_risk,
        "market_mode": market_mode,
        "market_mode_note": market_mode_note,
        "priority_trade": {
            "lead": trade_lead,
            "entry": round(entry, 2) if entry is not None else None,
            "take_profit": round(tp, 2) if tp is not None else None,
            "stop_loss": round(sl, 2) if sl is not None else None,
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
                        "o": round(b["o"], 2), "h": round(b["h"], 2),
                        "l": round(b["l"], 2), "c": round(b["c"], 2),
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
            "fear_greed_value": fear_greed_value,
            "fear_greed_label": fear_greed_label,
            "funding_rate_pct": round(funding_rate_pct, 4) if funding_rate_pct is not None else None,
            "btc_dominance_pct": round(btc_dominance_pct, 2) if btc_dominance_pct is not None else None,
        },
        "commentary": commentary,
        "market_context": market_context,
        "disclaimer": "本データはルールベースの参考情報であり、投資成果を保証するものではありません。",
    }


def main():
    out_path = os.path.join(os.path.dirname(__file__), "..", "signal.json")
    out_path = os.path.abspath(out_path)

    try:
        signal = build_signal()
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] シグナル計算に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)
    print(f"書き出し完了: {out_path}")
    print(json.dumps(signal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
