#!/usr/bin/env python3
"""
AI BTC研究所 - 本日のAIシグナル 自動計算スクリプト

やっていること（概要）:
  1. Binance の公開API（無料・キー不要・公式）から
     BTC/USDT の価格（1分足・5分足・15分足・1時間足）を取得する。毎回実行時に取得。
  2. Binance Futuresの資金調達率（レバレッジ過熱度の指標）、
     Fear & Greed Index（市場心理指数）、CoinGeckoのBTCドミナンス（市場内シェア）を
     補助データとして取得する（いずれも無料・キー不要）。
  3. 5分・15分・1時間足の3つで線形回帰チャネルを計算し、3つとも「チャネル中心から
     ±1.3σ以上その方向に偏っている」状態(momentum_direction)が一致した時だけ、
     上位足の方向（押し目買い/戻り売りの候補方向）を確定する。
  4. その方向候補が確定している時だけ、1分足がその方向に逆行してチャネル際まで
     達し、そこから戻り始めたタイミング(detect_reversal_setup)を検出し、
     検出できた瞬間だけ実際のSELL/BUYシグナルとして確定する（それ以外はWAIT）。
  5. Entry = 直近1分足終値。TP/SLは、1分足の逆行の谷/山（測定値幅）を基準に算出する。
  6. 結果を signal.json として書き出す（GitHub Actionsがコミットし、
     WordPress側からraw.githubusercontent.com経由で読み込む）。

設計方針:
  - ブラックボックスなAI予測ではなく、「なぜその判定になったか」を
    誰でも追える単純な統計ルール（回帰チャネル）にしている。
  - このロジック（5分・15分・1時間足の方向一致＋1分足の逆行からの戻り）は、
    AI FX研究所（USD/JPY版）で2025-01〜2026-07の19ヶ月・実データバックテストにより
    検証済み（勝率70.3%・PF1.88）のものをBTC/USDTに移植している。ただしバックテストは
    USD/JPYで行ったものであり、値動きの性質が異なる暗号資産で同等の成績が出るとは
    限らない点に留意（値動きが荒い・ボラティリティが高い等の違いがある）。
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
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

SYMBOL = "BTCUSDT"

# 回帰チャネル計算に使う直近バーの本数
LOOKBACK = 100

# チャネル内での位置（sigma単位）がこれを超えたら「その方向に強く偏っている」とみなす
EDGE_THRESHOLD = 1.3

# 1分足の「逆行からの戻り」判定パラメータ（USD/JPY版のバックテストで検証済みの値をそのまま採用）
REVERT_WINDOW = 10       # 直近何本(分)以内に逆行の谷/山を探すか
REVERT_MIN_PIPS = 3.0    # 「戻り出した」とみなす最低反発幅
SL_BUFFER_PIPS = 2.0     # 逆行の谷/山からSLまでの余白

# USD/JPYは1pips=0.01円だが、BTC/USDTは価格スケールが全く異なる（数万〜数十万ドル）ため、
# 固定pips幅ではなくドル建ての固定幅を使う。BTC/USDTの通常の値動き（1分足で数十ドル単位）を
# 踏まえ、REVERT_MIN_USDT・SL_BUFFER_USDTという名前で同じ役割の定数を別途持つ。
REVERT_MIN_USDT = 15.0
SL_BUFFER_USDT = 10.0


def http_get_json(url, retries=3, wait_sec=5, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                return json.loads(res.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            time.sleep(wait_sec)
    raise RuntimeError(f"取得に失敗しました: {url} ({last_err})")


def fetch_klines(interval, limit):
    """
    Binance公式APIからBTC/USDTのローソク足を取得し、
    [{"t":timestamp,"o":始値,"h":高値,"l":安値,"c":終値}, ...] を古い順で返す。
    interval例: "1m" "5m" "15m" "1h"
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


def momentum_direction(ch):
    """
    5分・15分・1時間足それぞれについて、「チャネル中心から何σ離れているか」(position)が
    ±EDGE_THRESHOLD(1.3σ)を超えていれば、その方向に強く偏っている(継続方向)とみなす。
    3つの時間足がこの判定で全て同じ方向になった時だけ、上位足の方向候補が確定する
    （build_signal参照）。
    """
    pos = ch["position"]
    if pos >= EDGE_THRESHOLD:
        return "UP"
    if pos <= -EDGE_THRESHOLD:
        return "DOWN"
    return "FLAT"


def detect_reversal_setup(bars, ch, direction):
    """
    directionは上位3時間足が一致した方向候補("BUY"/"SELL")。1分足がこの方向とは
    逆に振れてチャネル際(±EDGE_THRESHOLD)まで達し、そこから戻り始めていれば、
    その谷(BUYの場合)/山(SELLの場合)の価格を返す。まだ戻り始めていない・戻り幅が
    REVERT_MIN_USDT未満・そもそもチャネル際まで達していない場合はNoneを返す。
    """
    if len(bars) < REVERT_WINDOW:
        return None
    recent = bars[-REVERT_WINDOW:]
    closes = [b["c"] for b in recent]
    latest = closes[-1]
    sigma = ch["sigma"]
    mid = ch["mid"]

    if direction == "BUY":
        trough_idx = min(range(len(closes)), key=lambda i: closes[i])
        trough = closes[trough_idx]
        if trough_idx == len(closes) - 1:
            return None  # 最新バーがまだ谷=反発が始まっていない
        trough_position = (trough - mid) / sigma
        if trough_position > -EDGE_THRESHOLD:
            return None  # チャネル下限際まで到達していない
        if (latest - trough) < REVERT_MIN_USDT:
            return None  # 戻り幅が不十分(ノイズ)
        return trough
    else:
        peak_idx = max(range(len(closes)), key=lambda i: closes[i])
        peak = closes[peak_idx]
        if peak_idx == len(closes) - 1:
            return None
        peak_position = (peak - mid) / sigma
        if peak_position < EDGE_THRESHOLD:
            return None
        if (peak - latest) < REVERT_MIN_USDT:
            return None
        return peak


def moving_average_trend(closes, short=10, long=30):
    """
    短期・長期の単純移動平均のクロスからトレンド方向を判定する（参考表示用）。
    差が0.05%未満なら方向感なし（FLAT）扱い。売買判定には使用しない。
    """
    if len(closes) < long:
        return "FLAT"
    short_ma = sum(closes[-short:]) / short
    long_ma = sum(closes[-long:]) / long
    if not long_ma:
        return "FLAT"
    diff_ratio = (short_ma - long_ma) / long_ma
    if diff_ratio > 0.0005:
        return "UP"
    if diff_ratio < -0.0005:
        return "DOWN"
    return "FLAT"


def build_market_context(bias, candidate, latest_price, day_change_pct,
                          fear_greed_value, fear_greed_label,
                          funding_rate_pct, btc_dominance_pct):
    """
    「今の相場環境」欄用の文章を、その時点の実データから自動生成する。
    固定文ではなく、価格・トレンドという生きた数値を毎回埋め込むため、
    時間が経っても内容が古びない（＝手動更新が要らない）設計にしている。

    bias: 最終的なシグナル("SELL"/"BUY"/"WAIT")
    candidate: 5分・15分・1時間足の方向一致だけで見た候補方向(一致していなければNone)。
      biasがWAITでもcandidateがある場合、「上位足は方向一致しているが1分足の
      反発シグナルがまだ点灯していない」ことを示せるため、単なるWAITより
      具体的な状況説明ができる。
    """
    change_txt = f"{day_change_pct:+.2f}%"

    if bias == "SELL":
        stance = "5分・15分・1時間足が揃って上値の重さを示す中、1分足が短期的な戻りから反落したタイミング"
        outlook = "目先は上値の重い展開が想定され、高値を追わず戻りを待つスタンスが機能しやすい局面。"
    elif bias == "BUY":
        stance = "5分・15分・1時間足が揃って下値の堅さを示す中、1分足が短期的な押し目から反発したタイミング"
        outlook = "目先は下値の堅い展開が想定され、押し目を焦らず拾うスタンスが機能しやすい局面。"
    elif candidate == "SELL":
        stance = "5分・15分・1時間足は戻り売り方向で揃っているが、1分足の反落シグナルはまだ点灯していない"
        outlook = "上位足の方向感は出ているため、1分足が戻り高値から反落するタイミングを待ちたい局面。"
    elif candidate == "BUY":
        stance = "5分・15分・1時間足は押し目買い方向で揃っているが、1分足の反発シグナルはまだ点灯していない"
        outlook = "上位足の方向感は出ているため、1分足が押し目安値から反発するタイミングを待ちたい局面。"
    else:
        stance = "5分・15分・1時間足の方向が揃っておらず、方向感に乏しいレンジ地合い"
        outlook = "明確な方向一致が出るまでは、無理に取りにいかず様子見が無難な局面。"

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


def build_daily_analysis(signal, timeframes):
    """詳細分析ページ用データを、同じ実データから毎回自動生成する。"""
    price = signal["latest_price"]
    bias = signal["signal"]["bias"]
    macro = signal["macro"]
    channels = {tf["key"]: tf["channel"] for tf in timeframes}
    h1 = channels["h1"]
    support = min(h1["lower"], price)
    resistance = max(h1["upper"], price)
    if bias == "BUY":
        conclusion = "押し目買い候補。ただし飛び乗らず、1分足の反発確認を優先。"
    elif bias == "SELL":
        conclusion = "戻り売り候補。ただし安値追いを避け、1分足の反落確認を優先。"
    else:
        conclusion = "方向一致が出るまで見送り。休むも相場を優先。"

    scenarios = [
        {"name": "上昇", "condition": f"${resistance:,.0f}を明確に上抜けて定着", "action": "押し目を待って買いを検討。高値への飛び乗りは避ける。"},
        {"name": "レンジ", "condition": f"${support:,.0f}〜${resistance:,.0f}で往来", "action": "中央では見送り、上下限で反発確認後のみ検討。"},
        {"name": "下落", "condition": f"${support:,.0f}を明確に下抜け", "action": "戻りを待って売りを検討。急落後の安値追いは避ける。"},
    ]
    return {
        "generated_at_utc": signal["generated_at_utc"],
        "title": "今日のBTC/USDT分析",
        "conclusion": conclusion,
        "market_context": signal["market_context"],
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "timeframes": signal["regression_channels"],
        "scenarios": scenarios,
        "fear_greed": {"value": macro.get("fear_greed_value"), "label": macro.get("fear_greed_label")},
        "funding_rate_pct": macro.get("funding_rate_pct"),
        "btc_dominance_pct": macro.get("btc_dominance_pct"),
        "priority_trade": signal["priority_trade"],
        "risk_note": "暗号資産は24時間取引され、短時間で大きく変動します。損切り水準と許容損失額を先に決めてください。",
    }


def load_trade_log(base_dir):
    """
    trade_log.json（リポジトリ直下、signal.jsonと同じ階層）を読み込む。
    過去のシグナル履歴（勝率・pnl検証用）を蓄積するファイルで、signal.jsonとは
    別ファイルにして肥大化を防いでいる。存在しない/壊れている場合は空の履歴から始める。
    """
    path = os.path.join(base_dir, "trade_log.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data.get("trades"), list):
            raise ValueError("trade_log.jsonの形式が不正です")
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, AttributeError):
        return {"trades": []}


def pnl_usdt_for(bias, entry, price):
    """bias方向での損益（USDT建て、正=利益、負=損失）を返す。"""
    diff = (entry - price) if bias == "SELL" else (price - entry)
    return round(diff, 2)


def update_trade_log(trade_log, bias, priority_trade, latest_price, confidence, now_iso):
    """
    ①オープン中の取引があれば、現在値がTP/SLに到達していないか確認して決着させる。
    ②オープン中の取引が無く、今回の判定がSELL/BUYであれば、新規にオープンとして記録する。
    同時に複数の建玉は持たない設計（両建てにしない）。
    戻り値は (trade_log, newly_opened)。newly_openedは②で実際に新規オープンした場合だけTrue。
    """
    trades = trade_log.get("trades", [])
    open_trade = trades[-1] if trades and trades[-1].get("status") == "OPEN" else None

    if open_trade is not None:
        ob = open_trade["bias"]
        tp = open_trade["take_profit"]
        sl = open_trade["stop_loss"]
        hit_tp = (latest_price <= tp) if ob == "SELL" else (latest_price >= tp)
        hit_sl = (latest_price >= sl) if ob == "SELL" else (latest_price <= sl)
        if hit_tp or hit_sl:
            open_trade["status"] = "WIN" if hit_tp else "LOSS"
            open_trade["closed_at_utc"] = now_iso
            open_trade["closed_price"] = round(latest_price, 2)
            open_trade["pnl_usdt"] = pnl_usdt_for(ob, open_trade["entry"], latest_price)
            open_trade = None  # 決着したので、この後の新規オープン判定に進める

    newly_opened = False
    if open_trade is None and bias in ("SELL", "BUY"):
        entry = priority_trade.get("entry")
        tp = priority_trade.get("take_profit")
        sl = priority_trade.get("stop_loss")
        if entry is not None and tp is not None and sl is not None:
            trades.append({
                "id": now_iso,
                "opened_at_utc": now_iso,
                "bias": bias,
                "entry": entry,
                "take_profit": tp,
                "stop_loss": sl,
                "confidence": confidence,
                "status": "OPEN",
                "closed_at_utc": None,
                "closed_price": None,
                "pnl_usdt": None,
            })
            newly_opened = True

    trade_log["trades"] = trades
    return trade_log, newly_opened


def compute_trade_stats(trades):
    """勝率・平均損益・プロフィットファクターを、決着済み（WIN/LOSS）の取引から算出する。"""
    closed = [t for t in trades if t.get("status") in ("WIN", "LOSS")]
    wins = [t for t in closed if t["status"] == "WIN"]
    losses = [t for t in closed if t["status"] == "LOSS"]
    total_closed = len(closed)

    gross_win = sum(t["pnl_usdt"] for t in wins)
    gross_loss = abs(sum(t["pnl_usdt"] for t in losses))

    return {
        "total_closed": total_closed,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / total_closed * 100, 1) if total_closed else None,
        "avg_win_usdt": round(gross_win / len(wins), 2) if wins else None,
        "avg_loss_usdt": round(-gross_loss / len(losses), 2) if losses else None,
        # 損失がまだ無い（＝分母ゼロ）場合はPF計算不能として扱い、無限大等の非JSON値を出さない。
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "total_usdt": round(sum(t["pnl_usdt"] for t in closed), 2) if closed else 0.0,
    }


def build_signal(out_path=None):
    now = datetime.now(timezone.utc)

    # --- BTC/USDT価格データ: Binance公式API（無料・キー不要・毎回取得） ---
    m1 = fetch_klines("1m", LOOKBACK)
    m5 = fetch_klines("5m", LOOKBACK)
    m15 = fetch_klines("15m", LOOKBACK)
    h1 = fetch_klines("1h", LOOKBACK)

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

    # --- 回帰チャネル: 5分・15分・1時間足で方向一致を判定、1分足で反発を検出 ---
    ch_1m = linear_regression_channel([b["c"] for b in m1])
    ch_5m = linear_regression_channel([b["c"] for b in m5])
    ch_15m = linear_regression_channel([b["c"] for b in m15])
    ch_1h = linear_regression_channel([b["c"] for b in h1])

    timeframes = [
        {"label": "5分足", "key": "m5", "channel": ch_5m, "bars": m5},
        {"label": "15分足", "key": "m15", "channel": ch_15m, "bars": m15},
        {"label": "1時間足", "key": "h1", "channel": ch_1h, "bars": h1},
    ]
    for tf in timeframes:
        tf["trend"] = moving_average_trend([b["c"] for b in tf["bars"]])
        tf["momentum"] = momentum_direction(tf["channel"])

    dirs = [tf["momentum"] for tf in timeframes]
    if dirs[0] == "UP" and dirs[1] == "UP" and dirs[2] == "UP":
        candidate = "BUY"
    elif dirs[0] == "DOWN" and dirs[1] == "DOWN" and dirs[2] == "DOWN":
        candidate = "SELL"
    else:
        candidate = None

    # 上位3時間足の方向が一致している時だけ、1分足の逆行からの戻りを調べる。
    extreme = detect_reversal_setup(m1, ch_1m, candidate) if candidate else None
    bias = candidate if (candidate and extreme is not None) else "WAIT"

    if bias in ("SELL", "BUY"):
        # 5分・15分・1時間足が全て一致している時しかbiasは確定しないため、
        # 一致度合いは常に3/3固定。代わりに、3時間足のチャネル際からの
        # 平均乖離度(avg_abs_pos)が大きいほど「強い一致」とみなして加点する。
        avg_abs_pos = sum(abs(tf["channel"]["position"]) for tf in timeframes) / len(timeframes)
        confidence = 50 + 30 + min(avg_abs_pos, 3.0) * 5
        confidence = max(50, min(95, round(confidence)))
        stars = max(1, min(5, round(confidence / 20)))
    else:
        confidence = 50
        stars = 2

    if candidate is not None:
        market_mode = "TREND"
        market_mode_note = "5分・15分・1時間足の方向が揃っており、方向感のある地合い。"
    else:
        market_mode = "RANGE"
        market_mode_note = "時間足ごとに方向が割れており、方向感に乏しいレンジ地合い。"

    latest_price = m1[-1]["c"] if m1 else ch_1h["latest"]
    day_change_pct = 0.0
    if len(h1) >= 24:
        base = h1[-24]["c"]
        if base:
            day_change_pct = (latest_price - base) / base * 100

    # 仮想通貨はFXよりも値動きが大きいため、しきい値はFX版より広めに設定
    # （急変動リスク＝短時間で大きく動いてロスカットが連鎖しやすい状況の目安）
    abs_change = abs(day_change_pct)
    volatility_risk = "HIGH" if abs_change >= 5.0 else ("MID" if abs_change >= 2.5 else "LOW")

    # Entry/TP/SLは、1分足の逆行の谷/山(extreme)を基準にした「測定値幅」で算出する。
    # SLはextremeの少し外側（このセットアップの前提が崩れる水準）、
    # TPはentryからextremeまでの距離を反対方向に伸ばした幅。
    if bias == "SELL":
        entry = latest_price
        move = abs(entry - extreme)
        sl = extreme + SL_BUFFER_USDT
        tp = entry - move
        trade_lead = "戻り売り ― 上位足の下降方向一致＋1分足の戻りからの反落"
    elif bias == "BUY":
        entry = latest_price
        move = abs(entry - extreme)
        sl = extreme - SL_BUFFER_USDT
        tp = entry + move
        trade_lead = "押し目買い ― 上位足の上昇方向一致＋1分足の押し目からの反発"
    else:
        entry = tp = sl = None
        if candidate == "SELL":
            trade_lead = "様子見 ― 上位足は戻り売り方向で一致、1分足の反落シグナル待ち"
        elif candidate == "BUY":
            trade_lead = "様子見 ― 上位足は押し目買い方向で一致、1分足の反発シグナル待ち"
        else:
            trade_lead = "様子見 ― 5分・15分・1時間足の方向が一致していない"

    reversal_setup = None
    if bias in ("SELL", "BUY"):
        reverted_amount = round((entry - extreme), 2) if bias == "BUY" else round((extreme - entry), 2)
        reversal_setup = {"extreme": round(extreme, 2), "reverted_usdt": reverted_amount}

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
            "1分足のタイミングを待つのが賢明。",
        ],
    }
    commentary = comments.get(bias, comments["WAIT"])[0]
    market_context = build_market_context(
        bias, candidate, latest_price, day_change_pct,
        fear_greed_value, fear_greed_label, funding_rate_pct, btc_dominance_pct,
    )

    result = {
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
        "reversal_setup": reversal_setup,
        "regression_channels": [
            {
                "key": tf["key"],
                "label": tf["label"],
                "position_sigma": round(tf["channel"]["position"], 2),
                "trend": tf["trend"],
                "mid": round(tf["channel"]["mid"], 2),
                "upper": round(tf["channel"]["upper"], 2),
                "lower": round(tf["channel"]["lower"], 2),
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
    result["daily_analysis"] = build_daily_analysis(result, timeframes)

    # trade_log.json（実績ページ用の履歴）はsignal.jsonとは別ファイルに直接書き出す。
    # ここで失敗しても、シグナル本体の計算・書き出しには影響させない。
    if out_path:
        try:
            base_dir = os.path.dirname(out_path)
            trade_log = load_trade_log(base_dir)
            trade_log, _newly_opened = update_trade_log(
                trade_log, bias, result["priority_trade"], latest_price, confidence, now.isoformat(),
            )
            trade_log["stats"] = compute_trade_stats(trade_log["trades"])
            trade_log["updated_at_utc"] = now.isoformat()
            with open(os.path.join(base_dir, "trade_log.json"), "w", encoding="utf-8") as f:
                json.dump(trade_log, f, ensure_ascii=False, indent=2)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] trade_log.jsonの更新に失敗しました（シグナル本体は継続します）: {e}", file=sys.stderr)

    return result


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
