"""Panel dashboard for MediaArchiving analytics."""
from __future__ import annotations

import ast
import io
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import arabic_reshaper
import numpy as np
import pandas as pd
import panel as pn
from bidi.algorithm import get_display
from matplotlib import font_manager
import matplotlib.pyplot as plt
from wordcloud import WordCloud

pn.extension("tabulator", sizing_mode="stretch_width")

# ===================== تنظیمات =====================
FALLBACK_EXCEL = Path("input.xlsx")
FALLBACK_SHEET = 0
STOPWORDS_PATH = Path("stop_word.txt")
OUTPUT_DIR_PATH = Path("outputs")
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

MIN_TOKEN_LEN = 2
TOP_K = None

COLUMNS_FOR_WORDCLOUD = [
    "Keyword Frequency",
    "Emotion Category",
    "Object Detection",
    "Face Detection",
    "Transcript",
    "Tone of Voice",
    "Desired Emotion",
    "Duration of Brand Exposure",
    "Appeal اقناعی",
    "Concept خلاق",
    "product_name",
    "brand_name",
    "نوع استراتژی اقناعی",
    "نمادها و اشیا",
]

MONTHS_FA = [
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
]
MONTH_MAP = {m: i + 1 for i, m in enumerate(MONTHS_FA)}

EXACT_COLS = {
    "color_dominance": "Color Dominance",
    "emotion_category": "Emotion Category",
    "desired_emotion": "Desired Emotion",
    "audience": "Insight مخاطب",
    "concept": "Concept خلاق",
    "tone": "Tone of Voice",
    "appeal": "Appeal اقناعی",
    "transcript": "Transcript",
    "symbols": "نمادها و اشیا",
    "colors": "Color Dominance",
    "music_mood": "mood موسیقی",
    "year": "year",
    "month": "month",
    "product": "product_name",
    "duration": "Duration",
    "avg_shot": "Average Shot Length",
    "scene_cnt": "Scene Count",
    "brand": "brand_name",
    "message": "Message اصلی",
    "ads": "ads",
    "persuasive": "نوع استراتژی اقناعی",
    "symbols_col": "نمادها و اشیا",
    "exposure": "Duration of Brand Exposure",
}


def pick_persian_font() -> str:
    candidates_by_name = [
        "Vazirmatn",
        "Vazir",
        "IRANSans",
        "Shabnam",
        "Sahel",
        "Yekan",
        "Tahoma",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    for name in candidates_by_name:
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if os.path.exists(path):
                return path
        except Exception:  # pragma: no cover - font lookup best effort
            pass
    for p in [r"C:\\Windows\\Fonts\\Yas.ttf", r"C:\\Windows\\Fonts\\iransans.ttf"]:
        if os.path.exists(p):
            return p
    return font_manager.findfont("DejaVu Sans")


FONT_PATH = pick_persian_font()


def rtl(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def shape_rtl(token: str) -> str:
    if not isinstance(token, str):
        return str(token)
    if not any("\u0600" <= ch <= "\u06FF" for ch in token):
        return token
    return rtl(token)


def norm_fa(s: str) -> str:
    s = str(s)
    s = s.replace("\u200c", "").replace("\u00A0", " ").strip()
    s = s.replace("ي", "ی").replace("ك", "ک")
    s = re.sub(r"[\u064B-\u0652]", "", s)
    return s


TOKEN_RE = re.compile(r"[0-9A-Za-z_]+|[\u0600-\u06FF]+")


def tokenize(txt: str) -> List[str]:
    txt = norm_fa(str(txt).lower())
    return TOKEN_RE.findall(txt)


def load_stopwords(path: Path) -> set:
    stops = set()
    if not path.exists():
        return {"و", "در", "به", "از", "که", "این", "آن", "برای", "با", "است", "را", "می"}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = norm_fa(s)
            toks = tokenize(s) if " " in s else [s]
            for t in toks:
                if len(t) >= MIN_TOKEN_LEN:
                    stops.add(t)
    return stops


PERSIAN_STOPS = load_stopwords(STOPWORDS_PATH)


def reload_stopwords() -> int:
    global PERSIAN_STOPS
    PERSIAN_STOPS = load_stopwords(STOPWORDS_PATH)
    return len(PERSIAN_STOPS)


def _maybe_parse(x):
    if isinstance(x, (list, tuple, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if (
            (s.startswith("[") and s.endswith("]"))
            or (s.startswith("{") and s.endswith("}"))
            or (s.startswith("(") and s.endswith(")"))
        ):
            for fn in (json.loads, ast.literal_eval):
                try:
                    return fn(s)
                except Exception:
                    continue
    return x


def _extract_tokens(x) -> List[Tuple[str, float]]:
    bag: List[Tuple[str, float]] = []
    x = _maybe_parse(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return bag

    if isinstance(x, dict):
        term = (
            x.get("term")
            or x.get("label")
            or x.get("word")
            or x.get("token")
            or x.get("speaker")
        )
        weight = x.get("freq") or x.get("count") or x.get("p") or x.get("probability") or 1
        try:
            weight = float(weight)
        except Exception:
            weight = 1.0
        if term is not None:
            terms = term if isinstance(term, (list, tuple)) else [term]
            for t in terms:
                for tok in tokenize(t):
                    bag.append((tok, weight))
        else:
            for v in x.values():
                bag += _extract_tokens(v)
        return bag

    if isinstance(x, (list, tuple)):
        for it in x:
            bag += _extract_tokens(it)
        return bag

    if isinstance(x, (int, float, np.integer, np.floating)):
        return bag

    for tok in tokenize(x):
        bag.append((tok, 1.0))
    return bag


def build_freq(df: pd.DataFrame, column: str, extra_stops=None, min_len=MIN_TOKEN_LEN, top=TOP_K) -> Counter:
    if column not in df.columns:
        return Counter()
    stops = set(PERSIAN_STOPS)
    if extra_stops:
        stops |= set(extra_stops)
    c = Counter()
    for v in df[column]:
        for tok, w in _extract_tokens(v):
            if len(tok) < min_len or tok in stops:
                continue
            c[tok] += float(w)
    return Counter(dict(c.most_common(top))) if top else c


def plot_wordcloud_rtl(freqs: Dict[str, float], title: str = None) -> Optional[plt.Figure]:
    if not freqs:
        return None
    shaped = {shape_rtl(k): v for k, v in freqs.items()}
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        font_path=FONT_PATH,
        collocations=False,
        prefer_horizontal=1.0,
    )
    wc.generate_from_frequencies(shaped)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(rtl(title))
    return fig


def pie_with_full_legend(values, labels, title: str, colors=None):
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _ = ax.pie(values, labels=None if len(values) > 5 else [rtl(l) for l in labels], startangle=90, colors=colors)
    if len(values) > 5:
        ax.legend(wedges, [rtl(l) for l in labels], loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(rtl(title))
    return fig


def parse_color_dominance(value) -> List[Tuple[str, float]]:
    parsed = _maybe_parse(value)
    result: List[Tuple[str, float]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                hx = item.get("hex") or item.get("color")
                wt = item.get("percent") or item.get("weight") or 0
                if hx:
                    try:
                        result.append((str(hx), float(wt)))
                    except Exception:
                        result.append((str(hx), 0.0))
    elif isinstance(parsed, dict):
        for k, v in parsed.items():
            try:
                result.append((str(k), float(v)))
            except Exception:
                continue
    else:
        tokens = re.findall(r"#?[0-9A-Fa-f]{6}", str(value))
        for tok in tokens:
            result.append((tok, 1.0))
    return result


def parse_seconds(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value)
    match = re.findall(r"\d+(?:\.\d+)?", s)
    if not match:
        return np.nan
    try:
        return float(match[0])
    except ValueError:
        return np.nan


def month_to_num(value) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.isdigit():
        num = int(s)
        return num if 1 <= num <= 12 else None
    return MONTH_MAP.get(s)


def count_tokens_from_col(df: pd.DataFrame, column: str, top: int = 15) -> Counter:
    if column not in df.columns:
        return Counter()
    c = Counter()
    for v in df[column].dropna():
        for tok, weight in _extract_tokens(v):
            if tok not in PERSIAN_STOPS:
                c[tok] += weight
    return Counter(dict(c.most_common(top)))


def draw_bar(counter: Counter, title: str) -> Optional[plt.Figure]:
    if not counter:
        return None
    items = counter.most_common()
    labels = [shape_rtl(k) for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(values)), values, color="#19647E")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(rtl(title))
    return fig


def draw_area_chart(df: pd.DataFrame, month_col: str, category_col: str, title: str) -> Optional[plt.Figure]:
    if month_col not in df.columns or category_col not in df.columns:
        return None
    sub = df.dropna(subset=[month_col, category_col])
    if sub.empty:
        return None
    sub["month_num"] = sub[month_col].map(month_to_num)
    sub = sub.dropna(subset=["month_num"])
    if sub.empty:
        return None
    pivot = sub.groupby(["month_num", category_col]).size().unstack(fill_value=0)
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(pivot.index, pivot.values.T, labels=[rtl(str(c)) for c in pivot.columns])
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([rtl(m) for m in MONTHS_FA])
    ax.set_title(rtl(title))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


def make_wordcloud_cards(df: pd.DataFrame) -> List[pn.layout.Panel]:
    cards = []
    for col in COLUMNS_FOR_WORDCLOUD:
        freqs = build_freq(df, col)
        fig = plot_wordcloud_rtl(freqs, title=col)
        cards.append(make_chart_card(fig, f"wordcloud_{col}.png", f"ابرکلمات {col}"))
    return cards


def emotion_pies(df: pd.DataFrame) -> List[pn.layout.Panel]:
    cards = []
    for col_key, title in [
        ("emotion_category", "Emotion Category (با حذف توقف‌واژه‌ها)"),
        ("desired_emotion", "Desired Emotion (با حذف توقف‌واژه‌ها)"),
    ]:
        column = EXACT_COLS.get(col_key)
        if not column or column not in df.columns:
            cards.append(make_chart_card(None, f"missing_{col_key}.png", title))
            continue
        cnt = Counter()
        for v in df[column].dropna():
            clean = str(v).replace("،", ",").replace("؛", ",").replace(";", ",")
            for t in tokenize(clean):
                if t not in PERSIAN_STOPS:
                    cnt[t] += 1
        fig = pie_with_full_legend([v for _, v in cnt.items()], [k for k, _ in cnt.items()], title)
        cards.append(make_chart_card(fig, f"pie_{col_key}.png", title))
    return cards


def color_charts(df: pd.DataFrame) -> List[pn.layout.Panel]:
    col_color_dom = EXACT_COLS["color_dominance"]
    cards = []
    if col_color_dom in df.columns:
        weight_by_hex = defaultdict(float)
        for v in df[col_color_dom].dropna():
            for hx, w in parse_color_dominance(v):
                weight_by_hex[str(hx)] += float(w)
        if weight_by_hex:
            items = sorted(weight_by_hex.items(), key=lambda x: x[1], reverse=True)
            labels = [i[0] for i in items]
            values = [i[1] for i in items]
            fig = pie_with_full_legend(values, labels, "توزیع رنگ‌های غالب")
            cards.append(make_chart_card(fig, "color_dominance.png", "توزیع رنگ‌های غالب"))
    cards.append(make_chart_card(build_color_stacked(df), "color_product.png", "رنگ غالب محصولات"))
    return cards


def build_color_stacked(df: pd.DataFrame) -> Optional[plt.Figure]:
    col_colors = EXACT_COLS["color_dominance"]
    col_prod = EXACT_COLS["product"]
    if col_colors not in df.columns or col_prod not in df.columns:
        return None
    rows = []
    for _, r in df[[col_prod, col_colors]].dropna(subset=[col_colors]).iterrows():
        prod = norm_fa(r[col_prod])
        for hx, w in parse_color_dominance(r[col_colors]):
            rows.append((prod, hx, w))
    if not rows:
        return None
    tab = pd.DataFrame(rows, columns=["product", "hex", "percent"])
    top_prods = tab.groupby("product")["percent"].sum().sort_values(ascending=False).head(6).index
    tab = tab[tab["product"].isin(top_prods)]
    pv = tab.pivot_table(index="product", columns="hex", values="percent", aggfunc="sum", fill_value=0)
    pv = pv.loc[sorted(pv.index, key=lambda x: -pv.loc[x].sum())]
    fig, ax = plt.subplots(figsize=(10, 5))
    pv.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xticklabels([rtl(x) for x in pv.index], rotation=30, ha="right")
    ax.set_title(rtl("توزیع رنگ‌های غالب در محصولات"))
    fig.tight_layout()
    return fig


def duration_charts(df: pd.DataFrame) -> List[pn.layout.Panel]:
    cards = []
    col_exposure = EXACT_COLS.get("exposure", "Duration of Brand Exposure")
    if col_exposure in df.columns:
        df["__exposure_sec"] = df[col_exposure].map(parse_seconds)
        bins = [0, 5, 10, 20, 30, np.inf]
        labels = [
            "کمتر از ۵ ثانیه",
            "۵ تا ۱۰ ثانیه",
            "۱۰ تا ۲۰ ثانیه",
            "۲۰ تا ۳۰ ثانیه",
            "بیش‌تر از ۳۰ ثانیه",
        ]
        df["__exposure_range"] = pd.cut(df["__exposure_sec"], bins=bins, labels=labels, right=False)
        vc = df["__exposure_range"].value_counts().sort_index()
        pct = (vc / vc.sum()) * 100
        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, _ = ax.pie(pct.values, startangle=90)
        ax.legend(wedges, [rtl(str(x)) for x in pct.index], loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(rtl("توزیع مدت زمان نمایش برند"))
        cards.append(make_chart_card(fig, "brand_exposure.png", "Duration of Brand Exposure"))
    duration_col = EXACT_COLS.get("duration")
    if duration_col and duration_col in df.columns:
        df["__duration_sec"] = df[duration_col].map(parse_seconds)
        fig, ax = plt.subplots(figsize=(8, 4))
        df["__duration_sec"].dropna().plot(kind="hist", bins=20, ax=ax, color="#E56B6F")
        ax.set_title(rtl("توزیع مدت زمان تیزرها"))
        cards.append(make_chart_card(fig, "duration_hist.png", "توزیع مدت زمان"))
    return cards


def product_charts(df: pd.DataFrame) -> List[pn.layout.Panel]:
    cards = []
    product_col = EXACT_COLS.get("product")
    if product_col in df.columns:
        vc = df[product_col].dropna().value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(vc)), vc.values, color="#4CAF50")
        ax.set_yticks(range(len(vc)))
        ax.set_yticklabels([rtl(str(x)) for x in vc.index])
        ax.set_title(rtl("محصولات پرتکرار"))
        cards.append(make_chart_card(fig, "product_bar.png", "محصولات پرتکرار"))
    ads_col = EXACT_COLS.get("ads")
    if ads_col in df.columns:
        vc = df[ads_col].dropna().value_counts().head(6)
        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, _ = ax.pie(vc.values, startangle=90)
        ax.legend(wedges, [rtl(str(x)) for x in vc.index], loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title(rtl("توزیع رسانه‌ها"))
        cards.append(make_chart_card(fig, "ads_pie.png", "توزیع رسانه‌ها"))
    return cards


def build_text_bars(df: pd.DataFrame) -> List[pn.layout.Panel]:
    cards = []
    targets = [
        ("audience", "واژه‌های Insight مخاطب"),
        ("concept", "واژه‌های Concept خلاق"),
        ("tone", "واژه‌های Tone of Voice"),
        ("appeal", "واژه‌های Appeal اقناعی"),
        ("transcript", "واژه‌های Transcript"),
        ("symbols", "نمادها و اشیا"),
        ("music_mood", "مودهای موسیقی"),
    ]
    for key, title in targets:
        column = EXACT_COLS.get(key)
        if not column:
            continue
        cnt = count_tokens_from_col(df, column)
        fig = draw_bar(cnt, title)
        cards.append(make_chart_card(fig, f"bar_{key}.png", title))
    return cards


def monthly_area_cards(df: pd.DataFrame) -> List[pn.layout.Panel]:
    cards = []
    for key, title in [
        ("product", "محصول × ماه"),
        ("tone", "Tone of Voice × ماه"),
    ]:
        column = EXACT_COLS.get(key)
        fig = draw_area_chart(df, EXACT_COLS.get("month", "month"), column, title)
        cards.append(make_chart_card(fig, f"area_{key}.png", title))
    return cards


def kpi_cards(df: pd.DataFrame) -> pn.Row:
    total_ads = len(df)
    scene_col = EXACT_COLS.get("scene_cnt")
    duration_col = EXACT_COLS.get("duration")
    emo_col = EXACT_COLS.get("emotion_category")
    avg_scene = (
        float(pd.to_numeric(df[scene_col], errors="coerce").mean()) if scene_col in df.columns else np.nan
    )
    avg_dur = (
        float(pd.to_numeric(df[duration_col].map(parse_seconds), errors="coerce").mean())
        if duration_col in df.columns
        else np.nan
    )
    most_emo = "-"
    if emo_col in df.columns:
        vals = []
        for v in df[emo_col].dropna():
            vals.extend(tokenize(v))
        if vals:
            most_emo = Counter(vals).most_common(1)[0][0]
    cards = []
    for title, value in [
        ("تعداد کل تیزرها", f"{total_ads}"),
        ("میانگین صحنه‌ها", f"{avg_scene:.1f}" if not np.isnan(avg_scene) else "-"),
        ("میانگین مدت زمان", f"{avg_dur:.1f} ثانیه" if not np.isnan(avg_dur) else "-"),
        ("احساس غالب", most_emo),
    ]:
        cards.append(
            pn.indicators.Number(
                name=title,
                value=value,
                format="{value}",
                styles={"font-family": "Vazirmatn, IRANSans, Arial"},
            )
        )
    return pn.Row(*cards)


def make_chart_card(fig: Optional[plt.Figure], filename: str, title: str):
    if fig is None:
        return pn.Card(pn.pane.Markdown(f"**{title}:** داده‌ای موجود نیست."), title=title)
    png_bytes = figure_to_png(fig)
    pane = pn.pane.PNG(png_bytes, sizing_mode="stretch_width")
    download = pn.widgets.FileDownload(
        label="دانلود نمودار",
        filename=filename,
        button_type="primary",
        data=png_bytes,
    )
    return pn.Card(pane, download, title=title)


def figure_to_png(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    data = buffer.getvalue()
    buffer.close()
    return data


@dataclass
class DashboardData:
    df: pd.DataFrame
    path: Optional[Path] = None


class Dashboard:
    def __init__(self):
        self.file_input = pn.widgets.FileInput(name="آپلود اکسل", accept=".xlsx,.xls")
        self.sheet_select = pn.widgets.IntInput(name="شماره شیت", value=FALLBACK_SHEET, start=0, end=50)
        self.load_button = pn.widgets.Button(name="بارگذاری داده", button_type="primary")
        self.status = pn.pane.Markdown("**منتظر انتخاب فایل...**")
        self.load_button.on_click(self._load_file)

        def _mark_ready(*_):
            self.status.object = "فایل آماده است"

        self.file_input.param.watch(_mark_ready, "value")
        self.data: Optional[DashboardData] = None

        self.tabs = pn.Tabs(sizing_mode="stretch_both")
        self.layout = pn.Column(
            pn.Row(self.file_input, self.sheet_select, self.load_button),
            self.status,
            self.tabs,
        )

    def _load_file(self, *_):
        if self.file_input.value:
            try:
                bio = io.BytesIO(self.file_input.value)
                df = pd.read_excel(bio, sheet_name=self.sheet_select.value)
                self.data = DashboardData(df=df)
                self.status.object = "✅ داده با موفقیت بارگذاری شد"
                self._build_dashboard()
            except Exception as exc:
                self.status.object = f"❌ خطا در خواندن فایل: {exc}"
        elif FALLBACK_EXCEL.exists():
            try:
                df = pd.read_excel(FALLBACK_EXCEL, sheet_name=self.sheet_select.value)
                self.data = DashboardData(df=df, path=FALLBACK_EXCEL)
                self.status.object = f"✅ داده از {FALLBACK_EXCEL} بارگذاری شد"
                self._build_dashboard()
            except Exception as exc:
                self.status.object = f"❌ خطا در خواندن فایل پیش‌فرض: {exc}"
        else:
            self.status.object = "⚠️ لطفاً فایل اکسل را بارگذاری کنید"

    def _build_dashboard(self):
        if not self.data:
            return
        df = self.data.df.copy()
        tabs_content = []
        tabs_content.append(("شاخص‌ها", pn.Column(kpi_cards(df))))

        wordcloud_tab = pn.Tabs(*[(col, card) for col, card in zip(COLUMNS_FOR_WORDCLOUD, make_wordcloud_cards(df))])
        tabs_content.append(("ابرکلمات", wordcloud_tab))

        color_tab = pn.Tabs(("تحلیل رنگ", pn.GridBox(*color_charts(df), ncols=2)))
        tabs_content.append(("رنگ و احساس", pn.Column(wordcloud_tab[0], pn.GridBox(*emotion_pies(df), ncols=2), color_tab["تحلیل رنگ"])))

        text_tab = pn.Tabs(("متنی", pn.GridBox(*build_text_bars(df), ncols=2)))
        tabs_content.append(("تحلیل متنی", text_tab))

        tabs_content.append(("مدت و رسانه", pn.GridBox(*duration_charts(df), *product_charts(df), ncols=2)))

        tabs_content.append(("ماهانه", pn.GridBox(*monthly_area_cards(df), ncols=1)))

        self.tabs.clear()
        for title, content in tabs_content:
            self.tabs.append((title, content))


def create_app():
    dash = Dashboard()
    return dash.layout


dashboard = create_app()

if __name__ == "__main__":
    pn.serve({"مدیا آرشیو": dashboard}, show=True, port=5006, autoreload=True)
