"""
Painel analítico interativo com visões coordenadas
Dados: Gapminder (1952-2007) — expectativa de vida, PIB per capita e população.

Mantra de Shneiderman:
  Overview first  -> mapa mundial + KPIs com todos os países
  Zoom and filter -> controle de ano, indicador, filtro de continentes, seleção (brushing)
  Details-on-demand -> tooltips enriquecidos + tabela filtrada pela seleção

Brushing & Linking:
  Uma seleção feita no mapa (clique), no ranking (clique) ou no gráfico de dispersão
  (caixa/laço) vira a seleção global. Todas as visões a destacam e a tabela é filtrada.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# Configuração geral
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Panorama mundial: saúde, renda e população",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#262730"        # texto principal
MUTED = "#6b7280"      # texto secundário
GRID = "#e5e7eb"       # linhas muito discretas
FONT = dict(family="Inter, Source Sans Pro, Arial, sans-serif", size=12, color=INK)

# Paleta de Okabe & Ito (segura para daltonismo) — uma cor por continente
CONT_COLORS = {
    "África": "#E69F00",
    "Américas": "#0072B2",
    "Ásia": "#009E73",
    "Europa": "#CC79A7",
    "Oceania": "#56B4E9",
}


def darken(hex_color: str, f: float = 0.6) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(int(r * f), int(g * f), int(b * f))


CONT_DARK = {k: darken(v) for k, v in CONT_COLORS.items()}  # rótulos e contornos (contraste >= 4,5:1)
CONT_PT = {
    "Africa": "África",
    "Americas": "Américas",
    "Asia": "Ásia",
    "Europe": "Europa",
    "Oceania": "Oceania",
}
SEQ_SCALE = "Viridis"  # sequencial perceptualmente uniforme e seguro para daltonismo

INDICATORS = {
    "lifeExp": dict(label="Expectativa de vida", unit="anos", log=False),
    "gdpPercap": dict(label="PIB per capita", unit="US$", log=True),
    "pop": dict(label="População", unit="hab.", log=True),
}
CHARTS = ("map", "bars", "scatter")  # visões que geram seleção (brushing)
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["autoScale2d", "resetScale2d"]}


# --------------------------------------------------------------------------------------
# Formatação (pt-BR)
# --------------------------------------------------------------------------------------
def fmt_num(x: float, dec: int = 0) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def n_sel(n: int) -> str:
    return "1 país selecionado" if n == 1 else f"{n} países selecionados"


def fmt_pop(x: float) -> str:
    if x >= 1e9:
        return f"{fmt_num(x / 1e9, 2)} bi"
    if x >= 1e6:
        return f"{fmt_num(x / 1e6, 1)} mi"
    if x >= 1e3:
        return f"{fmt_num(x / 1e3, 0)} mil"
    return fmt_num(x)


def fmt_value(key: str, x: float) -> str:
    if key == "lifeExp":
        return f"{fmt_num(x, 1)} anos"
    if key == "gdpPercap":
        return f"US$ {fmt_num(x)}"
    return fmt_pop(x)


def fmt_short(key: str, x: float) -> str:
    if key == "lifeExp":
        return fmt_num(x, 1)
    if key == "gdpPercap":
        return fmt_num(x)
    return fmt_pop(x)


def tick_pop(v: float) -> str:
    for div, unit in ((1e9, "bi"), (1e6, "mi"), (1e3, "mil")):
        if v >= div:
            n = v / div
            return f"{fmt_num(n, 0 if abs(n - round(n)) < 1e-9 else 1)} {unit}"
    return fmt_num(v)


def log_ticks(key: str, lo: float, hi: float):
    """Marcas legíveis para eixos logarítmicos (valores reais, posições em log10)."""
    base = {"pop": (1e5, 1e9), "gdpPercap": (100, 1e5)}[key]
    fine, coarse = [], []
    k = base[0]
    while k <= base[1] * 1.001:
        for m in (1, 2, 3, 5):
            fine.append(k * m)
        coarse.extend([k, 3 * k])
        k *= 10
    inside = lambda vals: [v for v in vals if lo * 0.999 <= v <= hi * 1.001]
    vals = inside(fine)
    if len(vals) > 7:
        vals = inside(coarse)
    labels = [tick_pop(v) if key == "pop" else fmt_short(key, v) for v in vals]
    return [math.log10(v) for v in vals], labels


# --------------------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = px.data.gapminder().copy()
    df["continente"] = df["continent"].map(CONT_PT)
    df = df.rename(columns={"country": "pais"})
    return df[["pais", "continente", "year", "lifeExp", "pop", "gdpPercap", "iso_alpha"]]


DATA = load_data()
YEARS = sorted(DATA["year"].unique().tolist())
ISO2COUNTRY = dict(zip(DATA["iso_alpha"], DATA["pais"]))
FIRST_YEAR = YEARS[0]


def wmean(df: pd.DataFrame, key: str) -> float:
    """Média ponderada pela população (PIB per capita = PIB total / população total)."""
    if df.empty:
        return float("nan")
    return float(np.average(df[key], weights=df["pop"]))


# --------------------------------------------------------------------------------------
# Estado da seleção global (brushing & linking)
# --------------------------------------------------------------------------------------
if "sel" not in st.session_state:
    st.session_state.sel = []  # países selecionados (fonte única da verdade)
if "ver" not in st.session_state:
    st.session_state.ver = {name: 0 for name in CHARTS}


def chart_key(name: str) -> str:
    return f"{name}_{st.session_state.ver[name]}"


def extract_countries(points: list[dict]) -> list[str]:
    out: list[str] = []
    for p in points:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)) and cd:
            out.append(str(cd[0]))
        elif isinstance(cd, str):
            out.append(cd)
        elif p.get("location") in ISO2COUNTRY:
            out.append(ISO2COUNTRY[p["location"]])
        elif isinstance(p.get("y"), str):
            out.append(p["y"])
    return list(dict.fromkeys(out))


def on_chart_select(name: str) -> None:
    """Seleção em uma visão -> seleção global; as outras visões reiniciam o próprio estado."""
    state = st.session_state.get(chart_key(name))
    pts = []
    if state:
        pts = (state.get("selection") or {}).get("points", []) if hasattr(state, "get") else []
    st.session_state.sel = extract_countries(pts)
    for other in CHARTS:
        if other != name:
            st.session_state.ver[other] += 1


def clear_selection() -> None:
    st.session_state.sel = []
    for name in CHARTS:
        st.session_state.ver[name] += 1


# --------------------------------------------------------------------------------------
# Barra lateral: zoom & filtro
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filtros")
    year = st.select_slider("Ano", options=YEARS, value=YEARS[-1])
    ind_key = st.radio(
        "Indicador no mapa e no ranking",
        options=list(INDICATORS),
        format_func=lambda k: INDICATORS[k]["label"],
    )
    conts = st.multiselect(
        "Continentes",
        options=list(CONT_COLORS),
        default=list(CONT_COLORS),
    )
    st.button("Limpar seleção", on_click=clear_selection, width="stretch")
    st.markdown("---")
    st.markdown("**Cores dos continentes**")
    st.markdown(
        " ".join(
            f"<span style='white-space:nowrap;margin-right:10px'>"
            f"<span style='display:inline-block;width:11px;height:11px;background:{c};"
            f"border-radius:2px;margin-right:4px;vertical-align:-1px'></span>{n}</span>"
            for n, c in CONT_COLORS.items()
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Como usar: clique em um país no mapa ou no ranking, ou arraste uma caixa/laço no "
        "gráfico de dispersão. Todas as visões destacam a seleção e a tabela é filtrada."
    )

IND = INDICATORS[ind_key]

if not conts:
    st.warning("Selecione ao menos um continente na barra lateral.")
    st.stop()

# --------------------------------------------------------------------------------------
# Subconjuntos
# --------------------------------------------------------------------------------------
universe_all = DATA[DATA["continente"].isin(conts)]
cur = universe_all[universe_all["year"] == year].copy()
base = universe_all[universe_all["year"] == FIRST_YEAR]

# valores derivados para tooltips e tabela
cur["rank"] = cur[ind_key].rank(ascending=False, method="min").astype(int)
first_vals = DATA[DATA["year"] == FIRST_YEAR].set_index("pais")[ind_key]
cur["chg"] = (cur[ind_key] / cur["pais"].map(first_vals) - 1) * 100

# seleção efetiva: só países que ainda estão no filtro de continentes
sel = [c for c in st.session_state.sel if c in set(cur["pais"])]
has_sel = bool(sel)
scope = cur[cur["pais"].isin(sel)] if has_sel else cur
scope_base = base[base["pais"].isin(sel)] if has_sel else base


def customdata(df: pd.DataFrame) -> np.ndarray:
    """Colunas: país, continente, vida, PIB, população, ranking, variação desde 1952."""
    n = len(cur)
    return np.column_stack(
        [
            df["pais"],
            df["continente"],
            df["lifeExp"].map(lambda v: fmt_num(v, 1)),
            df["gdpPercap"].map(lambda v: fmt_num(v)),
            df["pop"].map(fmt_pop),
            df["rank"].map(lambda r: f"{r}º de {n}"),
            df["chg"].map(lambda v: ("+" if v >= 0 else "") + fmt_num(v, 0) + "%"),
        ]
    )


TOOLTIP = (
    "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
    "Expectativa de vida: %{customdata[2]} anos<br>"
    "PIB per capita: US$ %{customdata[3]}<br>"
    "População: %{customdata[4]}<br>"
    f"Ranking ({IND['label'].lower()}): %{{customdata[5]}}<br>"
    f"Variação desde {FIRST_YEAR}: %{{customdata[6]}}"
    "<extra></extra>"
)


def base_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font=dict(color=INK, size=12), bordercolor=GRID),
        **kw,
    )
    fig.update_xaxes(automargin=True, title_standoff=8)
    fig.update_yaxes(automargin=True, title_standoff=8)
    return fig


def sel_idx(df: pd.DataFrame):
    """Índices a destacar (ou None, quando não há seleção -> nada é esmaecido)."""
    if not has_sel:
        return None
    return [i for i, c in enumerate(df["pais"]) if c in sel]


DIM = dict(marker=dict(opacity=0.22))
FULL = dict(marker=dict(opacity=1))

# --------------------------------------------------------------------------------------
# Cabeçalho + KPIs (visão geral primeiro)
# --------------------------------------------------------------------------------------
st.markdown("## Panorama mundial: saúde, renda e população")
n_total = len(cur)
st.caption(
    f"{year} · "
    + (f"{n_sel(len(sel))} (de {n_total})" if has_sel else f"{n_total} países no filtro")
    + " · fonte: Gapminder"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Países", fmt_num(len(scope)))
k2.metric("População", fmt_pop(scope["pop"].sum()))
le, le0 = wmean(scope, "lifeExp"), wmean(scope_base, "lifeExp")
k3.metric(
    "Expectativa de vida (média ponderada)",
    f"{fmt_num(le, 1)} anos",
    delta=None if year == FIRST_YEAR else f"{'+' if le >= le0 else ''}{fmt_num(le - le0, 1)} anos desde {FIRST_YEAR}",
    delta_color="off",
)
gd, gd0 = wmean(scope, "gdpPercap"), wmean(scope_base, "gdpPercap")
k4.metric(
    "PIB per capita (média ponderada)",
    f"US$ {fmt_num(gd)}",
    delta=None if year == FIRST_YEAR else f"{'+' if gd >= gd0 else ''}{fmt_num((gd / gd0 - 1) * 100, 0)}% desde {FIRST_YEAR}",
    delta_color="off",
)

# --------------------------------------------------------------------------------------
# Linha 1: mapa (visão geral espacial) + ranking (distribuição categórica)
# --------------------------------------------------------------------------------------
col_map, col_bar = st.columns([3, 2], gap="large")

with col_map:
    st.markdown(f"**Mapa · {IND['label']}** ({IND['unit']})")
    z = np.log10(cur[ind_key]) if IND["log"] else cur[ind_key]
    zmin, zmax = float(z.min()), float(z.max())
    cb = dict(thickness=10, len=0.7, outlinewidth=0, tickfont=dict(size=11, color=MUTED), x=0.02, y=0.3, xanchor="left")
    if IND["log"]:
        lo_v, hi_v = 10 ** zmin, 10 ** zmax
        tv, tt = log_ticks(ind_key, lo_v, hi_v)
        cb.update(tickvals=tv, ticktext=tt)
    fig_map = go.Figure(
        go.Choropleth(
            locations=cur["iso_alpha"],
            z=z,
            zmin=zmin,
            zmax=zmax,
            colorscale=SEQ_SCALE,
            reversescale=False,
            marker=dict(line=dict(color="white", width=0.4)),
            colorbar=cb,
            customdata=customdata(cur),
            hovertemplate=TOOLTIP,
            selectedpoints=sel_idx(cur),
            selected=FULL,
            unselected=DIM,
        )
    )
    fig_map.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=False,
        showcountries=False,
        showland=True,
        landcolor="#f3f4f6",
        showocean=False,
        bgcolor="rgba(0,0,0,0)",
        lataxis_range=[-58, 85],
    )
    base_layout(fig_map, 400)
    st.plotly_chart(
        fig_map,
        key=chart_key("map"),
        on_select=lambda: on_chart_select("map"),
        selection_mode="points",
        theme=None,
        config=CONFIG,
        width="stretch",
    )

with col_bar:
    st.markdown(f"**Ranking · {IND['label']}** ({IND['unit']})")
    top = cur.nlargest(12, ind_key)
    extra = cur[cur["pais"].isin(sel) & ~cur["pais"].isin(top["pais"])]
    rk = pd.concat([top, extra]).sort_values(ind_key, ascending=False)
    if len(rk) > 24:
        rk = rk.nlargest(24, ind_key)
    rk = rk.reset_index(drop=True)
    fig_bar = go.Figure(
        go.Bar(
            x=rk[ind_key],
            y=rk["pais"],
            orientation="h",
            marker=dict(color=[CONT_COLORS[c] for c in rk["continente"]]),
            text=[fmt_short(ind_key, v) for v in rk[ind_key]],
            textposition="outside",
            textfont=dict(size=11, color=INK),
            cliponaxis=False,
            customdata=customdata(rk),
            hovertemplate=TOOLTIP,
            selectedpoints=sel_idx(rk),
            selected=FULL,
            unselected=DIM,
        )
    )
    fig_bar.update_yaxes(
        autorange="reversed", showgrid=False, zeroline=False, showline=False, ticks="", tickfont=dict(size=11)
    )
    fig_bar.update_xaxes(visible=False, range=[0, float(rk[ind_key].max()) * 1.22])
    base_layout(fig_bar, 400, bargap=0.28)
    st.plotly_chart(
        fig_bar,
        key=chart_key("bars"),
        on_select=lambda: on_chart_select("bars"),
        selection_mode="points",
        theme=None,
        config=CONFIG,
        width="stretch",
    )

# --------------------------------------------------------------------------------------
# Linha 2: dispersão (brushing por caixa/laço) + evolução temporal
# --------------------------------------------------------------------------------------
col_sc, col_ts = st.columns(2, gap="large")

with col_sc:
    st.markdown("**Renda × saúde** (tamanho = população) · arraste para selecionar")
    fig_sc = go.Figure()
    for cont, color in CONT_COLORS.items():
        color_dark = CONT_DARK[cont]
        sub = cur[cur["continente"] == cont].reset_index(drop=True)
        if sub.empty:
            continue
        fig_sc.add_trace(
            go.Scatter(
                x=sub["gdpPercap"],
                y=sub["lifeExp"],
                mode="markers",
                name=cont,
                marker=dict(
                    color=color,
                    size=sub["pop"],
                    sizemode="area",
                    sizeref=2.0 * 1.4e9 / (46**2),
                    sizemin=4,
                    line=dict(color=color_dark, width=0.8),
                ),
                customdata=customdata(sub),
                hovertemplate=TOOLTIP,
                selectedpoints=sel_idx(sub),
                selected=FULL,
                unselected=dict(marker=dict(opacity=0.18)),
                opacity=0.85,
            )
        )
    tv, tt = log_ticks("gdpPercap", 250, 120000)
    fig_sc.update_xaxes(
        type="log",
        range=[math.log10(230), math.log10(130000)],
        tickvals=[10 ** v for v in tv],
        ticktext=tt,
        title=dict(text="PIB per capita (US$, escala log)", font=dict(size=11, color=MUTED)),
        showgrid=False,
        zeroline=False,
        linecolor=GRID,
    )
    fig_sc.update_yaxes(
        range=[20, 87],
        title=dict(text="Expectativa de vida (anos)", font=dict(size=11, color=MUTED)),
        showgrid=False,
        zeroline=False,
        linecolor=GRID,
    )
    base_layout(fig_sc, 380, dragmode="select")
    st.plotly_chart(
        fig_sc,
        key=chart_key("scatter"),
        on_select=lambda: on_chart_select("scatter"),
        selection_mode=("points", "box", "lasso"),
        theme=None,
        config=CONFIG,
        width="stretch",
    )

with col_ts:
    st.markdown(
        f"**Evolução · {IND['label']}** "
        + (f"({n_sel(len(sel))})" if has_sel else "(média por continente)")
    )
    fig_ts = go.Figure()
    plotted: list[float] = []

    def add_line(name: str, g: pd.DataFrame, col: str, color: str, width: float, opacity: float, label: bool,
                 text_color: str = INK):
        g = g.sort_values("year")
        plotted.extend(g[col].tolist())
        fig_ts.add_trace(
            go.Scatter(
                x=g["year"],
                y=g[col],
                mode="lines",
                line=dict(color=color, width=width),
                opacity=opacity,
                customdata=[[name, fmt_value(ind_key, v)] for v in g[col]],
                hovertemplate="<b>%{customdata[0]}</b> · %{x}<br>%{customdata[1]}<extra></extra>",
            )
        )
        if label:
            last_y = g[col].iloc[-1]
            fig_ts.add_annotation(
                x=g["year"].iloc[-1],
                y=math.log10(last_y) if IND["log"] else last_y,
                text=name,
                showarrow=False,
                xanchor="left",
                xshift=5,
                font=dict(size=11, color=text_color),
            )

    if has_sel:
        hist = universe_all[universe_all["pais"].isin(sel)]
        many = len(sel) > 10
        for pais, g in hist.groupby("pais"):
            add_line(pais, g, ind_key, CONT_COLORS[g["continente"].iloc[0]],
                     1.2 if many else 2.2, 0.55 if many else 1.0, label=len(sel) <= 8,
                     text_color=CONT_DARK[g["continente"].iloc[0]])
        if many:  # destaca o sinal: média da seleção (total, no caso de população)
            rows = [
                dict(year=yr, val=g["pop"].sum() if ind_key == "pop" else wmean(g, ind_key))
                for yr, g in hist.groupby("year")
            ]
            add_line("Seleção (média ponderada)" if ind_key != "pop" else "Seleção (total)",
                     pd.DataFrame(rows), "val", INK, 3.2, 1.0, label=False)
    else:
        rows = []
        for (cont, yr), g in universe_all.groupby(["continente", "year"]):
            val = g["pop"].sum() if ind_key == "pop" else wmean(g, ind_key)
            rows.append(dict(continente=cont, year=yr, val=val))
        agg = pd.DataFrame(rows)
        for cont, g in agg.groupby("continente"):
            add_line(cont, g, "val", CONT_COLORS[cont], 2.6, 1.0, label=True, text_color=CONT_DARK[cont])

    fig_ts.add_vline(x=year, line_dash="dot", line_color=MUTED, line_width=1)
    ts_kw = dict(
        showgrid=False,
        zeroline=False,
        linecolor=GRID,
        title=dict(text=f"{IND['label']} ({IND['unit']})", font=dict(size=11, color=MUTED)),
    )
    if IND["log"]:
        fig_ts.update_yaxes(type="log", **ts_kw)
        tvals, ttxt = log_ticks(ind_key, min(plotted) / 1.6, max(plotted) * 1.6)
        if tvals:
            fig_ts.update_yaxes(tickvals=[10 ** v for v in tvals], ticktext=ttxt)
    else:
        fig_ts.update_yaxes(**ts_kw)
    fig_ts.update_xaxes(range=[FIRST_YEAR - 1, YEARS[-1] + 14], tickvals=YEARS[::2], showgrid=False, linecolor=GRID)
    base_layout(fig_ts, 380)
    st.plotly_chart(fig_ts, key="ts", theme=None, config=CONFIG, width="stretch")

# --------------------------------------------------------------------------------------
# Detalhes sob demanda: tabela filtrada pela seleção
# --------------------------------------------------------------------------------------
st.markdown(
    "**Detalhes · " + (n_sel(len(sel)) if has_sel else "todos os países do filtro") + "**"
)
tbl = scope[["pais", "continente", "lifeExp", "gdpPercap", "pop", "chg"]].sort_values("pop", ascending=False)
st.dataframe(
    tbl,
    hide_index=True,
    width="stretch",
    height=min(320, 38 + 35 * len(tbl)),
    column_config={
        "pais": st.column_config.TextColumn("País"),
        "continente": st.column_config.TextColumn("Continente"),
        "lifeExp": st.column_config.NumberColumn("Expectativa de vida (anos)", format="%.1f"),
        "gdpPercap": st.column_config.NumberColumn("PIB per capita (US$)", format="localized"),
        "pop": st.column_config.NumberColumn("População", format="localized"),
        "chg": st.column_config.NumberColumn(f"{IND['label']}: variação desde {FIRST_YEAR} (%)", format="%+.0f"),
    },
)
