from __future__ import annotations

import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -------------------------------------------------------------------
# FY Quarter helpers
# -------------------------------------------------------------------
def _fy_year_and_quarter(dt, fy_start_month: int = 10) -> tuple[int, int]:
    """Return (FY_year, FY_quarter) for a datetime-like object.

    Default FY is US Gov-style: FY starts Oct 1 (fy_start_month=10):
      Q1 = Oct-Dec, Q2 = Jan-Mar, Q3 = Apr-Jun, Q4 = Jul-Sep.
    """
    if dt is None:
        raise ValueError("dt must not be None")
    ts = pd.to_datetime(dt)
    m = int(ts.month)
    y = int(ts.year)

    # FY year rolls over at fy_start_month
    fy_year = y + (1 if m >= fy_start_month else 0)

    # Compute fiscal month index (1..12) where 1 == fy_start_month
    fiscal_month = ((m - fy_start_month) % 12) + 1
    fy_quarter = int((fiscal_month - 1) // 3) + 1
    return fy_year, fy_quarter


def _quarter_start_for_date(dt, fy_start_month: int = 10) -> pd.Timestamp:
    """Return the fiscal-quarter start date for dt (as a Timestamp)."""
    ts = pd.to_datetime(dt)
    fy_year, fy_q = _fy_year_and_quarter(ts, fy_start_month=fy_start_month)

    # Determine calendar month of FY quarter start
    # FY Q1 starts at fy_start_month, Q2 at fy_start_month+3, ...
    start_month = ((fy_start_month - 1 + (fy_q - 1) * 3) % 12) + 1

    # Determine calendar year of that quarter start
    # If start_month is >= fy_start_month, it's in prior calendar year for that FY
    start_year = fy_year - 1 if start_month >= fy_start_month else fy_year
    return pd.Timestamp(start_year, start_month, 1)


def make_fy_quarter_ticks(
    x_min,
    x_max,
    fy_start_month: int = 10,
    two_digit_fy: bool = True,
    label_fmt: str = "FY{fy} Q{q}",
) -> tuple[list[pd.Timestamp], list[str]]:
    """Generate quarterly tick positions + labels between x_min and x_max (inclusive)."""
    x0 = pd.to_datetime(x_min)
    x1 = pd.to_datetime(x_max)

    if x1 < x0:
        x0, x1 = x1, x0

    cur = _quarter_start_for_date(x0, fy_start_month=fy_start_month)
    # ensure we cover the start if x0 is already at quarter start; OK either way
    ticks = []
    labels = []

    # step in 3-month increments using pandas DateOffset
    step = pd.DateOffset(months=3)

    while cur <= x1:
        fy_year, fy_q = _fy_year_and_quarter(cur, fy_start_month=fy_start_month)
        fy_label = f"{fy_year % 100:02d}" if two_digit_fy else str(fy_year)
        label = label_fmt.format(fy=fy_label, q=fy_q, FY=fy_label, Q=fy_q, year=fy_year)
        ticks.append(cur)
        labels.append(label)
        cur = cur + step

    return ticks, labels


def apply_fy_quarter_xaxis(
    fig: go.Figure,
    x_min=None,
    x_max=None,
    fy_start_month: int = 10,
    two_digit_fy: bool = True,
    label_fmt: str = "FY{fy} Q{q}",
) -> go.Figure:
    """Apply FY-quarter tick labels to the main x-axis using tickvals/ticktext."""
    if x_min is None or x_max is None:
        # attempt to infer from figure data
        xs = []
        for tr in fig.data:
            if hasattr(tr, "x") and tr.x is not None:
                try:
                    xs.extend([pd.to_datetime(v) for v in tr.x if v is not None])
                except Exception:
                    pass
        if xs:
            x_min = min(xs)
            x_max = max(xs)

    if x_min is None or x_max is None:
        return fig  # nothing to do

    tickvals, ticktext = make_fy_quarter_ticks(
        x_min, x_max, fy_start_month=fy_start_month, two_digit_fy=two_digit_fy, label_fmt=label_fmt
    )

    tickvals = [pd.to_datetime(v).to_pydatetime() for v in tickvals]

    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=ticktext,
    )
    return fig


# -----------------------------
# Paths
# -----------------------------
def get_project_root() -> Path:
    """
    Infer the project root as three levels up from this file.

    Assumes layout:

        project/
          ├── data/
          ├── figures/
          └── src/survey_pipeline/analysis_utils.py
    """
    return Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Paths:
    """
    Convenience container for common project paths.
    """

    root: Path
    data: Path
    figures: Path
    raw: Path
    interim: Path
    processed: Path


def get_paths(root: Path | None = None) -> Paths:
    """
    Build a Paths object for this project.

    Parameters
    ----------
    root : Path, optional
        If provided, use this as the project root instead of inferring
        it from the location of this file.
    """
    r = root or get_project_root()
    d = r / "data"
    return Paths(
        root=r,
        data=d,
        figures=r / "figures",
        raw=d / "0_raw",
        interim=d / "1_interim",
        processed=d / "2_processed",
    )


# -----------------------------
# Analysis
# -----------------------------

NS = {"msp": "http://schemas.microsoft.com/project"}


def _parse_bool(x: str | None):
    if x == "1":
        return True
    if x == "0":
        return False
    return pd.NA


def read_mspdi_tasks_and_links(xml_path: str | Path):
    """
    Read MSPDI (MS Project XML) and return:
      - tasks_df: UID, name, start, finish, outline_level, wbs, summary, milestone, predecessors (list)
      - links_df: successor_uid, predecessor_uid, type, lag, lag_format
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tasks_el = root.find("msp:Tasks", NS)
    if tasks_el is None:
        raise ValueError("No <Tasks> section found. Is this an MSPDI XML export?")

    tasks_rows = []
    links_rows = []

    for t in tasks_el.findall("msp:Task", NS):

        def get(tag: str, t=t) -> str | None:
            return t.findtext(f"msp:{tag}", default=None, namespaces=NS)

        uid_txt = get("UID")
        if uid_txt is None:
            continue
        uid = int(uid_txt) if uid_txt.isdigit() else uid_txt

        # Some exports include "null" placeholder tasks
        if _parse_bool(get("IsNull")) is True:
            continue

        tasks_rows.append(
            {
                "uid": uid,
                "name": get("Name"),
                "start": pd.to_datetime(get("Start"), errors="coerce"),
                "finish": pd.to_datetime(get("Finish"), errors="coerce"),
                "outline_level": pd.to_numeric(get("OutlineLevel"), errors="coerce").astype(int),
                "wbs": get("WBS"),
                "summary": _parse_bool(get("Summary")),
                "milestone": _parse_bool(get("Milestone")),
            }
        )

        # Predecessor edges live under each Task
        for pl in t.findall("msp:PredecessorLink", NS):
            pred_txt = pl.findtext("msp:PredecessorUID", default=None, namespaces=NS)
            pred_uid = int(pred_txt) if (pred_txt or "").isdigit() else pred_txt

            links_rows.append(
                {
                    "successor_uid": uid,
                    "predecessor_uid": pred_uid,
                    "type": pd.to_numeric(
                        pl.findtext("msp:Type", default=None, namespaces=NS), errors="coerce"
                    ).astype(int),
                    "lag": pd.to_numeric(
                        pl.findtext("msp:LinkLag", default=None, namespaces=NS), errors="coerce"
                    ).astype(int),
                    "lag_format": pd.to_numeric(
                        pl.findtext("msp:LagFormat", default=None, namespaces=NS), errors="coerce"
                    ).astype(int),
                }
            )

    tasks_df = pd.DataFrame(tasks_rows)
    links_df = pd.DataFrame(links_rows)

    # Aggregate predecessor info onto tasks_df (nice for hover / debugging)
    if not links_df.empty:
        cols = ["predecessor_uid", "type", "lag", "lag_format"]
        preds = (
            links_df.sort_values(["successor_uid", "predecessor_uid"])
            .groupby("successor_uid", sort=False)[cols]
            .apply(lambda g: g.to_dict("records"))
            .rename("predecessors")
            .reset_index()
        )
        tasks_df = tasks_df.merge(preds, left_on="uid", right_on="successor_uid", how="left").drop(
            columns=["successor_uid"]
        )
    else:
        tasks_df["predecessors"] = pd.NA

    return tasks_df, links_df


def _wrap_html(s: object, width: int) -> str:
    s = "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
    if width <= 0:
        return s
    return "<br>".join(textwrap.wrap(s, width=width, break_long_words=False))


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    # Accept already-rgba/rgb strings
    if not isinstance(hex_color, str):
        return str(hex_color)
    c = hex_color.strip()
    if c.lower().startswith("rgba(") or c.lower().startswith("rgb("):
        # Don't try to rewrite; just return as-is
        return c
    if not c.startswith("#") or len(c) not in (4, 7):
        return c
    if len(c) == 4:
        r = int(c[1] * 2, 16)
        g = int(c[2] * 2, 16)
        b = int(c[3] * 2, 16)
    else:
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
    a = 1.0 if alpha is None or (isinstance(alpha, float) and pd.isna(alpha)) else float(alpha)
    a = min(1.0, max(0.0, a))
    return f"rgba({r},{g},{b},{a})"


RoadmapDD = str | Path | pd.DataFrame


def load_roadmap_dd(
    dd: RoadmapDD,
    *,
    dd_uid_col: str = "uid",
    # IMPORTANT: preserves literal strings like "NA" instead of converting to NaN
    keep_default_na: bool = False,
    # pass-through if you want to override later; usually leave alone
    csv_kwargs: dict | None = None,
    # column names (so apply-defaults can add/fill them on the DD too)
    y_col: str = "y_id",
    y_label: str = "y_label",
    include_col: str = "include",
    item_type_col: str = "item_type",
    color_col: str = "color",
    shape_col: str = "shape",
    fill_alpha_col: str = "fill_alpha",
    bar_pattern_col: str = "bar_pattern",
    bar_line_color_col: str = "bar_line_color",
    bar_line_width_col: str = "bar_line_width",
    marker_size_col: str = "marker_size",
    marker_line_color_col: str = "marker_line_color",
    marker_line_width_col: str = "marker_line_width",
    dd_milestone_col: str = "milestone",
) -> pd.DataFrame:
    """
    Load the roadmap data dictionary (CSV path or DataFrame) and apply defaults.

    Notes
    -----
    - Uses keep_default_na=False by default so literal strings like "NA" aren't
      auto-cast to NaN by pandas.
    - Applies default columns/values in an idempotent way.
    """
    if isinstance(dd, (str, Path)):
        kwargs = dict(keep_default_na=keep_default_na)
        if csv_kwargs:
            kwargs.update(csv_kwargs)
        dd_df = pd.read_csv(dd, **kwargs)
    else:
        dd_df = dd.copy()

    if dd_uid_col not in dd_df.columns:
        raise ValueError(f"dd missing required column: {dd_uid_col}")

    dd_df = apply_roadmap_defaults(
        dd_df,
        kind="dd",
        dd_uid_col=dd_uid_col,
        y_col=y_col,
        y_label=y_label,
        include_col=include_col,
        item_type_col=item_type_col,
        color_col=color_col,
        shape_col=shape_col,
        fill_alpha_col=fill_alpha_col,
        bar_pattern_col=bar_pattern_col,
        bar_line_color_col=bar_line_color_col,
        bar_line_width_col=bar_line_width_col,
        marker_size_col=marker_size_col,
        marker_line_color_col=marker_line_color_col,
        marker_line_width_col=marker_line_width_col,
        dd_milestone_col=dd_milestone_col,
        # DD generally doesn't need the debug helper col
        create_is_milestone_col=False,
    )

    return dd_df


def apply_roadmap_defaults(
    df: pd.DataFrame,
    *,
    kind: Literal["dd", "merged"],
    # df column names (only used when present)
    df_uid_col: str = "uid",
    df_name_col: str = "name",
    df_milestone_col: str = "milestone",
    # dd column names
    dd_uid_col: str = "uid",
    dd_milestone_col: str = "milestone",
    # roadmap cols
    y_col: str = "y_id",
    y_label: str = "y_label",
    include_col: str = "include",
    item_type_col: str = "item_type",
    color_col: str = "color",
    shape_col: str = "shape",
    fill_alpha_col: str = "fill_alpha",
    bar_pattern_col: str = "bar_pattern",
    bar_line_color_col: str = "bar_line_color",
    bar_line_width_col: str = "bar_line_width",
    marker_size_col: str = "marker_size",
    marker_line_color_col: str = "marker_line_color",
    marker_line_width_col: str = "marker_line_width",
    # behavior
    create_is_milestone_col: bool = True,
) -> pd.DataFrame:
    """
    Apply default values / NA fillers to either:
      - the DD itself (kind="dd"), or
      - the merged MS Project + DD dataframe (kind="merged").

    Design goals:
      - idempotent (safe to call multiple times)
      - only fills truly missing values (NaN/NA and blank strings)
      - avoids clobbering literal strings like "NA"
    """
    out = df.copy()

    def _is_missing(series: pd.Series) -> pd.Series:
        """Treat NaN/NA as missing; treat blank strings as missing too."""
        if series.dtype == "object":
            s = series.astype("string")
            return series.isna() | (s.str.strip() == "")
        return series.isna()

    # ----------------------------
    # include flag
    # ----------------------------
    if include_col not in out.columns:
        out[include_col] = True
    else:
        out.loc[out[include_col].isna(), include_col] = True
    out[include_col] = out[include_col].astype(bool)

    # ----------------------------
    # milestone boolean (for default styling + item_type fallback)
    # ----------------------------
    if dd_milestone_col in out.columns:
        is_ms = pd.Series(out[dd_milestone_col])
    elif df_milestone_col in out.columns:
        is_ms = pd.Series(out[df_milestone_col])
    else:
        is_ms = pd.Series(False, index=out.index)

    is_ms = is_ms.fillna(False).astype(bool)

    if create_is_milestone_col and kind == "merged":
        out["_is_milestone"] = is_ms

    # ----------------------------
    # item_type fallback
    # ----------------------------
    default_item_type = is_ms.map(lambda m: "milestone" if m else "task")

    if item_type_col not in out.columns:
        out[item_type_col] = default_item_type
    else:
        missing = _is_missing(out[item_type_col])
        out.loc[missing, item_type_col] = default_item_type.loc[missing]

    # ----------------------------
    # y columns fallback
    # ----------------------------
    if y_col not in out.columns:
        # merged: prefer df uid; dd: prefer dd uid; otherwise index
        if df_uid_col in out.columns:
            out[y_col] = out[df_uid_col].astype(str)
        elif dd_uid_col in out.columns:
            out[y_col] = out[dd_uid_col].astype(str)
        else:
            out[y_col] = out.index.astype(str)
    out[y_col] = out[y_col].astype(str)

    if y_label not in out.columns:
        # merged: fall back to df name if present; dd: fall back to y_id
        if df_name_col in out.columns:
            out[y_label] = out[df_name_col].astype(str)
        else:
            out[y_label] = out[y_col].astype(str)
    else:
        # only replace *true* NaNs; don't overwrite literal "NA"
        out.loc[out[y_label].isna(), y_label] = ""
        out[y_label] = out[y_label].astype(str)

    # ----------------------------
    # style defaults
    # ----------------------------
    default_color = is_ms.map(lambda m: "#FECB52" if m else "#636EFA")
    if color_col not in out.columns:
        out[color_col] = default_color
    else:
        missing = _is_missing(out[color_col])
        out.loc[missing, color_col] = default_color.loc[missing]

    default_shape = is_ms.map(lambda m: "star" if m else "circle")
    if shape_col not in out.columns:
        out[shape_col] = default_shape
    else:
        missing = _is_missing(out[shape_col])
        out.loc[missing, shape_col] = default_shape.loc[missing]

    if fill_alpha_col not in out.columns:
        out[fill_alpha_col] = 1.0
    out[fill_alpha_col] = pd.to_numeric(out[fill_alpha_col], errors="coerce").fillna(1.0)

    if bar_pattern_col not in out.columns:
        out[bar_pattern_col] = ""
    out.loc[out[bar_pattern_col].isna(), bar_pattern_col] = ""
    out[bar_pattern_col] = out[bar_pattern_col].astype(str)

    if bar_line_color_col not in out.columns:
        out[bar_line_color_col] = "#333333"
    else:
        out.loc[_is_missing(out[bar_line_color_col]), bar_line_color_col] = "#333333"

    if bar_line_width_col not in out.columns:
        out[bar_line_width_col] = 1
    out[bar_line_width_col] = pd.to_numeric(out[bar_line_width_col], errors="coerce").fillna(1)

    if marker_size_col not in out.columns:
        out[marker_size_col] = 14
    out[marker_size_col] = pd.to_numeric(out[marker_size_col], errors="coerce").fillna(14)

    if marker_line_color_col not in out.columns:
        out[marker_line_color_col] = "black"
    else:
        out.loc[_is_missing(out[marker_line_color_col]), marker_line_color_col] = "black"

    if marker_line_width_col not in out.columns:
        out[marker_line_width_col] = 1
    out[marker_line_width_col] = pd.to_numeric(out[marker_line_width_col], errors="coerce").fillna(
        1
    )

    return out


def merge_msproject_with_roadmap_dd(
    df: pd.DataFrame,
    dd: RoadmapDD,
    *,
    # Source DF column names
    df_uid_col: str = "uid",
    df_name_col: str = "name",
    df_start_col: str = "start",
    df_finish_col: str = "finish",
    df_milestone_col: str = "milestone",
    # DD / merge column names
    dd_uid_col: str = "uid",
    # roadmap cols
    y_col: str = "y_id",
    y_label: str = "y_label",
    include_col: str = "include",
    item_type_col: str = "item_type",
    color_col: str = "color",
    shape_col: str = "shape",
    fill_alpha_col: str = "fill_alpha",
    bar_pattern_col: str = "bar_pattern",
    bar_line_color_col: str = "bar_line_color",
    bar_line_width_col: str = "bar_line_width",
    marker_size_col: str = "marker_size",
    marker_line_color_col: str = "marker_line_color",
    marker_line_width_col: str = "marker_line_width",
    dd_milestone_col: str = "milestone",
    # CSV read behavior (preserve literal "NA")
    keep_default_na: bool = False,
    csv_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge an MS Project-export dataframe with the roadmap data dictionary.

    Returns
    -------
    dfm : pd.DataFrame
        merged dataframe, with defaults applied
    dd_df : pd.DataFrame
        loaded data dictionary (with defaults applied)
    """
    # validate minimum columns in df
    for col in (df_uid_col, df_start_col, df_finish_col, df_name_col):
        if col not in df.columns:
            raise ValueError(f"df missing required column: {col}")

    dd_df = load_roadmap_dd(
        dd,
        dd_uid_col=dd_uid_col,
        keep_default_na=keep_default_na,
        csv_kwargs=csv_kwargs,
        y_col=y_col,
        y_label=y_label,
        include_col=include_col,
        item_type_col=item_type_col,
        color_col=color_col,
        shape_col=shape_col,
        fill_alpha_col=fill_alpha_col,
        bar_pattern_col=bar_pattern_col,
        bar_line_color_col=bar_line_color_col,
        bar_line_width_col=bar_line_width_col,
        marker_size_col=marker_size_col,
        marker_line_color_col=marker_line_color_col,
        marker_line_width_col=marker_line_width_col,
        dd_milestone_col=dd_milestone_col,
    )

    # merge
    dfm = df.merge(
        dd_df,
        how="left",
        left_on=df_uid_col,
        right_on=dd_uid_col,
        suffixes=("", "_dd"),
    )

    # apply defaults on merged (again) — fills missing styling/item_type/etc from merge result
    dfm = apply_roadmap_defaults(
        dfm,
        kind="merged",
        df_uid_col=df_uid_col,
        df_name_col=df_name_col,
        df_milestone_col=df_milestone_col,
        dd_uid_col=dd_uid_col,
        dd_milestone_col=dd_milestone_col,
        y_col=y_col,
        y_label=y_label,
        include_col=include_col,
        item_type_col=item_type_col,
        color_col=color_col,
        shape_col=shape_col,
        fill_alpha_col=fill_alpha_col,
        bar_pattern_col=bar_pattern_col,
        bar_line_color_col=bar_line_color_col,
        bar_line_width_col=bar_line_width_col,
        marker_size_col=marker_size_col,
        marker_line_color_col=marker_line_color_col,
        marker_line_width_col=marker_line_width_col,
        create_is_milestone_col=True,
    )

    return dfm, dd_df


def build_roadmap_timeline_from_dd(
    df: pd.DataFrame,
    dd: str | Path | pd.DataFrame,
    *,
    # ----------------------------
    # Source DF column names
    # ----------------------------
    df_uid_col: str = "uid",
    df_name_col: str = "name",
    df_start_col: str = "start",
    df_finish_col: str = "finish",
    df_milestone_col: str = "milestone",
    # ----------------------------
    # Data Dictionary column names (defaults match v4)
    # ----------------------------
    dd_uid_col: str = "uid",
    y_col: str = "y_id",
    y_label: str = "y_label",
    milestone_text_col: str = "y_label_short",
    include_col: str = "include",
    item_type_col: str = "item_type",
    color_col: str = "color",
    shape_col: str = "shape",
    fill_alpha_col: str = "fill_alpha",
    bar_pattern_col: str = "bar_pattern",
    bar_line_color_col: str = "bar_line_color",
    bar_line_width_col: str = "bar_line_width",
    marker_size_col: str = "marker_size",
    marker_line_color_col: str = "marker_line_color",
    marker_line_width_col: str = "marker_line_width",
    dd_milestone_col: str = "milestone",
    # ----------------------------
    # Labeling / styling behavior
    # ----------------------------
    bar_text_col: str | None = None,
    bar_text_short_col: str | None = "y_label_short",
    # -------------------------
    # X-axis FY quarter formatting
    # -------------------------
    xaxis_fy_quarters: bool = False,
    fy_start_month: int = 10,
    fy_two_digit: bool = True,
    fy_label_fmt: str = "FY{fy} Q{q}",
    bar_text_wrap_width: int = 24,
    show_milestone_text: bool = True,
    milestone_text_position: str = "top center",
    show_milestone_legend: bool = False,
    # ----------------------------
    # Layout
    # ----------------------------
    show_rangeslider: bool = True,
    height_per_row: int = 18,
    autorange_reversed: bool = True,
) -> tuple[go.Figure, pd.DataFrame]:
    """
    Build a Plotly roadmap timeline using a "data dictionary" (style CSV) to drive visuals.

    Key idea:
      - `df` provides schedule truth (start/finish, task names, etc.)
      - `dd` provides presentation (lanes, labels, colors, symbols, etc.)
      - This function merges them and renders:
          * Bars for item_type == "task"
          * Markers (and optional text) for item_type == "milestone"

    All column names are configurable via function inputs.
    Defaults are aligned to the DoD demo v4 data dictionary format.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form task table. Must contain at least: df_uid_col, df_start_col, df_finish_col, df_name_col.
    dd : Union[str, Path, pd.DataFrame]
        Data dictionary (CSV path or DataFrame). Must contain dd_uid_col and (typically) y_col / y_label.
    y_col / y_label : str
        Category ID and visible y-axis label (lane label).
    milestone_text_col : str
        Short label to show next to milestone markers.
    include_col : str
        If present, rows with include == False are dropped from the plot.
    item_type_col : str
        If present, drives task vs milestone. If missing, derived from milestone boolean.
    """

    dfm, dd_df = merge_msproject_with_roadmap_dd(
        df,
        dd,
        df_uid_col=df_uid_col,
        df_name_col=df_name_col,
        df_start_col=df_start_col,
        df_finish_col=df_finish_col,
        df_milestone_col=df_milestone_col,
        dd_uid_col=dd_uid_col,
        y_col=y_col,
        y_label=y_label,
        include_col=include_col,
        item_type_col=item_type_col,
        color_col=color_col,
        shape_col=shape_col,
        fill_alpha_col=fill_alpha_col,
        bar_pattern_col=bar_pattern_col,
        bar_line_color_col=bar_line_color_col,
        bar_line_width_col=bar_line_width_col,
        marker_size_col=marker_size_col,
        marker_line_color_col=marker_line_color_col,
        marker_line_width_col=marker_line_width_col,
        dd_milestone_col=dd_milestone_col,
        keep_default_na=False,
    )

    # --- choose bar label column (defaults to df task name) ---
    if bar_text_col is None:
        bar_text_col = df_name_col
    if bar_text_col not in dfm.columns:
        # don't crash; just use name
        bar_text_col = df_name_col

    # --- filter includes ---
    dfm_plot = dfm[dfm[include_col]].copy()

    # --- split tasks vs milestones ---
    df_tasks = dfm_plot[
        (dfm_plot[item_type_col].astype(str).str.lower() == "task")
        & dfm_plot[df_start_col].notna()
        & dfm_plot[df_finish_col].notna()
    ].copy()

    df_ms = dfm_plot[
        (dfm_plot[item_type_col].astype(str).str.lower() == "milestone")
        & dfm_plot[df_start_col].notna()
    ].copy()

    # --- bar text (wrapped) ---
    if not df_tasks.empty:
        # Prefer a short label for tasks when provided (e.g., y_label_short in the data dictionary),
        # but fall back to the default bar_text_col (typically df_name_col).
        if bar_text_short_col and (bar_text_short_col in df_tasks.columns):
            _short = df_tasks[bar_text_short_col]
            use_short = _short.notna() & (_short.astype(str).str.strip() != "")
            base = df_tasks[bar_text_col]
            text_series = base.where(~use_short, _short)
        else:
            text_series = df_tasks[bar_text_col]

        df_tasks["_bar_text"] = text_series.map(lambda s: _wrap_html(s, bar_text_wrap_width))
    else:
        df_tasks["_bar_text"] = []

    # --- build bars ---
    fig = px.timeline(
        df_tasks,
        x_start=df_start_col,
        x_end=df_finish_col,
        y=y_col,
        text="_bar_text" if not df_tasks.empty else None,
    )

    # apply per-bar styles (color, alpha, line, pattern)
    if not df_tasks.empty:
        bar_colors = [
            _hex_to_rgba(c, a)
            for c, a in zip(
                df_tasks[color_col].tolist(), df_tasks[fill_alpha_col].tolist(), strict=True
            )
        ]
        fig.update_traces(
            marker=dict(
                color=bar_colors,
                line=dict(
                    color=df_tasks[bar_line_color_col].tolist(),
                    width=df_tasks[bar_line_width_col].tolist(),
                ),
                pattern=dict(shape=df_tasks[bar_pattern_col].tolist()),
            ),
            texttemplate="%{text}",
            textposition="inside",
            insidetextanchor="middle",
            cliponaxis=False,
            selector=dict(type="bar"),
        )

    # --- y ordering + labels from data dictionary (unique lanes) ---
    present_y = set(dfm_plot[y_col].astype(str).unique().tolist())
    if y_col in dd_df.columns:
        if y_label in dd_df.columns:
            y_meta = (
                dd_df[[y_col, y_label]]
                .dropna(subset=[y_col])
                .drop_duplicates(y_col, keep="first")
                .copy()
            )
        else:
            y_meta = (
                dd_df[[y_col]].dropna(subset=[y_col]).drop_duplicates(y_col, keep="first").copy()
            )
            y_meta[y_label] = y_meta[y_col]
        y_meta[y_col] = y_meta[y_col].astype(str)
        y_meta = y_meta[y_meta[y_col].isin(present_y)]
        y_order = y_meta[y_col].tolist()
        y_ticktext = y_meta[y_label].fillna("").astype(str).tolist()
    else:
        # fallback: order from plotted data
        y_order = list(dict.fromkeys(dfm_plot[y_col].astype(str).tolist()))
        y_ticktext = y_order

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=y_order,
        tickmode="array",
        tickvals=y_order,
        ticktext=y_ticktext,
        autorange="reversed" if autorange_reversed else True,
    )

    # --- overlay milestones ---
    if not df_ms.empty:
        # milestone labels: prefer dd column milestone_text_col, fallback to df name
        if milestone_text_col in df_ms.columns:
            ms_text = df_ms[milestone_text_col].fillna("").astype(str).tolist()
        else:
            ms_text = df_ms[df_name_col].astype(str).tolist()

        mode = "markers+text" if show_milestone_text else "markers"

        fig.add_trace(
            go.Scatter(
                x=df_ms[df_start_col],
                y=df_ms[y_col].astype(str),
                mode=mode,
                text=ms_text if show_milestone_text else None,
                textposition=milestone_text_position,
                marker=dict(
                    symbol=df_ms[shape_col].tolist(),
                    color=df_ms[color_col].tolist(),
                    size=df_ms[marker_size_col].astype(float).tolist(),
                    line=dict(
                        color=df_ms[marker_line_color_col].tolist(),
                        width=df_ms[marker_line_width_col].astype(float).tolist(),
                    ),
                ),
                hovertext=df_ms[df_name_col].astype(str).tolist(),
                hovertemplate="%{hovertext}<extra></extra>",
                name="Milestones",
                showlegend=bool(show_milestone_legend),
            )
        )

    # --- layout ---
    fig.update_layout(
        height=max(500, height_per_row * max(1, len(y_order))),
        xaxis_rangeslider_visible=show_rangeslider,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Optional: show FY quarter ticks on the x-axis
    if xaxis_fy_quarters:
        try:
            x_min = pd.to_datetime(dfm_plot[df_start_col]).min()
            x_max = pd.to_datetime(dfm_plot[df_finish_col]).max()
            if pd.isna(x_max):
                x_max = pd.to_datetime(dfm_plot[df_start_col]).max()
            apply_fy_quarter_xaxis(
                fig,
                x_min=x_min,
                x_max=x_max,
                fy_start_month=fy_start_month,
                two_digit_fy=fy_two_digit,
                label_fmt=fy_label_fmt,
            )
        except Exception:
            # Never crash plotting if tick formatting fails
            pass

    return fig, dfm_plot
