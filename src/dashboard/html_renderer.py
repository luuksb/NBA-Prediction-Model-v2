"""src/dashboard/html_renderer.py — Pure HTML-building helpers for the dashboard.

No Streamlit imports; all functions return HTML strings.  These helpers are
imported by ``scripts/run_dashboard.py`` and rendered via
``streamlit.components.v1.html`` or ``st.markdown``.
"""

from __future__ import annotations

import pandas as pd

from src.dashboard.data_loader import logo_url

# ---------------------------------------------------------------------------
# NBA team colors  (primary, secondary, tertiary)
# Source: official NBA brand guidelines; equivalent to nbacolors.com values.
# ---------------------------------------------------------------------------
TEAM_COLORS: dict[str, tuple[str, str, str]] = {
    "ATL": ("#E03A3E", "#C1D32F", "#26282A"),  # Hawks
    "BOS": ("#007A33", "#BA9653", "#000000"),  # Celtics
    "BKN": ("#000000", "#FFFFFF", "#AAAAAA"),  # Nets
    "CHA": ("#1D1160", "#00788C", "#A1A1A4"),  # Hornets
    "CHI": ("#CE1141", "#000000", "#FFFFFF"),  # Bulls
    "CLE": ("#860038", "#FDBB30", "#002D62"),  # Cavaliers
    "DAL": ("#00538C", "#002B5E", "#B8C4CA"),  # Mavericks
    "DEN": ("#0E2240", "#FEC524", "#8B2131"),  # Nuggets
    "DET": ("#C8102E", "#1D428A", "#BEC0C2"),  # Pistons
    "GSW": ("#1D428A", "#FFC72C", "#FFFFFF"),  # Warriors
    "HOU": ("#CE1141", "#000000", "#C4CED4"),  # Rockets
    "IND": ("#002D62", "#FDBB30", "#BEC0C2"),  # Pacers
    "LAC": ("#C8102E", "#1D428A", "#BEC0C2"),  # Clippers
    "LAL": ("#552583", "#FDB927", "#000000"),  # Lakers
    "MEM": ("#5D76A9", "#12173F", "#F5B112"),  # Grizzlies
    "MIA": ("#98002E", "#F9A01B", "#000000"),  # Heat
    "MIL": ("#00471B", "#EEE1C6", "#000000"),  # Bucks
    "MIN": ("#0C2340", "#236192", "#78BE20"),  # Timberwolves
    "NOP": ("#0C2340", "#C8102E", "#85714D"),  # Pelicans
    "NYK": ("#006BB6", "#F58426", "#BEC0C2"),  # Knicks
    "OKC": ("#007AC1", "#EF3B24", "#002D62"),  # Thunder
    "ORL": ("#0077C0", "#C4CED4", "#000000"),  # Magic
    "PHI": ("#006BB6", "#ED174C", "#002B5C"),  # 76ers
    "PHX": ("#1D1160", "#E56020", "#F9AD1B"),  # Suns
    "POR": ("#E03A3E", "#000000", "#FFFFFF"),  # Trail Blazers
    "SAC": ("#5A2D81", "#63727A", "#000000"),  # Kings
    "SAS": ("#C4CED4", "#000000", "#FFFFFF"),  # Spurs
    "TOR": ("#CE1141", "#000000", "#A1A1A4"),  # Raptors
    "UTA": ("#002B5C", "#00471B", "#F9A01B"),  # Jazz
    "WAS": ("#002B5C", "#E31837", "#C4CED4"),  # Wizards
}


def team_node_html(
    node: dict,
    is_champion: bool = False,
    prob_mode: str = "Matchup Win %",
    champ_prob: float | None = None,
    is_finals: bool = False,
) -> str:
    """Render a single team card as HTML (2K26 pill style).

    Args:
        node: Dict with keys ``abbrev``, ``seed``, ``logo_url``, ``cond_win_prob``.
        is_champion: Whether this team is the predicted champion.
        prob_mode: ``"Matchup Win %"`` or ``"Championship %"``.
        champ_prob: Championship probability (used when prob_mode is championship).
        is_finals: Whether this card is shown in the Finals matchup.

    Returns:
        HTML string for the team card.
    """
    abbrev = node["abbrev"]
    seed = node["seed"]
    url = node["logo_url"]
    if prob_mode == "Championship %" and champ_prob is not None:
        prob_pct = f"🏆 {champ_prob:.0%}" if is_champion else f"{champ_prob:.0%}"
    else:
        prob_pct = f"{node['cond_win_prob']:.0%}"
    finalist_cls = " finalist" if (is_champion and prob_mode == "Championship %") else ""
    champ_cls = " champion" if is_champion else ""

    # Primary team color → gradient fading right so dark logo area blends in
    primary = TEAM_COLORS.get(abbrev, ("#1a3a5c", "#0a1929", "#0a1929"))[0]
    pill_bg = (
        f"linear-gradient(90deg, {primary} 0%, {primary}cc 55%, {primary}55 85%, transparent 100%)"
    )
    # Darker solid version of primary for logo background: 40% opacity over dark base
    logo_bg = f"{primary}66"

    logo = (
        f'<img class="team-logo" src="{url}" '
        f"onerror=\"this.style.display='none';this.nextSibling.style.display='flex';\" "
        f'alt="{abbrev}">'
        f'<span class="team-logo-fallback" style="display:none">{abbrev}</span>'
    )
    return (
        f'<div class="team-node{champ_cls}{finalist_cls}">'
        f'<div class="team-pill" style="background:{pill_bg}">'
        f'<div class="team-seed-badge">{seed}</div>'
        f'<span class="team-abbrev">{abbrev}</span>'
        f"</div>"
        f'<div class="team-logo-area" style="background:{logo_bg}">{logo}</div>'
        f'<span class="team-prob">{prob_pct}</span>'
        f"</div>"
    )


def matchup_html(
    matchup: dict,
    champion_abbrev: str | None = None,
    is_finals: bool = False,
    prob_mode: str = "Matchup Win %",
    champ_probs: dict | None = None,
) -> str:
    """Render a matchup (two team cards) as HTML.

    Args:
        matchup: Dict with keys ``high`` and ``low``, each a team-node dict.
        champion_abbrev: Abbreviation of the predicted champion (may be None).
        is_finals: Whether this matchup is the Finals.
        prob_mode: Probability display mode forwarded to :func:`team_node_html`.
        champ_probs: Dict mapping team abbreviation → championship probability.

    Returns:
        HTML string for the matchup.
    """
    hi = matchup["high"]
    lo = matchup["low"]
    hi_champ = is_finals and hi["abbrev"] == champion_abbrev
    lo_champ = is_finals and lo["abbrev"] == champion_abbrev
    hi_cp = champ_probs.get(hi["abbrev"]) if champ_probs else None
    lo_cp = champ_probs.get(lo["abbrev"]) if champ_probs else None
    return (
        f'<div class="matchup">'
        f"{team_node_html(hi, is_champion=hi_champ, prob_mode=prob_mode, champ_prob=hi_cp, is_finals=is_finals)}"
        f"{team_node_html(lo, is_champion=lo_champ, prob_mode=prob_mode, champ_prob=lo_cp, is_finals=is_finals)}"
        f"</div>"
    )


def round_col_html(
    matchups: list[dict],
    label: str,
    champion_abbrev: str | None = None,
    is_finals: bool = False,
    css_class: str = "",
) -> str:
    """Render one round column (label + all matchups).

    Args:
        matchups: List of matchup dicts for this round column.
        label: Round label shown in the column header.
        champion_abbrev: Predicted champion abbreviation (may be None).
        is_finals: Whether this column is the Finals.
        css_class: Extra CSS class(es) to add to the round-col div.

    Returns:
        HTML string for the round column.
    """
    rendered = [
        matchup_html(m, champion_abbrev=champion_abbrev, is_finals=is_finals) for m in matchups
    ]
    # Wrap consecutive pairs in .matchup-pair for bracket-arm connectors.
    inner = ""
    for i in range(0, len(rendered), 2):
        pair = rendered[i : i + 2]
        if len(pair) == 2:
            inner += f'<div class="matchup-pair">{"".join(pair)}</div>'
        else:
            inner += pair[0]
    extra_cls = f" {css_class}" if css_class else ""
    return (
        f'<div class="round-col{extra_cls}">' f'<div class="round-col-body">{inner}</div>' f"</div>"
    )


def render_bracket_html_canvas(
    bracket: dict,
    colors: dict,
    prob_mode: str = "Matchup Win %",
    champ_probs: dict | None = None,
) -> str:
    """Build the full bracket HTML string using an absolutely-positioned canvas.

    All 15 matchup boxes (W R1×4, W R2×2, W CF×1, Finals×1, E CF×1, E R2×2,
    E R1×4) are placed at explicit (x, y) pixel coordinates inside a single
    positioned container.

    Args:
        bracket: Bracket structure produced by
            :func:`src.dashboard.bracket_builder.build_bracket_structure`.
        colors: UI colors dict from dashboard config (keys: ``background``,
            ``champion_border``).
        prob_mode: Probability display mode (``"Matchup Win %"`` or
            ``"Championship %"``).
        champ_probs: Dict mapping team abbreviation → championship probability.

    Returns:
        Self-contained HTML document string.
    """
    # ── Layout constants ──────────────────────────────────────────────────────
    CANVAS_WIDTH = 900  # px, full bracket width
    CANVAS_HEIGHT = 550  # px
    BOX_W = 130  # px, matchup box width
    BOX_H = 105  # px, matchup box height (two 36px team rows)

    # ── X positions per round column ─────────────────────────────────────────
    west_r1_x = 0
    west_r2_x = int(CANVAS_WIDTH * 0.16)  # 211
    west_cf_x = int(CANVAS_WIDTH * 0.22)  # 382
    finals_x = int(CANVAS_WIDTH * 0.50 - BOX_W / 2)  # 567
    east_cf_x = CANVAS_WIDTH - west_cf_x - BOX_W  # 753
    east_r2_x = CANVAS_WIDTH - west_r2_x - BOX_W  # 924
    east_r1_x = CANVAS_WIDTH - BOX_W  # 1135

    # ── Y positions: distribute 4 R1 slots evenly, then center upward ────────
    _pad = 20
    _slot = (CANVAS_HEIGHT - 2 * _pad) / 4  # 170 px per R1 slot
    r1_centers = [_pad + i * _slot + _slot / 2 for i in range(4)]
    # R2 center = midpoint between the two R1 centers that feed it
    r2_centers = [
        (r1_centers[0] + r1_centers[1]) / 2,  # 190
        (r1_centers[2] + r1_centers[3]) / 2,  # 530
    ]
    cf_center = (r2_centers[0] + r2_centers[1]) / 2  # 360
    finals_center = cf_center  # 360

    def _box_y(center: float) -> int:
        return int(center - BOX_H / 2)

    r1_y = [_box_y(c) for c in r1_centers]  # [69, 239, 409, 579]
    r2_y = [_box_y(c) for c in r2_centers]  # [154, 494]
    cf_y = _box_y(cf_center)  # 324
    finals_y = _box_y(finals_center)  # 324

    # ── Extract bracket data ─────────────────────────────────────────────────
    champ = bracket.get("champion")
    champ_abbrev = champ["abbrev"] if champ else None

    # R1 display order [1v8, 4v5, 2v7, 3v6] (raw indices 0,3,1,2) keeps adjacent
    # pairs feeding the same R2 slot so midpoints align vertically.
    west_r1_raw = bracket["west"][1]
    west_r1_ordered = [west_r1_raw[0], west_r1_raw[3], west_r1_raw[1], west_r1_raw[2]]
    west_r2_list = bracket["west"][2]
    west_cf = bracket["west"][3][0]

    east_r1_raw = bracket["east"][1]
    east_r1_ordered = [east_r1_raw[0], east_r1_raw[3], east_r1_raw[1], east_r1_raw[2]]
    east_r2_list = bracket["east"][2]
    east_cf = bracket["east"][3][0]

    finals_matchup = bracket["finals"][4][0]

    # ── Position specs: (x, y, matchup_dict, is_finals) ──────────────────────
    box_specs = []
    for i, m in enumerate(west_r1_ordered):
        box_specs.append((west_r1_x, r1_y[i], m, False))
    for i, m in enumerate(west_r2_list):
        box_specs.append((west_r2_x, r2_y[i], m, False))
    box_specs.append((west_cf_x, cf_y, west_cf, False))
    box_specs.append((finals_x, finals_y, finals_matchup, True))
    box_specs.append((east_cf_x, cf_y, east_cf, False))
    for i, m in enumerate(east_r2_list):
        box_specs.append((east_r2_x, r2_y[i], m, False))
    for i, m in enumerate(east_r1_ordered):
        box_specs.append((east_r1_x, r1_y[i], m, False))

    # ── Generate positioned matchup boxes ────────────────────────────────────
    html_boxes = "".join(
        (
            f'<div style="position:absolute;left:{x}px;top:{y}px;width:{BOX_W}px;z-index:1;'
            f'transform:scale(1.3);transform-origin:center center">'
            if is_finals
            else f'<div style="position:absolute;left:{x}px;top:{y}px;width:{BOX_W}px;z-index:1">'
        )
        + f"{matchup_html(m, champion_abbrev=champ_abbrev, is_finals=is_finals, prob_mode=prob_mode, champ_probs=champ_probs)}"
        + f"</div>"
        for x, y, m, is_finals in box_specs
    )

    # ── SVG connector lines (rendered behind boxes via z-index:0) ────────────
    # BOX_H (105) positions each box but the actual rendered height is
    # 2 team-node rows (36px each) + 3px gap = 75px. Derive connector y-centres
    # from actual box tops so stubs exit the visual midpoint between the two team
    # rows, not the lower portion of the box.
    _ACTUAL_H = 2 * 36 + 3  # 75px
    _r1_cy = [y + _ACTUAL_H / 2 for y in r1_y]  # actual centre of each R1 box
    _r2_cy = [y + _ACTUAL_H / 2 for y in r2_y]  # actual centre of each R2 box
    _cf_cy = cf_y + _ACTUAL_H / 2
    _fin_cy = finals_y + _ACTUAL_H / 2

    _S = 'stroke="#3d5a80" stroke-width="1.5" fill="none" opacity="0.85"'

    def _arm_right(
        src_rx: float, dst_lx: float, top_cy: float, bot_cy: float, dst_cy: float, box_height: float
    ) -> str:
        """Bracket arm extending rightward: two stubs → vertical gather bar → bridge to dest."""
        gx = src_rx + dst_lx
        gy_top = dst_cy - box_height / 3
        gy_bot = dst_cy + box_height / 3
        return (
            f'<line x1="{src_rx:.1f}" y1="{top_cy:.1f}" x2="{gx:.1f}" y2="{top_cy:.1f}" {_S}/>'
            f'<line x1="{src_rx:.1f}" y1="{bot_cy:.1f}" x2="{gx:.1f}" y2="{bot_cy:.1f}" {_S}/>'
            f'<line x1="{gx:.1f}" y1="{top_cy:.1f}" x2="{gx:.1f}" y2="{gy_top:.1f}" {_S}/>'
            f'<line x1="{gx:.1f}" y1="{bot_cy:.1f}" x2="{gx:.1f}" y2="{gy_bot:.1f}" {_S}/>'
        )

    def _arm_left(
        src_lx: float, dst_rx: float, top_cy: float, bot_cy: float, dst_cy: float, box_height: float
    ) -> str:
        """Bracket arm extending leftward: two stubs → vertical gather bar → bridge to dest."""
        gx = src_lx - dst_rx
        gy_top = dst_cy - box_height / 3
        gy_bot = dst_cy + box_height / 3
        return (
            f'<line x1="{src_lx:.1f}" y1="{top_cy:.1f}" x2="{dst_rx:.1f}" y2="{top_cy:.1f}" {_S}/>'
            f'<line x1="{src_lx:.1f}" y1="{bot_cy:.1f}" x2="{dst_rx:.1f}" y2="{bot_cy:.1f}" {_S}/>'
            f'<line x1="{dst_rx:.1f}" y1="{top_cy:.1f}" x2="{dst_rx:.1f}" y2="{gy_top:.1f}" {_S}/>'
            f'<line x1="{dst_rx:.1f}" y1="{bot_cy:.1f}" x2="{dst_rx:.1f}" y2="{gy_bot:.1f}" {_S}/>'
        )

    def _hline(x1: float, x2: float, y: float) -> str:
        return f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" {_S}/>'

    svg_lines = ""
    # West: R1→R2 (two pairs)
    svg_lines += _arm_right(
        west_r1_x + BOX_W, west_r2_x - BOX_W / 2, _r1_cy[0], _r1_cy[1], _r2_cy[0], BOX_H
    )
    svg_lines += _arm_right(
        west_r1_x + BOX_W, west_r2_x - BOX_W / 2, _r1_cy[2], _r1_cy[3], _r2_cy[1], BOX_H
    )
    # West: R2→CF
    svg_lines += _arm_right(
        west_r2_x + BOX_W, west_cf_x - BOX_W * 1.3, _r2_cy[0], _r2_cy[1], _cf_cy, BOX_H
    )
    # West: CF→Finals (single horizontal)
    svg_lines += _hline(west_cf_x + BOX_W, finals_x - 0.3 * BOX_W / 2, _cf_cy)
    # East: R1→R2 (two pairs, arms grow leftward)
    svg_lines += _arm_left(east_r1_x, east_r2_x + BOX_W / 2, _r1_cy[0], _r1_cy[1], _r2_cy[0], BOX_H)
    svg_lines += _arm_left(east_r1_x, east_r2_x + BOX_W / 2, _r1_cy[2], _r1_cy[3], _r2_cy[1], BOX_H)
    # East: R2→CF
    svg_lines += _arm_left(east_r2_x, east_cf_x + BOX_W * 0.2, _r2_cy[0], _r2_cy[1], _cf_cy, BOX_H)
    # East: CF→Finals (single horizontal)
    svg_lines += _hline(east_cf_x, finals_x + 1.15 * BOX_W, _cf_cy)

    svg_el = (
        f'<svg style="position:absolute;top:0;left:0;width:{CANVAS_WIDTH}px;height:{CANVAS_HEIGHT}px;'
        f'z-index:0;pointer-events:none;overflow:visible">'
        f"{svg_lines}"
        f"</svg>"
    )

    # ── CSS scoped to team-node components (iframe has no parent page styles) ─
    champ_border = colors["champion_border"]
    _css = f"""
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{ background: {colors['background']}; overflow-x: auto; overflow-y: hidden;
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
      .matchup {{ display: flex; flex-direction: column; gap: 3px; width: 100%; }}
      .team-node {{
        display: flex; flex-direction: row; align-items: stretch;
        background: #1e2a45; border: none; border-radius: 8px;
        min-height: 36px; position: relative; width: 100%; overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      }}
      .team-node:hover {{ box-shadow: 0 2px 12px rgba(90,180,255,0.22); }}
      .team-node.champion {{
        box-shadow: 0 0 12px 3px rgba(255,215,0,0.32);
        outline: 1px solid {champ_border};
      }}
      .team-node.finalist {{
        box-shadow: 0 0 12px 3px rgba(255,215,0,0.32);
        outline: 1px solid {champ_border};
      }}
      .team-pill {{
        display: flex; flex-direction: row; align-items: center; flex: 1; min-width: 0;
      }}
      .team-seed-badge {{
        width: 18px; min-width: 18px; align-self: stretch;
        display: flex; align-items: center; justify-content: center;
        font-size: 9px; font-weight: 900; color: rgba(255,255,255,0.85);
        background: rgba(0,0,0,0.28); flex-shrink: 0;
      }}
      .team-abbrev {{
        font-size: 10px; font-weight: 800; color: #ffffff;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        text-transform: uppercase; letter-spacing: 0.04em; padding: 0 4px 0 5px;
      }}
      .team-logo-area {{
        width: 30px; min-width: 30px; align-self: stretch;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
      }}
      .team-logo {{ width: 22px; height: 22px; object-fit: contain; }}
      .team-logo-fallback {{
        width: 22px; height: 22px; display: flex; align-items: center;
        justify-content: center; font-size: 7px; font-weight: 700; color: #8fa3c1;
      }}
      .team-prob {{
        font-size: 10px; font-weight: 700; color: #8fa3c1; white-space: nowrap;
        flex-shrink: 0; align-self: center; padding: 0 5px; min-width: 28px;
        text-align: right;
      }}
      .team-node.champion .team-prob {{ color: {champ_border}; }}
      .team-node.finalist .team-prob {{ color: {champ_border}; }}
    """

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>{_css}</style>
</head>
<body>
  <div id="outer" style="width:100%;overflow:hidden;position:relative">
    <div id="canvas" style="position:relative;width:{CANVAS_WIDTH}px;height:{CANVAS_HEIGHT}px;transform-origin:top left">
      {svg_el}
      {html_boxes}
    </div>
  </div>
  <script>
    var CW = {CANVAS_WIDTH}, CH = {CANVAS_HEIGHT};
    var canvas = document.getElementById('canvas');
    var outer  = document.getElementById('outer');
    function applyScale() {{
      var w = outer.clientWidth || document.documentElement.clientWidth;
      if (w < 10) {{ setTimeout(applyScale, 50); return; }}
      var scale = w / CW;
      canvas.style.transform = 'scale(' + scale + ')';
      var newH = Math.ceil(CH * scale) + 4;
      outer.style.height = (newH - 4) + 'px';
      // st.components.v1.html() ignores setFrameHeight messages, so resize the
      // parent iframe element directly. Works because Streamlit runs same-origin.
      try {{
        var frames = window.parent.document.querySelectorAll('iframe');
        for (var i = 0; i < frames.length; i++) {{
          if (frames[i].contentWindow === window) {{
            frames[i].style.height = newH + 'px';
            frames[i].style.minHeight = newH + 'px';
            break;
          }}
        }}
      }} catch(e) {{}}
    }}
    applyScale();
    setTimeout(applyScale, 100);
    window.addEventListener('resize', applyScale);
  </script>
</body>
</html>"""


def render_champ_prob_chart_html(
    champ_df: pd.DataFrame,
    seeds: dict[str, list[str]],
) -> str:
    """Return self-contained HTML for the championship probability bar chart.

    Bars are sorted descending by probability, colored West=blue / East=red,
    with team logo images along the x-axis.

    Args:
        champ_df: DataFrame with columns ``team`` and ``championship_prob``.
        seeds: Dict mapping ``"west"`` / ``"east"`` to lists of team abbreviations.

    Returns:
        Self-contained HTML string.
    """
    west_set = set(seeds.get("west", []))
    sorted_df = champ_df.sort_values("championship_prob", ascending=False).reset_index(drop=True)

    max_prob = float(sorted_df["championship_prob"].max()) or 1.0
    MAX_BAR_H = 160

    bar_cells = ""
    logo_cells = ""
    for _, row in sorted_df.iterrows():
        team = str(row["team"])
        prob = float(row["championship_prob"])
        color = "#3a68b5" if team in west_set else "#c04060"
        bar_h = max(2, int(MAX_BAR_H * prob / max_prob))
        pct_label = f"{prob:.1%}" if prob < 0.005 else f"{prob:.0%}"
        url = logo_url(team)

        bar_cells += (
            f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;min-width:0;padding:0 1px">'
            f'<span style="font-size:9px;color:#8fa3c1;margin-bottom:2px;white-space:nowrap">{pct_label}</span>'
            f'<div style="width:80%;height:{bar_h}px;background:{color};border-radius:4px 4px 0 0"></div>'
            f"</div>"
        )
        logo_cells += (
            f'<div style="flex:1;display:flex;flex-direction:column;align-items:center;min-width:0;padding:0 1px">'
            f'<img src="{url}" style="width:22px;height:22px;object-fit:contain" '
            f"onerror=\"this.style.visibility='hidden'\">"
            f'<span style="font-size:8px;color:#8fa3c1;margin-top:1px">{team}</span>'
            f"</div>"
        )

    return (
        f'<div style="font-family:sans-serif;background:transparent;width:100%">'
        f'<div style="display:flex;align-items:flex-end;width:100%;height:{MAX_BAR_H + 22}px;'
        f'border-bottom:1px solid #2a3a54">'
        f"{bar_cells}"
        f"</div>"
        f'<div style="display:flex;width:100%;margin-top:4px">'
        f"{logo_cells}"
        f"</div>"
        f"</div>"
    )
