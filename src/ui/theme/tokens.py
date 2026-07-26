"""Design tokens — the single source of every colour, size and duration in the UI.

Transcribed 1:1 from docs/DESIGN_SYSTEM.md sections 1.1-1.4 plus the concrete
values the component specs (section 2) pin down. The acceptance criterion is
literal: "ни одного цвета и отступа вне раздела 1", so widgets import from here
and never write a hex literal or a magic pixel of their own.

Deliberately Qt-free: the values are plain strings and numbers so they can be
unit-tested and embedded in stylesheets without a QApplication. Anything that
needs a QColor/QFont goes through ui.theme.fonts / ui.theme.icons.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1.1 Colour
# ---------------------------------------------------------------------------
# Every colour is opaque: the window has its own backdrop and there are no
# translucent layers anywhere in the system (brief section 3).


class Color:
    # Backgrounds
    BG_WINDOW = "#0A0C18"          # window backdrop, transcript area
    BG_TITLEBAR = "#0D1020"        # title bar (drag zone)
    BG_SURFACE = "#161824"         # action buttons, panels, fields, menu
    BG_SURFACE_HOVER = "#1E2130"   # hover on any surface
    BG_SURFACE_ACTIVE = "#252940"  # press; selected menu item
    BG_CARD = "#12151E"            # answer card, history entry

    # Borders
    BORDER = "#232634"             # 1 px borders: window, panels, fields, cards
    BORDER_STRONG = "#2E3244"      # border on hover, menu separator

    # Text
    TEXT_PRIMARY = "#E8EAF2"       # committed transcript, model answer
    TEXT_LABEL = "#DCDEE8"         # button/menu labels, headings
    TEXT_SECONDARY = "#767889"     # button second line, statuses, hints
    TEXT_TIMESTAMP = "#5F6377"     # gutter timestamp
    TEXT_PRELIMINARY = "#6C7086"   # provisional hypothesis (italic)

    # Accent — the only one in the system
    ACCENT = "#6D4FE6"
    ACCENT_HOVER = "#7D63EA"
    ACCENT_ACTIVE = "#5B3FD1"
    ACCENT_SUBTLE = "#1B1733"      # icon plate in the answer card, notification bg
    ACCENT_ON = "#C9BEFF"          # text/icon on ACCENT_SUBTLE
    ON_ACCENT = "#FFFFFF"          # text on a filled accent (selected segment)

    # State
    STATE_GOOD = "#22C55E"         # audio flowing, recording, model online
    STATE_WARN = "#F5C518"         # recording paused; model offline/loading
    STATE_ERROR = "#EF4450"        # capture error, broken journal, fatal
    STATE_IDLE = "#3A4051"         # silence, indicator off — grey, not a colour

    # Fatal error block
    ERROR_BG = "#2A1218"
    ERROR_BORDER = "#4A2029"
    ERROR_TEXT = "#FFD9DE"         # label on the single "Закрыть" button

    # Keyboard focus (settings window only — the overlay never takes focus)
    FOCUS_RING = "#6D4FE6"

    # Control states, section 2.2. Hover/press reuse BG_SURFACE_*; disabled and
    # the brighter hover text are their own values.
    TEXT_LABEL_HOVER = "#E8EAF2"
    TEXT_SECONDARY_HOVER = "#8A8CA0"
    DISABLED_BG = "#111320"
    DISABLED_BORDER = "#1A1C28"
    DISABLED_TEXT = "#4E5162"
    DISABLED_TEXT_SECONDARY = "#3A3D4C"


# ---------------------------------------------------------------------------
# 1.2 Typography
# ---------------------------------------------------------------------------
# IBM Plex Sans for the interface, IBM Plex Mono for clock digits and hotkeys
# only: a monospaced timestamp does not "breathe" as its digits change, and that
# is the most frequent repaint in the window. Both ship in assets/fonts.

FAMILY_SANS = "IBM Plex Sans"
FAMILY_MONO = "IBM Plex Mono"

# Used only if the bundled .ttf files are missing (ui.theme.fonts falls back).
FALLBACK_SANS = "Segoe UI"
FALLBACK_MONO = "Consolas"

WEIGHT_REGULAR = 400
WEIGHT_MEDIUM = 500
WEIGHT_SEMIBOLD = 600

# Smallest size the system allows; nothing may be set below it.
MIN_FONT_PX = 10.5


@dataclass(frozen=True, slots=True)
class TextStyle:
    """One row of the typography table: family, size/leading in px, weight, colour."""

    family: str
    size_px: float
    weight: int
    line_px: float
    color: str
    italic: bool = False


class Type:
    TRANSCRIPT = TextStyle(FAMILY_SANS, 13, WEIGHT_REGULAR, 20, Color.TEXT_PRIMARY)
    TRANSCRIPT_PRELIMINARY = TextStyle(
        FAMILY_SANS, 13, WEIGHT_REGULAR, 20, Color.TEXT_PRELIMINARY, italic=True
    )
    TIMESTAMP = TextStyle(FAMILY_MONO, 11, WEIGHT_REGULAR, 20, Color.TEXT_TIMESTAMP)
    PANEL_TITLE = TextStyle(FAMILY_SANS, 12, WEIGHT_SEMIBOLD, 16, Color.TEXT_LABEL)
    BUTTON_LABEL = TextStyle(FAMILY_SANS, 12, WEIGHT_MEDIUM, 15, Color.TEXT_LABEL)
    BUTTON_SUB = TextStyle(FAMILY_SANS, 10.5, WEIGHT_REGULAR, 13, Color.TEXT_SECONDARY)
    STATUS = TextStyle(FAMILY_SANS, 11, WEIGHT_MEDIUM, 14, Color.TEXT_SECONDARY)
    ANSWER = TextStyle(FAMILY_SANS, 12.5, WEIGHT_REGULAR, 18, Color.TEXT_PRIMARY)
    MENU_ITEM = TextStyle(FAMILY_SANS, 12, WEIGHT_REGULAR, 16, Color.TEXT_LABEL)
    HOTKEY = TextStyle(FAMILY_MONO, 11, WEIGHT_REGULAR, 16, Color.TEXT_SECONDARY)
    # Compact mode overrides
    TRANSCRIPT_COMPACT = TextStyle(FAMILY_SANS, 12, WEIGHT_REGULAR, 17, Color.TEXT_PRIMARY)
    TRANSCRIPT_COMPACT_PRELIMINARY = TextStyle(
        FAMILY_SANS, 12, WEIGHT_REGULAR, 17, Color.TEXT_PRELIMINARY, italic=True
    )
    TIMESTAMP_COMPACT = TextStyle(FAMILY_MONO, 10.5, WEIGHT_REGULAR, 17, Color.TEXT_TIMESTAMP)
    # Windows that are not the overlay
    WINDOW_TITLE = TextStyle(FAMILY_SANS, 14, WEIGHT_SEMIBOLD, 18, Color.TEXT_LABEL)
    FIELD_LABEL = TextStyle(FAMILY_SANS, 12, WEIGHT_REGULAR, 16, Color.TEXT_LABEL)
    FIELD_VALUE = TextStyle(FAMILY_SANS, 11.5, WEIGHT_REGULAR, 16, Color.TEXT_LABEL)
    HISTORY_TITLE = TextStyle(FAMILY_SANS, 12, WEIGHT_MEDIUM, 16, Color.TEXT_LABEL)
    HISTORY_PREVIEW = TextStyle(FAMILY_SANS, 10.5, WEIGHT_REGULAR, 14, Color.TEXT_SECONDARY)


# ---------------------------------------------------------------------------
# 1.3 Spacing, radii, borders
# ---------------------------------------------------------------------------
# Base unit 4 px. No other values are allowed; the single exception is an
# optical nudge of an icon by +-1 px.


class Space:
    S1 = 4    # inside a pill, icon-to-text
    S2 = 8    # between buttons, compact padding
    S3 = 12   # window padding (normal mode)
    S4 = 16   # between blocks inside a panel
    S5 = 20   # between settings sections
    S6 = 24   # top/bottom of the empty screen


class Radius:
    WINDOW = 10
    PANEL = 8      # panels, cards, menu
    CONTROL = 6    # buttons, fields, segments
    SMALL = 4      # menu item, icon plate inside a button... see section 2.2/2.5
    PLATE = 3      # accent plate behind a button icon
    PILL = 999


class Border:
    WIDTH = 1      # everywhere; there is no 2 px border in the system


class Size:
    DOT = 8              # status dot diameter
    ICON = 16            # normal mode
    ICON_COMPACT = 14
    ICON_STROKE = 1.5
    HIT_MIN = 24         # smallest clickable box
    INDICATOR_SLOT = 14  # fixed icon slot in the title bar: never resizes


# ---------------------------------------------------------------------------
# 1.4 Motion
# ---------------------------------------------------------------------------
# Only opacity is animated, and only where it cannot move the transcript. No
# animation changes the geometry of the feed. Default curve: OutCubic.


class Motion:
    HOVER_MS = 90
    CARD_MS = 120
    NOTIFICATION_MS = 180
    STATUS_MS = 120
    WINDOW_TOGGLE_MS = 140
    MODE_SWITCH_MS = 160
    CAPTURE_BAR_MS = 200      # one step of the 3-bar capture animation
    THINKING_DOT_MS = 260     # one step of the 3-dot "модель думает" rhythm
    STATUS_LIFETIME_MS = 4000
    NOTIFICATION_LIFETIME_MS = 5000
    TOOLTIP_DELAY_MS = 400
    ANSWER_AUTOCLOSE_MS = 30000
    CARET_BLINK_MS = 1000


# ---------------------------------------------------------------------------
# 3. Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LayoutMode:
    """Every geometry number of one window mode, section 3."""

    name: str
    width: int
    height: int
    titlebar_h: int
    padding: int
    gutter_w: int          # timestamp gutter
    gutter_gap: int        # gutter to text
    action_h: int          # action row height
    action_gap: int        # gap between buttons
    icon: int
    title_gap: int         # gap between title bar items
    answer_max_h: int      # answer card cap
    answer_pad_v: int
    answer_pad_h: int
    transcript: TextStyle
    preliminary: TextStyle
    timestamp: TextStyle
    ts_format: str
    with_labels: bool      # indicator captions and button captions visible


# 600, not the specified 520: the title bar carries three labelled indicators,
# the status zone, the menu and a close button, and at 520 the zone was down to
# ~45 px — narrower than a single word. 600 leaves it ~125. Everything else
# (heights, paddings, the gutter) is unchanged, and compact stays 360.
NORMAL = LayoutMode(
    name="normal",
    width=600,
    height=340,
    titlebar_h=36,
    padding=Space.S3,
    gutter_w=56,
    gutter_gap=Space.S3,
    action_h=40,
    action_gap=Space.S2,
    icon=Size.ICON,
    title_gap=14,
    answer_max_h=160,
    answer_pad_v=10,
    answer_pad_h=Space.S3,
    transcript=Type.TRANSCRIPT,
    preliminary=Type.TRANSCRIPT_PRELIMINARY,
    timestamp=Type.TIMESTAMP,
    ts_format="%H:%M:%S",
    with_labels=True,
)

COMPACT = LayoutMode(
    name="compact",
    width=360,
    height=200,
    titlebar_h=28,
    padding=Space.S2,
    gutter_w=44,
    gutter_gap=Space.S2,
    action_h=28,
    action_gap=6,
    icon=Size.ICON_COMPACT,
    title_gap=Space.S2,
    answer_max_h=96,
    answer_pad_v=Space.S2,
    answer_pad_h=10,
    transcript=Type.TRANSCRIPT_COMPACT,
    preliminary=Type.TRANSCRIPT_COMPACT_PRELIMINARY,
    timestamp=Type.TIMESTAMP_COMPACT,
    ts_format="%H:%M",
    with_labels=False,
)

# Reserved zone for the status message in the title bar. It is always there, so
# a message appearing moves nothing (section 2.6 / the no-shift check).
STATUS_ZONE_W = 180
STATUS_ZONE_H = 20

# Satellite windows
NOTIFICATION_W, NOTIFICATION_H = 240, 56
NOTIFICATION_MARGIN = Space.S4     # from the screen edges
NOTIFICATION_STRIPE = 3            # accent stripe on the left
FATAL_W, FATAL_H = 320, 200
# The companion window that keeps the live original while the main feed shows
# only finished translations (§7.8). Narrower and shorter than compact: it is
# glanced at, not read.
SOURCE_W, SOURCE_H = 360, 168
ONBOARDING_W, ONBOARDING_H = 420, 260
SETTINGS_W, SETTINGS_H = 420, 470
HISTORY_W, HISTORY_H = 380, 460
MENU_MAX_W = 240
MENU_ITEM_H = 30

# Section 2.3: a reply is broken into a new gutter-less block after this many
# lines, and the whole feed keeps at most this many replies rendered.
MAX_LINES_PER_REPLY = 4
