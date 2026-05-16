"""Style tokens shared across pages."""

# Full rainbow gradient for the BirdML heading. The 200% background-size +
# rainbow_shift keyframes give the title its slow shimmer.
GRADIENT_TITLE = (
    "linear-gradient(90deg, "
    "#ef4444 0%, #f97316 14%, #eab308 28%, #22c55e 42%, "
    "#06b6d4 56%, #3b82f6 70%, #a855f7 84%, #ec4899 100%)"
)

# Layered background: local asset first, Wikimedia fallback if that 404s,
# green-violet gradient as a final fallback if both images fail to load.
BACKGROUND_IMAGE = (
    "url('/macaws_2.jpg'), "
    "url('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/"
    "Cacatua_galerita_-Cape_York_Peninsula%2C_Queensland%2C_Australia-8.jpg/"
    "1920px-Cacatua_galerita_-Cape_York_Peninsula%2C_Queensland%2C_Australia-8.jpg'), "
    "linear-gradient(135deg, #064e3b 0%, #14532d 40%, #312e81 100%)"
)

# Glass-card recipe used by the navbar pill and stub "coming soon" cards
# so all surfaces share the same translucent look.
GLASS_BG = "rgba(15, 13, 24, 0.42)"
GLASS_BORDER = "1px solid rgba(167, 139, 250, 0.35)"
GLASS_BLUR = "blur(14px)"

# Accent used for active nav state and card highlights.
ACCENT = "rgba(167, 139, 250, 0.25)"
ACCENT_HOVER = "rgba(167, 139, 250, 0.15)"
