"""Shared page chrome: background image, dark overlay, navbar, content slot.

Wrapping each page in page_layout() guarantees all three routes share the
exact same background, overlay, and navigation — no per-page duplication.
"""

import reflex as rx

from ..styles import BACKGROUND_IMAGE
from .navbar import navbar

# Detect Instagram's in-app browser (it injects "Instagram" into the user
# agent) and show a dismissable banner asking the user to open the site
# in their real browser. The in-app webview has limited vertical space
# and CSS/JS quirks; pushing users out to Safari/Chrome gives them the
# full experience. iOS and Android have different "open externally" menus
# so the instructions text branches on platform.
_INSTAGRAM_BANNER_JS = """
(function() {
  const ua = window.navigator.userAgent || '';
  if (!/Instagram/i.test(ua)) return;
  if (document.getElementById('ig-browser-banner')) return;

  const isIOS = /iPhone|iPad|iPod/i.test(ua);
  const howTo = isIOS
    ? "Tap <strong>\\u2022\\u2022\\u2022</strong> (top right) \\u2192 <strong>Open in external browser</strong>"
    : "Tap <strong>\\u22ee</strong> (top right) \\u2192 <strong>Open in Chrome</strong>";

  const banner = document.createElement('div');
  banner.id = 'ig-browser-banner';
  banner.style.cssText = [
    'position:fixed','top:0','left:0','right:0','z-index:9999',
    'background:rgba(15,13,24,0.96)',
    'backdrop-filter:blur(12px)','-webkit-backdrop-filter:blur(12px)',
    'color:white','padding:12px 44px 12px 16px',
    'font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",system-ui,sans-serif',
    'font-size:13px','line-height:1.45','text-align:center',
    'border-bottom:1px solid rgba(167,139,250,0.4)',
    'box-shadow:0 6px 24px rgba(0,0,0,0.5)'
  ].join(';');
  banner.innerHTML =
    '<span>For the best experience: ' + howTo + '</span>' +
    '<button id=\"ig-banner-close\" aria-label=\"Dismiss\" style=\"' +
      'position:absolute;right:6px;top:50%;transform:translateY(-50%);' +
      'background:none;border:none;color:rgba(255,255,255,0.75);' +
      'cursor:pointer;font-size:22px;line-height:1;padding:6px 10px;' +
      'font-weight:600;\">\\u00d7</button>';

  document.body.prepend(banner);
  document.getElementById('ig-banner-close')
    .addEventListener('click', function() { banner.remove(); });
})();
"""


def page_layout(content: rx.Component) -> rx.Component:
    """Wrap a page's content with the shared background + overlay + navbar."""
    return rx.box(
        # Inline JS that adds an "open in external browser" banner only when
        # the visitor is inside Instagram's in-app webview. No-op everywhere
        # else, so Safari/Chrome users never see it.
        rx.script(_INSTAGRAM_BANNER_JS),
        # Dark gradient overlay sits above the background image so text and
        # glass cards stay legible regardless of which photo loads.
        rx.vstack(
            navbar(),
            # Top-aligned content wrapper: every page anchors its heading at
            # the same vertical position below the navbar regardless of how
            # much content sits underneath. flex=1 still lets the wrapper
            # expand so the dark gradient overlay fills the full viewport.
            rx.flex(
                content,
                width="100%",
                flex="1",
                justify="center",
                align="start",
            ),
            spacing="0",
            min_height="100vh",
            width="100%",
            background=(
                "linear-gradient(180deg, "
                "rgba(8, 6, 16, 0.55) 0%, "
                "rgba(8, 6, 16, 0.78) 100%)"
            ),
        ),
        background_image=BACKGROUND_IMAGE,
        background_size="cover, cover, auto",
        background_repeat="no-repeat",
        background_position="center 45%",
        background_attachment="fixed",
        background_color="#0a1a14",
        min_height="100vh",
        width="100%",
    )
