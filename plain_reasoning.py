TEMPLATES = {
    "GIFT_Nifty_return": "GIFT Nifty moved {val:+.2f}% overnight, which historically leads to a {dir} open.",
    "SP500_return":      "S&P 500 closed {val:+.2f}%, a signal that correlates with Nifty opening {dir}.",
    "Nasdaq_return":     "Nasdaq closed {val:+.2f}%, adding {sentiment} pressure on tech-heavy Nifty stocks.",
    "India_VIX":         "India VIX is at {val:.1f} — {vix_label} volatility, which {vix_effect} signal clarity.",
    "USDINR_return":     "USD/INR moved {val:+.2f}%, indicating {fx_label} for foreign flows into Indian markets.",
    "Crude_return":      "Crude oil {crude_dir} {val:.1f}%, which {crude_effect} input costs for Indian markets.",
}

def _vix_label(val):
    if val < 15: return "low"
    if val < 20: return "moderate"
    return "high"

def _vix_effect(val):
    return "improves" if val < 20 else "reduces"

def _fx_label(val):
    return "rupee weakness — negative" if val > 0 else "rupee strength — positive"

def _crude_dir(val):
    return "rose" if val > 0 else "fell"

def _crude_effect(val):
    return "raises" if val > 0 else "lowers"

def generate_reasoning(shap_features: list, live_data: dict, direction: str) -> list:
    sentences = []
    sentiment = "positive" if direction == "UP" else "negative"
    for feat, shap_val in shap_features[:3]:
        val = live_data.get(feat, 0)
        tmpl = TEMPLATES.get(feat)
        if not tmpl:
            sentences.append(f"{feat} contributed a SHAP value of {shap_val:+.3f} toward the {direction} call.")
            continue
        try:
            s = tmpl.format(
                val=val, dir=direction, sentiment=sentiment,
                vix_label=_vix_label(val), vix_effect=_vix_effect(val),
                fx_label=_fx_label(val), crude_dir=_crude_dir(val),
                crude_effect=_crude_effect(val)
            )
            sentences.append(s)
        except Exception:
            sentences.append(f"{feat} is a key driver of today's {direction} prediction.")
    return sentences