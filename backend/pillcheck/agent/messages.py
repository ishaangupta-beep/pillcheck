"""Message templates. Patient-facing text is available in English and Hindi.

The rule planner always uses these. The Bedrock planner writes its own wording,
but the policy layer falls back to these templates whenever generated text
fails the safety checks.
"""
from __future__ import annotations

PATIENT = {
    "dose_recorded": {
        "en": "Thank you! Your {time} {med} has been recorded.",
        "hi": "धन्यवाद! आपकी {time} बजे की {med} दर्ज हो गई है।",
    },
    "dose_recorded_late": {
        "en": "Thank you. Your {time} {med} has been recorded (taken late).",
        "hi": "धन्यवाद। आपकी {time} बजे की {med} दर्ज हो गई है (देर से ली गई)।",
    },
    "not_taken_yet": {
        "en": "I don't see your {time} {med} taken yet. It is due now.",
        "hi": "आपकी {time} बजे की {med} अभी तक ली हुई नहीं दिख रही। इसका समय हो गया है।",
    },
    "nothing_new": {
        "en": "No new tablet was removed. Your next {med} is at {time}.",
        "hi": "कोई नई गोली नहीं निकली। आपकी अगली {med} {time} बजे है।",
    },
    "duplicate": {
        "en": "This photo was already sent earlier. Please take a new photo of the strip now.",
        "hi": "यह फ़ोटो पहले भेजी जा चुकी है। कृपया अभी पत्ते की नई फ़ोटो लें।",
    },
    "unrecognised": {
        "en": "I could not recognise this strip. Please photograph the whole strip, printed side up, on a plain surface.",
        "hi": "यह पत्ता पहचान में नहीं आया। कृपया पूरा पत्ता, छपी हुई तरफ़ ऊपर रखकर, सादी सतह पर फ़ोटो लें।",
    },
    "wrong_medication": {
        "en": "This strip is {matched}. The medicine due now is {expected}.",
        "hi": "यह पत्ता {matched} का है। अभी {expected} लेने का समय है।",
    },
    "extra_removed": {
        "en": "More tablets were removed than scheduled. Your caregiver has been informed. If you feel unwell, please contact your doctor.",
        "hi": "निर्धारित से ज़्यादा गोलियाँ निकाली गई हैं। आपके परिजन को सूचित कर दिया गया है। तबीयत ठीक न लगे तो डॉक्टर से संपर्क करें।",
    },
    "expired": {
        "en": "The expiry date on this {med} strip ({exp}) has passed. Your caregiver has been told and will check it with you.",
        "hi": "इस {med} पत्ते की एक्सपायरी तारीख ({exp}) निकल चुकी है। आपके परिजन को बता दिया गया है, वे आपके साथ इसे देखेंगे।",
    },
    "reminder": {
        "en": "Reminder: your {time} {med} is due. After taking it, please send a photo of the strip.",
        "hi": "याद दिलाना: आपकी {time} बजे की {med} का समय हो गया है। लेने के बाद पत्ते की फ़ोटो भेजें।",
    },
    "unclear": {
        "en": "Some pockets were hard to see. Please take one more photo, from straight above, without the flash.",
        "hi": "कुछ खाने साफ़ नहीं दिखे। कृपया ऊपर से, फ़्लैश के बिना, एक और फ़ोटो लें।",
    },
}

CAREGIVER = {
    "extra_removed": "{patient} removed {n} more {med} tablet(s) than scheduled ({now}). Possible double dose: please check in with them. The evidence photo is in the app.",
    "wrong_medication": "{patient} removed a {matched} tablet when {expected} was due ({now}). Please check which medicine was taken.",
    "missed": "{patient} has not taken the {time} {med} ({minutes} min overdue). Reminders were sent.",
    "photo_reuse": "{patient} sent a photo that was already submitted earlier ({now}). It was not counted as a dose.",
    "confirm_dose": "Two photos of {patient}'s {med} strip were unclear. Please look at the evidence and confirm whether the {time} dose was taken.",
    "new_strip": "{patient}'s {med} photo shows tablets back in pockets that were empty. Has a new strip been started? If so, register it in the app.",
    "expired": "{patient}'s {med} strip expired in {exp}. Please replace it before the next dose; the photo it was read from is in the app.",
    "expiring_soon": "{patient}'s {med} strip expires in {exp}. Plan a fresh strip before then.",
    "refill": "{med} for {patient} is running low: {left} tablet(s) left, about {days} day(s). Approve to mark a refill as needed.",
}


def patient_text(key: str, lang: str, **kw) -> str:
    t = PATIENT[key]
    return t.get(lang, t["en"]).format(**kw)


def caregiver_text(key: str, **kw) -> str:
    return CAREGIVER[key].format(**kw)


def fmt_time(hhmm: str) -> str:
    h, m = (int(x) for x in hhmm.split(":"))
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {suffix}"


def fmt_expiry(yyyy_mm: str) -> str:
    import calendar
    y, m = yyyy_mm.split("-")
    return f"{calendar.month_abbr[int(m)]} {y}"
