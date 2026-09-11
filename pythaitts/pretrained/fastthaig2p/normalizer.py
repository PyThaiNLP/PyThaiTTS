"""Text normalization for Thai TTS — converts numbers, abbreviations, and
special characters to speakable Thai text before tokenization. In order:
- expand maiyamok
- convert thai numerals to arabic
- read emails
- read english abbreviations
- read units
- read symbols
- read time patterns
- read phone numbers
- read alphanumeric letters together like A10G
- read comma numbers
- read numbers
- read thai abbreviations; put here due to possible confusion with time, units
- read residual latins
"""

from __future__ import annotations
import re
from pythainlp.util import expand_maiyamok as _expand_maiyamok_pythainlp

# pythainlp defers its dictionary load to the first call — pay that cost at
# import time rather than on the first online request containing ๆ (~330ms)
_expand_maiyamok_pythainlp("ๆ")

THAI_DIGITS = {
    "0": "ศูนย์",
    "1": "หนึ่ง",
    "2": "สอง",
    "3": "สาม",
    "4": "สี่",
    "5": "ห้า",
    "6": "หก",
    "7": "เจ็ด",
    "8": "แปด",
    "9": "เก้า",
}

THAI_NUMERAL_MAP = {
    "๐": "0", "๑": "1", "๒": "2", "๓": "3", "๔": "4",
    "๕": "5", "๖": "6", "๗": "7", "๘": "8", "๙": "9",
}

LETTER_TO_THAI = {
    "A": "เอ", "B": "บี", "C": "ซี", "D": "ดี", "E": "อี",
    "F": "เอฟ", "G": "จี", "H": "เอช", "I": "ไอ", "J": "เจ",
    "K": "เค", "L": "แอล", "M": "เอ็ม", "N": "เอ็น", "O": "โอ",
    "P": "พี", "Q": "คิว", "R": "อาร์", "S": "เอส", "T": "ที",
    "U": "ยู", "V": "วี", "W": "ดับเบิลยู", "X": "เอ็กซ์",
    "Y": "วาย", "Z": "แซด",
}

# we omit one-letter abbreviations to prevent ambiguous cases
# such as ม. being เมตร and มหาวิทยาลัย
ABBREVIATIONS = {
    # Months
    "ม.ค.": "มกราคม", "ก.พ.": "กุมภาพันธ์", "มี.ค.": "มีนาคม",
    "เม.ย.": "เมษายน", "พ.ค.": "พฤษภาคม", "มิ.ย.": "มิถุนายน",
    "ก.ค.": "กรกฎาคม", "ส.ค.": "สิงหาคม", "ก.ย.": "กันยายน",
    "ต.ค.": "ตุลาคม", "พ.ย.": "พฤศจิกายน", "ธ.ค.": "ธันวาคม",
    # Eras
    "พ.ศ.": "พุทธศักราช", "ค.ศ.": "คริสต์ศักราช",
    # Titles
    ## Common Civil & Academic
    "น.ส.": "นางสาว",
    "นส.": "นางสาว",
    "ดร.": "ด็อกเตอร์",
    # "ศ.": "ศาสตราจารย์",
    "ผศ.": "ผู้ช่วยศาสตราจารย์",
    "รศ.": "รองศาสตราจารย์",
    "ดร.": "ด็อกเตอร์",
    ## Medical & Professional
    "นพ.": "นายแพทย์",
    "พญ.": "แพทย์หญิง",
    "ทพ.": "ทันตแพทย์",
    "ทพญ.": "ทันตแพทย์หญิง",
    "น.สพ.": "นายสััตวแพทย์",
    "สพ.ญ.": "สัตวแพทย์หญิง",
    "ภก.": "เภสัชกร",
    "ภญ.": "เภสัชกรหญิง",
    "ทนพ.": "เทคนิคการแพทย์",
    "กภ.": "กายภาพบำบัด",
    ## Army (Officers)
    "พล.อ.": "พลเอก",
    "พล.ท.": "พลโท",
    "พล.ต.": "พลตรี",
    "พ.อ.": "พันเอก",
    "พ.ท.": "พันโท",
    "พ.ต.": "พันตรี",
    "ร.อ.": "ร้อยเอก",
    "ร.ท.": "ร้อยโท",
    "ร.ต.": "ร้อยตรี",
    ## Army (NCOs)
    "จ.ส.อ.": "จ่าสิบเอก",
    "จ.ส.ท.": "จ่าสิบโท",
    "จ.ส.ต.": "จ่าสิบตรี",
    "ส.อ.": "สิบเอก",
    "ส.ท.": "สิบโท",
    "ส.ต.": "สิบตรี",
    ## Police
    "พล.ต.อ.": "พลตำรวจเอก",
    "พล.ต.ท.": "พลตำรวจโท",
    "พล.ต.ต.": "พลตำรวจตรี",
    "พ.ต.อ.": "พันตำรวจเอก",
    "พ.ต.ท.": "พันตำรวจโท",
    "พ.ต.ต.": "พันตำรวจตรี",
    "ร.ต.อ.": "ร้อยตำรวจเอก",
    "ร.ต.ท.": "ร้อยตำรวจโท",
    "ร.ต.ต.": "ร้อยตำรวจตรี",
    "ด.ต.": "ดาบตำรวจ",
    ## Navy (Officers & NCOs); เรือเอก/โท/ตรี ซ้ำกับ ร้อยเอก/โท/ตรี
    "พล.ร.อ.": "พลเรือเอก",
    "พล.ร.ท.": "พลเรือโท",
    "พล.ร.ต.": "พลเรือตรี",
    "น.อ.": "นาวาเอก",
    "น.ท.": "นาวาโท",
    "น.ต.": "นาวาตรี",
    "พ.จ.อ.": "พันจ่าเอก",
    "พ.จ.ท.": "พันจ่าโท",
    "พ.จ.ต.": "พันจ่าตรี",
    "จ.อ.": "จ่าเอก",
    "จ.ท.": "จ่าโท",
    "จ.ต.": "จ่าตรี",
    ## Air Force (Officers & NCOs); นาวาอากาศเอก/โท/ตรี ซ้ำกับ นาวาเอก/โท/ตรี
    "พล.อ.อ.": "พลอากาศเอก",
    "พล.อ.ท.": "พลอากาศโท",
    "พล.อ.ต.": "พลอากาศตรี",
    "พ.อ.อ.": "พันจ่าอากาศเอก",
    "พ.อ.ท.": "พันจ่าอากาศโท",
    "พ.อ.ต.": "พันจ่าอากาศตรี",
    ## Royal & Noble
    "ม.จ.": "หม่อมเจ้า",
    "ม.ร.ว.": "หม่อมราชวงศ์",
    "ม.ล.": "หม่อมหลวง",
    # Common
    "กทม.": "กรุงเทพมหานคร",
    "รร.": "โรงเรียน", "ร.ร.": "โรงเรียน",
    "รพ.": "โรงพยาบาล", "ร.พ.": "โรงพยาบาล",
    "บจก.": "บริษัทจำกัด", 
    "ฯลฯ": "เป็นต้น",
}

SYMBOLS = {
    "%": "เปอร์เซ็นต์",
    "°C": "องศาเซลเซียส",
    "°F": "องศาฟาเรนไฮต์",
    "°": "องศา",
    "@": " แอท ",
    "/": " ทับ ",
}

#exclude letter-by-letter readings as it's already handled by other rules
#exclude single-letter abbr since can be confusing
ENGLISH_ABBREVS = {
    # Finance
    "thb": "บาท", "usd": "ดอลลาร์", "eur": "ยูโร",
    "vat": "แวต","pin": "พิน", 
    # "atm": "เอทีเอ็ม", "gdp": "จีดีพี",
    
    # Tech
    # "gps": "จีพีเอส",
    # "usb": "ยูเอสบี", "cpu": "ซีพียู", "gpu": "จีพียู", 
    # "sms": "เอสเอ็มเอส","qr": "คิวอาร์", "ai": "เอไอ", 
    "ram": "แรม", "wifi": "ไวไฟ", "otp": "โอทีพี",
    # "llm": "แอว แอว เอ็ม",
    # "it": "ไอที", "id": "ไอดี",
    # "url": "ยูอาร์แอล", "pdf": "พีดีเอฟ",
    # "tv": "ทีวี", "vip": "วีไอพี",

    # Health
    "covid": "โควิด", 
    # "icu": "ไอซียู", "opd": "โอพีดี",

    # Orgs
    # "un": "ยูเอ็น", "who": "ดับเบิลยูเอชโอ",
    # "nba": "เอ็นบีเอ", 
    "fifa": "ฟีฟ่า",

    # Brands / internet words commonly read as words, not letters
    "line": "ไลน์", "facebook": "เฟซบุ๊ก", "instagram": "อินสตาแกรม",
    "amazon":"อมาซอน", "twitter": "ทวิตเตอร์",
    "google": "กูเกิล", "youtube": "ยูทูบ", "tiktok": "ติ๊กต็อก",
    "gmail": "จีเมล", "hotmail": "ฮอตเมล",
    "email": "อีเมล", "com": "คอม", "net": "เน็ต",
    "app": "แอป",
    "lazada": "ลาซาด้า",
    "shopee": "ช้อปปี้",
    "grab": "แกร็บ",
    "uber": "อูเบอร์",
    "whatsapp": "วอทส์แอป",
    "paypal": "เพย์พาล",
    "promptpay": "พร้อมเพย์",
    "truemoney": "ทรูมันนี่",
}

#also comment out single-letter units even though there is regex lookahead
#to prevent things like 214/12 ม.10 becoming 214/12 เมตร 10
UNITS = {
    "km": "กิโลเมตร", 
    "cm": "เซนติเมตร", 
    "mm": "มิลลิเมตร", 
    # "m": "เมตร",
    "ml": "มิลลิลิตร", 
    "kwh": "กิโลวัตต์ชั่วโมง", 
    "kw": "กิโลวัตต์", 
    # "w": "วัตต์",
    "mb": "เมกะไบต์", "gb": "กิกะไบต์", "tb": "เทราไบต์", "kb": "กิโลไบต์",
    "mbps": "เมกะบิตต่อวินาที",
    ## Length
    "กม.": "กิโลเมตร",
    # "ม.": "เมตร",
    "ซม.": "เซนติเมตร",
    "มม.": "มิลลิเมตร",
    # "ว.": "วา",
    ## Area
    "ตร.กม.": "ตารางกิโลเมตร",
    "ตร.ม.": "ตารางเมตร",
    "ตร.ซม.": "ตารางเซนติเมตร",
    "ตร.มม.": "ตารางมิลลิเมตร",
    "ตร.ว.": "ตารางวา",
    ## Volume
    "ลบ.ม.": "ลูกบาศก์เมตร",
    "ลบ.ซม.": "ลูกบาศก์เซนติเมตร",
    # "ล.": "ลิตร",
    "มล.": "มิลลิลิตร",
    "กล.": "กิโลลิตร",
    ## Weight
    # "ต.": "เมตริกตัน",
    "กก.": "กิโลกรัม",
    # "ก.": "กรัม",
    "มก.": "มิลลิกรัม",
    "kg": "กิโลกรัม", 
    # "g": "กรัม", 
    "mg": "มิลลิกรัม",
    # "t": "เมตริกตัน",
}

#replace longest symbol first
_SYMBOL_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(SYMBOLS.keys(), key=len, reverse=True))
)
#replace longest abbreviation first
_ENGLISH_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ENGLISH_ABBREVS.keys(), key=len, reverse=True)) + r")",
    re.IGNORECASE
)
#replace longest unit first
_UNIT_PATTERN = re.compile(
    r"(?<=\d)\s*(" + "|".join(re.escape(k) for k in sorted(UNITS.keys(), key=len, reverse=True)) + r")"
)

_THAI_NUMERAL_PATTERN = re.compile(r"[๐-๙]+")
# optional trailing นาฬิกา / น. is consumed so "8:00 น." doesn't duplicate it
_TIME_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s*(?:นาฬิกา|น\.))?")
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_PATTERN = re.compile(r"\d{2,4}[-\.]\d{3,4}[-\.]\d{3,4}")
_ALPHANUM_ID_PATTERN = re.compile(
    r"[A-Za-z]+[\-]?\d+[\-\dA-Za-z]*|\d+[\-]?[A-Za-z]+[\-\dA-Za-z]*"
)
_COMMA_NUMBER_PATTERN = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
#replace longest abbreviation first
_ABBREV_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(ABBREVIATIONS.keys(), key=len, reverse=True))
)
_LATIN_RESIDUE_PATTERN = re.compile(r"[A-Za-z]+")


def _number_group_to_thai(n: int) -> str:
    if n == 0:
        return ""
    parts = []
    remaining = n
    for place in [100000, 10000, 1000, 100, 10, 1]:
        digit = remaining // place
        remaining = remaining % place
        if digit == 0:
            continue
        if place == 1:
            if n > 1 and digit == 1:
                parts.append("เอ็ด")
            else:
                parts.append(THAI_DIGITS[str(digit)])
        elif place == 10:
            if digit == 1:
                parts.append("สิบ")
            elif digit == 2:
                parts.append("ยี่สิบ")
            else:
                parts.append(THAI_DIGITS[str(digit)] + "สิบ")
        elif place == 100:
            parts.append(THAI_DIGITS[str(digit)] + "ร้อย")
        elif place == 1000:
            parts.append(THAI_DIGITS[str(digit)] + "พัน")
        elif place == 10000:
            parts.append(THAI_DIGITS[str(digit)] + "หมื่น")
        elif place == 100000:
            parts.append(THAI_DIGITS[str(digit)] + "แสน")
    return "".join(parts)


def _integer_to_thai(n: int) -> str:
    if n == 0:
        return "ศูนย์"
    if n < 0:
        return "ลบ" + _integer_to_thai(-n)
    parts = []
    million_count = 0
    while n > 0:
        group = n % 1000000
        n = n // 1000000
        if group > 0:
            group_text = _number_group_to_thai(group)
            suffix = "ล้าน" * million_count
            parts.append(group_text + suffix)
        million_count += 1
    return "".join(reversed(parts))


def _decimal_to_thai(text: str) -> str:
    if "." in text:
        integer_part, decimal_part = text.split(".", 1)
        integer_part = integer_part or "0"
        int_thai = _integer_to_thai(int(integer_part))
        dec_thai = "".join(THAI_DIGITS[d] for d in decimal_part)
        return int_thai + "จุด" + dec_thai
    return _integer_to_thai(int(text))


def _digits_to_thai(digits: str) -> str:
    return "".join(THAI_DIGITS[d] for d in digits)


_EMAIL_SEPARATORS = {"@": " แอท ", ".": " ดอท ", "-": " ขีด ", "_": " ขีดล่าง "}
# words read as words in an address; anything else is spelled letter-by-letter
_EMAIL_WORDS = {
    "gmail": "จีเมล", "hotmail": "ฮอตเมล", "yahoo": "ยาฮู",
    "outlook": "เอาต์ลุก", "com": "คอม", "net": "เน็ต", "org": "ออร์ก",
    # "co": "ซีโอ", "th": "ทีเอช", 
    "mail": "เมล", "email": "อีเมล",
}


def _email_to_thai(match: re.Match) -> str:
    """somchai.w@gmail.com → Thai readout with แอท/ดอท. Runs before
    abbreviation expansion so dots can't false-match Thai abbreviations."""
    out = []
    token = ""

    def flush():
        nonlocal token
        if token:
            out.append(
                _EMAIL_WORDS.get(token.lower())
                or "".join(
                    THAI_DIGITS[c] if c.isdigit() else LETTER_TO_THAI.get(c.upper(), c)
                    for c in token
                )
            )
            token = ""

    for char in match.group(0):
        if char in _EMAIL_SEPARATORS:
            flush()
            out.append(_EMAIL_SEPARATORS[char])
        else:
            token += char
    flush()
    return "".join(out)


def _time_to_thai(match: re.Match) -> str:
    hours, minutes = int(match.group(1)), int(match.group(2))
    result = _integer_to_thai(hours) + "นาฬิกา"
    if minutes:
        result += _integer_to_thai(minutes) + "นาที"
    return result


def _phone_number_to_thai(match: re.Match) -> str:
    phone = match.group(0)
    parts = re.split(r"[-.]", phone)
    return " ".join(_digits_to_thai(p) for p in parts)


def _alphanum_to_thai(match: re.Match) -> str:
    token = match.group(0)
    result = []
    for char in token:
        if char.isdigit():
            result.append(THAI_DIGITS[char])
        elif char.upper() in LETTER_TO_THAI:
            result.append(LETTER_TO_THAI[char.upper()])
        elif char == "-":
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)


def _strip_commas_and_convert(match: re.Match) -> str:
    return _decimal_to_thai(match.group(0).replace(",", ""))


def _number_or_id_to_thai(match: re.Match) -> str:
    text = match.group(0)
    raw = text.lstrip("-")
    negative = text.startswith("-")

    if "." in raw:
        integer_part, _ = raw.split(".", 1)
    else:
        integer_part = raw

    if len(integer_part) >= 7:
        result = _digits_to_thai(raw.replace(".", "จุด").replace("-", ""))
        if "." in raw:
            int_p, dec_p = raw.split(".", 1)
            result = _digits_to_thai(int_p) + "จุด" + _digits_to_thai(dec_p)
        else:
            result = _digits_to_thai(integer_part)
        if negative:
            result = "ลบ" + result
        return result

    return _decimal_to_thai(text)


def _expand_abbreviations(text: str) -> str:
    return _ABBREV_PATTERN.sub(lambda m: ABBREVIATIONS[m.group(0)], text)


def _spell_latin_residue(text: str) -> str:
    """Spell out any remaining Latin letters Thai-style (A → เอ, B → บี).

    Runs last, so known words/abbreviations have already been expanded;
    what's left is codes like "AA", "SAVE", or plan names like "แผน A".
    """
    return _LATIN_RESIDUE_PATTERN.sub(
        lambda m: "".join(LETTER_TO_THAI[c.upper()] for c in m.group(0)), text
    )


def _expand_symbols(text: str) -> str:
    return _SYMBOL_PATTERN.sub(lambda m: SYMBOLS[m.group(0)], text)


def _expand_english_abbrevs(text: str) -> str:
    return _ENGLISH_ABBREV_PATTERN.sub(lambda m: ENGLISH_ABBREVS[m.group(0).lower()], text)


def _expand_units(text: str) -> str:
    return _UNIT_PATTERN.sub(lambda m: UNITS[m.group(1)], text)


def _expand_maiyamok(text: str) -> str:
    if "ๆ" not in text:
        return text
    return "".join(_expand_maiyamok_pythainlp(text))


def _thai_numerals_to_arabic(text: str) -> str:
    for thai, arabic in THAI_NUMERAL_MAP.items():
        text = text.replace(thai, arabic)
    return text


def normalize(text: str) -> str:
    """Normalize text for TTS: convert numbers, abbreviations, and special
    characters to speakable Thai words."""
    text = _expand_maiyamok(text)
    text = _thai_numerals_to_arabic(text)
    text = _EMAIL_PATTERN.sub(_email_to_thai, text)
    text = _expand_english_abbrevs(text)
    text = _expand_units(text)
    text = _expand_symbols(text)
    text = _TIME_PATTERN.sub(_time_to_thai, text)
    text = _PHONE_PATTERN.sub(_phone_number_to_thai, text)
    text = _ALPHANUM_ID_PATTERN.sub(_alphanum_to_thai, text)
    text = _COMMA_NUMBER_PATTERN.sub(_strip_commas_and_convert, text)
    text = _NUMBER_PATTERN.sub(_number_or_id_to_thai, text)
    text = _expand_abbreviations(text)
    text = _spell_latin_residue(text)
    # leftover hyphens (ชื่อ-นามสกุล) — connectors, not speakable tokens
    text = text.replace("-", " ")
    return text
