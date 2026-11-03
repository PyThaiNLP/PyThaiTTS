# -*- coding: utf-8 -*-
"""English-to-Thai-IPA Grapheme-to-Phoneme (G2P) for Thai-style English TTS.

Converts English words into English IPA and maps English IPA phonemes into
Wiktionary-convention Thai IPA phonemes (/syl.la.ble˧/) so that Thai TTS models
(such as FastThaiG2P + Kokoro) pronounce English code-switched text naturally
in Thai style (Thanglish).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Built-in Dictionary of Common & Irregular English Words -> English IPA
# ---------------------------------------------------------------------------
_EN_IPA_DICT: Dict[str, str] = {
    # Pronouns, articles, prepositions, conjunctions, auxiliaries
    "a": "eɪ",
    "an": "æn",
    "the": "ðə",
    "i": "aɪ",
    "me": "miː",
    "my": "maɪ",
    "mine": "maɪn",
    "you": "juː",
    "your": "jɔːɹ",
    "yours": "jɔːɹz",
    "he": "hiː",
    "him": "hɪm",
    "his": "hɪz",
    "she": "ʃiː",
    "her": "hɜːɹ",
    "hers": "hɜːɹz",
    "it": "ɪt",
    "its": "ɪts",
    "we": "wiː",
    "us": "ʌs",
    "our": "aʊ.ɚ",
    "ours": "aʊ.ɚz",
    "they": "ðeɪ",
    "them": "ðɛm",
    "their": "ðɛəɹ",
    "theirs": "ðɛəɹz",
    "this": "ðɪs",
    "that": "ðæt",
    "these": "ðiːz",
    "those": "ðoʊz",
    "who": "huː",
    "whom": "huːm",
    "whose": "huːz",
    "what": "wɒt",
    "which": "wɪʧ",
    "where": "wɛəɹ",
    "when": "wɛn",
    "why": "waɪ",
    "how": "haʊ",
    "am": "æm",
    "is": "ɪz",
    "are": "ɑːɹ",
    "was": "wɒz",
    "were": "wɜːɹ",
    "be": "biː",
    "been": "biːn",
    "being": "ˈbiː.ɪŋ",
    "have": "hæv",
    "has": "hæz",
    "had": "hæd",
    "having": "ˈhæv.ɪŋ",
    "do": "duː",
    "does": "dʌz",
    "did": "dɪd",
    "done": "dʌn",
    "doing": "ˈduː.ɪŋ",
    "will": "wɪl",
    "would": "wʊd",
    "shall": "ʃæl",
    "should": "ʃʊd",
    "can": "kæn",
    "cannot": "ˈkæn.ɒt",
    "could": "kʊd",
    "may": "meɪ",
    "might": "maɪt",
    "must": "mʌst",
    "in": "ɪn",
    "on": "ɒn",
    "at": "æt",
    "to": "tuː",
    "for": "fɔːɹ",
    "of": "ɒv",
    "off": "ɒf",
    "from": "fɹɒm",
    "with": "wɪθ",
    "without": "wɪˈðaʊt",
    "by": "baɪ",
    "about": "əˈbaʊt",
    "into": "ˈɪn.tuː",
    "onto": "ˈɒn.tuː",
    "over": "ˈoʊ.vɚ",
    "under": "ˈʌn.dɚ",
    "after": "ˈɑːf.tɚ",
    "before": "bɪˈfɔːɹ",
    "between": "bɪˈtwiːn",
    "among": "əˈmʌŋ",
    "through": "θɹuː",
    "during": "ˈdjʊə.ɹɪŋ",
    "until": "ʌnˈtɪl",
    "against": "əˈɡɛnst",
    "around": "əˈɹaʊnd",
    "and": "ænd",
    "or": "ɔːɹ",
    "but": "bʌt",
    "if": "ɪf",
    "because": "bɪˈkɒz",
    "so": "soʊ",
    "than": "ðæn",
    "then": "ðɛn",
    "though": "ðoʊ",
    "although": "ɔːlˈðoʊ",
    "while": "waɪl",
    "not": "nɒt",
    "no": "noʊ",
    "yes": "jɛs",
    "yeah": "jɛə",
    "ok": "oʊˈkeɪ",
    "okay": "oʊˈkeɪ",
    "all": "ɔːl",
    "any": "ˈɛn.i",
    "some": "sʌm",
    "many": "ˈmɛn.i",
    "much": "mʌʧ",
    "more": "mɔːɹ",
    "most": "moʊst",
    "other": "ˈʌð.ɚ",
    "another": "əˈnʌð.ɚ",
    "such": "sʌʧ",
    "only": "ˈoʊn.li",
    "own": "oʊn",
    "same": "seɪm",
    "very": "ˈvɛɹ.i",
    "just": "ʤʌst",
    "now": "naʊ",
    "here": "hɪəɹ",
    "there": "ðɛəɹ",
    "always": "ˈɔːl.weɪz",
    "never": "ˈnɛv.ɚ",
    "often": "ˈɒf.ən",
    "sometimes": "ˈsʌm.taɪmz",
    "again": "əˈɡɛn",
    "already": "ɔːlˈɹɛd.i",
    "still": "stɪl",
    "also": "ˈɔːl.soʊ",
    "too": "tuː",
    "well": "wɛl",
    "even": "ˈiː.vən",
    "back": "bæk",
    "down": "daʊn",
    "up": "ʌp",
    "out": "aʊt",
    "away": "əˈweɪ",
    "today": "təˈdeɪ",
    "tomorrow": "təˈmɒɹ.oʊ",
    "yesterday": "ˈjɛs.tɚ.deɪ",
    "tonight": "təˈnaɪt",

    # Greetings & Everyday Words
    "hello": "həˈloʊ",
    "hi": "haɪ",
    "hey": "heɪ",
    "bye": "baɪ",
    "goodbye": "ɡʊdˈbaɪ",
    "welcome": "ˈwɛl.kʌm",
    "thank": "θæŋk",
    "thanks": "θæŋks",
    "please": "pliːz",
    "sorry": "ˈsɒɹ.i",
    "excuse": "ɪkˈskjuːs",
    "world": "wɜːld",
    "people": "ˈpiː.pəl",
    "person": "ˈpɜː.sən",
    "man": "mæn",
    "men": "mɛn",
    "woman": "ˈwʊm.ən",
    "women": "ˈwɪm.ɪn",
    "child": "ʧaɪld",
    "children": "ˈʧɪl.dɹən",
    "friend": "fɹɛnd",
    "friends": "fɹɛndz",
    "family": "ˈfæm.ə.li",
    "father": "ˈfɑː.ðɚ",
    "mother": "ˈmʌð.ɚ",
    "brother": "ˈbɹʌð.ɚ",
    "sister": "ˈsɪs.tɚ",
    "baby": "ˈbeɪ.bi",
    "boy": "bɔɪ",
    "girl": "ɡɜːl",
    "guy": "ɡaɪ",
    "guys": "ɡaɪz",
    "time": "taɪm",
    "day": "deɪ",
    "week": "wiːk",
    "month": "mʌnθ",
    "year": "jɪəɹ",
    "hour": "ˈaʊ.ɚ",
    "minute": "ˈmɪn.ɪt",
    "second": "ˈsɛk.ənd",
    "morning": "ˈmɔː.nɪŋ",
    "afternoon": "ˌɑːf.tɚˈnuːn",
    "evening": "ˈiːv.nɪŋ",
    "night": "naɪt",
    "weekend": "ˈwiːk.ɛnd",
    "holiday": "ˈhɒl.ɪ.deɪ",
    "birthday": "ˈbɜːθ.deɪ",
    "party": "ˈpɑː.ti",
    "life": "laɪf",
    "live": "laɪv",
    "love": "lʌv",
    "like": "laɪk",
    "good": "ɡʊd",
    "better": "ˈbɛt.ɚ",
    "best": "bɛst",
    "bad": "bæd",
    "worse": "wɜːs",
    "worst": "wɜːst",
    "great": "ɡɹeɪt",
    "nice": "naɪs",
    "cool": "kuːl",
    "fine": "faɪn",
    "happy": "ˈhæp.i",
    "sad": "sæd",
    "easy": "ˈiː.zi",
    "hard": "hɑːd",
    "fast": "fɑːst",
    "slow": "sloʊ",
    "new": "njuː",
    "old": "oʊld",
    "young": "jʌŋ",
    "big": "bɪɡ",
    "small": "smɔːl",
    "large": "lɑːʤ",
    "little": "ˈlɪt.əl",
    "long": "lɒŋ",
    "short": "ʃɔːt",
    "high": "haɪ",
    "low": "loʊ",
    "hot": "hɒt",
    "cold": "koʊld",
    "warm": "wɔːm",
    "full": "fʊl",
    "empty": "ˈɛmp.ti",
    "free": "fɹiː",
    "busy": "ˈbɪz.i",
    "ready": "ˈɹɛd.i",
    "sure": "ʃʊəɹ",
    "real": "ɹiːl",
    "really": "ˈɹiː.ə.li",
    "true": "tɹuː",
    "false": "fɔːls",
    "right": "ɹaɪt",
    "wrong": "ɹɒŋ",
    "left": "lɛft",
    "first": "fɜːst",
    "last": "lɑːst",
    "next": "nɛkst",
    "One": "wʌn",
    "one": "wʌn",
    "two": "tuː",
    "three": "θɹiː",
    "four": "fɔːɹ",
    "five": "faɪv",
    "six": "sɪks",
    "seven": "ˈsɛv.ən",
    "eight": "eɪt",
    "nine": "naɪn",
    "ten": "tɛn",
    "zero": "ˈzɪə.ɹoʊ",
    "hundred": "ˈhʌn.dɹəd",
    "thousand": "ˈθaʊ.zənd",
    "million": "ˈmɪl.jən",
    "billion": "ˈbɪl.jən",
    "half": "hɑːf",
    "once": "wʌns",
    "twice": "twaɪs",

    # Common Verbs (especially irregular / silent-letter ones)
    "go": "ɡoʊ",
    "going": "ˈɡoʊ.ɪŋ",
    "gone": "ɡɒn",
    "went": "wɛnt",
    "come": "kʌm",
    "coming": "ˈkʌm.ɪŋ",
    "came": "keɪm",
    "get": "ɡɛt",
    "getting": "ˈɡɛt.ɪŋ",
    "got": "ɡɒt",
    "make": "meɪk",
    "making": "ˈmeɪ.kɪŋ",
    "made": "meɪd",
    "know": "noʊ",
    "known": "noʊn",
    "knew": "njuː",
    "think": "θɪŋk",
    "thought": "θɔːt",
    "thinking": "ˈθɪŋ.kɪŋ",
    "take": "teɪk",
    "taking": "ˈteɪ.kɪŋ",
    "took": "tʊk",
    "taken": "ˈteɪ.kən",
    "see": "siː",
    "saw": "sɔː",
    "seen": "siːn",
    "look": "lʊk",
    "looking": "ˈlʊk.ɪŋ",
    "want": "wɒnt",
    "give": "ɡɪv",
    "given": "ˈɡɪv.ən",
    "gave": "ɡeɪv",
    "use": "juːz",
    "using": "ˈjuː.zɪŋ",
    "used": "juːzd",
    "find": "faɪnd",
    "found": "faʊnd",
    "tell": "tɛl",
    "told": "toʊld",
    "ask": "ɑːsk",
    "work": "wɜːk",
    "working": "ˈwɜː.kɪŋ",
    "call": "kɔːl",
    "calling": "ˈkɔː.lɪŋ",
    "try": "tɹaɪ",
    "trying": "ˈtɹaɪ.ɪŋ",
    "need": "niːd",
    "feel": "fiːl",
    "feeling": "ˈfiː.lɪŋ",
    "become": "bɪˈkʌm",
    "leave": "liːv",
    "left": "lɛft",
    "put": "pʊt",
    "mean": "miːn",
    "keep": "kiːp",
    "let": "lɛt",
    "begin": "bɪˈɡɪn",
    "seem": "siːm",
    "help": "hɛlp",
    "talk": "tɔːk",
    "talking": "ˈtɔː.kɪŋ",
    "walk": "wɔːk",
    "walking": "ˈwɔː.kɪŋ",
    "turn": "tɜːn",
    "start": "stɑːt",
    "starting": "ˈstɑː.tɪŋ",
    "stop": "stɒp",
    "show": "ʃoʊ",
    "hear": "hɪəɹ",
    "heard": "hɜːd",
    "play": "pleɪ",
    "playing": "ˈpleɪ.ɪŋ",
    "run": "ɹʌn",
    "running": "ˈɹʌn.ɪŋ",
    "move": "muːv",
    "moving": "ˈmuː.vɪŋ",
    "believe": "bɪˈliːv",
    "hold": "hoʊld",
    "bring": "bɹɪŋ",
    "brought": "bɹɔːt",
    "happen": "ˈhæp.ən",
    "write": "ɹaɪt",
    "writing": "ˈɹaɪ.tɪŋ",
    "written": "ˈɹɪt.ən",
    "wrote": "ɹoʊt",
    "provide": "pɹəˈvaɪd",
    "sit": "sɪt",
    "stand": "stænd",
    "lose": "luːz",
    "lost": "lɒst",
    "pay": "peɪ",
    "paid": "peɪd",
    "meet": "miːt",
    "meeting": "ˈmiː.tɪŋ",
    "include": "ɪnˈkluːd",
    "continue": "kənˈtɪn.juː",
    "set": "sɛt",
    "setting": "ˈsɛt.ɪŋ",
    "settings": "ˈsɛt.ɪŋz",
    "learn": "lɜːn",
    "learning": "ˈlɜː.nɪŋ",
    "change": "ʧeɪnʤ",
    "lead": "liːd",
    "leader": "ˈliː.dɚ",
    "understand": "ˌʌn.dɚˈstænd",
    "watch": "wɒʧ",
    "follow": "ˈfɒl.oʊ",
    "following": "ˈfɒl.oʊ.ɪŋ",
    "follower": "ˈfɒl.oʊ.ɚ",
    "create": "kɹiˈeɪt",
    "creator": "kɹiˈeɪ.tɚ",
    "speak": "spiːk",
    "speaking": "ˈspiː.kɪŋ",
    "speaker": "ˈspiː.kɚ",
    "read": "ɹiːd",
    "reading": "ˈɹiː.dɪŋ",
    "allow": "əˈlaʊ",
    "add": "æd",
    "spend": "spɛnd",
    "grow": "ɡɹoʊ",
    "open": "ˈoʊ.pən",
    "close": "kloʊz",
    "win": "wɪn",
    "offer": "ˈɒf.ɚ",
    "remember": "ɹɪˈmɛm.bɚ",
    "consider": "kənˈsɪd.ɚ",
    "appear": "əˈpɪəɹ",
    "buy": "baɪ",
    "bought": "bɔːt",
    "wait": "weɪt",
    "waiting": "ˈweɪ.tɪŋ",
    "serve": "sɜːv",
    "die": "daɪ",
    "send": "sɛnd",
    "expect": "ɪkˈspɛkt",
    "build": "bɪld",
    "built": "bɪlt",
    "building": "ˈbɪl.dɪŋ",
    "stay": "steɪ",
    "fall": "fɔːl",
    "cut": "kʌt",
    "reach": "ɹiːʧ",
    "kill": "kɪl",
    "remain": "ɹɪˈmeɪn",
    "suggest": "səˈʤɛst",
    "raise": "ɹeɪz",
    "pass": "pɑːs",
    "sell": "sɛl",
    "sale": "seɪl",
    "sales": "seɪlz",
    "require": "ɹɪˈkwaɪ.ɚ",
    "report": "ɹɪˈpɔːt",
    "decide": "dɪˈsaɪd",
    "pull": "pʊl",
    "push": "pʊʃ",
    "return": "ɹɪˈtɜːn",
    "explain": "ɪkˈspleɪn",
    "hope": "hoʊp",
    "develop": "dɪˈvɛl.əp",
    "developer": "dɪˈvɛl.ə.pɚ",
    "development": "dɪˈvɛl.əp.mənt",
    "carry": "ˈkæɹ.i",
    "break": "bɹeɪk",
    "receive": "ɹɪˈsiːv",
    "agree": "əˈɡɹiː",
    "support": "səˈpɔːt",
    "hit": "hɪt",
    "produce": "pɹəˈdjuːs",
    "eat": "iːt",
    "cover": "ˈkʌv.ɚ",
    "catch": "kæʧ",
    "draw": "dɹɔː",
    "choose": "ʧuːz",

    # Technology, Computing, Business & Daily Loanwords in Thai
    "computer": "kəmˈpjuː.tɚ",
    "internet": "ˈɪn.tɚ.nɛt",
    "website": "ˈwɛb.saɪt",
    "web": "wɛb",
    "site": "saɪt",
    "online": "ˈɒnˈlaɪn",
    "offline": "ˈɒfˈlaɪn",
    "software": "ˈsɒft.wɛəɹ",
    "hardware": "ˈhɑːd.wɛəɹ",
    "network": "ˈnɛt.wɜːk",
    "system": "ˈsɪs.təm",
    "server": "ˈsɜː.vɚ",
    "service": "ˈsɜː.vɪs",
    "client": "ˈklaɪ.ənt",
    "user": "ˈjuː.zɚ",
    "admin": "ˈæd.mɪn",
    "login": "ˈlɒɡ.ɪn",
    "logout": "ˈlɒɡ.aʊt",
    "password": "ˈpɑːs.wɜːd",
    "account": "əˈkaʊnt",
    "profile": "ˈpɹoʊ.faɪl",
    "data": "ˈdeɪ.tə",
    "database": "ˈdeɪ.tə.beɪs",
    "cloud": "klaʊd",
    "file": "faɪl",
    "files": "faɪlz",
    "folder": "ˈfoʊl.dɚ",
    "upload": "ˈʌp.loʊd",
    "download": "ˈdaʊn.loʊd",
    "update": "ˈʌp.deɪt",
    "upgrade": "ˈʌp.ɡɹeɪd",
    "install": "ɪnˈstɔːl",
    "delete": "dɪˈliːt",
    "remove": "ɹɪˈmuːv",
    "search": "sɜːʧ",
    "filter": "ˈfɪl.tɚ",
    "sort": "sɔːt",
    "click": "klɪk",
    "button": "ˈbʌt.ən",
    "screen": "skɹiːn",
    "display": "dɪˈspleɪ",
    "monitor": "ˈmɒn.ɪ.tɚ",
    "keyboard": "ˈkiː.bɔːd",
    "mouse": "maʊs",
    "printer": "ˈpɹɪn.tɚ",
    "scanner": "ˈskæn.ɚ",
    "camera": "ˈkæm.ɹə",
    "video": "ˈvɪd.i.oʊ",
    "audio": "ˈɔː.di.oʊ",
    "radio": "ˈɹeɪ.di.oʊ",
    "media": "ˈmiː.di.ə",
    "social": "ˈsoʊ.ʃəl",
    "photo": "ˈfoʊ.toʊ",
    "image": "ˈɪm.ɪʤ",
    "picture": "ˈpɪk.ʧɚ",
    "graphic": "ˈɡɹæf.ɪk",
    "graph": "ɡɹɑːf",
    "chart": "ʧɑːt",
    "table": "ˈteɪ.bəl",
    "list": "lɪst",
    "item": "ˈaɪ.təm",
    "detail": "ˈdiː.teɪl",
    "details": "ˈdiː.teɪlz",
    "info": "ˈɪn.foʊ",
    "information": "ˌɪn.fɚˈmeɪ.ʃən",
    "document": "ˈdɒk.jʊ.mənt",
    "format": "ˈfɔː.mæt",
    "text": "tɛkst",
    "font": "fɒnt",
    "style": "staɪl",
    "color": "ˈkʌl.ɚ",
    "colour": "ˈkʌl.ɚ",
    "design": "dɪˈzaɪn",
    "designer": "dɪˈzaɪ.nɚ",
    "layout": "ˈleɪ.aʊt",
    "template": "ˈtɛm.pleɪt",
    "page": "peɪʤ",
    "view": "vjuː",
    "model": "ˈmɒd.əl",
    "train": "tɹeɪn",
    "training": "ˈtɹeɪ.nɪŋ",
    "test": "tɛst",
    "testing": "ˈtɛs.tɪŋ",
    "code": "koʊd",
    "coding": "ˈkoʊ.dɪŋ",
    "script": "skɹɪpt",
    "debug": "diːˈbʌɡ",
    "error": "ˈɛɹ.ɚ",
    "bug": "bʌɡ",
    "issue": "ˈɪʃ.uː",
    "feature": "ˈfiː.ʧɚ",
    "version": "ˈvɜː.ʃən",
    "release": "ɹɪˈliːs",
    "deploy": "dɪˈplɔɪ",
    "package": "ˈpæk.ɪʤ",
    "module": "ˈmɒd.juːl",
    "library": "ˈlaɪ.bɹə.ɹi",
    "framework": "ˈfɹeɪm.wɜːk",
    "platform": "ˈplæt.fɔːm",
    "application": "ˌæp.lɪˈkeɪ.ʃən",
    "program": "ˈpɹoʊ.ɡɹæm",
    "programmer": "ˈpɹoʊ.ɡɹæm.ɚ",
    "project": "ˈpɹoʊ.ʤɛkt",
    "product": "ˈpɹoʊ.dʌkt",
    "production": "pɹoʊˈdʌk.ʃən",
    "process": "ˈpɹoʊ.sɛs",
    "status": "ˈsteɪ.təs",
    "option": "ˈɒp.ʃən",
    "menu": "ˈmɛn.juː",
    "home": "hoʊm",
    "link": "lɪŋk",
    "url": "ˌjuː.ɑːɹˈɛl",
    "browser": "ˈbɹaʊ.zɚ",
    "engine": "ˈɛn.ʤɪn",
    "engineer": "ˌɛn.ʤɪˈnɪəɹ",
    "engineering": "ˌɛn.ʤɪˈnɪə.ɹɪŋ",
    "technology": "tɛkˈnɒl.ə.ʤi",
    "tech": "tɛk",
    "digital": "ˈdɪʤ.ɪ.təl",
    "mobile": "ˈmoʊ.baɪl",
    "phone": "foʊn",
    "smartphone": "ˈsmɑːt.foʊn",
    "device": "dɪˈvaɪs",
    "battery": "ˈbæt.ɚ.i",
    "power": "ˈpaʊ.ɚ",
    "energy": "ˈɛn.ɚ.ʤi",
    "charge": "ʧɑːʤ",
    "charger": "ˈʧɑː.ʤɚ",
    "cable": "ˈkeɪ.bəl",
    "adapter": "əˈdæp.tɚ",
    "sensor": "ˈsɛn.sɚ",
    "signal": "ˈsɪɡ.nəl",
    "wireless": "ˈwaɪə.ləs",
    "bluetooth": "ˈbluː.tuːθ",
    "channel": "ˈʧæn.əl",
    "stream": "stɹiːm",
    "streaming": "ˈstɹiː.mɪŋ",
    "podcast": "ˈpɒd.kɑːst",
    "content": "ˈkɒn.tɛnt",
    "comment": "ˈkɒm.ɛnt",
    "share": "ʃɛəɹ",
    "post": "poʊst",
    "story": "ˈstɔː.ɹi",
    "feed": "fiːd",
    "trend": "tɹɛnd",
    "trending": "ˈtɹɛn.dɪŋ",
    "viral": "ˈvaɪ.ɹəl",
    "brand": "bɹænd",
    "market": "ˈmɑː.kɪt",
    "marketing": "ˈmɑː.kɪ.tɪŋ",
    "business": "ˈbɪz.nəs",
    "company": "ˈkʌm.pə.ni",
    "office": "ˈɒf.ɪs",
    "team": "tiːm",
    "group": "ɡɹuːp",
    "member": "ˈmɛm.bɚ",
    "manager": "ˈmæn.ɪ.ʤɚ",
    "management": "ˈmæn.ɪʤ.mənt",
    "director": "dɪˈɹɛk.tɚ",
    "customer": "ˈkʌs.tə.mɚ",
    "partner": "ˈpɑːt.nɚ",
    "center": "ˈsɛn.tɚ",
    "centre": "ˈsɛn.tɚ",
    "department": "dɪˈpɑːt.mənt",
    "target": "ˈtɑː.ɡɪt",
    "budget": "ˈbʌʤ.ɪt",
    "profit": "ˈpɹɒf.ɪt",
    "cost": "kɒst",
    "price": "pɹaɪs",
    "order": "ˈɔː.dɚ",
    "payment": "ˈpeɪ.mənt",
    "discount": "ˈdɪs.kaʊnt",
    "promotion": "pɹəˈmoʊ.ʃən",
    "coupon": "ˈkuː.pɒn",
    "voucher": "ˈvaʊ.ʧɚ",
    "point": "pɔɪnt",
    "reward": "ɹɪˈwɔːd",
    "bonus": "ˈboʊ.nəs",
    "cash": "kæʃ",
    "credit": "ˈkɹɛd.ɪt",
    "debit": "ˈdɛb.ɪt",
    "card": "kɑːd",
    "bank": "bæŋk",
    "banking": "ˈbæŋ.kɪŋ",
    "transfer": "ˈtɹæns.fɜːɹ",
    "wallet": "ˈwɒl.ɪt",
    "money": "ˈmʌn.i",
    "tax": "tæks",
    "invoice": "ˈɪn.vɔɪs",
    "receipt": "ɹɪˈsiːt",
    "summary": "ˈsʌm.ɚ.i",
    "analysis": "əˈnæl.ə.sɪs",
    "analytics": "ˌæn.əˈlɪt.ɪks",
    "dashboard": "ˈdæʃ.bɔːd",
    "schedule": "ˈskɛʤ.uːl",
    "calendar": "ˈkæl.ɪn.dɚ",
    "event": "ɪˈvɛnt",
    "workshop": "ˈwɜːk.ʃɒp",
    "seminar": "ˈsɛm.ɪ.nɑːɹ",
    "course": "kɔːs",
    "class": "klɑːs",
    "lesson": "ˈlɛs.ən",
    "student": "ˈstjuː.dənt",
    "teacher": "ˈtiː.ʧɚ",
    "school": "skuːl",
    "college": "ˈkɒl.ɪʤ",
    "university": "ˌjuː.nɪˈvɜː.sɪ.ti",
    "science": "ˈsaɪ.əns",
    "math": "mæθ",
    "medical": "ˈmɛd.ɪ.kəl",
    "health": "hɛlθ",
    "hospital": "ˈhɒs.pɪ.təl",
    "doctor": "ˈdɒk.tɚ",
    "nurse": "nɜːs",
    "clinic": "ˈklɪn.ɪk",
    "vaccine": "ˈvæk.siːn",
    "virus": "ˈvaɪ.ɹəs",
    "vitamin": "ˈvaɪ.tə.mɪn",
    "protein": "ˈpɹoʊ.tiːn",
    "hotel": "hoʊˈtɛl",
    "booking": "ˈbʊk.ɪŋ",
    "ticket": "ˈtɪk.ɪt",
    "flight": "flaɪt",
    "travel": "ˈtɹæv.əl",
    "tour": "tʊəɹ",
    "trip": "tɹɪp",
    "guide": "ɡaɪd",
    "map": "mæp",
    "route": "ɹuːt",
    "location": "loʊˈkeɪ.ʃən",
    "address": "ˈæd.ɹɛs",
    "contact": "ˈkɒn.tækt",
    "message": "ˈmɛs.ɪʤ",
    "chat": "ʧæt",
    "voice": "vɔɪs",
    "music": "ˈmjuː.zɪk",
    "song": "sɒŋ",
    "movie": "ˈmuː.vi",
    "film": "fɪlm",
    "game": "ɡeɪm",
    "gaming": "ˈɡeɪ.mɪŋ",
    "gamer": "ˈɡeɪ.mɚ",
    "sport": "spɔːt",
    "football": "ˈfʊt.bɔːl",
    "golf": "ɡɒlf",
    "tennis": "ˈtɛn.ɪs",
    "coffee": "ˈkɒf.i",
    "tea": "tiː",
    "water": "ˈwɔː.tɚ",
    "milk": "mɪlk",
    "juice": "ʤuːs",
    "beer": "bɪəɹ",
    "wine": "waɪn",
    "food": "fuːd",
    "menu": "ˈmɛn.juː",
    "restaurant": "ˈɹɛs.tɹɒnt",
    "cafe": "ˈkæf.eɪ",
    "shop": "ʃɒp",
    "shopping": "ˈʃɒp.ɪŋ",
    "store": "stɔːɹ",
    "mall": "mɔːl",
    "supermarket": "ˈsuː.pɚ.mɑː.kɪt",
    "car": "kɑːɹ",
    "bus": "bʌs",
    "taxi": "ˈtæk.si",
    "station": "ˈsteɪ.ʃən",
    "airport": "ˈɛə.pɔːt",
    "room": "ɹuːm",
    "house": "haʊs",
    "condo": "ˈkɒn.doʊ",
    "city": "ˈsɪt.i",
    "country": "ˈkʌn.tɹi",
    "world": "wɜːld",
    "nature": "ˈneɪ.ʧɚ",
    "future": "ˈfjuː.ʧɚ",
    "culture": "ˈkʌl.ʧɚ",
    "action": "ˈæk.ʃən",
    "section": "ˈsɛk.ʃən",
    "function": "ˈfʌŋk.ʃən",
    "question": "ˈkwɛs.ʧən",
    "answer": "ˈɑːn.sɚ",
    "problem": "ˈpɹɒb.ləm",
    "solution": "səˈluː.ʃən",
    "idea": "aɪˈdiː.ə",
    "concept": "ˈkɒn.sɛpt",
    "topic": "ˈtɒp.ɪk",
    "example": "ɪɡˈzɑːm.pəl",
    "sample": "ˈsɑːm.pəl",
    "result": "ɹɪˈzʌlt",
    "level": "ˈlɛv.əl",
    "normal": "ˈnɔː.məl",
    "standard": "ˈstæn.dɚd",
    "special": "ˈspɛʃ.əl",
    "general": "ˈʤɛn.ɹəl",
    "public": "ˈpʌb.lɪk",
    "private": "ˈpɹaɪ.vət",
    "local": "ˈloʊ.kəl",
    "global": "ˈɡloʊ.bəl",
    "international": "ˌɪn.tɚˈnæʃ.ən.əl",
    "national": "ˈnæʃ.ən.əl",
    "personal": "ˈpɜː.sən.əl",
    "original": "əˈɹɪʤ.ɪ.nəl",
    "final": "ˈfaɪ.nəl",
    "total": "ˈtoʊ.təl",
    "percent": "pɚˈsɛnt",
    "number": "ˈnʌm.bɚ",
    "value": "ˈvæl.juː",
    "quality": "ˈkwɒl.ɪ.ti",
    "security": "sɪˈkjʊə.ɹɪ.ti",
    "community": "kəˈmjuː.nɪ.ti",
    "activity": "ækˈtɪv.ɪ.ti",
    "university": "ˌjuː.nɪˈvɜː.sɪ.ti",
    "copy": "ˈkɒp.i",

    # Programming & Popular Tech Names
    "python": "ˈpaɪ.θɒn",
    "java": "ˈʤɑː.və",
    "javascript": "ˈʤɑː.və.skɹɪpt",
    "typescript": "ˈtaɪp.skɹɪpt",
    "react": "ɹiˈækt",
    "node": "noʊd",
    "docker": "ˈdɒk.ɚ",
    "linux": "ˈlɪn.ʌks",
    "windows": "ˈwɪn.doʊz",
    "apple": "ˈæp.əl",
    "iphone": "ˈaɪ.foʊn",
    "ipad": "ˈaɪ.pæd",
    "macbook": "ˈmæk.bʊk",
    "android": "ˈæn.dɹɔɪd",
    "samsung": "ˈsæm.sʌŋ",
    "galaxy": "ˈɡæl.ək.si",
    "google": "ˈɡuː.ɡəl",
    "microsoft": "ˈmaɪ.kɹoʊ.sɒft",
    "amazon": "ˈæm.ə.zɒn",
    "facebook": "ˈfeɪs.bʊk",
    "instagram": "ˈɪn.stə.ɡɹæm",
    "youtube": "ˈjuː.tjuːb",
    "tiktok": "ˈtɪk.tɒk",
    "twitter": "ˈtwɪt.ɚ",
    "netflix": "ˈnɛt.flɪks",
    "spotify": "ˈspɒt.ɪ.faɪ",
    "realtime": "ˈɹiːl.taɪm",
    "token": "ˈtoʊ.kən",
    "tokens": "ˈtoʊ.kənz",
    "prompt": "pɹɒmpt",
    "agent": "ˈeɪ.ʤənt",
    "agents": "ˈeɪ.ʤənts",
    "workflow": "ˈwɜːk.floʊ",
    "pipeline": "ˈpaɪp.laɪn",
    "audio": "ˈɔː.di.oʊ",
    "speech": "spiːʧ",
    "synthesis": "ˈsɪn.θə.sɪs",
    "phoneme": "ˈfoʊ.niːm",
    "phonemes": "ˈfoʊ.niːmz",
}


# ---------------------------------------------------------------------------
# 2. Rule-Based English Grapheme-to-Phoneme (OOV Fallback)
# ---------------------------------------------------------------------------
_VOWEL_LETTERS = set("aeiouy")


def _is_vowel_letter(ch: str) -> bool:
    return ch in _VOWEL_LETTERS


def _rule_based_en_to_ipa(word: str) -> str:
    """Convert an out-of-vocabulary English word to approximate English IPA.

    Uses morphological prefix/suffix decomposition, silent-letter rules,
    magic-e detection, digraph/trigraph rules, and syllable-aware vowel rules.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""

    if w in _EN_IPA_DICT:
        return _EN_IPA_DICT[w]

    # Check regular plural / 3rd person -s / -es / -ies or past -ed / -ing on known stems
    if len(w) > 3:
        if w.endswith("ies") and w[:-3] + "y" in _EN_IPA_DICT:
            base_ipa = _EN_IPA_DICT[w[:-3] + "y"]
            if base_ipa.endswith("i"):
                return base_ipa + "z"
        if w.endswith("es") and w[:-2] in _EN_IPA_DICT:
            base_ipa = _EN_IPA_DICT[w[:-2]]
            return base_ipa + (".ɪz" if base_ipa[-1] in "szʃʒʧʤ" else "z")
        if w.endswith("s") and not w.endswith("ss") and w[:-1] in _EN_IPA_DICT:
            base_ipa = _EN_IPA_DICT[w[:-1]]
            return base_ipa + ("s" if base_ipa[-1] in "ptkfθ" else "z")
        if w.endswith("ing"):
            if w[:-3] in _EN_IPA_DICT:
                return _EN_IPA_DICT[w[:-3]] + ".ɪŋ"
            if w[:-3] + "e" in _EN_IPA_DICT:
                return _EN_IPA_DICT[w[:-3] + "e"] + ".ɪŋ"
            if len(w) > 5 and w[-4] == w[-5] and w[:-4] in _EN_IPA_DICT:
                return _EN_IPA_DICT[w[:-4]] + ".ɪŋ"
        if w.endswith("ed"):
            if w[:-2] in _EN_IPA_DICT:
                base_ipa = _EN_IPA_DICT[w[:-2]]
                return base_ipa + (".ɪd" if base_ipa[-1] in "td" else "d")
            if w[:-1] in _EN_IPA_DICT:
                base_ipa = _EN_IPA_DICT[w[:-1]]
                return base_ipa + (".ɪd" if base_ipa[-1] in "td" else "d")
        if w.endswith("er"):
            if w[:-2] in _EN_IPA_DICT:
                return _EN_IPA_DICT[w[:-2]] + ".ɚ"
            if w[:-1] in _EN_IPA_DICT:
                return _EN_IPA_DICT[w[:-1]] + ".ɚ"
        if w.endswith("ly") and w[:-2] in _EN_IPA_DICT:
            return _EN_IPA_DICT[w[:-2]] + ".li"

    # Silent initial letters
    if w.startswith(("kn", "gn", "pn")) and len(w) > 2:
        w = w[1:]
    elif w.startswith("wr") and len(w) > 2:
        w = w[1:]
    elif w.startswith("ps") and len(w) > 2 and w[2] in _VOWEL_LETTERS:
        w = w[1:]
    elif w.startswith("wh"):
        if w.startswith(("who", "whom", "whose", "whole")):
            w = "h" + w[2:]
        else:
            w = "w" + w[2:]

    # Silent final -mb / -mn
    if len(w) > 2 and w.endswith(("mb", "mn")):
        w = w[:-1]

    # Strip common suffixes and save their IPA tails
    suffix_ipas: List[str] = []
    _SUFFIX_RULES = [
        ("isation", "aɪ.ˈzeɪ.ʃən"),
        ("ization", "aɪ.ˈzeɪ.ʃən"),
        ("ication", "ɪ.ˈkeɪ.ʃən"),
        ("ational", "ˈeɪ.ʃən.əl"),
        ("fulness", "fəl.nəs"),
        ("lessness", "ləs.nəs"),
        ("ousness", "əs.nəs"),
        ("iveness", "ɪv.nəs"),
        ("ability", "ə.ˈbɪl.ɪ.ti"),
        ("ibility", "ɪ.ˈbɪl.ɪ.ti"),
        ("ology", "ˈɒl.ə.ʤi"),
        ("ography", "ˈɒɡ.ɹə.fi"),
        ("tion", "ʃən"),
        ("sion", "ʃən"),
        ("cian", "ʃən"),
        ("ture", "ʧɚ"),
        ("sure", "ʒɚ"),
        ("cious", "ʃəs"),
        ("tious", "ʃəs"),
        ("ious", "i.əs"),
        ("eous", "i.əs"),
        ("ment", "mənt"),
        ("ness", "nəs"),
        ("less", "ləs"),
        ("able", "ə.bəl"),
        ("ible", "ɪ.bəl"),
        ("ally", "ə.li"),
        ("ical", "ɪ.kəl"),
        ("ful", "fʊl"),
        ("ous", "əs"),
        ("ive", "ɪv"),
        ("ing", "ɪŋ"),
        ("est", "ɪst"),
        ("ist", "ɪst"),
        ("ism", "ɪ.zəm"),
        ("ize", "aɪz"),
        ("ise", "aɪz"),
        ("ify", "ɪ.faɪ"),
        ("ity", "ɪ.ti"),
    ]
    for suf, suf_ipa in _SUFFIX_RULES:
        if len(w) >= len(suf) + 2 and w.endswith(suf):
            stem = w[: -len(suf)]
            if any(c in _VOWEL_LETTERS for c in stem):
                w = stem
                suffix_ipas.insert(0, suf_ipa)
                break

    # If the remaining stem is now in _EN_IPA_DICT, use it directly
    if w in _EN_IPA_DICT:
        stem_ipa = _EN_IPA_DICT[w]
        if suffix_ipas:
            return stem_ipa + "." + ".".join(suffix_ipas)
        return stem_ipa

    # Detect magic-e at the end of the stem: ...V + C + 'e'
    magic_e = False
    if (
        len(w) >= 3
        and w.endswith("e")
        and not w.endswith(("ee", "ie", "oe", "ue", "ye", "le", "re", "ce", "ge", "se", "ve", "ze"))
        and w[-2] not in _VOWEL_LETTERS
        and w[-3] in _VOWEL_LETTERS
    ):
        magic_e = True
        w = w[:-1]
    elif (
        len(w) >= 4
        and w.endswith(("ce", "ge", "se", "ve", "ze"))
        and w[-3] in _VOWEL_LETTERS
        and (len(w) < 4 or w[-4] not in _VOWEL_LETTERS)
    ):
        magic_e = True

    # Scan left-to-right with multi-character phoneme rules
    out: List[str] = []
    i = 0
    n = len(w)

    while i < n:
        rem = w[i:]

        # 4-letter patterns
        if rem.startswith("eigh"):
            out.append("eɪ")
            i += 4
            continue
        if rem.startswith("ough"):
            out.append("oʊ")
            i += 4
            continue
        if rem.startswith("augh"):
            out.append("ɔː")
            i += 4
            continue

        # 3-letter patterns
        if rem.startswith("igh"):
            out.append("aɪ")
            i += 3
            continue
        if rem.startswith("tch"):
            out.append("ʧ")
            i += 3
            continue
        if rem.startswith("dge"):
            out.append("ʤ")
            i += 3
            continue
        if rem.startswith(("ear", "eer")):
            out.append("ɪə")
            i += 3
            continue
        if rem.startswith(("air", "are")):
            out.append("ɛə")
            i += 3
            continue
        if rem.startswith("ore"):
            out.append("ɔː")
            i += 3
            continue
        if rem.startswith("our"):
            out.append("aʊ.ɚ")
            i += 3
            continue
        if rem.startswith("ire"):
            out.append("aɪ.ɚ")
            i += 3
            continue
        if rem.startswith("ure"):
            out.append("ʊə")
            i += 3
            continue

        # 2-letter consonant digraphs
        if rem.startswith("sh"):
            out.append("ʃ")
            i += 2
            continue
        if rem.startswith("ch"):
            out.append("ʧ")
            i += 2
            continue
        if rem.startswith("ph"):
            out.append("f")
            i += 2
            continue
        if rem.startswith("th"):
            out.append("θ")
            i += 2
            continue
        if rem.startswith("ck"):
            out.append("k")
            i += 2
            continue
        if rem.startswith("qu"):
            out.append("kw")
            i += 2
            continue
        if rem.startswith("ng"):
            out.append("ŋ")
            i += 2
            continue
        if rem.startswith("nk"):
            out.append("ŋk")
            i += 2
            continue
        if rem.startswith("gh"):
            if i == 0:
                out.append("ɡ")
            i += 2
            continue

        # 2-letter vowel digraphs & r-colored vowels
        if rem.startswith(("ee", "ea")):
            out.append("iː")
            i += 2
            continue
        if rem.startswith("oo"):
            if rem.startswith(("ook", "ood", "oot")):
                out.append("ʊ")
            else:
                out.append("uː")
            i += 2
            continue
        if rem.startswith(("ai", "ay", "ei", "ey")):
            out.append("eɪ")
            i += 2
            continue
        if rem.startswith(("oi", "oy")):
            out.append("ɔɪ")
            i += 2
            continue
        if rem.startswith("ou"):
            out.append("aʊ")
            i += 2
            continue
        if rem.startswith("ow"):
            # Final -ow in multi-syllable words is usually oʊ (window, follow, yellow)
            if i + 2 == n and i > 1:
                out.append("oʊ")
            else:
                out.append("aʊ")
            i += 2
            continue
        if rem.startswith(("au", "aw")):
            out.append("ɔː")
            i += 2
            continue
        if rem.startswith(("oa", "oe")):
            out.append("oʊ")
            i += 2
            continue
        if rem.startswith(("ew", "ue", "ui")):
            out.append("uː")
            i += 2
            continue
        if rem.startswith("ie"):
            if i + 2 == n:
                out.append("aɪ" if n <= 4 else "iː")
            else:
                out.append("iː")
            i += 2
            continue

        # Vowel + r (when not followed by another vowel)
        if len(rem) >= 2 and rem[1] == "r" and (len(rem) == 2 or rem[2] not in _VOWEL_LETTERS):
            if rem[0] == "a":
                out.append("ɑː")
                i += 2
                continue
            if rem[0] == "o":
                out.append("ɔː")
                i += 2
                continue
            if rem[0] in ("e", "i", "u"):
                out.append("ɚ" if i > 0 and i + 2 == n else "ɜː")
                i += 2
                continue

        # Final -le after a consonant (apple, table, google)
        if i + 2 == n and rem == "le" and i > 0 and w[i - 1] not in _VOWEL_LETTERS:
            out.append("əl")
            i += 2
            continue

        # Double consonants (bb, dd, ff, gg, ll, mm, nn, pp, rr, ss, tt, zz)
        if len(rem) >= 2 and rem[0] == rem[1] and rem[0] not in _VOWEL_LETTERS:
            if rem.startswith("cc") and len(rem) >= 3 and rem[2] in "eiy":
                out.append("ks")
                i += 2
                continue
            # Emit single consonant and advance by 2
            ch = rem[0]
            rem = ch + rem[2:]
            i += 1
        else:
            ch = rem[0]

        # Single-letter rules
        is_last_vowel = magic_e and (i == n - 2 or (i == n - 3 and w.endswith(("ce", "ge", "se", "ve", "ze"))))

        if ch == "a":
            if is_last_vowel:
                out.append("eɪ")
            elif i + 1 == n:
                out.append("ə")
            elif rem.startswith(("all", "alk", "alt", "ald")):
                out.append("ɔː")
                if rem.startswith("alk"):
                    out.append("k")
                    i += 3
                    continue
            else:
                out.append("æ")
        elif ch == "e":
            if i + 1 == n and n > 2:
                # Silent final -e
                pass
            elif is_last_vowel or (i + 1 == n and n <= 2):
                out.append("iː")
            else:
                out.append("ɛ")
        elif ch == "i":
            if is_last_vowel or rem.startswith(("ind", "ild", "ign")):
                out.append("aɪ")
            else:
                out.append("ɪ")
        elif ch == "o":
            if is_last_vowel or i + 1 == n or rem.startswith(("old", "olt", "ost", "oll")):
                out.append("oʊ")
            else:
                out.append("ɒ")
        elif ch == "u":
            if is_last_vowel:
                out.append("juː")
            elif rem.startswith(("ull", "ush", "ut")) and w in ("put", "push", "pull", "full", "bull"):
                out.append("ʊ")
            else:
                out.append("ʌ")
        elif ch == "y":
            if i == 0:
                out.append("j")
            elif i + 1 == n:
                out.append("aɪ" if n <= 3 else "i")
            elif is_last_vowel:
                out.append("aɪ")
            else:
                out.append("ɪ")
        elif ch == "c":
            if len(rem) >= 2 and rem[1] in "eiy":
                out.append("s")
            else:
                out.append("k")
        elif ch == "g":
            if len(rem) >= 2 and rem[1] in "eiy":
                out.append("ʤ")
            else:
                out.append("ɡ")
        elif ch == "x":
            if i == 0:
                out.append("z")
            else:
                out.append("ks")
        elif ch == "s":
            if i > 0 and i + 1 < n and w[i - 1] in _VOWEL_LETTERS and w[i + 1] in _VOWEL_LETTERS:
                out.append("z")
            else:
                out.append("s")
        elif ch == "r":
            out.append("ɹ")
        elif ch == "j":
            out.append("ʤ")
        elif ch in "bdfhklmnptvwz":
            out.append(ch)

        i += 1

    stem_ipa = "".join(out)
    if suffix_ipas:
        if stem_ipa:
            return "ˈ" + stem_ipa + "." + ".".join(suffix_ipas)
        return ".".join(suffix_ipas)
    return "ˈ" + stem_ipa if stem_ipa else ""


# ---------------------------------------------------------------------------
# 3. English-to-IPA Entry Point (Dictionary + Rule-Based + Optional espeak-ng)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def en_to_ipa(word: str, use_espeak: bool = False) -> str:
    """Convert an English word (or phrase) to English IPA.

    :param str word: English word or text
    :param bool use_espeak: Whether to try espeak-ng CLI when available (default: False)
    :return: English IPA string (e.g. '/həˈloʊ/')
    """
    if not word or not word.strip():
        return ""

    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", word)
    if not tokens:
        return ""

    results: List[str] = []
    for tok in tokens:
        clean = tok.lower().replace("'", "")
        if not clean:
            continue

        if clean in _EN_IPA_DICT:
            results.append(f"/{_EN_IPA_DICT[clean]}/")
            continue

        if use_espeak and shutil.which("espeak-ng"):
            try:
                proc = subprocess.run(
                    ["espeak-ng", "-q", "--ipa", "-v", "en-us", clean],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                espeak_out = proc.stdout.strip()
                if espeak_out:
                    results.append(f"/{espeak_out}/")
                    continue
            except Exception:
                pass

        ipa = _rule_based_en_to_ipa(clean)
        if ipa:
            results.append(f"/{ipa}/")

    return " ".join(results)


# ---------------------------------------------------------------------------
# 4. English IPA -> Thai IPA Phoneme Converter (Thai-Style Adaptation)
# ---------------------------------------------------------------------------
# Ordered vowel tokens in normalized English IPA (longest first)
_EN_VOWEL_TOKENS = (
    "aɪ", "ɑɪ", "ʌɪ",
    "aʊ", "ɑʊ",
    "eɪ", "ɛɪ",
    "oʊ", "əʊ",
    "ɔɪ", "oɪ",
    "ɪə", "iə",
    "ɛə", "eə",
    "ʊə", "uə",
    "iː", "uː", "ɑː", "ɔː", "ɜː",
    "ɚ", "ɝ",
    "ɪ", "ᵻ", "i", "ʊ", "u", "ɛ", "e", "æ", "ɑ", "ɒ", "ɔ", "ʌ", "ɐ", "ə", "a", "o",
)

_EN_VOWEL_SET = set(_EN_VOWEL_TOKENS)

_EN_CONSONANT_TOKENS = (
    "ʧ", "ʤ",
    "p", "b", "t", "d", "k", "ɡ",
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "m", "n", "ŋ", "l", "ɹ", "j", "w", "ʔ",
)

_EN_CONSONANT_SET = set(_EN_CONSONANT_TOKENS)

# Valid 2-consonant English onset clusters (excluding s-clusters, handled separately)
_VALID_EN_ONSET_CLUSTERS = {
    ("p", "ɹ"), ("p", "l"), ("p", "j"),
    ("b", "ɹ"), ("b", "l"), ("b", "j"),
    ("t", "ɹ"), ("t", "j"), ("t", "w"),
    ("d", "ɹ"), ("d", "j"), ("d", "w"),
    ("k", "ɹ"), ("k", "l"), ("k", "w"), ("k", "j"),
    ("ɡ", "ɹ"), ("ɡ", "l"), ("ɡ", "w"), ("ɡ", "j"),
    ("f", "ɹ"), ("f", "l"), ("f", "j"),
    ("v", "j"), ("m", "j"), ("n", "j"), ("h", "j"),
    ("θ", "ɹ"), ("ʃ", "ɹ"),
}

# Short lax English vowels that trigger consonant gemination in stressed open syllables
_SHORT_LAX_EN_VOWELS = {"ɪ", "ɛ", "æ", "ɒ", "ʌ", "ʊ"}

# Valid Thai onset clusters in FastThaiG2P's ipa.json
_VALID_THAI_CLUSTERS = {
    "kr", "kl", "kw",
    "kʰr", "kʰl", "kʰw",
    "pr", "pl",
    "pʰr", "pʰl",
    "tr", "tʰr",
    "br", "bl", "dr", "fr", "fl",
}

# Thai low-class onset consonants (for loanword tone assignment)
_LOW_CLASS_ONSETS = {
    "kʰ", "kʰr", "kʰl", "kʰw",
    "tʰ", "tʰr",
    "pʰ", "pʰr", "pʰl",
    "t͡ɕʰ", "f", "fr", "fl", "s", "h",
    "m", "n", "ŋ", "r", "l", "j", "w",
}


def _normalize_en_ipa_word(ipa_word: str) -> str:
    """Normalize a single English IPA word string prior to tokenization."""
    s = ipa_word.strip().strip("/[]")
    if not s:
        return ""

    # Pre-composed affricates and tie-bars
    s = s.replace("t͡ʃ", "ʧ").replace("t͜ʃ", "ʧ").replace("tʃ", "ʧ")
    s = s.replace("d͡ʒ", "ʤ").replace("d͜ʒ", "ʤ").replace("dʒ", "ʤ")
    s = s.replace("g", "ɡ")
    s = s.replace("ɾ", "ɹ").replace("ɻ", "ɹ").replace("r", "ɹ")
    s = s.replace("ɫ", "l")

    # Syllabic consonants (n̩, m̩, l̩)
    s = s.replace("n̩", "ən").replace("m̩", "əm").replace("l̩", "əl")

    # Normalize r-colored diphthongs & vowels when followed by non-vowel (coda r)
    # Match 3-char diphthongs (ɪəɹ, ɛəɹ, ʊəɹ) before 2-char əɹ!
    s = re.sub(r"ɪə?ɹ(?![aeiouɑɒɔɛɪʊʌəɚɝæ])", "ɪə", s)
    s = re.sub(r"[ɛe]ə?ɹ(?![aeiouɑɒɔɛɪʊʌəɚɝæ])", "ɛə", s)
    s = re.sub(r"ʊə?ɹ(?![aeiouɑɒɔɛɪʊʌəɚɝæ])", "ʊə", s)
    s = re.sub(r"ɑː?ɹ(?![aeiouɑɒɔɛɪʊʌəɚɝæ])", "ɑː", s)
    s = re.sub(r"[ɔo]ː?ɹ(?![aeiouɑɒɔɛɪʊʌəɚɝæ])", "ɔː", s)
    s = re.sub(r"[ɜɝ]ː?ɹ?(?![aeiouɑɒɔɛɪʊʌəɚɝæ])", "ɜː", s)
    s = re.sub(r"əɹ(?![aeiouɑɒɔɛɪʊʌəɚɝæ])", "ɚ", s)

    return s


def _tokenize_en_ipa(s: str) -> List[str]:
    """Tokenize normalized English IPA into phoneme units + boundary markers."""
    tokens: List[str] = []
    i = 0
    n = len(s)
    all_units = (
        ("ˈ", "ˌ", ".")
        + _EN_VOWEL_TOKENS
        + _EN_CONSONANT_TOKENS
    )

    while i < n:
        if s[i].isspace():
            i += 1
            continue
        matched = False
        for unit in all_units:
            if s.startswith(unit, i):
                tokens.append(unit)
                i += len(unit)
                matched = True
                break
        if not matched:
            # Skip unknown diacritics (e.g. ː, ʰ, ̃)
            i += 1
    return tokens


def _syllabify_en_ipa(
    tokens: List[str],
) -> List[Tuple[int, List[str], str, List[str]]]:
    """Group English IPA phoneme tokens into syllables: (stress, onset, vowel, coda).

    stress: 1 = primary ('ˈ'), 2 = secondary ('ˌ'), 0 = unstressed.
    """
    items: List[Tuple[str, str]] = []  # ('B', marker) | ('C', tok) | ('V', tok)
    for t in tokens:
        if t in ("ˈ", "ˌ", "."):
            items.append(("B", t))
        elif t in _EN_VOWEL_SET:
            items.append(("V", t))
        elif t in _EN_CONSONANT_SET:
            items.append(("C", t))

    v_indices = [idx for idx, (kind, _) in enumerate(items) if kind == "V"]
    if not v_indices:
        return []

    syllables: List[Tuple[int, List[str], str, List[str]]] = []
    num_v = len(v_indices)

    onsets: List[List[str]] = [[] for _ in range(num_v)]
    codas: List[List[str]] = [[] for _ in range(num_v)]
    stresses: List[int] = [0 for _ in range(num_v)]
    vowels: List[str] = [items[v_idx][1] for v_idx in v_indices]

    # Leading items before first vowel
    for kind, val in items[: v_indices[0]]:
        if kind == "B":
            if val == "ˈ":
                stresses[0] = 1
            elif val == "ˌ" and stresses[0] == 0:
                stresses[0] = 2
        elif kind == "C":
            onsets[0].append(val)

    # Trailing items after last vowel
    for kind, val in items[v_indices[-1] + 1 :]:
        if kind == "C":
            codas[-1].append(val)

    # Intervocalic stretches between v_indices[k] and v_indices[k + 1]
    for k in range(num_v - 1):
        mid = items[v_indices[k] + 1 : v_indices[k + 1]]
        b_pos = -1
        for idx, (kind, val) in enumerate(mid):
            if kind == "B":
                b_pos = idx
                if val == "ˈ":
                    stresses[k + 1] = 1
                elif val == "ˌ" and stresses[k + 1] == 0:
                    stresses[k + 1] = 2

        if b_pos != -1:
            # Explicit boundary exists
            coda_c = [val for kind, val in mid[:b_pos] if kind == "C"]
            onset_c = [val for kind, val in mid[b_pos + 1 :] if kind == "C"]

            if not onset_c and coda_c:
                last_c = coda_c[-1]
                if vowels[k] in _SHORT_LAX_EN_VOWELS or vowels[k] == "ə":
                    # Geminate: keep in coda of k AND copy to onset of k+1 (e.g. ˈkɒp.i -> kɔp.piː)
                    onset_c = [last_c]
                else:
                    onset_c = [coda_c.pop()]
            elif len(onset_c) == 1 and not coda_c and vowels[k] in _SHORT_LAX_EN_VOWELS and stresses[k] == 1:
                if onset_c[0] in ("p", "b", "t", "d", "k", "ɡ", "f", "m", "n"):
                    coda_c = [onset_c[0]]

            codas[k].extend(coda_c)
            onsets[k + 1].extend(onset_c)
        else:
            # No explicit boundary: apply Maximal Onset Principle on consonants
            cons = [val for kind, val in mid if kind == "C"]
            if not cons:
                if vowels[k] in ("aʊ", "oʊ", "əʊ", "uː", "ʊ"):
                    onsets[k + 1].append("w")
                elif vowels[k] in ("aɪ", "eɪ", "ɔɪ"):
                    onsets[k + 1].append("j")
            elif len(cons) == 1:
                c = cons[0]
                if vowels[k] in _SHORT_LAX_EN_VOWELS and (stresses[k] == 1 or k == 0):
                    if c in ("p", "b", "t", "d", "k", "ɡ", "f", "m", "n"):
                        codas[k].append(c)
                onsets[k + 1].append(c)
            else:
                if tuple(cons[-2:]) in _VALID_EN_ONSET_CLUSTERS:
                    codas[k].extend(cons[:-2])
                    onsets[k + 1].extend(cons[-2:])
                elif len(cons) >= 2 and cons[-2] == "s" and cons[-1] in ("p", "t", "k"):
                    codas[k].extend(cons[:-1])
                    onsets[k + 1].append(cons[-1])
                else:
                    codas[k].extend(cons[:-1])
                    onsets[k + 1].append(cons[-1])

    if all(s == 0 for s in stresses) and stresses:
        stresses[0] = 1

    for k in range(num_v):
        syllables.append((stresses[k], onsets[k], vowels[k], codas[k]))

    return syllables


def _map_en_onset_to_thai(
    onset: List[str],
    after_s: bool = False,
    is_word_initial: bool = True,
    stress: int = 1,
) -> str:
    """Map an English syllable onset consonant list to a Thai IPA onset string."""
    if not onset:
        return "ʔ"

    def _map_single(c: str, in_cluster: bool = False) -> str:
        if c == "p":
            if after_s or (not is_word_initial and stress == 0 and not in_cluster):
                return "p"
            return "pʰ"
        if c == "t":
            if after_s or (not is_word_initial and stress == 0 and not in_cluster):
                return "t"
            return "tʰ"
        if c == "k":
            return "k" if after_s else "kʰ"
        if c == "b":
            return "b"
        if c == "d":
            return "d"
        if c == "ɡ":
            return "k"
        if c == "ʧ":
            return "t͡ɕʰ"
        if c == "ʤ":
            return "t͡ɕ"
        if c == "f":
            return "f"
        if c == "v":
            return "w"
        if c == "θ":
            return "tʰ"
        if c == "ð":
            return "d" if is_word_initial else "tʰ"
        if c in ("s", "z"):
            return "s"
        if c in ("ʃ", "ʒ"):
            return "t͡ɕʰ"
        if c == "h":
            return "h"
        if c == "m":
            return "m"
        if c == "n":
            return "n"
        if c == "ŋ":
            return "ŋ"
        if c == "l":
            return "l"
        if c == "ɹ":
            return "r"
        if c == "j":
            return "j"
        if c == "w":
            return "w"
        return "ʔ"

    if len(onset) == 1:
        return _map_single(onset[0])

    # 2-consonant onset cluster
    if onset[0] == "p" and onset[1] == "ɹ":
        return "pr"
    c1 = _map_single(onset[0], in_cluster=True)
    c2 = _map_single(onset[1], in_cluster=True)
    cand = c1 + c2
    if cand in _VALID_THAI_CLUSTERS:
        return cand
    return c1


def _map_en_coda_to_thai(coda: List[str]) -> str:
    """Map an English coda consonant list to a single Thai IPA coda."""
    # Filter out post-vocalic r
    c_list = [c for c in coda if c != "ɹ"]
    if not c_list:
        return ""

    pair = tuple(c_list[:2])
    # Nasal + stop/fricative clusters keep the nasal in Thai loanwords
    if c_list[0] == "m" and len(c_list) >= 2 and c_list[1] in ("p", "b", "f", "z", "s"):
        return "m"
    if c_list[0] == "n" and len(c_list) >= 2 and c_list[1] in ("t", "d", "s", "z", "ʧ", "ʤ", "θ"):
        return "n"
    if c_list[0] == "ŋ" and len(c_list) >= 2 and c_list[1] in ("k", "ɡ", "z", "s"):
        return "ŋ"

    # Special obstruent clusters
    if pair in (("k", "s"), ("ɡ", "z"), ("k", "t"), ("s", "k")):
        return "k̚"
    if pair in (("s", "t"), ("f", "t") if False else ("s", "t"), ("t", "s"), ("d", "z")):
        return "t̚"
    if pair in (("f", "t"), ("p", "t"), ("s", "p"), ("p", "s"), ("b", "z")):
        return "p̚"
    if pair in (("l", "d"), ("l", "t"), ("l", "z"), ("l", "s")):
        return "n"

    # Default: map the first coda consonant
    c = c_list[0]
    if c in ("p", "b", "f", "v"):
        return "p̚"
    if c in ("t", "d", "θ", "ð", "s", "z", "ʃ", "ʒ", "ʧ", "ʤ"):
        return "t̚"
    if c in ("k", "ɡ"):
        return "k̚"
    if c == "m":
        return "m"
    if c in ("n", "l"):
        return "n"
    if c == "ŋ":
        return "ŋ"
    if c == "j":
        return "j"
    if c == "w":
        return "w"
    return ""


def _assign_thai_tone(
    onset: str,
    vowel: str,
    coda: str,
    stress: int,
    is_first: bool,
    is_last: bool,
    total_syls: int,
) -> str:
    """Assign a Thai tone (˧, ˨˩, ˥˩, ˦˥, ˩˩˦) for Thai-style English pronunciation."""
    is_dead = coda in ("k̚", "t̚", "p̚", "ʔ") or (coda == "" and vowel in ("a", "i", "u"))
    is_long = "ː" in vowel or "̯" in vowel
    is_low_onset = onset in _LOW_CLASS_ONSETS

    if is_dead:
        if coda == "" and vowel in ("a", "i", "u"):
            return "˨˩"
        if not is_long:
            # Short vowel + stop coda (k̚, t̚, p̚)
            if is_low_onset or total_syls == 1 or (is_first and stress > 0):
                return "˦˥"
            return "˨˩"
        else:
            # Long vowel + stop coda
            if is_low_onset and (vowel in ("ɤː", "ɔː") or total_syls == 1):
                return "˦˥"
            return "˨˩"

    # Live syllable (open long vowel or sonorant coda m, n, ŋ, j, w)
    if total_syls == 1:
        return "˧"
    if not is_last:
        return "˧"
    # Final live syllable in a multi-syllable word:
    # Primary-stressed final syllable takes mid tone (e.g. online, cartoon, hello);
    # unstressed final syllable takes falling tone (e.g. computer, meeting, copy, google).
    if stress == 1:
        return "˧"
    return "˥˩"


def _convert_single_en_ipa_word(en_ipa_word: str) -> str:
    """Convert a single word's English IPA string into Wiktionary Thai IPA `/syl.syl/`."""
    norm = _normalize_en_ipa_word(en_ipa_word)
    if not norm:
        return ""
    tokens = _tokenize_en_ipa(norm)
    if not tokens:
        return ""

    raw_syllables = _syllabify_en_ipa(tokens)
    if not raw_syllables:
        return ""

    # Expand syllables (handling s-cluster epenthesis)
    expanded: List[Tuple[int, str, str, str]] = []  # (stress, thai_onset, thai_vowel, thai_coda)
    num_raw = len(raw_syllables)

    for idx, (stress, onset, vowel, coda) in enumerate(raw_syllables):
        is_first_raw = idx == 0
        is_last_raw = idx == num_raw - 1

        # 1. Check s + consonant onset cluster -> epenthetic 'sa˨˩'
        after_s = False
        if len(onset) >= 2 and onset[0] == "s":
            expanded.append((0, "s", "a", ""))
            onset = onset[1:]
            after_s = True

        # 2. Check C + j + uː/ʊ -> C + iw (e.g. computer -> kʰɔm.pʰiw.tɤː, new -> niw, cute -> kʰiw)
        if len(onset) >= 2 and onset[-1] == "j" and vowel in ("uː", "u", "ʊ"):
            onset = onset[:-1]
            thai_onset = _map_en_onset_to_thai(
                onset, after_s=after_s, is_word_initial=is_first_raw, stress=stress
            )
            expanded.append((stress, thai_onset, "i", "w"))
            continue

        thai_onset = _map_en_onset_to_thai(
            onset, after_s=after_s, is_word_initial=is_first_raw, stress=stress
        )

        # 3. Diphthongs with offglides (aɪ, aʊ, ɔɪ) where the offglide becomes the Thai coda
        if vowel in ("aɪ", "ɑɪ", "ʌɪ"):
            thai_v = "aː" if (not coda or coda[0] in ("l", "n", "m", "d", "v", "z")) else "a"
            expanded.append((stress, thai_onset, thai_v, "j"))
            continue
        if vowel in ("aʊ", "ɑʊ"):
            thai_v = "aː" if (not coda or coda[0] in ("l", "n", "m", "d", "v", "z")) else "a"
            expanded.append((stress, thai_onset, thai_v, "w"))
            continue
        if vowel in ("ɔɪ", "oɪ"):
            expanded.append((stress, thai_onset, "ɔː", "j"))
            continue

        # 4. Map coda
        thai_coda = _map_en_coda_to_thai(coda)

        # If coda had 'ɹ' that was stripped, lengthen short vowels
        had_r_coda = "ɹ" in coda

        # 5. Map vowel
        if vowel == "iː":
            thai_v = "iː"
        elif vowel == "i":
            thai_v = "iː" if not thai_coda else "i"
        elif vowel in ("ɪ", "ᵻ"):
            if had_r_coda:
                thai_v = "ia̯"
            elif not thai_coda and is_last_raw:
                thai_v = "iː"
            else:
                thai_v = "i"
        elif vowel in ("eɪ", "ɛɪ"):
            thai_v = "eː"
        elif vowel in ("ɛ", "e"):
            if had_r_coda:
                thai_v = "ɛː"
            elif not thai_coda:
                thai_v = "eː"
            else:
                thai_v = "e"
        elif vowel == "æ":
            if not thai_coda or thai_coda == "n":
                thai_v = "ɛː"
            else:
                thai_v = "ɛ"
        elif vowel in ("ɑː", "ɑ", "a"):
            if vowel == "ɑː" or had_r_coda or not thai_coda:
                thai_v = "aː"
            else:
                thai_v = "a"
        elif vowel == "ɒ":
            if had_r_coda or not thai_coda:
                thai_v = "ɔː"
            else:
                thai_v = "ɔ"
        elif vowel in ("ɔː", "ɔ"):
            thai_v = "ɔː"
        elif vowel in ("oʊ", "əʊ", "o"):
            thai_v = "oː"
        elif vowel == "ʊ":
            if had_r_coda:
                thai_v = "ua̯"
            else:
                thai_v = "u"
        elif vowel in ("uː", "u"):
            thai_v = "uː"
        elif vowel in ("ʌ", "ɐ"):
            thai_v = "a"
        elif vowel in ("ɜː", "ɝ", "ɚ"):
            thai_v = "ɤː"
        elif vowel == "ə":
            if had_r_coda:
                thai_v = "ɤː"
            elif not thai_coda:
                if is_last_raw:
                    thai_v = "aː"
                elif thai_onset == "h" and idx + 1 < num_raw and raw_syllables[idx + 1][1] == ["l"]:
                    # e.g. hello /həˈloʊ/ -> he˧.loː˧
                    thai_v = "eː"
                else:
                    thai_v = "a"
            elif coda and coda[0] == "l":
                thai_v = "ɤː"
            elif thai_coda in ("m", "n") and thai_onset in ("kʰ", "k") and is_first_raw:
                # com- / con- prefix (computer, control, connect)
                thai_v = "ɔ"
            else:
                thai_v = "a"
        elif vowel in ("ɪə", "iə"):
            thai_v = "ia̯"
        elif vowel in ("ɛə", "eə"):
            thai_v = "ɛː"
        elif vowel in ("ʊə", "uə"):
            thai_v = "ua̯"
        else:
            thai_v = "aː" if not thai_coda else "a"

        # Enforce Thai phonotactics: mid/open-mid short vowels cannot be open without coda
        if not thai_coda and thai_v in ("e", "ɛ", "o", "ɔ", "ɤ", "ɯ"):
            thai_v = thai_v + "ː"

        expanded.append((stress, thai_onset, thai_v, thai_coda))

    # 6. Assign tones and join with '.'
    total_exp = len(expanded)
    thai_syls: List[str] = []
    for s_idx, (stress, t_onset, t_vowel, t_coda) in enumerate(expanded):
        tone = _assign_thai_tone(
            onset=t_onset,
            vowel=t_vowel,
            coda=t_coda,
            stress=stress,
            is_first=(s_idx == 0),
            is_last=(s_idx == total_exp - 1),
            total_syls=total_exp,
        )
        thai_syls.append(f"{t_onset}{t_vowel}{t_coda}{tone}")

    if not thai_syls:
        return ""
    return "/" + ".".join(thai_syls) + "/"


def en_ipa_to_thai_ipa(en_ipa: str) -> str:
    """Convert an English IPA string into Wiktionary-style Thai IPA phonemes.

    Adapts English consonants, vowels, consonant clusters, codas, and stress
    into Thai phonological structure (/syl.la.ble˧/) so Thai TTS models speak
    English words with natural Thai-style pronunciation.

    :param str en_ipa: English IPA string (e.g. '/həˈloʊ/' or '/kəmˈpjuːtɚ/')
    :return: Thai Wiktionary IPA string (e.g. '/heː˧.loː˧/' or '/kʰɔm˧.pʰiw˧.tɤː˥˩/')
    """
    if not en_ipa or not en_ipa.strip():
        return ""

    # Split into word-level IPA chunks (either slash-delimited or space-delimited)
    slash_chunks = re.findall(r"/[^/]+/", en_ipa)
    if slash_chunks:
        words = slash_chunks
    else:
        words = en_ipa.strip().split()

    out = []
    for w in words:
        converted = _convert_single_en_ipa_word(w)
        if converted:
            out.append(converted)
    return " ".join(out)


@lru_cache(maxsize=4096)
def en_g2p(word: str, use_espeak: bool = False) -> str:
    """Convert English word(s) to Thai-style Wiktionary IPA (/syl.la.ble˧/).

    Pipeline: English text -> English IPA -> Thai IPA.

    :param str word: English word or text
    :param bool use_espeak: Whether to use espeak-ng when available (default: False)
    :return: Thai Wiktionary IPA string
    """
    en_ipa = en_to_ipa(word, use_espeak=use_espeak)
    if not en_ipa:
        return word
    thai_ipa = en_ipa_to_thai_ipa(en_ipa)
    return thai_ipa if thai_ipa else word


class EnglishG2P:
    """English-to-Thai-IPA G2P converter for speaking English in Thai style."""

    def __init__(self, use_espeak: bool = False):
        self.use_espeak = use_espeak

    def to_en_ipa(self, text: str) -> str:
        """Convert English text to English IPA."""
        return en_to_ipa(text, use_espeak=self.use_espeak)

    def to_thai_ipa(self, en_ipa: str) -> str:
        """Convert English IPA phoneme string to Thai Wiktionary IPA."""
        return en_ipa_to_thai_ipa(en_ipa)

    def convert(self, text: str) -> str:
        """Convert English text directly to Thai Wiktionary IPA."""
        return en_g2p(text, use_espeak=self.use_espeak)

    def __call__(self, text: str) -> str:
        return self.convert(text)
