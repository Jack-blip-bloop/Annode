"""Asking the AI what to mark. It gets the TEXT of the pages (numbered lines), not pictures: that is much cheaper,
and the program already knows where every word is, so it only needs the AI to say WHICH words.

Same options as the OneNote annotator: a focus (general, or a specific request like 'every time clothes are
mentioned'), the teacher's instructions, which mark styles to use, notes, margin symbols, a summary box, amounts.
It is a study aid: it never answers its own questions or writes assignment text."""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .textlayer import numbered_lines

MODEL = os.environ.get("ANNOTATOR_MODEL", "gemini-3.5-flash-lite")
# What the model costs per million tokens, in US dollars. CHECK THESE against Google's current price list and set the
# environment variables if they differ: every cost figure the app shows is computed from them.
PRICE_IN_PER_M = float(os.environ.get("PRICE_IN_PER_M", "0.10"))
PRICE_OUT_PER_M = float(os.environ.get("PRICE_OUT_PER_M", "0.40"))

MAX_WORDS_PER_CALL = 3500      # pages are sent a few at a time, up to about this many words per request
MAX_PAGES_PER_CALL = 4
PARALLEL_CALLS = 4

STYLES = ["underline", "double", "squiggle", "circle", "box", "highlight"]
ROLE_ORDER = ["main_idea", "key_term", "definition", "fact", "question", "key_quote"]
ROLE_INFO = {
    "main_idea": ("the clause that states the central point of a section or paragraph", ["highlight", "box", "underline"]),
    "key_term": ("an important vocabulary word or short term the reader must know", ["circle", "underline", "highlight"]),
    "definition": ("a definition, formula, rule or final answer", ["box", "underline", "highlight"]),
    "fact": ("a name, date, number or other specific fact worth remembering", ["circle", "underline"]),
    "question": ("something unclear, doubtful, or that raises a question worth checking", ["squiggle", "double", "circle"]),
    "key_quote": ("the single most important statement on the page (at most one per page)", ["double", "underline", "box"]),
}
PER_PAGE = {"few": "2 to 4", "normal": "4 to 7", "lots": "7 to 11"}
NOTES_PER_PAGE = {"few": "1", "normal": "2 or 3", "many": "3 or 4"}
SYMBOL_HELP = {"?": "something confusing, or a question you'd ask", "!": "something surprising or really important",
               "star": "the most important idea on the page (once at most)", "arrow": "an example, or a cause leading to an effect"}


def load_key():
    """In order: STUDY_ANNOTATOR_GEMINI_KEY (set this on a server), gemini_key.txt next to this project, the key saved
    by the OneNote annotator on this computer, then GEMINI_API_KEY. (This computer has an old, invalid GEMINI_API_KEY
    in its environment, so that one comes last.)"""
    key = os.environ.get("STUDY_ANNOTATOR_GEMINI_KEY")
    if key:
        return key.strip()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "gemini_key.txt")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    p = os.path.join(os.path.dirname(here), "onenote-annotator", "config.json")
    if os.path.exists(p):
        k = json.load(open(p, encoding="utf-8")).get("api_key")
        if k:
            return k.strip()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    raise RuntimeError("No Gemini API key: set the STUDY_ANNOTATOR_GEMINI_KEY environment variable.")


def role_styles(options):
    """A mark style for each kind of idea, from the styles that are switched on (avoiding repeats where possible)."""
    enabled = [s for s in STYLES if s in (options.get("styles") or [])] or ["underline"]
    used, out = set(), {}
    for role in ROLE_ORDER:
        prefs = ROLE_INFO[role][1]
        pick = (next((p for p in prefs if p in enabled and p not in used), None)
                or next((p for p in prefs if p in enabled), None)
                or next((s for s in enabled if s not in used), None) or enabled[0])
        out[role] = pick
        used.add(pick)
    return out


# ---------------------------------------------------------------- 1. what kind of request is this?
REQUEST_SCHEMA = {
    "type": "object",
    "properties": {"mode": {"type": "string", "enum": ["general", "targeted"]}, "search_terms": {"type": "array", "items": {"type": "string"}}},
    "required": ["mode", "search_terms"],
}


def classify_request(client, focus, sample, usage):
    """Decided ONCE for the whole document, so every part of a long reading is treated the same way."""
    if not focus.strip():
        return {"mode": "general", "search_terms": []}
    system = "\n".join([
        "A student typed a request saying what to annotate in a reading. Decide what kind it is:",
        "  - 'targeted': it asks for something SPECIFIC: every mention of a word, thing, person, place, theme or idea",
        "    (e.g. 'find every time clothes are mentioned', 'only the dates', 'circle every character name', 'highlight",
        "    all the evidence about money', 'just the verbs').",
        "  - 'general': it only says what the reading is for, or asks for a normal study annotation ('annotate for a",
        "    test', 'anything important', 'it's homework', 'key ideas and vocab', 'the author's argument').",
        "For targeted requests, search_terms = the exact words and word forms that count as a match and are likely to",
        "appear (e.g. for clothes: clothes, clothing, clothed, shirt, coat, dress, wore, wearing). Leave it empty if",
        "matches aren't single words or short phrases (e.g. 'evidence about money'), or for general requests.",
    ])
    return _call(client, f"Request: {focus.strip()[:800]}\n\nThe reading begins: {sample[:1500]}", system,
                 REQUEST_SCHEMA, usage)


# ---------------------------------------------------------------- 2. marks, notes and symbols, a few pages at a time
def page_prompt(options, request):
    rs = role_styles(options)
    targeted = request.get("mode") == "targeted"
    focus = (options.get("focus") or "").strip()
    teacher = (options.get("teacher") or "").strip()
    p = [
        "You annotate a reading for a student, the way a strong student marks up a printed text in pencil.",
        "This is a STUDY AID: point at what matters and ask questions that make them think. NEVER answer questions,",
        "never summarize so they don't have to read, never write anything that could be handed in as their own work.",
        "You get the text of some pages, each line numbered like [12]. Return one entry per page with its page number.",
        "",
    ]
    if targeted:
        p += [
            "THE STUDENT'S REQUEST (do exactly this and NOTHING else):", "<<<", focus[:800], ">>>",
            "  - Mark EVERY place on these pages that matches. Missing one is a mistake. Repeats are wanted: mark the same",
            "    word again every time it appears. There is no limit on the number of marks.",
            "  - Mark ONLY matches: no main ideas, terms, facts or quotes that aren't what was asked for.",
            "  - Count something as a match if it really is about the thing asked for, not unrelated words.",
            "  - Quote just the matching word or short phrase (1 to 6 words). Give all matches the role key_term.",
            "  - If the request says how to mark them (circle, underline, highlight, box...), set every mark's style to",
            "    that; otherwise style 'auto'.",
            "  - notes and symbols: ONLY if the request asks for them, and only about what it asks for. Otherwise empty.",
        ]
    else:
        p += [
            "MARKS: for each important idea, the exact words to mark and what kind of thing it is (role). The role decides",
            "how it's drawn:",
        ]
        p += [f"    {r}: {ROLE_INFO[r][0]}  ->  {rs[r]}" for r in ROLE_ORDER]
        p += [
            f"  - About {PER_PAGE.get(options.get('amount'), PER_PAGE['normal'])} marks per page with real text. Be selective:",
            "    what a student needs for a test or discussion. Use different roles.",
            "  - How much to quote: main_idea / key_quote: the key clause, 4 to 12 words (never a whole long sentence).",
            "    key_term / fact: 1 to 4 words. definition: the term plus its meaning, up to 12 words. question: the few",
            "    words that are confusing.",
            "  - Never mark the same words twice. Don't mark headings, running headers, page numbers, captions, footnotes",
            "    or reference lists.",
        ]
        if focus:
            p += ["  - The student wants to focus on this (it decides what counts as important):", "    <<<", "    " + focus[:800], "    >>>"]
    p += [
        "  - For each mark: line (the [number] where the words start), quote (copied EXACTLY from that line; it may run",
        "    onto the next line), role, and style ('auto' unless instructions below say how to mark it).",
        "",
    ]
    if not targeted:
        if options.get("notes", True):
            p += [
                f"NOTES ({NOTES_PER_PAGE.get(options.get('notes_amount'), '2 or 3')} per page with real text), written in the",
                "margin next to the line they're about, like a student's pencil notes (at most 12 words each):",
                "  - Notice things: a question a sentence raises, something surprising, a link to another part of the",
                "    reading, a claim that needs evidence, a definition in your own words.",
                "  - Start questions with 'Q:'. Others can start with '!', 'Key:', 'Link:', 'Why?', 'Ex:'. Never answers.",
                "  - Each: text, line, and anchor_quote (2 to 8 exact words from that line).",
            ]
        else:
            p.append("NOTES: not wanted. Return empty notes lists.")
        if options.get("symbols") or teacher:
            p += ["", "MARGIN SYMBOLS (small marks in the margin beside a line):"]
            p += [f"    {k}: {v}" for k, v in SYMBOL_HELP.items()]
            p += ["  - Each: symbol, line, quote (2 to 6 exact words from that line). Different lines from each other." +
                  (" 1 to 3 per page." if options.get("symbols") else
                   " Only if the teacher's instructions ask for them; otherwise empty.")]
        else:
            p.append("MARGIN SYMBOLS: not wanted. Return empty symbols lists.")
    if teacher:
        p += ["", "THE TEACHER'S ANNOTATION INSTRUCTIONS. Follow these exactly; they win over everything above"
              + (" except the student's specific request" if targeted else "") + ":",
              "<<<", teacher[:3000], ">>>",
              "  - If they say to mark EVERY one of something (every date, every name...), mark all of them on every page,",
              "    even past the usual number of marks.",
              "  - If they say to mark something a particular way, set that mark's style to it: underline, double",
              "    (double underline), squiggle (wavy line), circle, box or highlight.",
              "  - If they ask for a ?, !, star or arrow next to things, those are MARGIN SYMBOLS: add them to symbols",
              "    (as many as the instructions call for).",
              "  - If they ask for questions or comments, write them as notes."]
    p += ["", "Pages with very little real text (title page, figure, reference list, index, blank) get empty lists."]
    return "\n".join(p)


PAGES_SCHEMA = {
    "type": "object",
    "properties": {
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "marks": {"type": "array", "items": {"type": "object", "properties": {
                        "role": {"type": "string", "enum": ROLE_ORDER}, "line": {"type": "integer"},
                        "quote": {"type": "string"}, "style": {"type": "string", "enum": ["auto"] + STYLES}},
                        "required": ["role", "line", "quote", "style"]}},
                    "notes": {"type": "array", "items": {"type": "object", "properties": {
                        "text": {"type": "string"}, "line": {"type": "integer"}, "anchor_quote": {"type": "string"}},
                        "required": ["text", "line", "anchor_quote"]}},
                    "symbols": {"type": "array", "items": {"type": "object", "properties": {
                        "symbol": {"type": "string", "enum": list(SYMBOL_HELP)}, "line": {"type": "integer"},
                        "quote": {"type": "string"}}, "required": ["symbol", "line", "quote"]}},
                },
                "required": ["page", "marks", "notes", "symbols"],
            },
        },
    },
    "required": ["pages"],
}


# ---------------------------------------------------------------- 3. the summary box (whole reading at once)
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"main_idea": {"type": "string"}, "key_terms": {"type": "array", "items": {"type": "string"}},
                   "question": {"type": "string"}},
    "required": ["main_idea", "key_terms", "question"],
}
SUMMARY_PROMPT = "\n".join([
    "Write a summary box for the top of a student's reading (a study aid; never answer the question):",
    "  - main_idea: the main idea of the whole reading in your own words, at most 14 words.",
    "  - key_terms: 2 or 3 of the most important terms, each as 'term - very short meaning' (at most 7 words).",
    "  - question: one good question the reading raises or a test might ask, at most 14 words.",
])


class Usage:
    """Tokens used (and what they cost), added up over every request of one job."""

    def __init__(self):
        self.tokens_in = self.tokens_out = self.calls = 0
        self._lock = threading.Lock()

    def add(self, resp):
        um = getattr(resp, "usage_metadata", None)
        if not um:
            return
        with self._lock:
            self.calls += 1
            self.tokens_in += um.prompt_token_count or 0
            self.tokens_out += (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)

    @property
    def cost_usd(self):
        return self.tokens_in / 1e6 * PRICE_IN_PER_M + self.tokens_out / 1e6 * PRICE_OUT_PER_M


def _call(client, text, system, schema, usage, temperature=0.3):
    last = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=[text],
                config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                                   response_schema=schema, temperature=temperature))
            usage.add(resp)
            return json.loads(resp.text)
        except genai_errors.APIError as e:
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (json.JSONDecodeError, ValueError) as e:
            last = e
        time.sleep(2 + attempt * 3)
    raise RuntimeError(f"The AI didn't answer properly ({type(last).__name__}). Please try again in a minute.")


def chunk_pages(pages):
    """Group the pages that have text into requests of a few pages each. pages = [(page number, words)]."""
    chunks, cur, n = [], [], 0
    for pno, words in pages:
        if len(words) < 8:
            continue
        if cur and (n + len(words) > MAX_WORDS_PER_CALL or len(cur) >= MAX_PAGES_PER_CALL):
            chunks.append(cur)
            cur, n = [], 0
        cur.append((pno, words))
        n += len(words)
    if cur:
        chunks.append(cur)
    return chunks


def _ask_pages(client, chunk, system, usage, context):
    parts = ["For context, the reading begins: " + context] if context else []
    parts += [f"=== Page {pno + 1} ===\n{numbered_lines(words)}" for pno, words in chunk]
    parts.append("Annotate these pages.")
    data = _call(client, "\n\n".join(parts), system, PAGES_SCHEMA, usage)
    out = {}
    for pg in data.get("pages", []):
        try:
            out[int(pg.get("page")) - 1] = pg
        except (TypeError, ValueError):
            pass
    return out


def annotate_pages(pages, options, api_key=None, progress=lambda done, total: None):
    """pages = [(page number, words)]. Returns (request, {page: {'marks', 'notes', 'symbols'}}, summary or None, Usage)."""
    client = genai.Client(api_key=api_key or load_key())
    usage = Usage()
    text_pages = [(p, w) for p, w in pages if len(w) >= 8]
    first = text_pages[0][1] if text_pages else []
    context = " ".join(w["text"] for w in first[:60])
    whole = "\n".join(" ".join(w["text"] for w in ws) for _, ws in text_pages)

    request = classify_request(client, options.get("focus") or "", whole, usage)
    system = page_prompt(options, request)
    chunks = chunk_pages(pages)
    results, done = {}, 0
    progress(0, len(chunks))
    with ThreadPoolExecutor(max_workers=PARALLEL_CALLS) as ex:
        summary_f = None
        if options.get("summary_box") and request.get("mode") != "targeted" and whole:
            summary_f = ex.submit(_call, client, whole[:120000], SUMMARY_PROMPT, SUMMARY_SCHEMA, usage)
        futs = [ex.submit(_ask_pages, client, c, system, usage, context if i else "") for i, c in enumerate(chunks)]
        for f in futs:
            results.update(f.result())
            done += 1
            progress(done, len(chunks))
        summary = summary_f.result() if summary_f else None
    return request, results, summary, usage
