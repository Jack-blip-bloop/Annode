# Study Annotator

Upload a reading (PDF, scan, or a photo of a page) and get it back annotated like a strong student would: main ideas
highlighted, key terms circled, important points underlined, and 2-4 questions per page in a new right-hand margin.
It's a study aid: it never answers the questions or writes assignment text.

The AI's marks are purple, on their own PDF layer ("AI annotations"), so students' own notes stand out and the AI's
can be hidden. The original pages are copied in unchanged.

## Run it

```
pip install -r requirements.txt          (plus Tesseract for scans)
python -m annotate_pdf reading.pdf       one file  -> reading-annotated.pdf
python web/app.py                        the website -> http://127.0.0.1:5000
python tests/batch_test.py [folder]      test a whole folder of readings, with a cost report
```

The Gemini key comes from `STUDY_ANNOTATOR_GEMINI_KEY`, or `gemini_key.txt` in this folder.

## Layout

- `annotate_pdf/textlayer.py` - words and their positions (the PDF's own text, or Tesseract OCR for scans)
- `annotate_pdf/ai.py` - asks Gemini what to mark (sends page text only, a few pages per request) and counts the cost
- `annotate_pdf/render.py` + `shapes.py` - draws the hand-drawn marks and handwritten margin questions
- `web/app.py` - the upload site: in-memory only, free daily page limit per anonymous visitor, usage/cost page at
  `/admin/usage?key=...`, privacy policy and terms

## Privacy

Files and results live only in memory and are deleted on download (or after 15 minutes). `web/usage.db` holds only
daily numbers per anonymous visitor (salted hash of the IP). Page text is sent to Google's Gemini API.

## Not built yet

Accounts and Stripe payments (waiting on accounts), and the OneNote add-in and other integrations (phase 4).
