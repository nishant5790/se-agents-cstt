# Extractor scripts

These are plain Python scripts. They do not use `pytest`.

Run them from the repository root with:

```powershell
python .\tests\test_text_extractor.py
python .\tests\test_xlsx_extractor.py
python .\tests\test_pdf_extractor.py
python .\tests\test_media_extractor.py
```

Each script prints a short trace when a `ContentBlock` is emitted and then dumps the final block list so you can see exactly how the block was created.

The xlsx, PDF, and media scripts rely on optional dependencies from `requirements.txt`:

- `openpyxl` for the xlsx extractor
- `pypdf` for the PDF extractor
- `vosk` plus `VOSK_MODEL_PATH` for the media extractor

The media script uses the offline Vosk path in `extractors.py`. If the model is not available, set `VOSK_MODEL_PATH` first.
