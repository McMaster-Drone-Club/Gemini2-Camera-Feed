# Gemini2-Camera-Feed

Minimal tests for the Orbbec Gemini‑2 using `pyorbbecsdk`.

## Files
- `main.py` 
    - basic pipeline test (requires `sudo` for higher perms)
    - from Gemini's documentation

- `distance.py` 
    - color + depth view with center‑point distance + rolling average
    - `ChatGPT` generated

## Run
```bash
sudo python3 main.py
sudo python3 distance.py
```

## Dependencies
```bash
pip3 install pyorbbecsdk numpy opencv-python
```