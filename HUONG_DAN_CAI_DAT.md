# ?? H??NG D?N C?I ??T & CH?Y TOOL

## ?? B??c 1: C?i ??t Python

Tool y?u c?u **Python 3.7+**

Ki?m tra phi?n b?n Python:
```bash
python3 --version
```

## ?? B??c 2: C?i ??t Th? Vi?n

### C?ch 1: D?ng requirements.txt (Khuy?n ngh?)
```bash
pip install -r requirements.txt
```

### C?ch 2: C?i t?ng package
```bash
pip install rich requests websocket-client pytz urllib3
```

ho?c d?ng pip3:
```bash
pip3 install rich requests websocket-client pytz urllib3
```

## ?? B??c 3: Ch?y Tool

```bash
python3 toolwsv9.py
```

ho?c:
```bash
python toolwsv9.py
```

## ?? Danh S?ch Th? Vi?n C?n Thi?t

| Th? Vi?n | Phi?n B?n | M?c ??ch |
|----------|-----------|----------|
| rich | >=13.0.0 | Giao di?n ??p trong terminal |
| requests | >=2.28.0 | HTTP requests |
| websocket-client | >=1.5.0 | WebSocket connection |
| pytz | >=2023.3 | Timezone (Asia/Ho_Chi_Minh) |
| urllib3 | >=1.26.0 | HTTP connection pooling |

## ?? X? L? L?i Th??ng G?p

### L?i 1: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'rich'
```
**Gi?i ph?p:**
```bash
pip install rich
```

### L?i 2: Permission denied
```
PermissionError: [Errno 13] Permission denied
```
**Gi?i ph?p:**
```bash
pip install --user rich requests websocket-client pytz urllib3
```

### L?i 3: pip kh?ng t?m th?y
```
pip: command not found
```
**Gi?i ph?p:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-pip

# macOS
python3 -m ensurepip

# Windows
py -m ensurepip
```

## ? Ki?m Tra C?i ??t

Ch?y l?nh n?y ?? ki?m tra:
```bash
python3 -c "import rich, requests, websocket, pytz; print('? T?t c? th? vi?n ?? ???c c?i ??t!')"
```

N?u kh?ng c? l?i, b?n ?? s?n s?ng!

## ?? Ch?y Tool

1. **Ch?y tool:**
   ```bash
   python3 toolwsv9.py
   ```

2. **Nh?p th?ng tin:**
   - D?n link game t? xworld.info
   - C?u h?nh s? BUILD ??t
   - C?u h?nh multiplier
   - C?c t?y ch?n kh?c (t?y ?)

3. **B?t ??u:**
   - Nh?n Enter
   - Theo d?i AI ho?t ??ng
   - Quan s?t k?t qu?

## ?? C?c File Quan Tr?ng

- `toolwsv9.py` - File ch?nh c?a tool
- `requirements.txt` - Danh s?ch th? vi?n c?n thi?t
- `NANG_CAP_ULTIMATE_AI.md` - T?i li?u chi ti?t v? n?ng c?p
- `SUMMARY.txt` - T?m t?t nhanh
- `escape_vip_ai_rebuild.log` - File log (t? ??ng t?o khi ch?y)

## ?? C?n Tr? Gi?p?

N?u g?p v?n ??:
1. Ki?m tra file log: `escape_vip_ai_rebuild.log`
2. ??m b?o Python >= 3.7
3. ??m b?o t?t c? th? vi?n ?? ???c c?i ??t
4. Ki?m tra k?t n?i internet
5. ??m b?o link game c?n h?p l?

---

**?? Ch?c b?n th?nh c?ng v?i ULTIMATE AI!**
