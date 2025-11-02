# ?? ULTRA AI - Tool Tho?t Hi?m Th?ng Minh Nh?t

## Gi?i thi?u

**ULTRA AI** l? phi?n b?n n?ng c?p si?u th?ng minh c?a tool ??t c??c Escape Master (XWorld), ???c trang b? c?ng ngh? AI ti?n ti?n nh?t.

## ? T?nh n?ng n?i b?t

### 1. ?? 10,000 C?ng Th?c Th?ng Minh
- **10,000 c?ng th?c** ?a d?ng v?i c?c chi?n l??c kh?c nhau:
  - Conservative (?u ti?n an to?n)
  - Aggressive (T?n c?ng m?nh)
  - Pattern-focused (T?p trung v?o m?u)
  - Momentum-based (D?a tr?n xu h??ng)
  - Balanced (C?n b?ng)

### 2. ?? Deep Learning
- **H?c s?u t? m?i k?t qu?**: AI t? ??ng ?i?u ch?nh tr?ng s? sau m?i v?n
- **Adaptive weights**: M?i c?ng th?c c? tr?ng s? ??ng t?ng/gi?m theo hi?u su?t
- **Confidence scoring**: ??nh gi? ?? tin c?y m?i d? ?o?n (0-95%)

### 3. ?? Pattern Recognition
- **Nh?n di?n m?u**: Ph?n t?ch 8 v?n g?n nh?t ?? t?m patterns
- **Pattern Memory**: L?u tr? 1,000 patterns th?nh c?ng
- **Anti-Pattern Detection**: Tr?nh 500 patterns d?n ??n th?t b?i
- **Sequence Learning**: H?c chu?i k?t qu? (ph?ng X th??ng theo sau ph?ng Y)

### 4. ?? Meta-Learning
- **H?c c?ch h?c**: T? ??ng ?i?u ch?nh t?c ?? h?c (Learning Rate)
- **Performance tracking**: Theo d?i 50 l?n d? ?o?n g?n nh?t c?a m?i c?ng th?c
- **Dynamic adaptation**: T?c ?? h?c thay ??i theo win rate
  - Win rate > 65% ? Gi?m LR ?? ?n ??nh
  - Win rate < 45% ? T?ng LR ?? th?ch nghi nhanh

### 5. ?? Adaptive Bet Sizing
- **?i?u ch?nh ti?n c??c theo confidence**:
  - Confidence ? 75%: T?ng c??c 10%
  - Confidence ? 40%: Gi?m c??c 20%
- ??m b?o an to?n khi kh?ng ch?c ch?n

### 6. ?? Advanced Features
- **Momentum Analysis**: Ph?n t?ch xu h??ng g?n ??y
- **Cycle Detection**: Ph?t hi?n chu k? c?a t?ng ph?ng
- **Risk Assessment**: ??nh gi? r?i ro t?ng h?p
- **Safety Score**: ?i?m an to?n cho m?i ph?ng
- **Sequence Correlation**: X?c su?t ph?ng X xu?t hi?n sau ph?ng Y

## ?? C?ch s? d?ng

### 1. C?i ??t dependencies
```bash
pip install requests websocket-client pytz rich
```

### 2. Ch?y tool
```bash
python toolwsv9.py
```

### 3. C?u h?nh
- **Nh?p link game**: D?n link t? xworld.info (ch?a userId & secretKey)
- **S? BUILD ??t**: S? ti?n c? b?n cho m?i v?n (VD: 1)
- **Nh?n sau khi thua**: H? s? nh?n martingale (VD: 2)
- **Ch?ng soi**: S? v?n ??t tr??c khi ngh? (0 = kh?ng ngh?)
- **Ngh? sau thua**: S? v?n ngh? sau khi thua (VD: 2)
- **Take Profit**: D?ng khi ??t l?i nhu?n m?c ti?u
- **Stop Loss**: D?ng khi l? ??n ng??ng

## ?? Hi?n th? UI

### Header
- ?? ULTRA AI stats (Confidence, Patterns, Learning Rate)
- S? d? BUILD/USDT/XWORLD
- Win/Loss streaks
- L?i/l? t?ch l?y

### Rooms Table
- Th?ng tin 8 ph?ng real-time
- S? ng??i ch?i, t?ng c??c
- Ph?ng d? ?o?n (? D? ?o?n)
- Ph?ng b? kill (? Kill)

### Analysis Panel
- Tr?ng th?i ph?n t?ch (ANALYZING/PREDICTED/RESULT)
- ??m ng??c th?i gian
- Confidence score
- Pattern signature

### Bet History
- 5 v?n g?n nh?t
- Thu?t to?n s? d?ng
- K?t qu? (Th?ng/Thua)
- S? ti?n ??t

## ?? C?ch ULTRA AI ho?t ??ng

### B??c 1: Feature Extraction
```
M?i ph?ng ???c ph?n t?ch 15+ features:
- Players, Bet, Bet per player (normalized)
- Survive score, Kill rate
- Recent penalty, Last kill penalty
- Hot/Cold scores
- Pattern strength, Sequence correlation
- Momentum, Bet variance
- Cycle score, Pattern confidence
- Safety score, Risk score
```

### B??c 2: Formula Voting
```
10,000 c?ng th?c vote cho ph?ng t?t nh?t:
- M?i c?ng th?c t?nh score cho 8 ph?ng
- Ch?n ph?ng c? score cao nh?t
- Vote ???c weighted theo adapt ? confidence
```

### B??c 3: Ensemble Aggregation
```
T?ng h?p votes:
- Weighted average c?a scores
- Consensus confidence (s? l??ng votes)
- Pattern confidence t? memory
- Combined confidence = 50% consensus + 50% pattern
```

### B??c 4: Final Selection
```
?i?u ch?nh cu?i:
- Boost ph?ng c? high confidence
- Boost ph?ng th??ng th?ng (pattern memory)
- Penalty ph?ng th??ng thua (anti-patterns)
- Boost safety score
- Penalty risk score
? Ch?n ph?ng c? final score cao nh?t
```

### B??c 5: Learning (sau k?t qu?)
```
C?p nh?t AI sau m?i v?n:
1. Update pattern/sequence memory
2. Reward/penalize formulas theo vote
   - Vote ??ng ? T?ng adapt + confidence
   - Vote sai ? Gi?m adapt + confidence
3. Update performance history
4. Meta-learning: ?i?u ch?nh learning rates
5. Sort formulas theo performance
```

## ?? ?u ?i?m so v?i c?c mode c?

| T?nh n?ng | VIP50 | VIP100 | VIP5000 | ULTRA AI |
|-----------|-------|--------|---------|----------|
| S? c?ng th?c | 50 | 100 | 5,000 | 10,000 |
| H?c t? k?t qu? | ? | ? | ? | ? |
| Pattern Recognition | ? | ? | ? | ? |
| Sequence Learning | ? | ? | ? | ? |
| Meta-Learning | ? | ? | ? | ? |
| Confidence Scoring | ? | ? | ? | ? |
| Memory System | ? | ? | ? | ? |
| Anti-Pattern Detection | ? | ? | ? | ? |
| Adaptive Bet Sizing | ? | ? | ? | ? |
| Dynamic Learning Rate | ? | ? | ? | ? |

## ?? Tham s? ?i?u ch?nh

### Learning Rate (META_LEARNING_RATE)
- M?c ??nh: 0.15
- T? ??ng ?i?u ch?nh d?a tr?n win rate
- Range: 0.08 - 0.25

### Confidence Threshold (CONFIDENCE_THRESHOLD)
- M?c ??nh: 0.7 (70%)
- Ng??ng tin c?y ?? t?ng/gi?m c??c

### Memory Size
- Pattern Memory: 1,000 patterns
- Anti-Patterns: 500 patterns
- Sequence Memory: Unlimited (defaultdict)

### Formula Performance
- Performance History: 50 l?n g?n nh?t/formula
- Adapt Range: 0.05 - 10.0
- Confidence Range: 0.05 - 0.95

## ?? L?u ?

1. **Kh?i t?o l?n ??u**: 10,000 c?ng th?c s? m?t ~2-3 gi?y
2. **Memory**: Tool s? d?ng ~100-200MB RAM
3. **Learning**: AI c?n ~20-30 v?n ?? "warm up"
4. **Best Performance**: Sau 100+ v?n, AI ??t hi?u su?t t?i ?u
5. **Backup**: Tool t? ??ng log v?o `escape_vip_ai_rebuild.log`

## ?? Tips ?? ??t hi?u qu? cao

1. **Ch?y li?n t?c**: AI c?ng ch?y l?u c?ng th?ng minh
2. **Martingale h?p l?**: Nh?n 2-2.5 l? t?i ?u
3. **Pause sau thua**: N?n set 1-2 v?n ?? tr?nh tilt
4. **Take Profit**: Ch?t l?i khi ??t +20-30% balance
5. **Stop Loss**: D?ng khi l? 10-15% ?? b?o v? v?n

## ?? Technical Details

### Architecture
```
Input (Room State) 
    ?
Feature Extraction (15+ features)
    ?
10,000 Formulas (Parallel Voting)
    ?
Ensemble Aggregation (Weighted)
    ?
Confidence Scoring
    ?
Memory Lookup (Patterns/Anti-patterns)
    ?
Final Selection
    ?
Adaptive Bet Sizing
    ?
Result ? Learning Loop
```

### Learning Algorithm
```python
# Pseudo-code
for each formula in FORMULAS:
    if formula voted correctly:
        formula.adapt *= (1 + learning_rate * 1.5)
        formula.confidence += 0.05
    else:
        formula.adapt *= (1 - learning_rate * 1.2)
        formula.confidence -= 0.08
    
    # Meta-learning
    if formula.win_rate > 0.6:
        formula.learning_rate *= 0.9  # Stable
    elif formula.win_rate < 0.4:
        formula.learning_rate *= 1.1  # Explore

# Global meta-learning
if overall_win_rate > 0.65:
    META_LEARNING_RATE *= 0.95  # Consolidate
elif overall_win_rate < 0.45:
    META_LEARNING_RATE *= 1.05  # Adapt faster
```

## ?? Troubleshooting

**Q: Tool ch?y ch?m?**
A: B?nh th??ng v?i 10,000 c?ng th?c. Optimization ?? ???c th?c hi?n.

**Q: Confidence th?p (<50%)?**
A: Ch? AI warm up 20-30 v?n. Sau ?? confidence s? t?ng d?n.

**Q: Thua nhi?u v?n ??u?**
A: AI ?ang h?c. Sau 50-100 v?n, t? l? th?ng s? c?i thi?n ??ng k?.

**Q: L?m sao xem performance?**
A: Check header: "AI Conf", "Patterns", "H?c" (Learning Rate)

## ?? License

Copyright by Duy Ho?ng | Ch?nh s?a by Kh?nh | N?ng c?p ULTRA AI by Claude

---

**?? ULTRA AI - Si?u Tr? Tu? Cho Tho?t Hi?m XWorld**

*"AI kh?ng ch? d? ?o?n, AI c?n h?c h?i v? ti?n h?a!"* ??
