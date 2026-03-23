# blackroad-quiz-platform

> Adaptive quiz platform with spaced repetition

Part of the [BlackRoad OS](https://blackroad.io) ecosystem — [BlackRoad-Education](https://github.com/BlackRoad-Education)

---

# BlackRoad Quiz Platform

Adaptive quiz platform with SM-2 spaced repetition algorithm and per-topic difficulty adjustment.

## Features

- **SM-2 Algorithm**: Full SuperMemo-2 implementation (easiness factor, intervals, repetitions)
- **Spaced Repetition**: get_next_review, update_review_schedule, get_due_questions
- **Adaptive Difficulty**: Per-user per-topic accuracy tracking with auto difficulty adjustment
- **Question Types**: Multiple choice, true/false, short answer, fill-blank, code
- **Attempt Tracking**: Start/submit/complete flow with scoring and pass/fail
- **Leaderboard**: Per-quiz rankings by score and speed
- **User Stats**: Accuracy, difficulty level, review stats per topic

## Usage

```bash
python quiz_platform.py
```

## License

Proprietary — BlackRoad OS, Inc. All rights reserved.
