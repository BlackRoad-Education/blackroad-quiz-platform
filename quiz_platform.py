"""
BlackRoad Quiz Platform
Adaptive quiz platform with SM-2 spaced repetition and difficulty adjustment.
"""

import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Difficulty(str, Enum):
    BEGINNER = "beginner"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    FILL_BLANK = "fill_blank"
    CODE = "code"


DIFFICULTY_ORDER = [
    Difficulty.BEGINNER, Difficulty.EASY, Difficulty.MEDIUM,
    Difficulty.HARD, Difficulty.EXPERT
]


# ---------------------------------------------------------------------------
# SM-2 Algorithm
# ---------------------------------------------------------------------------

@dataclass
class SM2Card:
    """SM-2 spaced repetition card state."""
    question_id: str
    user_id: str
    repetitions: int = 0
    easiness: float = 2.5        # EF — starts at 2.5
    interval_days: float = 1.0
    next_review_ts: float = field(default_factory=time.time)
    last_reviewed_ts: Optional[float] = None
    total_reviews: int = 0
    correct_streak: int = 0

    def update(self, quality: int):
        """
        Update card based on recall quality (0-5):
          5 = perfect, effortless recall
          4 = correct, small hesitation
          3 = correct, significant hesitation
          2 = incorrect, correct felt easy
          1 = incorrect, correct felt hard
          0 = complete blackout
        """
        quality = max(0, min(5, quality))
        self.total_reviews += 1
        self.last_reviewed_ts = time.time()

        if quality >= 3:
            # Correct response
            self.correct_streak += 1
            if self.repetitions == 0:
                self.interval_days = 1.0
            elif self.repetitions == 1:
                self.interval_days = 6.0
            else:
                self.interval_days = round(self.interval_days * self.easiness, 1)
            self.repetitions += 1
        else:
            # Incorrect — reset
            self.correct_streak = 0
            self.repetitions = 0
            self.interval_days = 1.0

        # Update easiness factor
        self.easiness = max(1.3, self.easiness + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        self.next_review_ts = time.time() + self.interval_days * 86400

    @property
    def is_due(self) -> bool:
        return time.time() >= self.next_review_ts

    @property
    def days_until_review(self) -> float:
        return max(0.0, (self.next_review_ts - time.time()) / 86400)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "user_id": self.user_id,
            "repetitions": self.repetitions,
            "easiness": round(self.easiness, 3),
            "interval_days": round(self.interval_days, 2),
            "next_review": round(self.next_review_ts, 2),
            "total_reviews": self.total_reviews,
            "correct_streak": self.correct_streak,
            "is_due": self.is_due,
            "days_until_review": round(self.days_until_review, 2),
        }


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------

@dataclass
class Question:
    """A quiz question."""
    question_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    topic: str = "general"
    type: QuestionType = QuestionType.MULTIPLE_CHOICE
    difficulty: Difficulty = Difficulty.MEDIUM
    text: str = ""
    options: List[str] = field(default_factory=list)
    correct_answer: str = ""
    explanation: str = ""
    hints: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    time_limit_s: Optional[int] = None
    points: int = 10
    created_at: float = field(default_factory=time.time)

    def check_answer(self, answer: str) -> bool:
        """Case-insensitive answer check."""
        return answer.strip().lower() == self.correct_answer.strip().lower()

    def to_dict(self) -> dict:
        return {
            "id": self.question_id,
            "topic": self.topic,
            "type": self.type.value,
            "difficulty": self.difficulty.value,
            "text": self.text,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "hints": self.hints,
            "tags": self.tags,
            "points": self.points,
        }


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------

@dataclass
class Quiz:
    """A collection of questions."""
    quiz_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    title: str = ""
    description: str = ""
    topic: str = "general"
    question_ids: List[str] = field(default_factory=list)
    time_limit_s: Optional[int] = None
    shuffle: bool = True
    pass_threshold: float = 0.7   # 70% correct to pass
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.quiz_id,
            "title": self.title,
            "description": self.description,
            "topic": self.topic,
            "question_count": len(self.question_ids),
            "pass_threshold": self.pass_threshold,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Attempt
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    question_id: str
    given_answer: str
    correct: bool
    time_taken_s: float = 0.0
    points_earned: int = 0
    hint_used: bool = False


@dataclass
class Attempt:
    """A user's attempt at a quiz."""
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    quiz_id: str = ""
    user_id: str = ""
    answers: List[Answer] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    passed: bool = False

    @property
    def score(self) -> int:
        return sum(a.points_earned for a in self.answers)

    @property
    def max_score(self) -> int:
        return sum(a.points_earned + (0 if a.correct else 10) for a in self.answers)

    @property
    def accuracy(self) -> float:
        if not self.answers:
            return 0.0
        return sum(1 for a in self.answers if a.correct) / len(self.answers)

    @property
    def duration_s(self) -> float:
        end = self.completed_at or time.time()
        return end - self.started_at

    def complete(self, pass_threshold: float = 0.7):
        self.completed_at = time.time()
        self.passed = self.accuracy >= pass_threshold

    def to_dict(self) -> dict:
        return {
            "id": self.attempt_id,
            "quiz_id": self.quiz_id,
            "user_id": self.user_id,
            "score": self.score,
            "accuracy": round(self.accuracy, 3),
            "passed": self.passed,
            "duration_s": round(self.duration_s, 2),
            "answers": len(self.answers),
            "correct": sum(1 for a in self.answers if a.correct),
        }


# ---------------------------------------------------------------------------
# User Stats
# ---------------------------------------------------------------------------

@dataclass
class UserTopicStats:
    user_id: str
    topic: str
    total_attempts: int = 0
    correct_count: int = 0
    current_difficulty: Difficulty = Difficulty.MEDIUM
    last_updated: float = field(default_factory=time.time)

    @property
    def accuracy(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.correct_count / self.total_attempts

    def record(self, correct: bool):
        self.total_attempts += 1
        if correct:
            self.correct_count += 1
        self.last_updated = time.time()

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "topic": self.topic,
            "total_attempts": self.total_attempts,
            "correct_count": self.correct_count,
            "accuracy": round(self.accuracy, 3),
            "current_difficulty": self.current_difficulty.value,
        }


# ---------------------------------------------------------------------------
# Quiz Platform
# ---------------------------------------------------------------------------

class QuizPlatform:
    """Main quiz platform manager with spaced repetition and adaptive difficulty."""

    def __init__(self):
        self._questions: Dict[str, Question] = {}
        self._quizzes: Dict[str, Quiz] = {}
        self._attempts: Dict[str, Attempt] = {}
        self._sm2_cards: Dict[str, SM2Card] = {}    # f"{user_id}:{question_id}"
        self._user_stats: Dict[str, UserTopicStats] = {}   # f"{user_id}:{topic}"
        self._active_attempts: Dict[str, Attempt] = {}   # user_id → active attempt

    # ------------------------------------------------------------------
    # Question management
    # ------------------------------------------------------------------

    def add_question(self, question: Question) -> Question:
        self._questions[question.question_id] = question
        return question

    def get_question(self, question_id: str) -> Optional[Question]:
        return self._questions.get(question_id)

    def list_questions(
        self,
        topic: Optional[str] = None,
        difficulty: Optional[Difficulty] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Question]:
        qs = list(self._questions.values())
        if topic:
            qs = [q for q in qs if q.topic == topic]
        if difficulty:
            qs = [q for q in qs if q.difficulty == difficulty]
        if tags:
            qs = [q for q in qs if any(t in q.tags for t in tags)]
        return qs

    # ------------------------------------------------------------------
    # Quiz management
    # ------------------------------------------------------------------

    def create_quiz(
        self,
        title: str,
        question_ids: List[str],
        topic: str = "general",
        description: str = "",
        pass_threshold: float = 0.7,
    ) -> Quiz:
        quiz = Quiz(
            title=title,
            description=description,
            topic=topic,
            question_ids=question_ids,
            pass_threshold=pass_threshold,
        )
        self._quizzes[quiz.quiz_id] = quiz
        return quiz

    def get_quiz(self, quiz_id: str) -> Optional[Quiz]:
        return self._quizzes.get(quiz_id)

    def list_quizzes(self, topic: Optional[str] = None) -> List[Quiz]:
        quizzes = list(self._quizzes.values())
        if topic:
            quizzes = [q for q in quizzes if q.topic == topic]
        return quizzes

    # ------------------------------------------------------------------
    # Attempt management
    # ------------------------------------------------------------------

    def start_attempt(self, quiz_id: str, user_id: str) -> Optional[Attempt]:
        quiz = self.get_quiz(quiz_id)
        if not quiz:
            return None
        attempt = Attempt(quiz_id=quiz_id, user_id=user_id)
        self._active_attempts[user_id] = attempt
        return attempt

    def submit_answer(
        self,
        user_id: str,
        question_id: str,
        given_answer: str,
        time_taken_s: float = 0.0,
        hint_used: bool = False,
    ) -> Optional[Answer]:
        attempt = self._active_attempts.get(user_id)
        if not attempt:
            return None
        question = self.get_question(question_id)
        if not question:
            return None

        correct = question.check_answer(given_answer)
        points = question.points if correct else 0
        if correct and hint_used:
            points = max(1, points // 2)

        answer = Answer(
            question_id=question_id,
            given_answer=given_answer,
            correct=correct,
            time_taken_s=time_taken_s,
            points_earned=points,
            hint_used=hint_used,
        )
        attempt.answers.append(answer)

        # Update user topic stats
        self._update_user_stats(user_id, question.topic, correct)
        # Update SM-2
        quality = 5 if correct and not hint_used else (3 if correct else 1)
        self.update_review_schedule(question_id, user_id, quality)

        return answer

    def complete_attempt(self, user_id: str) -> Optional[Attempt]:
        attempt = self._active_attempts.pop(user_id, None)
        if not attempt:
            return None
        quiz = self.get_quiz(attempt.quiz_id)
        threshold = quiz.pass_threshold if quiz else 0.7
        attempt.complete(threshold)
        self._attempts[attempt.attempt_id] = attempt
        return attempt

    def get_attempt_history(self, user_id: str) -> List[Attempt]:
        return [a for a in self._attempts.values() if a.user_id == user_id]

    # ------------------------------------------------------------------
    # Spaced Repetition (SM-2)
    # ------------------------------------------------------------------

    def _card_key(self, question_id: str, user_id: str) -> str:
        return f"{user_id}:{question_id}"

    def _get_or_create_card(self, question_id: str, user_id: str) -> SM2Card:
        key = self._card_key(question_id, user_id)
        if key not in self._sm2_cards:
            self._sm2_cards[key] = SM2Card(question_id=question_id, user_id=user_id)
        return self._sm2_cards[key]

    def update_review_schedule(self, question_id: str, user_id: str, quality: int):
        """Update SM-2 card for a question after review."""
        card = self._get_or_create_card(question_id, user_id)
        card.update(quality)

    def get_next_review(self, question_id: str, user_id: str) -> dict:
        """Get review schedule for a specific question."""
        card = self._get_or_create_card(question_id, user_id)
        return card.to_dict()

    def get_due_questions(self, user_id: str, limit: int = 20) -> List[Question]:
        """Get questions due for review for a user, sorted by overdue-ness."""
        due = []
        now = time.time()
        for key, card in self._sm2_cards.items():
            if not key.startswith(f"{user_id}:"):
                continue
            if card.is_due:
                question = self.get_question(card.question_id)
                if question:
                    overdue = now - card.next_review_ts
                    due.append((overdue, question))
        due.sort(key=lambda x: -x[0])
        return [q for _, q in due[:limit]]

    def get_review_stats(self, user_id: str) -> dict:
        cards = [c for k, c in self._sm2_cards.items() if k.startswith(f"{user_id}:")]
        due_count = sum(1 for c in cards if c.is_due)
        avg_easiness = sum(c.easiness for c in cards) / len(cards) if cards else 0
        return {
            "total_cards": len(cards),
            "due_now": due_count,
            "avg_easiness": round(avg_easiness, 3),
            "total_reviews": sum(c.total_reviews for c in cards),
        }

    # ------------------------------------------------------------------
    # Adaptive difficulty
    # ------------------------------------------------------------------

    def _stats_key(self, user_id: str, topic: str) -> str:
        return f"{user_id}:{topic}"

    def _update_user_stats(self, user_id: str, topic: str, correct: bool):
        key = self._stats_key(user_id, topic)
        if key not in self._user_stats:
            self._user_stats[key] = UserTopicStats(user_id=user_id, topic=topic)
        stats = self._user_stats[key]
        stats.record(correct)
        # Adjust difficulty based on sliding window accuracy
        if stats.total_attempts >= 5:
            self.adjust_difficulty(user_id, topic)

    def adjust_difficulty(self, user_id: str, topic: str) -> Difficulty:
        """
        Adjust user's difficulty for a topic based on rolling accuracy.
        >80% accuracy → increase difficulty
        <50% accuracy → decrease difficulty
        Otherwise     → maintain
        """
        key = self._stats_key(user_id, topic)
        if key not in self._user_stats:
            return Difficulty.MEDIUM
        stats = self._user_stats[key]
        idx = DIFFICULTY_ORDER.index(stats.current_difficulty)

        if stats.accuracy > 0.80 and idx < len(DIFFICULTY_ORDER) - 1:
            stats.current_difficulty = DIFFICULTY_ORDER[idx + 1]
        elif stats.accuracy < 0.50 and idx > 0:
            stats.current_difficulty = DIFFICULTY_ORDER[idx - 1]

        return stats.current_difficulty

    def get_next_question(self, user_id: str, topic: str) -> Optional[Question]:
        """
        Get the next best question for a user:
        1. Check SM-2 due queue first
        2. Fall back to adaptive difficulty questions
        """
        due = self.get_due_questions(user_id, limit=5)
        topic_due = [q for q in due if q.topic == topic]
        if topic_due:
            return topic_due[0]

        # Adaptive difficulty selection
        difficulty = self.get_user_difficulty(user_id, topic)
        candidates = self.list_questions(topic=topic, difficulty=difficulty)
        if not candidates:
            # Expand to adjacent difficulties
            idx = DIFFICULTY_ORDER.index(difficulty)
            adjacent = []
            if idx > 0:
                adjacent += self.list_questions(topic=topic, difficulty=DIFFICULTY_ORDER[idx - 1])
            if idx < len(DIFFICULTY_ORDER) - 1:
                adjacent += self.list_questions(topic=topic, difficulty=DIFFICULTY_ORDER[idx + 1])
            candidates = adjacent
        if candidates:
            return random.choice(candidates)
        return None

    def get_user_difficulty(self, user_id: str, topic: str) -> Difficulty:
        key = self._stats_key(user_id, topic)
        stats = self._user_stats.get(key)
        return stats.current_difficulty if stats else Difficulty.MEDIUM

    def get_user_stats(self, user_id: str) -> dict:
        stats_list = [s for k, s in self._user_stats.items() if k.startswith(f"{user_id}:")]
        return {
            "user_id": user_id,
            "topics": [s.to_dict() for s in stats_list],
            "review_stats": self.get_review_stats(user_id),
            "attempts": len(self.get_attempt_history(user_id)),
        }

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    def get_leaderboard(self, quiz_id: str, limit: int = 10) -> List[dict]:
        attempts = [a for a in self._attempts.values() if a.quiz_id == quiz_id and a.completed_at]
        best_by_user: Dict[str, Attempt] = {}
        for attempt in attempts:
            uid = attempt.user_id
            if uid not in best_by_user or attempt.score > best_by_user[uid].score:
                best_by_user[uid] = attempt
        ranked = sorted(best_by_user.values(), key=lambda a: (-a.score, a.duration_s))
        return [
            {
                "rank": i + 1,
                "user_id": a.user_id,
                "score": a.score,
                "accuracy": round(a.accuracy * 100, 1),
                "duration_s": round(a.duration_s, 2),
                "passed": a.passed,
            }
            for i, a in enumerate(ranked[:limit])
        ]

    # ------------------------------------------------------------------
    # Statistics & export
    # ------------------------------------------------------------------

    def platform_stats(self) -> dict:
        return {
            "questions": len(self._questions),
            "quizzes": len(self._quizzes),
            "attempts": len(self._attempts),
            "active_attempts": len(self._active_attempts),
            "sm2_cards": len(self._sm2_cards),
            "users": len(set(a.user_id for a in self._attempts.values())),
        }


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

def make_sample_questions(topic: str = "python", count: int = 20) -> List[Question]:
    """Generate sample quiz questions for testing."""
    python_questions = [
        ("What is the output of `print(type([]))` in Python?", "<class 'list'>",
         ["<class 'list'>", "<class 'array'>", "<class 'tuple'>", "list"],
         Difficulty.BEGINNER),
        ("Which keyword is used to define a generator function?", "yield",
         ["yield", "return", "generate", "next"],
         Difficulty.EASY),
        ("What does the `*args` syntax do?", "Collects extra positional arguments",
         ["Collects extra positional arguments", "Multiplies arguments", "Unpacks a list", "Defines optional args"],
         Difficulty.MEDIUM),
        ("What is the time complexity of dict lookup in Python?", "O(1)",
         ["O(1)", "O(n)", "O(log n)", "O(n²)"],
         Difficulty.MEDIUM),
        ("What is a Python descriptor?", "An object that defines __get__, __set__, or __delete__",
         ["An object that defines __get__, __set__, or __delete__",
          "A type annotation",
          "A property decorator",
          "A class decorator"],
         Difficulty.EXPERT),
    ]
    questions = []
    for i in range(count):
        q_data = python_questions[i % len(python_questions)]
        diff_idx = i % len(DIFFICULTY_ORDER)
        q = Question(
            topic=topic,
            type=QuestionType.MULTIPLE_CHOICE,
            difficulty=DIFFICULTY_ORDER[diff_idx],
            text=f"[Q{i+1}] {q_data[0]}",
            options=q_data[2] if len(q_data) > 2 else [],
            correct_answer=q_data[1],
            explanation=f"Explanation for question {i+1}.",
            points=10 * (diff_idx + 1),
            tags=[topic, DIFFICULTY_ORDER[diff_idx].value],
        )
        questions.append(q)
    return questions


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== BlackRoad Quiz Platform Demo ===")
    platform = QuizPlatform()

    # Add questions
    qs = make_sample_questions("python", 20)
    for q in qs:
        platform.add_question(q)
    print(f"\nAdded {len(qs)} questions")

    # Create quiz
    all_ids = [q.question_id for q in qs[:10]]
    quiz = platform.create_quiz("Python Fundamentals", all_ids, topic="python", pass_threshold=0.6)
    print(f"Created quiz: {quiz.title} ({quiz.quiz_id})")

    # Simulate a user attempt
    user_id = "user_alice"
    attempt = platform.start_attempt(quiz.quiz_id, user_id)
    print(f"\n[Attempt started for {user_id}]")

    q_list = platform.list_questions(topic="python")[:10]
    for q in q_list:
        # Simulate ~70% correct
        correct = random.random() < 0.7
        answer = q.correct_answer if correct else "wrong answer"
        platform.submit_answer(user_id, q.question_id, answer, time_taken_s=random.uniform(2, 15))

    completed = platform.complete_attempt(user_id)
    print(f"Attempt complete: score={completed.score}, accuracy={completed.accuracy:.0%}, passed={completed.passed}")

    # SM-2 stats
    print(f"\nSM-2 review stats: {platform.get_review_stats(user_id)}")

    # Due questions
    due = platform.get_due_questions(user_id, limit=5)
    print(f"Questions due for review: {len(due)}")

    # Adaptive difficulty
    diff = platform.adjust_difficulty(user_id, "python")
    print(f"Adjusted difficulty for 'python': {diff.value}")

    # Second user
    user2 = "user_bob"
    attempt2 = platform.start_attempt(quiz.quiz_id, user2)
    for q in q_list:
        correct = random.random() < 0.9    # Bob is smarter
        answer = q.correct_answer if correct else "wrong"
        platform.submit_answer(user2, q.question_id, answer)
    platform.complete_attempt(user2)

    # Leaderboard
    print(f"\nLeaderboard for quiz:")
    for entry in platform.get_leaderboard(quiz.quiz_id):
        print(f"  #{entry['rank']} {entry['user_id']}: {entry['score']}pts ({entry['accuracy']}%)")

    # Next question suggestion
    next_q = platform.get_next_question(user_id, "python")
    if next_q:
        print(f"\nNext suggested question for {user_id}: [{next_q.difficulty.value}] {next_q.text[:60]}...")

    # User stats
    print(f"\nUser stats: {platform.get_user_stats(user_id)}")

    # SM-2 card inspect
    if q_list:
        card_info = platform.get_next_review(q_list[0].question_id, user_id)
        print(f"\nSM-2 card for Q1: {card_info}")

    print(f"\nPlatform stats: {platform.platform_stats()}")


if __name__ == "__main__":
    demo()
