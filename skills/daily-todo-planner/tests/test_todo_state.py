from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "skills" / "daily-todo-planner" / "scripts" / "todo_state.py"


class TodoStateTest(unittest.TestCase):
    def run_cli(self, *args: str) -> dict:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            check=True,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        return json.loads(proc.stdout)

    def write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_plan_infers_quadrants_orders_schedule_and_flags_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            tasks_path = base / "tasks.json"
            plan_path = base / "plan.json"
            md_path = base / "plan.md"
            self.write_json(
                tasks_path,
                {
                    "constraints": {
                        "work_start": "09:00",
                        "work_end": "12:30",
                        "buffer_ratio": 0,
                        "default_lunch": False,
                        "fixed_blocks": [{"label": "Standup", "start": "10:00", "end": "10:30"}],
                    },
                    "tasks": [
                        {
                            "id": "research-react-compiler",
                            "title": "调研 React Compiler GitHub 项目",
                            "outcome": "形成采用风险判断",
                            "importance": 5,
                            "urgency": 2,
                            "estimated_minutes": 60,
                            "type": "research",
                            "tags": ["github", "research"],
                            "sources": ["https://github.com/facebook/react"],
                        },
                        {
                            "id": "reply-partner",
                            "title": "回复合作方阻塞消息",
                            "outcome": "对方能继续推进",
                            "importance": 4,
                            "urgency": 5,
                            "estimated_minutes": 30,
                            "type": "communication",
                            "tags": ["message"],
                        },
                        {
                            "id": "submit-invoice",
                            "title": "提交今天到期的报销单",
                            "outcome": "报销单进入审批",
                            "importance": 2,
                            "urgency": 5,
                            "estimated_minutes": 20,
                            "type": "admin",
                            "tags": ["finance"],
                        },
                        {
                            "id": "clean-reading-list",
                            "title": "整理稍后阅读列表",
                            "outcome": "删除过期链接",
                            "importance": 1,
                            "urgency": 1,
                            "estimated_minutes": 25,
                            "type": "admin",
                            "tags": ["cleanup"],
                        },
                    ],
                },
            )

            self.run_cli(
                "--state-dir",
                str(base / "state"),
                "plan",
                "--tasks",
                str(tasks_path),
                "--date",
                "2026-06-15",
                "--output",
                str(plan_path),
                "--markdown",
                str(md_path),
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            by_id = {task["id"]: task for task in plan["scheduled"]}

            self.assertEqual([task["id"] for task in plan["scheduled"]], ["reply-partner", "submit-invoice", "research-react-compiler", "clean-reading-list"])
            self.assertEqual(by_id["reply-partner"]["quadrant"], "Q1")
            self.assertEqual(by_id["research-react-compiler"]["quadrant"], "Q2")
            self.assertEqual(by_id["submit-invoice"]["quadrant"], "Q3")
            self.assertEqual(by_id["clean-reading-list"]["quadrant"], "Q4")
            self.assertEqual(by_id["submit-invoice"]["start"], "09:30")
            self.assertEqual(by_id["research-react-compiler"]["start"], "10:30")
            self.assertEqual(by_id["research-react-compiler"]["end"], "11:30")
            self.assertEqual([task["id"] for task in plan["research_prefetch"]], ["research-react-compiler"])
            self.assertTrue(by_id["research-react-compiler"]["research_pack_path"].endswith("research-react-compiler.md"))
            self.assertIn("prefetch", md_path.read_text(encoding="utf-8"))

    def test_review_learning_adjusts_next_plan_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            state_dir = base / "state"
            first_tasks = base / "first_tasks.json"
            first_plan = base / "first_plan.json"
            actuals = base / "actuals.json"
            second_tasks = base / "second_tasks.json"
            second_plan = base / "second_plan.json"

            self.write_json(
                first_tasks,
                {
                    "constraints": {"work_start": "09:00", "work_end": "12:00", "buffer_ratio": 0, "default_lunch": False},
                    "tasks": [
                        {
                            "id": "paper-survey",
                            "title": "阅读并评估一篇论文",
                            "importance": 5,
                            "urgency": 2,
                            "estimated_minutes": 60,
                            "type": "research",
                            "tags": ["paper", "research"],
                        }
                    ],
                },
            )
            self.run_cli("--state-dir", str(state_dir), "plan", "--tasks", str(first_tasks), "--date", "2026-06-15", "--output", str(first_plan))
            self.write_json(actuals, {"actuals": [{"id": "paper-survey", "status": "done", "actual_minutes": 90}]})
            review = self.run_cli("--state-dir", str(state_dir), "review", "--actuals", str(actuals), "--plan", str(first_plan))

            self.assertEqual(review["profile"]["task_count"], 1)
            self.assertEqual(review["profile"]["types"]["research"]["mean_multiplier"], 1.5)

            self.write_json(
                second_tasks,
                {
                    "constraints": {"work_start": "09:00", "work_end": "12:00", "buffer_ratio": 0, "default_lunch": False},
                    "tasks": [
                        {
                            "id": "github-survey",
                            "title": "研究一个 GitHub 项目",
                            "importance": 5,
                            "urgency": 2,
                            "estimated_minutes": 60,
                            "type": "research",
                            "tags": ["github", "research"],
                        }
                    ],
                },
            )
            self.run_cli("--state-dir", str(state_dir), "plan", "--tasks", str(second_tasks), "--date", "2026-06-16", "--output", str(second_plan))
            plan = json.loads(second_plan.read_text(encoding="utf-8"))

            self.assertEqual(plan["scheduled"][0]["adjusted_estimate_minutes"], 90)
            self.assertEqual(plan["scheduled"][0]["estimate_multiplier"], 1.5)

    def test_prefetch_creates_research_pack_for_scheduled_research_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            state_dir = base / "state"
            tasks_path = base / "tasks.json"
            plan_path = base / "plan.json"
            answers_path = base / "answers.json"
            self.write_json(
                tasks_path,
                {
                    "constraints": {"work_start": "09:00", "work_end": "11:00", "buffer_ratio": 0, "default_lunch": False},
                    "tasks": [
                        {
                            "id": "repo-prefetch",
                            "title": "研究一个 GitHub 项目",
                            "outcome": "给出是否值得采用的短结论",
                            "importance": 5,
                            "urgency": 2,
                            "estimated_minutes": 45,
                            "type": "research",
                            "tags": ["github", "research"],
                            "sources": ["https://github.com/example/project"],
                        }
                    ],
                },
            )
            self.run_cli("--state-dir", str(state_dir), "plan", "--tasks", str(tasks_path), "--date", "2026-06-15", "--output", str(plan_path))
            self.write_json(
                answers_path,
                {
                    "answers": [
                        {
                            "id": "repo-prefetch",
                            "short_answer": "该项目适合作为候选，但需要先验证许可证和维护活跃度。",
                            "evidence": ["README shows active setup path", "Recent release exists"],
                            "risks": ["API surface may still change"],
                            "next_actions": ["Clone repo and run smoke test"],
                        }
                    ]
                },
            )
            result = self.run_cli("--state-dir", str(state_dir), "prefetch", "--plan", str(plan_path), "--answers", str(answers_path))

            self.assertEqual(result["packs"][0]["status"], "created")
            pack_path = Path(result["packs"][0]["path"])
            content = pack_path.read_text(encoding="utf-8")
            self.assertIn("# Research Pack: 研究一个 GitHub 项目", content)
            self.assertIn("Scheduled handoff: 09:00", content)
            self.assertIn("该项目适合作为候选", content)
            self.assertIn("Recent release exists", content)
            self.assertIn("Clone repo and run smoke test", content)


if __name__ == "__main__":
    unittest.main()
