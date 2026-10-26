from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


class ProjectCardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = README.read_text(encoding="utf-8")

    def test_required_sections_are_present(self) -> None:
        required_sections = (
            "## Исследовательский вопрос",
            "## Данные и срез",
            "## Эталон и метод",
            "## Метрика и критерий",
            "## Риск и границы",
            "## Следующий шаг",
            "## Что нужно уточнить",
            "## Проверка перед сохранением",
        )

        for section in required_sections:
            with self.subTest(section=section):
                self.assertIn(section, self.readme)

    def test_required_fields_are_filled(self) -> None:
        required_fields = (
            "TextWorldExpress",
            'gameName="sorting"',
            "Apache License 2.0",
            "Эталонный исход или прокси:",
            "Оцениваемый оценщик:",
            "Базовое сравнение:",
            "Метрика ошибки:",
            "Численный критерий успеха:",
            "Возможная ошибка оценщика:",
            "За границами проекта:",
            "Ответственный и срок:",
        )

        for field in required_fields:
            with self.subTest(field=field):
                self.assertIn(field, self.readme)

    def test_metric_is_computable(self) -> None:
        self.assertIn("disagreement_rate = (FP + FN) / N", self.readme)
        self.assertIn("`FP + FN`", self.readme)
        self.assertIn("перестановки `N`", self.readme)

    def test_control_questions_are_answered(self) -> None:
        answers = [
            line
            for line in self.readme.splitlines()
            if line.startswith(("1. **", "2. **", "3. **", "4. **"))
            and " Да:" in line
        ]
        self.assertEqual(len(answers), 4)


if __name__ == "__main__":
    unittest.main()
