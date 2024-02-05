import unittest

import llm_utils


class TestCalculateCost(unittest.TestCase):
    def test_calculate_cost(self):
        self.assertAlmostEqual(
            llm_utils.calculate_cost(1000, 2000, "gpt-3.5-turbo"), 0.0055
        )
        self.assertAlmostEqual(
            llm_utils.calculate_cost(1000, 2000, "gpt-3.5-turbo-16k"), 0.011
        )
        self.assertAlmostEqual(llm_utils.calculate_cost(1000, 2000, "gpt-4"), 0.15)
        self.assertAlmostEqual(llm_utils.calculate_cost(1000, 2000, "gpt-4-32k"), 0.3)
        self.assertAlmostEqual(llm_utils.calculate_cost(1000, 2000, "gpt-4-32k"), 0.3)
        self.assertAlmostEqual(
            llm_utils.calculate_cost(10000, 20000, "gpt-3.5-turbo-1106"), 0.05
        )
        self.assertAlmostEqual(
            llm_utils.calculate_cost(10000, 2000, "gpt-4-1106-preview"), 0.16
        )
        self.assertRaises(ValueError, llm_utils.calculate_cost, 0, 0, "not-an-llm")


class TestWordWrap(unittest.TestCase):
    def test_word_wrap_except_code_blocks(self):
        self.assertEqual(llm_utils.word_wrap_except_code_blocks(""), "")
        self.assertEqual(llm_utils.word_wrap_except_code_blocks("a"), "a")
        self.assertEqual(llm_utils.word_wrap_except_code_blocks("a\nb"), "a\nb")
        self.assertEqual(llm_utils.word_wrap_except_code_blocks("a\n\nb"), "a\n\nb")
        self.assertEqual(llm_utils.word_wrap_except_code_blocks("a\n\n\nb"), "a\n\nb")
        self.assertEqual(
            llm_utils.word_wrap_except_code_blocks(
                "A very long line that should get broken up after a certain number of characters (specifically, 80)."
            ),
            "A very long line that should get broken up after a certain number of characters\n(specifically, 80).",
        )
        self.assertEqual(
            llm_utils.word_wrap_except_code_blocks(
                "AVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongWord."
            ),
            "AVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongWor\nd.",
        )
        self.assertEqual(
            llm_utils.word_wrap_except_code_blocks(
                "A VeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongWord."
            ),
            "A VeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongWo\nrd.",
        )
        self.assertEqual(
            llm_utils.word_wrap_except_code_blocks(
                "AVeryVeryVeryVeryVeryVeryVeryVeryVeryLongFirstWord AVeryVeryVeryVeryVeryLongSecondWord."
            ),
            "AVeryVeryVeryVeryVeryVeryVeryVeryVeryLongFirstWord\nAVeryVeryVeryVeryVeryLongSecondWord.",
        )

        self.assertEqual(
            llm_utils.word_wrap_except_code_blocks(
                "The answer to your problem is the following:\n"
                "```python\n"
                "a = b + c # Some inserted comment.\n"
                "def f():\n"
                "    pass # Indentation should be kept.\n"
                "```"
            ),
            "The answer to your problem is the following:\n"
            "\n"
            "```python\n"
            "a = b + c # Some inserted comment.\n"
            "def f():\n"
            "    pass # Indentation should be kept.\n"
            "```",
        )

        self.assertEqual(
            llm_utils.word_wrap_except_code_blocks(
                "First, I make a long introduction as to why the code is wrong. I may even apologize I didn't answer right the first time. The answer to your problem is the following:\n"
                "I'll add a second line here to make sure the line break is kept.\n"
                "```python\n"
                "a = b + c # Some inserted comment. Another long description basically saying the previous code was trash.\n"
                "```\n"
                "\n"
                "```\n"
                "A second code block here for the sake of it. Again, this should not wrap whatsoever and should stay as a single line.\n"
                "```"
            ),
            "First, I make a long introduction as to why the code is wrong. I may even\napologize I didn't answer right the first time. The answer to your problem is\nthe following:\n"
            "I'll add a second line here to make sure the line break is kept.\n"
            "\n"
            "```python\n"
            "a = b + c # Some inserted comment. Another long description basically saying the previous code was trash.\n"
            "```\n"
            "\n"
            "```\n"
            "A second code block here for the sake of it. Again, this should not wrap whatsoever and should stay as a single line.\n"
            "```",
        )


class TestNumberGroupOfLines(unittest.TestCase):
    def test_with_strip(self):
        self.assertEqual(llm_utils.number_group_of_lines([], 1, True), "")
        self.assertEqual(
            llm_utils.number_group_of_lines(["A", "B"], 2, True), "2 A\n3 B"
        )
        self.assertEqual(
            llm_utils.number_group_of_lines(["A", "B"], 11, True), "11 A\n12 B"
        )
        self.assertEqual(
            llm_utils.number_group_of_lines(["A", "B"], 9, True), " 9 A\n10 B"
        )
        self.assertEqual(
            llm_utils.number_group_of_lines(["", "A", "B"], 8, True), " 9 A\n10 B"
        )
        self.assertEqual(
            llm_utils.number_group_of_lines(["", "A", "B", ""], 8, True), " 9 A\n10 B"
        )
        self.assertEqual(
            llm_utils.number_group_of_lines(["A", "B", "", "", "", ""], 9, True),
            " 9 A\n10 B",
        )

    def test_without_strip(self):
        self.assertEqual(llm_utils.number_group_of_lines([], 1, False), "")
        self.assertEqual(
            llm_utils.number_group_of_lines(["A", "B"], 2, False), "2 A\n3 B"
        )
        self.assertEqual(
            llm_utils.number_group_of_lines(["A", "B"], 11, False), "11 A\n12 B"
        )
        self.assertEqual(
            llm_utils.number_group_of_lines(["A", "B"], 9, False), " 9 A\n10 B"
        )
        self.assertEqual(
            llm_utils.number_group_of_lines(["", "A", "B"], 8, False), " 8 \n 9 A\n10 B"
        )
        self.assertEqual(
            llm_utils.number_group_of_lines(["", "A", "B", ""], 8, False),
            " 8 \n 9 A\n10 B\n11 ",
        )
        self.assertEqual(
            llm_utils.number_group_of_lines(["A", "B", "", "", "", ""], 9, False),
            " 9 A\n10 B\n11 \n12 \n13 \n14 ",
        )


if __name__ == "__main__":
    unittest.main()
