from __future__ import annotations

import unittest

from qwen3_ollama import _extract_message_content


class Qwen3OllamaTests(unittest.TestCase):
    def test_extract_message_content_prefers_content(self):
        body = {
            "message": {
                "content": '{"ok": true}',
                "thinking": 'ignored',
            }
        }
        self.assertEqual(_extract_message_content(body), '{"ok": true}')


if __name__ == "__main__":
    unittest.main()
