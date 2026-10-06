"""Check persisted conversation discovery and metadata operations."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.conversations import ConversationStore


class TestConversationStore(unittest.TestCase):
    """Conversation metadata uses the shared grouped-state format."""

    def test_new_grouped_conversation_is_empty(self) -> None:
        """The empty-state check reads the canonical resume-state mapping."""

        with TemporaryDirectory() as temporary:
            store = ConversationStore(Path(temporary))
            conversation = store.ensure()

            self.assertTrue(store.is_empty(conversation.conversation_id))


if __name__ == "__main__":
    unittest.main()
