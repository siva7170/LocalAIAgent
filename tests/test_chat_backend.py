import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai.chat_backend import _get_llama_cpp_model, chat
from config import (
    LLAMA_CPP,
    LLAMA_CPP_N_CTX,
    LLAMA_CPP_N_GPU_LAYERS,
    MODEL,
    OLLAMA,
)


class ChatBackendTests(unittest.TestCase):

    def test_ollama_backend_preserves_json_chat_options(self):
        messages = [{"role": "user", "content": "make a plan"}]
        chat_method = Mock(
            return_value=[
                SimpleNamespace(message=SimpleNamespace(content="{")),
                SimpleNamespace(message=SimpleNamespace(content="}")),
            ]
        )
        client = SimpleNamespace(chat=chat_method)

        with patch("ai.chat_backend.USE", OLLAMA), patch(
            "ai.chat_backend._get_ollama_client", return_value=client
        ) as get_client:
            result = chat(messages, temperature=0.1, json_mode=True)

        self.assertEqual(result, "{}")
        get_client.assert_called_once()
        chat_method.assert_called_once_with(
            model=MODEL,
            messages=messages,
            options={"temperature": 0.1},
            stream=True,
            format="json",
        )

    def test_llama_cpp_backend_calls_loaded_model(self):
        messages = [{"role": "user", "content": "make a plan"}]
        create_chat_completion = Mock(
            return_value=iter([
                {"choices": [{"delta": {"content": "{"}}]},
                {"choices": [{"delta": {"content": "}"}}]},
            ])
        )
        model = SimpleNamespace(
            create_chat_completion=create_chat_completion
        )

        with patch("ai.chat_backend.USE", LLAMA_CPP), patch(
            "ai.chat_backend._get_llama_cpp_model", return_value=model
        ) as get_model:
            result = chat(messages, temperature=0.1, json_mode=True)

        self.assertEqual(result, "{}")
        get_model.assert_called_once()
        create_chat_completion.assert_called_once_with(
            messages=messages,
            temperature=0.1,
            stream=True,
            response_format={"type": "json_object"},
        )

    def test_llama_cpp_backend_omits_json_format_for_normal_chat(self):
        create_chat_completion = Mock(
            return_value=iter([
                {"choices": [{"delta": {"content": "done"}}]},
            ])
        )
        model = SimpleNamespace(
            create_chat_completion=create_chat_completion
        )

        with patch("ai.chat_backend.USE", LLAMA_CPP), patch(
            "ai.chat_backend._get_llama_cpp_model", return_value=model
        ):
            result = chat(
                [{"role": "user", "content": "respond"}],
                temperature=0.3,
            )

        self.assertEqual(result, "done")
        create_chat_completion.assert_called_once_with(
            messages=[{"role": "user", "content": "respond"}],
            temperature=0.3,
            stream=True,
        )

    def test_llama_cpp_backend_loads_configured_gguf_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.Q4_K_M.gguf"
            model_path.touch()
            llama_module = types.ModuleType("llama_cpp")
            llama_constructor = Mock(return_value=object())
            setattr(llama_module, "Llama", llama_constructor)

            _get_llama_cpp_model.cache_clear()
            try:
                with patch(
                    "ai.chat_backend.LLAMA_CPP_MODEL_PATH",
                    str(model_path),
                ), patch.dict("sys.modules", {"llama_cpp": llama_module}):
                    model = _get_llama_cpp_model()
            finally:
                _get_llama_cpp_model.cache_clear()

        self.assertIsNotNone(model)
        llama_constructor.assert_called_once_with(
            model_path=str(model_path),
            n_ctx=LLAMA_CPP_N_CTX,
            n_gpu_layers=LLAMA_CPP_N_GPU_LAYERS,
            verbose=False,
        )

    def test_unknown_backend_fails_explicitly(self):
        with patch("ai.chat_backend.USE", "unknown"):
            with self.assertRaisesRegex(ValueError, "Unsupported model backend"):
                chat([], temperature=0.1)


if __name__ == "__main__":
    unittest.main()
