import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from click.testing import CliRunner

import simple_recorder
from src.config import Config


def _status_result(ready: bool, returncode: int = 0) -> subprocess.CompletedProcess:
    missing = [] if ready else ["nemotron-3-diarization/example.mlmodelc"]
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=json.dumps({
            "ready": ready,
            "cache_directory": "/private/tmp/isolated/models/speaker-diarization",
            "required_models": ["nemotron-3-diarization/example.mlmodelc"],
            "missing_models": missing,
        }) + "\n",
        stderr="",
    )


class SpeakerModelCliTests(unittest.TestCase):
    def test_status_reports_missing_models_as_a_successful_read(self):
        sidecar_result = subprocess.CompletedProcess(
            args=[],
            returncode=3,
            stdout=json.dumps({
                "ready": False,
                "cache_directory": "/private/tmp/isolated/models/speaker-diarization",
                "required_models": ["sortformer/example.mlmodelc"],
                "missing_models": ["sortformer/example.mlmodelc"],
            }) + "\n",
            stderr="",
        )
        with patch("src.transcriber._resolve_steno_diarize", return_value="/fake/steno-diarize"), \
             patch("subprocess.run", return_value=sidecar_result) as run:
            result = CliRunner().invoke(simple_recorder.speaker_model_status)

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertTrue(payload["success"])
        self.assertFalse(payload["ready"])
        run.assert_called_once_with(
            ["/fake/steno-diarize", "model-status"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=ANY,
        )

    def test_prepare_failure_is_structured_and_nonzero(self):
        sidecar_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="download failed\n"
        )
        with patch("src.transcriber._resolve_steno_diarize", return_value="/fake/steno-diarize"), \
             patch("subprocess.run", return_value=sidecar_result):
            result = CliRunner().invoke(simple_recorder.prepare_speaker_models)

        self.assertEqual(result.exit_code, 1, result.output)
        payload = json.loads(result.output)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"], "Speaker diarization model setup failed")
        self.assertNotIn("download failed", result.output)

    def test_prepare_accepts_coreml_diagnostics_around_json(self):
        payload = {
            "ready": True,
            "cache_directory": "/private/tmp/isolated/models/speaker-diarization",
            "required_models": ["sortformer/example.mlmodelc"],
            "missing_models": [],
        }
        sidecar_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "E5RT encountered an STL exception. msg = unordered_map::at: key not found."
                + json.dumps(payload)
                + "\nMetal teardown warning\n"
            ),
            stderr="steno-diarize: preparing speaker diarization models\n",
        )
        with patch("src.transcriber._resolve_steno_diarize", return_value="/fake/steno-diarize"), \
             patch("subprocess.run", return_value=sidecar_result):
            result = CliRunner().invoke(simple_recorder.prepare_speaker_models)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output), {"success": True, **payload})
        self.assertNotIn("E5RT", result.output)
        self.assertNotIn("Metal", result.output)

    def test_status_without_sidecar_is_structured(self):
        with patch("src.transcriber._resolve_steno_diarize", return_value=None):
            result = CliRunner().invoke(simple_recorder.speaker_model_status)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            json.loads(result.output),
            {
                "success": False,
                "ready": False,
                "error": "Speaker diarization is unavailable on this system",
            },
        )


class SpeakerModelEngineTests(unittest.TestCase):
    """The sidecar's model commands target the same engine meeting
    processing runs: the saved setting, or an explicit --engine while
    Settings prepares a choice it has not saved yet."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.config_path = Path(tmp.name) / "config.json"
        self.config = Config(config_path=self.config_path)
        patcher = patch("src.config.get_config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("STENOAI_DIARIZE_ENGINE", None)

    def _invoke(self, command, args, sidecar_result):
        with patch("src.transcriber._resolve_steno_diarize", return_value="/fake/steno-diarize"), \
             patch("subprocess.run", return_value=sidecar_result) as run:
            result = CliRunner().invoke(command, args)
        return result, run

    def test_status_targets_the_saved_engine(self):
        self.config._config["diarization_engine"] = "nemotron3"
        result, run = self._invoke(simple_recorder.speaker_model_status, [], _status_result(True))

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run.call_args.kwargs["env"]["STENOAI_DIARIZE_ENGINE"], "nemotron3")

    def test_prepare_engine_option_overrides_the_saved_engine(self):
        result, run = self._invoke(
            simple_recorder.prepare_speaker_models, ["--engine", "nemotron3"], _status_result(True)
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(self.config.get_diarization_engine(), "sortformer")
        self.assertEqual(run.call_args.args[0], ["/fake/steno-diarize", "prepare-models"])
        self.assertEqual(run.call_args.kwargs["env"]["STENOAI_DIARIZE_ENGINE"], "nemotron3")

    def test_engine_option_rejects_unknown_engines(self):
        result, run = self._invoke(
            simple_recorder.prepare_speaker_models, ["--engine", "pyannote"], _status_result(True)
        )

        self.assertNotEqual(result.exit_code, 0)
        run.assert_not_called()

    def test_set_nemotron3_is_refused_until_its_models_are_ready(self):
        result, run = self._invoke(
            simple_recorder.set_diarization_engine_cmd, ["nemotron3"], _status_result(False, returncode=3)
        )

        payload = json.loads(result.output)
        self.assertFalse(payload["success"])
        self.assertFalse(payload["models_ready"])
        self.assertEqual(run.call_args.kwargs["env"]["STENOAI_DIARIZE_ENGINE"], "nemotron3")
        self.assertEqual(Config(config_path=self.config_path).get_diarization_engine(), "sortformer")

    def test_set_nemotron3_persists_once_its_models_are_ready(self):
        result, _run = self._invoke(
            simple_recorder.set_diarization_engine_cmd, ["nemotron3"], _status_result(True)
        )

        self.assertEqual(json.loads(result.output), {"success": True, "engine": "nemotron3"})
        self.assertEqual(Config(config_path=self.config_path).get_diarization_engine(), "nemotron3")

    def test_set_sortformer_never_needs_the_sidecar(self):
        self.config.set_diarization_engine("nemotron3")
        result, run = self._invoke(
            simple_recorder.set_diarization_engine_cmd, ["sortformer"], _status_result(False)
        )

        self.assertEqual(json.loads(result.output), {"success": True, "engine": "sortformer"})
        run.assert_not_called()
        self.assertEqual(Config(config_path=self.config_path).get_diarization_engine(), "sortformer")

    def test_set_rejects_unknown_engines(self):
        result, run = self._invoke(
            simple_recorder.set_diarization_engine_cmd, ["pyannote"], _status_result(True)
        )

        payload = json.loads(result.output)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["valid_engines"], ["sortformer", "nemotron3"])
        run.assert_not_called()

    def test_cli_engine_choices_match_the_config_registry(self):
        choice = next(
            param for param in simple_recorder.prepare_speaker_models.params
            if param.name == "engine"
        )
        self.assertEqual(tuple(choice.type.choices), Config.VALID_DIARIZATION_ENGINES)


if __name__ == "__main__":
    unittest.main()
