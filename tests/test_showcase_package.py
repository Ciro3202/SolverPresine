from __future__ import annotations

import json
import unittest

from presine.cli import ROOT, build_parser
from presine.policies.abstraction_resolving.abstract import load_blueprint


class ShowcasePackageTests(unittest.TestCase):
    def test_all_heads_up_blueprints_load(self) -> None:
        for round_size in range(2, 6):
            path = ROOT / "models" / "heads-up" / f"r{round_size}" / "blueprint.json"
            policy = load_blueprint(path)
            self.assertIsNotNone(policy)

    def test_multiplayer_bundle_is_present(self) -> None:
        path = ROOT / "models" / "multiplayer" / "3p" / "strategy-bundle.pt"
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 0)

    def test_cli_accepts_supported_modes(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["--players", "2"]).players, 2)
        self.assertEqual(parser.parse_args(["--players", "3"]).players, 3)

    def test_blueprints_are_strict_json(self) -> None:
        for path in (ROOT / "models" / "heads-up").glob("r*/blueprint.json"):
            with path.open("r", encoding="utf-8") as handle:
                self.assertIsInstance(json.load(handle), dict)


if __name__ == "__main__":
    unittest.main()
