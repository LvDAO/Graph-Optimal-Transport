from __future__ import annotations

from conan import ConanFile


class GraphOTConan(ConanFile):
    name = "graphot-core"
    version = "0.0.3"
    package_type = "application"

    settings = "os", "arch", "build_type"
    generators = "CMakeDeps"

    requires = (
        "pybind11/2.13.6",
    )

    def configure(self) -> None:
        self.options["pybind11/*"].shared = False
