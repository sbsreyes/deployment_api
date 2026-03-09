#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from setuptools import find_packages, setup

# ── Metadatos del paquete ─────────────────────────────────────────────────────
NAME            = "subscription-api"
DESCRIPTION     = "API de predicción de Subscription Status — Shopping Behavior."
EMAIL           = "tu_email@uniandes.edu.co"
AUTHOR          = "tu_nombre"
REQUIRES_PYTHON = ">=3.10.0"

ROOT_DIR         = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
MODEL_DIR        = ROOT_DIR / "model"

# Leer versión desde model/VERSION
about = {}
with open(MODEL_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return [
            line for line in fd.read().splitlines()
            if line and not line.startswith("#")
        ]


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    # Incluye tanto el paquete del modelo como el de la app FastAPI
    packages=find_packages(exclude=("tests",)),
    package_data={
        "model": [
            "VERSION",
            "config.yml",
            "datasets/*.csv",
            "trained/*.pkl",
        ]
    },
    install_requires=list_reqs(),
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
