"""
generate_requirements_api.py
----------------------------
Génère automatiquement requirements-api.txt à partir de environment.yml.
Ce script est la seule source de vérité pour les dépendances Docker.

Exclusions intentionnelles :
- streamlit : appartient au service Hugging Face, pas à l'API EC2

Usage :
    python generate_requirements_api.py
"""

import yaml
import re

EXCLUDED = {"streamlit"}
INPUT_FILE = "environment.yml"
OUTPUT_FILE = "requirements-api.txt"


def conda_to_pip_version(spec: str) -> str:
    """Convertit la syntaxe conda (=) en syntaxe pip (==)."""
    # "pandas=2.3.3" → "pandas==2.3.3"
    # "uvicorn" (sans version) → "uvicorn"
    return re.sub(r"(?<![=!<>])=(?!=)", "==", spec)


def extract_dependencies(env_file: str) -> list[str]:
    with open(env_file, "r") as f:
        env = yaml.safe_load(f)

    deps = []
    for dep in env.get("dependencies", []):
        if isinstance(dep, str):
            # Dépendance conda standard : on la convertit pour pip
            name = dep.split("=")[0].split(">")[0].split("<")[0].lower()
            if name in EXCLUDED or name in ("python", "pip", "setuptools"):
                continue
            deps.append(conda_to_pip_version(dep))
        elif isinstance(dep, dict) and "pip" in dep:
            # Bloc pip: dans environment.yml
            for pip_dep in dep["pip"]:
                if pip_dep.startswith("-e") or pip_dep.startswith("-r"):
                    continue  # on gère -e . séparément dans le Dockerfile
                name = pip_dep.split("==")[0].split(">=")[0].lower()
                if name in EXCLUDED:
                    continue
                deps.append(pip_dep)

    return deps


def main():
    deps = extract_dependencies(INPUT_FILE)

    with open(OUTPUT_FILE, "w") as f:
        f.write("# Auto-généré depuis environment.yml par generate_requirements_api.py\n")
        f.write("# NE PAS MODIFIER MANUELLEMENT — relancer le script si les dépendances changent.\n\n")
        for dep in deps:
            f.write(dep + "\n")

    print(f"✅ {OUTPUT_FILE} généré avec {len(deps)} dépendances.")
    print(f"   Exclusions : {EXCLUDED}")


if __name__ == "__main__":
    main()
