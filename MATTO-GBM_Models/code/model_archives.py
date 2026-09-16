"""Utilities for preparing MATTO-GBM model archives for inference."""
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from zipfile import BadZipFile, ZipFile

REQUIRED_MODEL_FILES = ("model.pkl", "preprocessor.pkl")


def _valid_model_dir(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_MODEL_FILES)


def ensure_model_directory(model_dir):
    """Return a ready model directory, extracting an adjacent ZIP if needed."""
    model_dir = Path(model_dir)
    if _valid_model_dir(model_dir):
        return model_dir
    if model_dir.is_dir():
        raise ValueError(f"Incomplete model folder: {model_dir}")

    archive_path = model_dir.with_name(model_dir.name + ".zip")
    if not archive_path.is_file():
        return model_dir

    install_model_archive(archive_path, model_dir.parent, expected_model=model_dir.name)
    return model_dir


def _locate_model_prefix(names, model_name):
    """Find a ZIP prefix containing the two files required by one model."""
    normalized = [n.replace("\\", "/") for n in names if not n.endswith("/")]
    candidates = set()
    for name in normalized:
        if name.endswith("/model.pkl") or name == "model.pkl":
            candidates.add(name[:-len("model.pkl")])

    valid = []
    for prefix in candidates:
        if all(prefix + filename in normalized for filename in REQUIRED_MODEL_FILES):
            clean_parts = [p for p in prefix.strip("/").split("/") if p]
            score = 0
            if model_name in clean_parts:
                score += 10
            if clean_parts and clean_parts[-1] == model_name:
                score += 5
            if not clean_parts:
                score += 1
            valid.append((score, len(clean_parts), prefix))
    if not valid:
        return None
    valid.sort(key=lambda item: (-item[0], item[1]))
    return valid[0][2]


def install_model_archive(archive_path, destination_root, expected_model=None):
    """Install model files from a ZIP into ``destination_root/<model name>``.

    The ZIP may contain the files at its root or below arbitrary parent folders.
    If ``expected_model`` is supplied, a folder bearing that name is preferred.
    """
    archive_path = Path(archive_path)
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    model_name = expected_model or archive_path.stem

    try:
        with ZipFile(archive_path) as archive:
            prefix = _locate_model_prefix(archive.namelist(), model_name)
            if prefix is None:
                raise ValueError(
                    f"{archive_path.name} does not contain both model.pkl and "
                    "preprocessor.pkl for a usable model."
                )
            # When a specific model is requested, make sure this ZIP actually
            # contains that model. Without this check, a ZIP containing another
            # valid model folder could be installed under the wrong model name.
            if expected_model is not None:
                clean_parts = [p for p in prefix.strip("/").split("/") if p]

                if prefix == "":
                    # A flat ZIP is only safe when its filename identifies the model.
                    stem = archive_path.stem
                    # Uploaded archives may be prefixed with an ordering token (e.g. 01_).
                    if stem[:3].endswith("_") and stem[:2].isdigit():
                        stem = stem[3:]
                    if stem != expected_model:
                        raise ValueError(
                            f"Flat archive {archive_path.name} does not identify "
                            f"expected model {expected_model}."
                        )
                elif expected_model not in clean_parts:
                    raise ValueError(
                        f"{archive_path.name} contains model files under "
                        f"{'/'.join(clean_parts)}, not under expected model "
                        f"{expected_model}."
                    )

            # If no model name was imposed, infer it from the archive path when possible.
            if expected_model is None:
                parts = [p for p in prefix.strip("/").split("/") if p]
                if parts:
                    model_name = parts[-1]

            target = destination_root / model_name
            if _valid_model_dir(target):
                return target
            if target.exists():
                shutil.rmtree(target)

            with tempfile.TemporaryDirectory(prefix=".model_extract_", dir=destination_root) as temp:
                staged = Path(temp) / model_name
                staged.mkdir(parents=True)
                for filename in REQUIRED_MODEL_FILES:
                    with archive.open(prefix + filename) as source, (staged / filename).open("wb") as dest:
                        shutil.copyfileobj(source, dest)
                staged.rename(target)
    except BadZipFile as exc:
        raise ValueError(f"{archive_path.name} is not a valid ZIP archive.") from exc

    return target


def install_archives_for_models(archive_paths, destination_root, model_names):
    """Find and install all requested named models from one or more ZIP files."""
    destination_root = Path(destination_root)
    archive_paths = [Path(p) for p in archive_paths]
    installed = {}

    for model_name in model_names:
        for archive_path in archive_paths:
            try:
                target = install_model_archive(archive_path, destination_root, expected_model=model_name)
            except ValueError:
                continue
            if _valid_model_dir(target):
                installed[model_name] = target
                break
    return installed


def zenodo_record_id(reference: str) -> str:
    """Extract a Zenodo record id from a numeric id, DOI or Zenodo URL."""
    reference = (reference or "").strip()
    if reference.isdigit():
        return reference
    parsed = urllib.parse.urlparse(reference if "://" in reference else "https://" + reference)
    text = parsed.path or reference
    parts = [p for p in text.split("/") if p]
    for i, part in enumerate(parts):
        if part in {"records", "record"} and i + 1 < len(parts) and parts[i + 1].isdigit():
            return parts[i + 1]
    # DOI forms such as 10.5281/zenodo.12345678
    for token in reference.replace("/", ".").split("."):
        if token.isdigit() and len(token) >= 4:
            candidate = token
    if 'candidate' in locals():
        return candidate
    raise ValueError("Enter a Zenodo record URL, DOI, or numeric record ID.")


def download_zenodo_zip_files(reference: str, destination: Path, log=None):
    """Download all ZIP attachments from a public Zenodo record."""
    record_id = zenodo_record_id(reference)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    api_url = f"https://zenodo.org/api/records/{record_id}"
    if log is not None:
        log.append(f"Downloading model list from Zenodo record {record_id}...")
    try:
        with urllib.request.urlopen(api_url, timeout=30) as response:
            metadata = json.load(response)
    except Exception as exc:
        raise RuntimeError(f"Could not read Zenodo record {record_id}: {exc}") from exc

    zip_files = [item for item in metadata.get("files", []) if item.get("key", "").lower().endswith(".zip")]
    if not zip_files:
        raise ValueError(f"Zenodo record {record_id} does not contain any .zip files.")

    downloaded = []
    for item in zip_files:
        filename = Path(item["key"]).name
        links = item.get("links", {})
        url = links.get("content") or links.get("self")
        if not url:
            continue
        target = destination / filename
        if log is not None:
            log.append(f"Downloading {filename} from Zenodo...")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:
            raise RuntimeError(f"Could not download {filename} from Zenodo: {exc}") from exc
        downloaded.append(target)

    if not downloaded:
        raise RuntimeError(f"No ZIP files could be downloaded from Zenodo record {record_id}.")
    return downloaded
