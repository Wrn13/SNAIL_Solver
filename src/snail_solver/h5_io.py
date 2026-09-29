"""Nested result documents <-> HDF5, with a JSON fallback for old files.

Run outputs are deeply nested documents (an operating-point record plus a
``stages`` tree of chevron axes and population traces). As HDF5, arrays stay
binary with their dtype and shape, are gzip-compressed, and are individually
addressable (``f["stages/rabi/chevrons/i00003/metric"]``); ``h5ls -r`` browses it.

The encoding is generic and reversible:

======================  =========================================================
Python                  HDF5
======================  =========================================================
``dict``                group (insertion order tracked)
``list`` of scalars     dataset tagged ``_container="list"`` (comes back a list)
``list`` (other)        group tagged ``_container="list"``, one member/attribute
                        per entry named ``i00000``, ``i00001``, ... (sorted on read)
``ndarray``             dataset (gzip for anything sizeable)
scalars, ``None``       attribute on the parent group; ``None`` is ``h5py.Empty``;
                        a long string spills to its own dataset (64 KB attr limit)
``bytes``               ``uint8`` dataset tagged ``_container="bytes"`` (figures)
anything else           ``str(value)`` attribute
======================  =========================================================

Scalars live in the parent's attributes so ``h5ls -v`` shows fitted coefficients
at a glance; only large things are datasets.

A ``file.h5:/group/path`` address is accepted wherever a path is, and
:func:`save_doc` on one APPENDS, leaving the file's other runs alone -- so a sweep
keeps its whole fan-out in one file. :func:`attach_figure` embeds rendered figures
in the run's own ``figures`` group so they travel with the data::

    python -m snail_solver.h5_io run.h5                    # what is in there
    python -m snail_solver.h5_io run.h5 --extract figs/    # PNGs back on disk

:func:`load_doc` sniffs the magic number (not the name), so HDF5 and old JSON
files both load. :func:`save_doc` dispatches on the suffix: ``.json`` writes JSON,
a bare name gets ``.h5``, anything else is HDF5.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

#: Written to every file's root as ``format``; bump when the mapping changes.
SCHEMA = "snail_solver/nested-1"

_MAGIC = b"\x89HDF\r\n\x1a\n"

#: Where :func:`attach_figure` keeps a document's figures, relative to its group.
FIGURES_GROUP = "figures"

#: Recorded on each embedded figure.
_MIME = {".png": "image/png", ".pdf": "application/pdf", ".svg": "image/svg+xml",
         ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

#: Attributes the encoding owns (real keys never start with an underscore).
_RESERVED = ("_container", "_len")

#: Strings longer than this go to a dataset: attributes live in the object header,
#: capped at 64 KB in the default libver, so a long traceback would fail the write.
_MAX_ATTR_STR = 4096

#: Compress datasets from this many elements up (below it gzip overhead dominates).
_COMPRESS_FROM = 64


def _h5py():
    """Import h5py, with an actionable message for a stale environment."""
    try:
        import h5py
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(
            "h5py is required to read/write run outputs as HDF5. Install it with "
            "`uv sync` (it is a project dependency), or write JSON instead by "
            "giving the output an explicit .json suffix.") from exc
    return h5py


def split_address(target: str) -> tuple:
    """``"run.h5:/runs/eta1p8"`` -> ``("run.h5", "runs/eta1p8")``.

    Splits on ``:/`` (not a bare ``:``, which is legal in a filename). Returns
    ``(target, None)`` when there is no group part.
    """
    text = str(target)
    i = text.rfind(":/")
    if i < 0:
        return text, None
    return text[:i], text[i + 2:].strip("/") or None


def is_hdf5(path: str) -> bool:
    """Whether `path` is an HDF5 file, by magic number rather than by name."""
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == _MAGIC
    except OSError:
        return False


def _resolve(path: str, group: Optional[str]) -> tuple:
    """Split an address; an explicit `group` wins over the address's."""
    file_path, addr_group = split_address(path)
    return file_path, group or addr_group


def _ensure_parent(path: str) -> None:
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _stamp_format(fh) -> None:
    if "format" not in fh.attrs:
        fh.attrs["format"] = SCHEMA


# ===========================================================================
# writing
# ===========================================================================
def _is_scalar(v: Any) -> bool:
    """Scalars stored as attributes: numbers, bools, strings and None."""
    return v is None or isinstance(v, (bool, int, float, complex, str,
                                       np.bool_, np.number))


def _flat_kind(seq) -> Optional[str]:
    """``"num"``, ``"str"`` or None -- can this sequence become one dataset?

    Only a homogeneous sequence may (``np.asarray(["a", 1])`` would round-trip
    ``1`` as ``"1"``), and none holding None. A rectangular numeric list OF LISTS
    also qualifies (JSON-era population traces; ``tolist()`` restores them). Lists
    of NDARRAYS deliberately do not: each element would come back as a list.
    """
    if len(seq) == 0:
        return None
    if all(isinstance(v, (bool, int, float, complex, np.bool_, np.number))
           and not isinstance(v, str) for v in seq):
        return "num"
    if all(isinstance(v, str) for v in seq):
        return "str"
    if all(isinstance(v, (list, tuple)) for v in seq):
        try:
            arr = np.asarray(seq)                             # ragged -> ValueError
        except (ValueError, TypeError):
            return None
        return "num" if arr.dtype.kind in "biufc" and arr.ndim >= 2 else None
    return None


def _put_attr(grp, key: str, value: Any) -> None:
    """Store a scalar (or None) as an attribute; long strings spill to a dataset."""
    h5py = _h5py()
    if value is None:
        grp.attrs.create(key, h5py.Empty("f"))
        return
    if isinstance(value, str) and len(value) > _MAX_ATTR_STR:
        grp.create_dataset(key, data=value, dtype=h5py.string_dtype())
        return
    if isinstance(value, (np.bool_, np.number)):
        value = value.item()
    grp.attrs[key] = value


def _put_array(grp, key: str, arr: np.ndarray, *, as_list: bool = False) -> None:
    """Store an ndarray as a dataset, gzipped once it is worth gzipping."""
    h5py = _h5py()
    if arr.dtype.kind in "US":                                # unicode / bytes
        ds = grp.create_dataset(key, data=arr.astype(object),
                                dtype=h5py.string_dtype())
    elif arr.size >= _COMPRESS_FROM and arr.ndim >= 1:
        ds = grp.create_dataset(key, data=arr, compression="gzip",
                                compression_opts=4, shuffle=True)
    else:
        ds = grp.create_dataset(key, data=arr)
    if as_list:
        ds.attrs["_container"] = "list"


def _put_bytes(grp, key: str, data: bytes) -> None:
    """Store a blob (a rendered figure) as a tagged ``uint8`` dataset, uncompressed
    (PNG/PDF already are)."""
    ds = grp.create_dataset(key, data=np.frombuffer(data, dtype=np.uint8))
    ds.attrs["_container"] = "bytes"
    return ds


def _write_value(grp, key: str, value: Any) -> None:
    """Write one (key, value) into the group `grp`, dispatching on the value."""
    if "/" in str(key):
        raise ValueError(f"HDF5 names cannot contain '/', got key {key!r}")
    key = str(key)

    if isinstance(value, (bytes, bytearray)):
        _put_bytes(grp, key, bytes(value))
    elif _is_scalar(value):
        _put_attr(grp, key, value)
    elif isinstance(value, np.ndarray):
        if value.dtype == object:                             # ragged / mixed
            _write_value(grp, key, value.tolist())
        elif value.ndim == 0:
            _put_attr(grp, key, value.item())
        else:
            _put_array(grp, key, value)
    elif isinstance(value, dict):
        _write_tree(grp.create_group(key, track_order=True), value)
    elif isinstance(value, (list, tuple)):
        seq = list(value)
        kind = _flat_kind(seq)
        if kind == "num":
            _put_array(grp, key, np.asarray(seq), as_list=True)
        elif kind == "str":
            _put_array(grp, key, np.asarray(seq, dtype=object), as_list=True)
        else:
            sub = grp.create_group(key, track_order=True)
            sub.attrs["_container"] = "list"
            sub.attrs["_len"] = len(seq)
            for i, item in enumerate(seq):
                _write_value(sub, f"i{i:05d}", item)
    else:                                                     # last resort
        _put_attr(grp, key, str(value))


def _write_tree(grp, tree: Dict[str, Any]) -> None:
    """Write every entry of a dict into an (already created) group."""
    for key, value in tree.items():
        _write_value(grp, key, value)


def save_tree(path: str, tree: Dict[str, Any],
              attrs: Optional[Dict[str, Any]] = None, *,
              group: Optional[str] = None) -> str:
    """Write a nested document to `path` (or a ``file.h5:/group`` address) as HDF5.

    `tree` keys must be strings without ``/``. `attrs` is provenance (command
    line, device, timestamp) written to the destination group's attributes,
    kept out of `tree` so it is never mistaken for data. With a `group`, the file
    is APPENDED to and only that group replaced (HDF5 does not reclaim the old
    group's space). Returns the address written.
    """
    h5py = _h5py()
    path, group = _resolve(path, group)
    _ensure_parent(path)
    with h5py.File(path, "a" if group else "w", track_order=True) as fh:
        _stamp_format(fh)
        if group:
            if group in fh:
                del fh[group]                                 # replace, don't merge
            dest = fh.create_group(group, track_order=True)
        else:
            dest = fh
        for k, v in (attrs or {}).items():
            _put_attr(dest, str(k), v)
        _write_tree(dest, tree)
    return f"{path}:/{group}" if group else path


# ===========================================================================
# reading
# ===========================================================================
def _read_attr(value: Any) -> Any:
    """Turn one stored attribute back into a Python value."""
    h5py = _h5py()
    if isinstance(value, h5py.Empty):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (np.bool_, np.number)):
        return value.item()
    if isinstance(value, np.ndarray):                         # h5py packs some
        return value.tolist()                                 # attrs as arrays
    return value


def _read_dataset(ds) -> Any:
    """Turn one dataset back into an ndarray, a list, or a scalar string."""
    h5py = _h5py()
    if h5py.check_string_dtype(ds.dtype):
        if ds.shape == ():
            return ds.asstr()[()]
        return list(ds.asstr()[...])
    arr = ds[...]
    kind = _read_attr(ds.attrs.get("_container"))
    if kind == "bytes":
        return arr.tobytes()
    if kind == "list":
        return arr.tolist()
    return arr


def _read_group(grp) -> Any:
    """Turn one group back into a dict, or into a list if it is tagged as one."""
    is_list = _read_attr(grp.attrs.get("_container")) == "list"
    items = {}
    for key, value in grp.attrs.items():
        if key not in _RESERVED and not (key == "format" and grp.name == "/"):
            items[key] = _read_attr(value)
    for key in grp.keys():
        node = grp[key]
        items[key] = (_read_group(node) if hasattr(node, "keys")
                      else _read_dataset(node))
    if is_list:
        return [items[k] for k in sorted(items)]
    return items


def load_tree(path: str, *, group: Optional[str] = None) -> Dict[str, Any]:
    """Read a document written by :func:`save_tree` back into nested dicts.

    With a group (address or `group`), only that group is read. Root provenance
    attributes come back as top-level keys, except ``format`` (it describes the
    file, not the run).
    """
    h5py = _h5py()
    path, group = _resolve(path, group)
    with h5py.File(path, "r") as fh:
        if group is not None and group not in fh:
            raise KeyError(f"{path} has no group {group!r} "
                           f"(top level: {', '.join(fh.keys()) or 'empty'})")
        return _read_group(fh[group] if group else fh)


def has_group(path: str, group: str) -> bool:
    """Whether an HDF5 file holds this group -- False for a JSON or missing file."""
    if not is_hdf5(path):
        return False
    h5py = _h5py()
    with h5py.File(path, "r") as fh:
        return group.strip("/") in fh


# ===========================================================================
# figures
# ===========================================================================
def _figures_group(fh, addr: Optional[str], *, create: bool):
    """The ``figures`` group of the document at `addr` (root when None)."""
    parent = fh
    if addr:
        if addr not in fh:
            if not create:
                return None
            parent = fh.create_group(addr, track_order=True)
        else:
            parent = fh[addr]
    if FIGURES_GROUP in parent:
        return parent[FIGURES_GROUP]
    return parent.create_group(FIGURES_GROUP, track_order=True) if create else None


def attach_figure(target: str, name: str, image_path: str) -> str:
    """Embed the rendered file `image_path` into the run file it belongs to.

    `target` is the run file or a ``file.h5:/runs/eta1p8`` address; the figure
    lands in that document's own ``figures`` group (file appended, never
    truncated). `name` says what it shows (``rabi``, ``chirp_ridge``); re-attaching
    replaces it (old bytes are not reclaimed -- ``h5repack`` to compact). The
    suffix picks the recorded MIME type. Returns the embedded figure's address.
    """
    h5py = _h5py()
    file_path, addr = split_address(target)
    with open(image_path, "rb") as fh:
        data = fh.read()
    with h5py.File(file_path, "a", track_order=True) as fh:
        _stamp_format(fh)
        figs = _figures_group(fh, addr, create=True)
        if name in figs:
            del figs[name]                                    # replace, don't merge
        ds = _put_bytes(figs, name, data)
        ds.attrs["filename"] = os.path.basename(image_path)
        ds.attrs["mime"] = _MIME.get(os.path.splitext(image_path)[1].lower(),
                                     "application/octet-stream")
    where = f"{addr}/{FIGURES_GROUP}" if addr else FIGURES_GROUP
    return f"{file_path}:/{where}/{name}"


def attach_figures(target: str, figures: Dict[str, str]) -> int:
    """Embed several figures; returns how many were stored.

    A figure that cannot be stored (missing, unreadable, locked) is skipped, never
    raised: it is not worth failing a finished run over.
    """
    n = 0
    for name, path in (figures or {}).items():
        if not path:
            continue
        try:
            attach_figure(target, name, path)
            n += 1
        except Exception:                                     # never fatal
            continue
    return n


def figure_names(target: str) -> list:
    """The names of the figures embedded in a document (``[]`` if none)."""
    file_path, addr = split_address(target)
    if not is_hdf5(file_path):
        return []
    h5py = _h5py()
    with h5py.File(file_path, "r") as fh:
        figs = _figures_group(fh, addr, create=False)
        return list(figs.keys()) if figs is not None else []


def extract_figures(target: str, outdir: str = ".",
                    names: Optional[Sequence[str]] = None) -> list:
    """Write a document's embedded figures back out under their original
    filenames; returns the paths written, in file order."""
    file_path, addr = split_address(target)
    out = []
    if not is_hdf5(file_path):                                # a JSON run has none
        return out
    h5py = _h5py()
    os.makedirs(outdir, exist_ok=True)
    with h5py.File(file_path, "r") as fh:
        figs = _figures_group(fh, addr, create=False)
        if figs is None:
            return out
        for name in (names if names is not None else figs.keys()):
            ds = figs[name]
            fname = _read_attr(ds.attrs.get("filename")) or f"{name}.png"
            path = os.path.join(outdir, fname)
            with open(path, "wb") as fp:
                fp.write(ds[...].tobytes())
            out.append(path)
    return out


# ===========================================================================
# format-dispatching front door
# ===========================================================================
def resolve_out_path(path: str) -> str:
    """The path :func:`save_doc` would write: a bare name gets ``.h5``."""
    root, ext = os.path.splitext(path)
    return path if ext else root + ".h5"


def save_doc(path: str, doc: Dict[str, Any],
             attrs: Optional[Dict[str, Any]] = None, *,
             group: Optional[str] = None) -> str:
    """Write `doc` as HDF5, or as JSON if `path` ends in ``.json``.

    With a group (or ``file.h5:/group`` address) the write appends into the file;
    a group on a ``.json`` target is a ValueError. Returns the address actually
    written (a bare name gains ``.h5``; a grouped write returns ``file:/group``).
    """
    file_path, group = _resolve(path, group)
    if os.path.splitext(file_path)[1].lower() == ".json":
        if group:
            raise ValueError(
                f"cannot write group {group!r} into the JSON file {file_path!r}: "
                f"only HDF5 holds several documents in one file")
        _ensure_parent(file_path)
        with open(file_path, "w") as fh:
            json.dump(dict(doc, **(attrs or {})), fh, indent=2, default=_plain)
        return file_path
    return save_tree(resolve_out_path(file_path), doc, attrs=attrs, group=group)


def load_doc(path: str, *, group: Optional[str] = None) -> Dict[str, Any]:
    """Read a document written by :func:`save_doc`, HDF5 or JSON (by magic
    number, not suffix). `path` may be a ``file.h5:/group`` address."""
    file_path, group = _resolve(path, group)
    if is_hdf5(file_path):
        return load_tree(file_path, group=group)
    if group:
        raise ValueError(f"{file_path} is not HDF5, so it has no group {group!r}")
    with open(file_path) as fh:
        return json.load(fh)


def _plain(o: Any) -> Any:
    """``json.dump`` fallback: arrays to lists, numpy scalars to Python ones."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ===========================================================================
# CLI -- look inside a run file, and get its figures back out
# ===========================================================================
def _describe(node, indent: str = "  ") -> None:
    """Print one level of an HDF5 file: groups, dataset shapes, scalars."""
    attrs = {k: v for k, v in node.attrs.items()
             if not k.startswith("_") and k != "format"}     # shown in the header
    for key, value in attrs.items():
        shown = _read_attr(value)
        if isinstance(shown, str) and len(shown) > 70:
            shown = shown[:67] + "..."
        print(f"{indent}{key} = {shown!r}")
    for key in node.keys():
        child = node[key]
        if hasattr(child, "keys"):
            kind = "list" if child.attrs.get("_container") == "list" else "group"
            print(f"{indent}{key}/  ({kind}, {len(child.keys())} members)")
        elif _read_attr(child.attrs.get("_container")) == "bytes":
            name = _read_attr(child.attrs.get("filename")) or key
            print(f"{indent}{key}  {child.size / 1024:.0f} KB  {name}")
        else:
            print(f"{indent}{key}  {child.shape} {child.dtype}")


def main() -> None:
    """CLI entry point: inspect a run file, or extract its figures."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m snail_solver.h5_io",
        description="Look inside a run file written by tune_up / tune_up_sweep / "
                    "post_chirp, or write its embedded figures back out as PNGs.")
    ap.add_argument("target", help="run file, or a file.h5:/runs/eta1p8 address")
    ap.add_argument("--extract", metavar="DIR", default=None,
                    help="write the embedded figures into DIR (they keep the "
                         "filenames they were rendered under)")
    ap.add_argument("--figure", action="append", default=None, metavar="NAME",
                    help="extract only this figure; repeatable")
    args = ap.parse_args()

    file_path, addr = split_address(args.target)
    if args.extract:
        paths = extract_figures(args.target, args.extract, names=args.figure)
        for path in paths:
            print(f"wrote {path}")
        if not paths:
            print(f"{args.target}: no embedded figures")
        return

    if not is_hdf5(file_path):
        raise SystemExit(f"{file_path} is not HDF5 (JSON runs hold no figures)")
    h5py = _h5py()
    with h5py.File(file_path, "r") as fh:
        node = fh[addr] if addr else fh
        print(f"{args.target}  [{_read_attr(fh.attrs.get('format'))}]")
        _describe(node)
    figs = figure_names(args.target)
    if figs:
        print(f"  figures: {', '.join(figs)}  "
              f"(--extract DIR writes them back out)")


if __name__ == "__main__":
    main()
