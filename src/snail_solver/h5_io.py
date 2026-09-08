"""Nested result documents <-> HDF5, with a JSON fallback for old files.

Why HDF5 for run outputs
------------------------
The tools here return one deeply nested document per run -- an operating-point
record plus a ``stages`` tree that carries every chevron's full offset axis,
time axis and population traces. Written as JSON that document is a wall of
decimal text: a single ``tune_up`` sweep is tens of MB of ``0.123456789012``,
it loses every array's dtype and shape (``np.asarray`` has to guess them back),
and nothing can read a single chevron out of it without parsing the whole file.

HDF5 fixes exactly those three things. Arrays stay binary and keep their dtype
and shape, they are gzip-compressed on the way out (float traces compress ~3-5x),
and the file is a directory tree -- ``h5ls -r run.h5`` lists every stage, and
``f["stages/rabi/chevrons/i00003/metric"]`` reads one chevron's envelope without
touching the rest. That is what makes a directory of runs browsable rather than
merely stored.

The encoding
------------
The mapping is generic (nothing here knows what a chevron is), reversible, and
chosen so the file reads well in ``h5ls``/HDFView rather than to be clever:

======================  =========================================================
Python                  HDF5
======================  =========================================================
``dict``                group (keys become members; insertion order is tracked)
``list`` of scalars     dataset, tagged ``_container="list"`` so it comes back
                        as a ``list`` and not an ``ndarray``
``list`` (anything      group tagged ``_container="list"``, with one member (or
else)                   attribute, for scalar elements) per entry named
                        ``i00000``, ``i00001``, ... -- sorted on read, so order
                        survives
``ndarray``             dataset (gzip for anything sizeable)
scalars, ``None``       attribute on the parent group; ``None`` is a native HDF5
                        null (``h5py.Empty``), and a long string spills to its
                        own dataset rather than blowing the 64 KB attribute limit
``bytes``               dataset of ``uint8`` tagged ``_container="bytes"`` --
                        this is what carries a rendered figure
anything else           ``str(value)`` attribute -- the same last resort
                        ``json.dump(..., default=...)`` used to take
======================  =========================================================

Scalars living in the PARENT's attributes (rather than as one-element datasets)
is what keeps the tree legible: ``h5ls -v run.h5/stages/chirp`` shows every
fitted coefficient at a glance, and only the genuinely large things are datasets.

One file, many runs
-------------------
A group inside a file is addressable as ``file.h5:/group/path`` wherever a path
is taken, and :func:`save_doc` on such an address APPENDS -- every other run in
the file is left alone. That is what lets a sweep keep its whole fan-out (the
summary, and each eta's complete tune-up) in ONE file, while
``tune_up --replot sweep.h5:/runs/eta1p8`` still replots a single point and reads
only that group.

Figures live in the run file too
--------------------------------
A run's figures are the same measurement as its arrays, so they are stored with
it rather than in a parallel ``figs/`` tree that has to be kept in step by hand:
:func:`attach_figure` puts each rendered PNG under the run's own ``figures``
group (``/figures`` for a single run, ``/runs/eta1p8/figures`` inside a sweep
file), keyed by what it shows -- ``rabi``, ``chirp_ridge``, ``post_chirp``. The
tools still write the ``--plot`` file on disk as before; the embedded copy is the
one that travels with the data, so a run file copied off the cluster carries its
own pictures and cannot be paired with the wrong ones.

They come back out as files with :func:`extract_figures`, or straight from the
command line::

    python -m snail_solver.h5_io run.h5                    # what is in there
    python -m snail_solver.h5_io run.h5 --extract figs/    # PNGs back on disk

Reading old runs
----------------
:func:`load_doc` sniffs the file's magic number, not its name, so every
``--replot`` accepts both the HDF5 files written now and every JSON file written
before -- and a JSON document round-trips through this module unchanged in
meaning (lists stay lists, ``None`` stays ``None``). :func:`save_doc` dispatches
on the SUFFIX instead: ``.json`` still writes JSON on request, a bare name gets
``.h5`` appended, and anything else is HDF5.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

#: Written to every file's root as ``format``; bump when the mapping changes.
SCHEMA = "snail_solver/nested-1"

#: Suffixes routed to HDF5 by :func:`save_doc`. Everything else that is not
#: ``.json`` also becomes HDF5 -- this list only documents the usual spellings.
H5_SUFFIXES = (".h5", ".hdf5", ".he5")

_MAGIC = b"\x89HDF\r\n\x1a\n"

#: Where :func:`attach_figure` keeps a document's rendered figures, relative to
#: that document's own group.
FIGURES_GROUP = "figures"

#: Recorded on each embedded figure so a reader (or a browser) knows what it is.
_MIME = {".png": "image/png", ".pdf": "application/pdf", ".svg": "image/svg+xml",
         ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

#: Attributes the encoding owns. Real keys never start with an underscore
#: (they are Python identifiers from the result dicts), so this cannot collide.
_RESERVED = ("_container", "_len")

#: Strings longer than this go to a dataset. HDF5 keeps attributes in the object
#: header, which is capped at 64 KB in the default (non-``latest``) libver; a long
#: log message or traceback in a result dict would otherwise fail the whole write.
_MAX_ATTR_STR = 4096

#: Compress datasets from this many elements up. Below it the gzip filter's own
#: per-chunk overhead is larger than what it saves.
_COMPRESS_FROM = 64


def _h5py():
    """Import h5py, with an actionable message if it is missing.

    It is a hard dependency of this project, but a stale environment (a cluster
    venv built before it was added) fails here rather than at import time of
    whichever tool happened to be run.
    """
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

    The separator is ``:/`` rather than a bare ``:`` so an ordinary path is never
    mistaken for an address -- a colon is legal in a filename, but ``:/`` inside
    one is not something these tools ever produce. Returns ``(target, None)``
    when there is no group part.
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


# ===========================================================================
# writing
# ===========================================================================
def _is_scalar(v: Any) -> bool:
    """Scalars stored as attributes: numbers, bools, strings and None."""
    return v is None or isinstance(v, (bool, int, float, complex, str,
                                       np.bool_, np.number))


def _flat_kind(seq) -> Optional[str]:
    """``"num"``, ``"str"`` or None -- can this sequence become one dataset?

    Only a homogeneous sequence may: ``np.asarray(["a", 1])`` silently makes an
    array of STRINGS, which would round-trip ``1`` back as ``"1"``. A sequence
    holding None cannot either (there is no null in a numeric dataset), so those
    fall through to the group form, where None is a real HDF5 null.

    A list OF LISTS gets the same treatment when it is rectangular and numeric --
    which is how a JSON run's population traces arrive, and how they are rebuilt
    by ``load_doc`` on an old file. One (n_offset, n_time) dataset per trace beats
    n_offset one-dimensional ones for both size and legibility, and ``tolist()``
    restores the nested lists exactly. Lists of NDARRAYS deliberately do NOT take
    this path: stacking them would round-trip each element back as a list, and a
    caller that reads ``.shape`` off one would break.
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
    """Store a scalar (or None) as an attribute of `grp`.

    Long strings spill into a dataset instead -- see :data:`_MAX_ATTR_STR`.
    """
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
    """Store an opaque blob (a rendered figure) as a tagged ``uint8`` dataset.

    Deliberately NOT gzipped: PNG and PDF are already compressed, so the filter
    would spend time to gain nothing.
    """
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
    """Write a nested document to `path` as HDF5.

    Parameters
    ----------
    path : str
        Output path, or a ``file.h5:/group`` address. The parent directory is
        created if needed.
    group : str, optional
        Write into this group instead of the root, APPENDING to an existing file
        and replacing only this group. That is how one file holds many runs. Note
        that HDF5 does not reclaim a replaced group's space, so a file rewritten
        over and over is worth deleting rather than overwriting.
    tree : dict
        The document. Keys must be strings without ``/``; values may be dicts,
        lists, ndarrays, scalars or None to any depth (see the module docstring).
    attrs : dict, optional
        Provenance written to the ROOT group alongside ``format`` -- e.g. the
        command line, the device, the timestamp. Kept out of `tree` so it can
        never be mistaken for data by a plotting function.

    Returns
    -------
    str
        The path written.
    """
    h5py = _h5py()
    path, addr_group = split_address(path)
    group = group or addr_group
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "a" if group else "w", track_order=True) as fh:
        if "format" not in fh.attrs:
            fh.attrs["format"] = SCHEMA
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
        out = list(ds.asstr()[...])
        return out
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

    `path` may be a ``file.h5:/group`` address (or `group` given separately), and
    then only that group is read -- pulling one run out of a sweep file does not
    pay for the rest of the file.

    Root provenance attributes (``format`` and whatever `attrs` carried) come
    back as ordinary top-level keys apart from ``format`` itself, which is
    dropped -- it describes the file, not the run.
    """
    h5py = _h5py()
    path, addr_group = split_address(path)
    group = group or addr_group
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
    """Embed a rendered figure into the run file it belongs to.

    The figure and the arrays it was drawn from are the same measurement, so
    keeping them in one file is what stops a run being paired with someone else's
    picture three months later. The ``--plot`` file on disk is still written; this
    is a copy that travels with the data.

    Parameters
    ----------
    target : str
        The run file, or a ``file.h5:/runs/eta1p8`` address -- the figure lands in
        THAT document's own ``figures`` group, so every run in a sweep file keeps
        its own. The file is appended to, never truncated.
    name : str
        What the figure shows (``rabi``, ``chirp_ridge``, ``post_chirp``), not a
        filename. Re-attaching under the same name replaces it -- though HDF5 does
        not reclaim the old bytes, so a file replotted many times is worth running
        through ``h5repack`` (or simply rewriting) to compact.
    image_path : str
        The rendered file to read. Any format; the suffix picks the recorded MIME
        type.

    Returns
    -------
    str
        The address of the embedded figure.
    """
    h5py = _h5py()
    file_path, addr = split_address(target)
    with open(image_path, "rb") as fh:
        data = fh.read()
    with h5py.File(file_path, "a", track_order=True) as fh:
        if "format" not in fh.attrs:
            fh.attrs["format"] = SCHEMA
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
    """Embed several figures at once; returns how many were stored.

    A figure that cannot be stored -- missing, unreadable, a file another process
    holds open -- is skipped rather than raised on: it is never worth failing a
    finished run over, which is the same rule the callers already apply to
    RENDERING one.
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
    """Write a document's embedded figures back out as files.

    Each keeps the filename it was rendered under, so a directory of extracted
    runs looks exactly like the ``figs/`` tree the tools write directly.

    Returns
    -------
    list of str
        The paths written, in file order.
    """
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
    """The path :func:`save_doc` would actually write for this name.

    A bare name (no suffix) gets ``.h5``; every other suffix is left alone. Split
    out so a CLI can tell the user where its output will land before running.
    """
    root, ext = os.path.splitext(path)
    return path if ext else root + ".h5"


def save_doc(path: str, doc: Dict[str, Any],
             attrs: Optional[Dict[str, Any]] = None, *,
             group: Optional[str] = None) -> str:
    """Write `doc` as HDF5, or as JSON if `path` ends in ``.json``.

    The suffix is the whole switch: HDF5 is the default for run outputs, and
    ``--out something.json`` remains available for a small document that wants
    to stay human-readable (or to feed a tool that has not been converted).

    With a `group` (or a ``file.h5:/group`` address) the write APPENDS into the
    file. JSON has no such thing, so asking for a group on a ``.json`` target is
    an error rather than a silently flattened file.

    Returns
    -------
    str
        The address actually written -- NOT necessarily `path`, since a bare name
        gains a ``.h5`` and a grouped write returns the full ``file:/group``.
    """
    file_path, addr_group = split_address(path)
    group = group or addr_group
    if os.path.splitext(file_path)[1].lower() == ".json":
        if group:
            raise ValueError(
                f"cannot write group {group!r} into the JSON file {file_path!r}: "
                f"only HDF5 holds several documents in one file")
        path = file_path
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(dict(doc, **(attrs or {})), fh, indent=2, default=_plain)
        return path
    return save_tree(resolve_out_path(file_path), doc, attrs=attrs, group=group)


def load_doc(path: str, *, group: Optional[str] = None) -> Dict[str, Any]:
    """Read a document written by :func:`save_doc`, HDF5 or JSON.

    Dispatches on the file's MAGIC NUMBER rather than its suffix, so a run
    renamed (or an old ``.json`` handed to a ``--replot`` that now defaults to
    HDF5) still loads. `path` may be a ``file.h5:/group`` address.
    """
    file_path, addr_group = split_address(path)
    group = group or addr_group
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
