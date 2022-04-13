"""
generic text-parsing functions designed for quickly yoinking metadata
from labels, particularly on networked and/or huge filesystems, without
sophisticated parsing. probably also useful for other things. sometimes
appropriate, sometimes not.
"""
from ast import literal_eval
from functools import cache
from io import BytesIO
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping, Optional, Union


def make_scraper(label_text: str) -> Callable:
    """
    makes a little function that scrapes a particular text for patterns
    """

    def scrape(pattern):
        result = re.search(pattern, label_text)
        if result is None:
            return None
        try:
            return literal_eval(result.group(1))
        except (ValueError, SyntaxError):
            return result.group(1)

    return scrape


def get_label_text(label: Union[Path, str], byte_size: int = 35592) -> str:
    """
    why not overload the label-getter? literally no reason
    """
    if isinstance(label, str):
        label = Path(label)
    with open(label, "rb") as file:
        return file.read(byte_size).decode()


# special-case block regex
SUBFRAME_GROUP = re.compile(r"SUBFRAME.*?END", re.DOTALL)


def scrape_subframe(label_text: str) -> tuple:
    """
    special case -- making a sequence from multiple pvl fields
    in a block. can generalize if necessary. output is:
    first line (row), first line sample (column),
    total lines, total line samples
    if image is not subframed, should be (1, 1, 1200, 1648)
    """
    return literal_eval(
        ",".join(
            re.findall(r"\d+", re.search(SUBFRAME_GROUP, label_text).group())
        )
    )


# cached versions for faster operation on networked filesystems.
cached_label_loader = cache(get_label_text)


@cache
def scrape_from_file(label, pattern):
    return make_scraper(cached_label_loader(label))(pattern)


# heuristic for max label size. we know it's not a real rule.
MAX_LABEL_SIZE = 500 * 1024


def head_file(fn_or_reader, nbytes):
    head_buffer = BytesIO()
    if not hasattr(fn_or_reader, "read"):
        fn_or_reader = open(fn_or_reader, 'rb')
    head_buffer.write(fn_or_reader.read(nbytes))
    fn_or_reader.close()
    head_buffer.seek(0)
    return head_buffer


KNOWN_LABEL_ENDINGS = (
    b'END\r\n',  # PVL
    b'\x00{3}',  # just null bytes
)


def trim_label(fn, max_size = MAX_LABEL_SIZE, raise_for_failure=False):
    head = head_file(fn, max_size).read()
    # TODO: add some logging or whatever i guess
    for ending in KNOWN_LABEL_ENDINGS:
        if (endmatch := re.search(ending, head)) is not None:
            return head[:endmatch.span()[1]].decode()
    if raise_for_failure:
        raise ValueError("couldn't find a label ending")
    return head.decode()


cached_label_trimmer = cache(trim_label)


def scrape_patterns(
    label: Union[Path, str],
    metadata_regex: Mapping,
    supplemental_search_function: Optional[Callable] = None,
    label_loader = cached_label_loader
) -> dict:
    """
    grabs all the patterns you specify from whatever.
    """
    label_text = label_loader(label)
    # little closure for neatness
    scrape = make_scraper(label_text)
    # general-case fields
    metadata = {
        field: scrape(pattern) for field, pattern in metadata_regex.items()
    }
    # optional insertion point for special cases that cannot be defined with
    # naive regex, default values, etc.
    if supplemental_search_function is not None:
        metadata |= supplemental_search_function(label, label_text)
    return metadata


def bulk_scrape_metadata(
    files: Iterable, pattern_scraper: Callable
) -> list[dict]:
    """
    scrapes metadata from all the files you pass it, and that's that
    """
    metaframe = []
    for file in files:
        metaframe.append(pattern_scraper(file))
    return metaframe


# cached filesystem functions for execution speed on networked filesystems
cached_ls = cache(os.listdir)
cached_exists = cache(os.path.exists)
