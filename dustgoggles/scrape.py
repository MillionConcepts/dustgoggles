"""
generic text-parsing functions designed for quickly yoinking metadata
from labels, particularly on networked and/or huge filesystems, without
sophisticated parsing. probably also useful for other things. sometimes
appropriate, sometimes not.
"""
from ast import literal_eval
from functools import cache
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


def scrape_patterns(
    label: Union[Path, str],
    metadata_regex: Mapping,
    supplemental_search_function: Optional[Callable] = None,
) -> dict:
    """
    grabs all the patterns you specify from whatever.
    """
    label_text = cached_label_loader(label)
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
