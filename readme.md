# dustgoggles 
*a marslab project*

[![DOI](https://zenodo.org/badge/393797006.svg)](https://zenodo.org/badge/latestdoi/393797006)

## Description

`dustgoggles` is a multi-featured utility library originally designed to
support other libraries in the *marslab* ecosystem. It contains a variety of
handy tools for parsing text, manipulating data structures, warping language,
and helping programs remember what they are doing. Highlights include:
* `composition`, a flexible, antiformal domain-specific functional language 
  framework (or, equivalently, a pipeline assembly system)
* `codex`, a set of in-memory storage objects to facilitate 
efficient interprocess communication in idiomatic Python   
* a family of "dig" functions (in `dustgoggles.structures`) that support 
  easy operations on horribly nested mappings (like a `dict` created from
  some horrible JSON returned by someone's very-helpful API)

*note: `dustgoggles` is not a discrete user-facing application. if you're 
looking for something like a 
[planetary data reader](www.github.com/millionconcepts/pdr) or an 
[image-processing system](www.github.com/millionconcepts/marslab), you've 
found one of its dependencies.*

## Installation

We recommend installing `dustgoggles` via `conda` from `conda-forge`:

`conda install -c conda-forge dustgoggles`

`dustgoggles` is also on PyPi and can be installed with `pip`:

`pip install dustgoggles`

Installing it from source is also straightforward:

`git clone git@github.com:MillionConcepts/dustgoggles.git; cd dustgoggles; pip install -e .`

The minimum supported version of Python is **3.9**.

## Feedback

Feedback is welcomed and encouraged. Because `dustgoggles` is a dependency of tactical software for 
the Mars Science Laboratory and Mars 2020 missions, it is possible that the content of your feedback might be 
covered by the MSL or M2020 Team Guidelines. In that case, please send an email to mstclair@millionconcepts.com. 
Otherwise, please file an Issue on this repository.

## Tests

Clone the repository, make sure `pytest` is installed in your environment, and run `pytest -vv` from the root 
directory of the cloned repository. 

Note that test coverage for `dustgoggles.composition` is currently provided only by downstream test suites; 
the most easily-accessible suite is in [this repository](www.github.com/millionconcepts/marslab). Integrated 
tests are planned.

----
The contents of this library are provided by Million Concepts (C. Million, M. St. Clair) 
under a BSD 3-Clause License. This license places very few restrictions on what you can 
do with this code. Questions related to copyright can be sent to chase@millionconcepts.com.
