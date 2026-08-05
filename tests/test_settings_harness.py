"""Checks on the settings tests' own harness, that run everywhere.

test_settings_gtk.py drives the window in a child interpreter, and that
child is a source string in the test file. So there are two interpreters
reading it: the outer one, which resolves the escapes in the string, and the
child, which then has to be handed valid Python. A "\\n" written as one
newline ends the outer string early, and what reaches the child is a broken
literal - which fails every test in that file at once, including the ones
that were passing before.

Every one of those tests is skipped where GTK is missing, so on a machine
without it the whole file goes green while the child would not even parse.
This file is deliberately not skipped: compiling that source needs no GTK, no
display and no window, and it is the check that would have caught it.
"""

from tests.test_settings_gtk import _CHILD


def test_the_child_interpreter_is_handed_valid_python():
    compile(_CHILD, "<test_settings_gtk._CHILD>", "exec")


def test_every_scenario_the_tests_ask_for_exists_in_the_child():
    """A typo in a scenario name is a test that proves nothing.

    The child runs a chain of `elif scenario ==` branches and falls off the
    end without complaint, printing OK for a scenario it never ran.
    """
    import re
    from pathlib import Path

    source = Path(__file__).with_name("test_settings_gtk.py").read_text(
        encoding="utf-8")
    handled = set(re.findall(r'scenario == "([a-z_]+)"', _CHILD))
    # _run(tmp_path, "name") is how every test names the one it wants.
    asked = set(re.findall(r'_run\(tmp_path, "([a-z_]+)"\)', source))
    assert asked, "no scenarios were found to check"
    assert asked <= handled, f"no branch in the child for {asked - handled}"
