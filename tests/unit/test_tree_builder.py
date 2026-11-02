from hybrid_rag.ingestion.tree_builder import TreeBuilder

URI = "file:///guide.md"


def test_first_h1_before_content_becomes_title() -> None:
    b = TreeBuilder(fallback_title="guide")
    b.heading("Guide", 1)
    b.block("paragraph", "Intro.")
    b.heading("Setup", 2)
    b.block("paragraph", "Run it.")
    tree = b.build(source_uri=URI)

    assert tree.title == "Guide"
    assert [blk.text for blk in tree.root.blocks] == ["Intro."]
    assert [s.heading for s in tree.root.children] == ["Setup"]


def test_fallback_title_when_content_precedes_heading() -> None:
    b = TreeBuilder(fallback_title="guide")
    b.block("paragraph", "Preamble.")
    b.heading("Real Heading", 1)
    tree = b.build(source_uri=URI)

    assert tree.title == "guide"
    assert [s.heading for s in tree.root.children] == ["Real Heading"]


def test_headings_nest_by_level_and_pop_on_shallower() -> None:
    b = TreeBuilder(fallback_title="t")
    b.block("paragraph", "x")
    for text, level in [("A", 2), ("A1", 3), ("B", 2)]:
        b.heading(text, level)
        b.block("paragraph", f"{text} body")
    tree = b.build(source_uri=URI)

    a, bb = tree.root.children
    assert (a.heading, bb.heading) == ("A", "B")
    assert [c.heading for c in a.children] == ["A1"]
    assert bb.children == ()


def test_skipped_levels_still_nest_under_nearest_ancestor() -> None:
    b = TreeBuilder(fallback_title="t")
    b.block("paragraph", "x")
    b.heading("A", 2)
    b.heading("Deep", 5)
    b.block("paragraph", "deep body")
    tree = b.build(source_uri=URI)

    assert tree.root.children[0].children[0].heading == "Deep"


def test_blank_blocks_and_headings_are_ignored() -> None:
    b = TreeBuilder(fallback_title="t")
    b.block("paragraph", "   \n ")
    b.heading("  ", 2)
    b.block("paragraph", "kept")
    tree = b.build(source_uri=URI)

    assert [blk.text for blk in tree.root.blocks] == ["kept"]
    assert tree.root.children == ()


def test_iter_blocks_yields_section_paths_in_document_order() -> None:
    b = TreeBuilder(fallback_title="t")
    b.block("paragraph", "root")
    b.heading("A", 2)
    b.block("paragraph", "a")
    b.heading("A1", 3)
    b.block("code", "a1", page=3)
    tree = b.build(source_uri=URI)

    assert [(p, blk.text) for p, blk in tree.iter_blocks()] == [
        ((), "root"),
        (("A",), "a"),
        (("A", "A1"), "a1"),
    ]
    assert next(blk for _, blk in tree.iter_blocks() if blk.kind == "code").page == 3
