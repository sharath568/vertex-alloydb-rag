from rag.chunk import chunk_markdown


def test_never_splits_across_headings():
    md = "# A\n\none two\n\n## B\n\nthree four\n\n# C\n\nfive"
    chunks = chunk_markdown(md)
    assert [c.heading for c in chunks] == ["A", "A > B", "C"]
    assert chunks[1].content == "three four"


def test_long_sections_are_windowed_with_overlap():
    para = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_markdown(f"# Big\n\n{para}", max_tokens=200, overlap_tokens=40)
    assert len(chunks) > 1
    first_tail = chunks[0].content.split()[-10:]
    assert all(w in chunks[1].content.split() for w in first_tail)
    assert all(len(c.content.split()) <= 150 for c in chunks)
