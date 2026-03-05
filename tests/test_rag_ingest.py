from ai_tutor.rag.ingest import ReferenceDoc, ingest_reference_corpus


def test_returns_list():
    docs = ingest_reference_corpus()
    assert isinstance(docs, list)


def test_non_empty():
    docs = ingest_reference_corpus()
    assert len(docs) > 0


def test_all_are_reference_docs():
    for doc in ingest_reference_corpus():
        assert isinstance(doc, ReferenceDoc)


def test_docs_have_required_fields():
    for doc in ingest_reference_corpus():
        assert doc.id
        assert doc.title
        assert doc.content


def test_ids_are_unique():
    docs = ingest_reference_corpus()
    ids = [doc.id for doc in docs]
    assert len(ids) == len(set(ids))
