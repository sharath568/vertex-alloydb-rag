from fastapi.testclient import TestClient


def test_ask_endpoint(monkeypatch, settings, embedder, indexed):
    from dataclasses import replace

    from rag import api
    from rag.answer import ModelAnswer

    def gen(sys, prompt):
        return ModelAnswer(answer="Send traffic to the previous revision [1].", citations=[1], grounded=True)

    monkeypatch.setattr(api, "_deps", lambda: (replace(settings, min_similarity=0.0), embedder, gen))
    client = TestClient(api.app)
    assert client.get("/healthz").json() == {"ok": True}
    r = client.post("/ask", json={"question": "How do I roll back?"}, headers={"X-Tenant-Id": "larkspur"})
    body = r.json()
    assert r.status_code == 200 and body["grounded"] is True
    assert body["sources"][0]["source_uri"].startswith("docs/larkspur/")
