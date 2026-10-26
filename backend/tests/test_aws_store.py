"""The DynamoDB + S3 store, against moto's AWS emulation."""
import pytest

moto = pytest.importorskip("moto")


def test_aws_store_roundtrip(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    with moto.mock_aws():
        import boto3

        from pillcheck.storage.aws import AwsStore, create_table
        create_table("pillcheck-test", "ap-south-1")
        boto3.client("s3", region_name="ap-south-1").create_bucket(
            Bucket="pc-test", CreateBucketConfiguration={"LocationConstraint": "ap-south-1"})
        store = AwsStore("pillcheck-test", "pc-test", "ap-south-1")

        store.put("checks", {"id": "c2", "patient_id": "p1", "created_at": "2026-10-21T04:00:00+00:00", "score": 2.5})
        store.put("checks", {"id": "c1", "patient_id": "p1", "created_at": "2026-10-21T03:00:00+00:00", "nested": {"x": [1, 2.25]}})
        store.put("checks", {"id": "c3", "patient_id": "p2", "created_at": "2026-10-21T05:00:00+00:00"})
        assert store.get("checks", "c1")["nested"] == {"x": [1, 2.25]}
        assert [d["id"] for d in store.list("checks", "p1")] == ["c1", "c2"]
        assert {d["id"] for d in store.list("checks")} == {"c1", "c2", "c3"}
        store.update("checks", "c2", status="ok")
        assert store.get("checks", "c2")["status"] == "ok" and store.get("checks", "nope") is None

        store.put_blob("checks/c1/photo.jpg", b"\xff\xd8jpeg", "image/jpeg")
        assert store.get_blob("checks/c1/photo.jpg") == b"\xff\xd8jpeg"
        assert store.get_blob("missing") is None
