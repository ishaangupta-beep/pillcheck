"""DynamoDB + S3 store for the deployed service.

Table layout (single table, on-demand capacity):
  pk = collection, sk = id                    -> get by id
  GSI "by_patient": gpk = "<collection>#<patient_id>", gsk = created_at
                                               -> one patient's docs, in time order
Photos, evidence images and reference features live in S3 with SSE.
"""
from __future__ import annotations

import json
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

from .base import Store, utcnow_iso


def _to_ddb(obj):
    return json.loads(json.dumps(obj), parse_float=Decimal)


def _from_ddb(obj):
    if isinstance(obj, list):
        return [_from_ddb(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _from_ddb(v) for k, v in obj.items()}
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    return obj


class AwsStore(Store):
    def __init__(self, table: str, bucket: str, region: str) -> None:
        self._table = boto3.resource("dynamodb", region_name=region).Table(table)
        self._s3 = boto3.client("s3", region_name=region)
        self._bucket = bucket

    def put(self, collection: str, doc: dict) -> dict:
        doc.setdefault("created_at", utcnow_iso())
        item = {"pk": collection, "sk": doc["id"], "body": _to_ddb(doc), "gsk": doc["created_at"]}
        if doc.get("patient_id"):
            item["gpk"] = f"{collection}#{doc['patient_id']}"
        self._table.put_item(Item=item)
        return doc

    def get(self, collection: str, doc_id: str) -> dict | None:
        item = self._table.get_item(Key={"pk": collection, "sk": doc_id}).get("Item")
        return _from_ddb(item["body"]) if item else None

    def list(self, collection: str, patient_id: str | None = None) -> list[dict]:
        items, kwargs = [], {}
        while True:
            if patient_id is None:
                resp = self._table.query(KeyConditionExpression=Key("pk").eq(collection), **kwargs)
            else:
                resp = self._table.query(IndexName="by_patient",
                                         KeyConditionExpression=Key("gpk").eq(f"{collection}#{patient_id}"), **kwargs)
            items += resp.get("Items", [])
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        docs = [_from_ddb(i["body"]) for i in items]
        return sorted(docs, key=lambda d: d.get("created_at", ""))

    def incr(self, key: str, ttl_seconds: int) -> int:
        import time
        resp = self._table.update_item(
            Key={"pk": "counters", "sk": key},
            UpdateExpression="ADD #v :one SET expires_at = if_not_exists(expires_at, :exp)",
            ExpressionAttributeNames={"#v": "value"},
            ExpressionAttributeValues={":one": 1, ":exp": int(time.time()) + ttl_seconds},
            ReturnValues="UPDATED_NEW")
        return int(resp["Attributes"]["value"])

    def put_blob(self, key: str, data: bytes, content_type: str) -> str:
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type,
                            ServerSideEncryption="AES256")
        return key

    def get_blob(self, key: str) -> bytes | None:
        try:
            return self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None


def create_table(table: str, region: str) -> None:
    """Create the DynamoDB table and index (used by the deploy script and tests)."""
    ddb = boto3.client("dynamodb", region_name=region)
    ddb.create_table(
        TableName=table,
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[{"AttributeName": n, "AttributeType": "S"} for n in ("pk", "sk", "gpk", "gsk")],
        KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
        GlobalSecondaryIndexes=[{
            "IndexName": "by_patient",
            "KeySchema": [{"AttributeName": "gpk", "KeyType": "HASH"}, {"AttributeName": "gsk", "KeyType": "RANGE"}],
            "Projection": {"ProjectionType": "ALL"},
        }],
    )
    ddb.get_waiter("table_exists").wait(TableName=table)
    ddb.update_time_to_live(TableName=table, TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"})
