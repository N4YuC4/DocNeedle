import json

import pytest
from httpx import ASGITransport, AsyncClient

from docneedle.extraction.verifier import ConfidenceCalculator
from docneedle.main import app


def test_confidence_calculator() -> None:
    # High confidence: found item
    conf_high = ConfidenceCalculator.calculate(
        found=True,
        value="31 Mart 2027",
        retrieval_score=0.03,
    )
    assert conf_high >= 0.8

    # Not found
    assert (
        ConfidenceCalculator.calculate(
            found=False,
            value=None,
            retrieval_score=0.0,
        )
        == 0.0
    )


@pytest.mark.asyncio
async def test_extract_endpoint_e2e() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sample_doc = (
            b"# Kira Sozlesmesi\n\n"
            b"Kiraci: Ahmet Yilmaz\n"
            b"Kira Bedeli: 25.000 TL\n"
            b"Baslangic Tarihi: 15 Haziran 2026\n"
        )
        schema = [
            {"key": "kiraci", "description": "Kiracı kimdir?", "type": "string"},
            {
                "key": "kira_bedeli",
                "description": "Aylık kira bedeli ne kadardır?",
                "type": "money",
            },
            {"key": "bilinmeyen", "description": "Depozito tutarı ne kadar?", "type": "money"},
        ]

        files = {
            "files": ("sozlesme.md", sample_doc, "text/markdown"),
        }
        data = {
            "schema": json.dumps(schema),
        }

        response = await client.post("/v1/extract", files=files, data=data)
        assert response.status_code == 200

        res_data = response.json()
        assert res_data["status"] == "completed"
        assert res_data["total_items"] == 3
        assert res_data["found_items"] >= 2

        results_by_key = {r["key"]: r for r in res_data["results"]}
        assert "kiraci" in results_by_key
        assert results_by_key["kiraci"]["found"] is True
        assert results_by_key["kiraci"]["confidence"] > 0.5
        assert results_by_key["kiraci"]["evidence"] is not None

        assert results_by_key["kira_bedeli"]["found"] is True
        assert "25.000" in str(results_by_key["kira_bedeli"]["value"])

        # bilinmeyen was not in document
        assert results_by_key["bilinmeyen"]["found"] is False


@pytest.mark.asyncio
async def test_official_document_identity_and_location_extraction() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        adli_sicil_doc = (
            b"ADLI SICIL VE ISTATISTIK GENEL MUDURLUGU\n"
            b"SORGULAMAYA ESAS KIMLIK BILGILERI\n"
            b"KIMLIK NUMARASI : 22334455667\n"
            b"AD, SOYAD : TOLGA, ARSLAN\n"
            b"ANNE ADI / BABA ADI : FATMA / HASAN\n"
            b"DOGUM YERI / TARIHI : ESKISEHIR / 15.05.1992\n"
            b"YUKARIDA KIMLIK BILGILERI BULUNAN KISININ ADLI SICIL KAYDI YOKTUR.\n"
        )
        ikametgah_doc = (
            b"YERLESIM YERI VE DIGER ADRES BELGESI\n"
            b"KIMLIK BILGILERI\n"
            b"ADI SOYADI : TOLGA ARSLAN\n"
            b"YERLESIM YERI BILGILERI\n"
            b"YERLESIM YERI ADRESI : Gaziosmanpasa Mah. Gunex Sk. No:8 Cankaya / ANKARA\n"
        )
        schema = [
            {"key": "madde_1", "description": "isim", "type": "string"},
            {"key": "madde_2", "description": "soyisim", "type": "string"},
            {"key": "madde_3", "description": "konum", "type": "string"},
            {"key": "madde_4", "description": "pasaport no", "type": "string"},
        ]

        files = [
            ("files", ("adli_sicil.txt", adli_sicil_doc, "text/plain")),
            ("files", ("ikametgah.txt", ikametgah_doc, "text/plain")),
        ]
        data = {
            "schema": json.dumps(schema),
        }

        response = await client.post("/v1/extract", files=files, data=data)
        assert response.status_code == 200

        res_data = response.json()
        assert res_data["status"] == "completed"
        results_by_key = {r["key"]: r for r in res_data["results"]}

        assert results_by_key["madde_1"]["found"] is True
        assert "TOLGA" in results_by_key["madde_1"]["value"]

        assert results_by_key["madde_2"]["found"] is True
        assert "ARSLAN" in results_by_key["madde_2"]["value"]

        assert results_by_key["madde_3"]["found"] is True
        assert "Gaziosmanpasa" in results_by_key["madde_3"]["value"]

        assert results_by_key["madde_4"]["found"] is False
