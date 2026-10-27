import io
import json

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageDraw

from docneedle.main import app


def _create_synthetic_invoice_image(
    invoice_no: str = "INV-77441",
    total: str = "12.500 USD",
    vendor: str = "ACME ENTERPRISE CORP",
) -> bytes:
    """Generate in-memory RGB PNG bytes of an invoice document."""
    img = Image.new("RGB", (900, 500), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), f"FATURA NO: {invoice_no}", fill="black")
    draw.text((50, 100), f"TOPLAM TUTAR: {total}", fill="black")
    draw.text((50, 150), f"TEDARIKCI: {vendor}", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_scanned_pdf(
    invoice_no: str = "INV-88552",
    total: str = "45.000 TL",
    vendor: str = "GLOBAL LOGISTICS LTD",
) -> bytes:
    """Generate in-memory PDF bytes containing a rasterized image page with zero digital text."""
    img = Image.new("RGB", (900, 500), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), f"FATURA NUMARASI: {invoice_no}", fill="black")
    draw.text((50, 100), f"ODEME TUTARI: {total}", fill="black")
    draw.text((50, 150), f"SATICI: {vendor}", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PDF")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_multimodal_png_image_extraction_e2e() -> None:
    """Verify that a standalone image document (.png) is embedded and extracted via LLM."""
    png_bytes = _create_synthetic_invoice_image(
        invoice_no="INV-77441",
        total="12.500 USD",
        vendor="ACME ENTERPRISE CORP",
    )
    schema = [
        {"key": "fatura_no", "description": "Fatura numarası nedir?", "type": "string"},
        {"key": "toplam_tutar", "description": "Toplam tutar ne kadardır?", "type": "money"},
        {"key": "olmayan_alan", "description": "Banka IBAN numarası nedir?", "type": "string"},
    ]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = [("files", ("invoice.png", png_bytes, "image/png"))]
        data = {"schema": json.dumps(schema)}
        response = await client.post("/v1/extract", files=files, data=data)
        assert response.status_code == 200

        res_data = response.json()
        assert res_data["status"] == "completed"
        assert res_data["total_items"] == 3

        results_by_key = {r["key"]: r for r in res_data["results"]}
        assert "fatura_no" in results_by_key
        assert results_by_key["fatura_no"]["found"] is True
        assert "77441" in results_by_key["fatura_no"]["value"]

        assert "toplam_tutar" in results_by_key
        assert results_by_key["toplam_tutar"]["found"] is True
        tot_val = str(results_by_key["toplam_tutar"]["value"])
        assert "12.500" in tot_val or "12500" in tot_val

        assert "olmayan_alan" in results_by_key
        assert results_by_key["olmayan_alan"]["found"] is False


@pytest.mark.asyncio
async def test_multimodal_scanned_pdf_extraction_e2e() -> None:
    """Verify that a scanned PDF without digital text is ingested as visual chunks and extracted."""
    pdf_bytes = _create_scanned_pdf(
        invoice_no="INV-88552",
        total="45.000 TL",
        vendor="GLOBAL LOGISTICS LTD",
    )
    schema = [
        {"key": "fatura_numarasi", "description": "Fatura numarası nedir?", "type": "string"},
        {"key": "odeme_tutari", "description": "Ödeme tutarı nedir?", "type": "money"},
    ]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = [("files", ("scanned.pdf", pdf_bytes, "application/pdf"))]
        data = {"schema": json.dumps(schema)}
        response = await client.post("/v1/extract", files=files, data=data)
        assert response.status_code == 200

        res_data = response.json()
        assert res_data["status"] == "completed"
        assert res_data["total_items"] == 2

        results_by_key = {r["key"]: r for r in res_data["results"]}
        assert "fatura_numarasi" in results_by_key
        assert results_by_key["fatura_numarasi"]["found"] is True
        assert "88552" in results_by_key["fatura_numarasi"]["value"]

        assert "odeme_tutari" in results_by_key
        assert results_by_key["odeme_tutari"]["found"] is True
        pay_val = str(results_by_key["odeme_tutari"]["value"])
        assert "45.000" in pay_val or "45000" in pay_val
