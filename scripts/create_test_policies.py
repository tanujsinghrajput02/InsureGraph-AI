
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "documents"


POLICIES = [
    {
        "filename": "sample_policy.pdf",
        "policy_number": "POL-1001",
        "customer_name": "Rahul Sharma",
        "insurer": "ABC Insurance",
        "annual_premium": "INR 15000",
        "coverage_amount": "INR 500000"
    },
    {
        "filename": "sample_policy_2.pdf",
        "policy_number": "POL-2002",
        "customer_name": "Priya Verma",
        "insurer": "SecureLife Insurance",
        "annual_premium": "INR 22000",
        "coverage_amount": "INR 800000"
    },
    {
        "filename": "sample_policy_3.pdf",
        "policy_number": "POL-3003",
        "customer_name": "Amit Patel",
        "insurer": "National Shield Insurance",
        "annual_premium": "INR 18000",
        "coverage_amount": "INR 650000"
    }
]


def create_policy_pdf(policy):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / policy["filename"]

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PolicyTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#17365D"),
        fontSize=19,
        alignment=1
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#17365D")
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15
    )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=22 * mm,
        leftMargin=22 * mm,
        topMargin=22 * mm,
        bottomMargin=22 * mm
    )

    rows = [
        ["Policy Number", policy["policy_number"]],
        ["Customer Name", policy["customer_name"]],
        ["Insurer", policy["insurer"]],
        ["Annual Premium", policy["annual_premium"]],
        ["Coverage Amount", policy["coverage_amount"]]
    ]

    table = Table(rows, colWidths=[55 * mm, 95 * mm])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9)
    ]))

    story = [
        Paragraph("SAMPLE INSURANCE POLICY", title_style),
        Spacer(1, 15),
        Paragraph("Policy Information", section_style),
        table,
        Spacer(1, 18),
        Paragraph("Policy Summary", section_style),
        Paragraph(
            f"This synthetic policy belongs to "
            f"{policy['customer_name']}. "
            f"The policy was issued by {policy['insurer']}. "
            f"The annual premium is {policy['annual_premium']} "
            f"and the coverage amount is {policy['coverage_amount']}.",
            body_style
        ),
        Spacer(1, 15),
        Paragraph(
            "Synthetic demonstration data only. "
            "Not a real insurance contract.",
            body_style
        )
    ]

    document.build(story)

    print(f"Created: {output_path.name}")


def main():
    for policy in POLICIES:
        create_policy_pdf(policy)

    print("\nAll synthetic policy documents created.")


if __name__ == "__main__":
    main()
