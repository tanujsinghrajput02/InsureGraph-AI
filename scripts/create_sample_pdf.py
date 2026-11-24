
import fitz
from pathlib import Path

output_path = Path("data/documents/sample_policy.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)

pdf = fitz.open()
page = pdf.new_page()

policy_text = """
ABC INSURANCE COMPANY
SAMPLE HEALTH INSURANCE POLICY

Policy Number: POL-1001
Customer Name: Rahul Sharma

Policy Details:
Insurer: ABC Insurance
Annual Premium: INR 15000
Coverage Amount: INR 500000

The policy provides health insurance coverage
to the registered customer Rahul Sharma.

Policy Number POL-1001 is issued by ABC Insurance.
The annual premium is INR 15000.
The total coverage amount is INR 500000.

This is synthetic data created for the InsureGraph AI project.
"""

page.insert_textbox(
    fitz.Rect(50, 50, 550, 750),
    policy_text,
    fontsize=12
)

pdf.save(output_path)
pdf.close()

print(f"Sample PDF created: {output_path}")
