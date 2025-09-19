import docx
import argparse


class DocxReader:
    """Parser for DOCX files that extracts sections, paragraphs and tables.

    Usage:
        reader = DocxReader(path)
        sections = reader.parse()
    """

    def __init__(self, docx_path: str):
        self.docx_path = docx_path

    def table_to_dict_list(self, table):
        """Convert a docx table to a list of dictionaries."""
        rows = list(table.rows)
        if not rows:
            return []

        headers = [cell.text.strip() for cell in rows[0].cells]
        data = []
        for row in rows[1:]:
            values = [cell.text.strip() for cell in row.cells]
            data.append(dict(zip(headers, values)))
        return data

    def parse(self):
        """Parse the DOCX and return a list of section dicts."""
        doc = docx.Document(self.docx_path)

        sections_data = []
        current_section = None

        for block in doc.element.body:
            if block.tag.endswith("p"):
                para = docx.text.paragraph.Paragraph(block, doc)
                text = para.text.strip()
                if not text:
                    continue
                style_name = para.style.name if para.style else "Unknown"

                # Consider Title and Heading styles as section starts. Also be a bit
                # lenient (e.g., localized style names that contain 'heading').
                is_heading = False
                lname = style_name.lower() if style_name else ""
                if lname.startswith("heading") or lname.startswith("title") or "heading" in lname:
                    is_heading = True

                if is_heading:
                    if current_section:
                        sections_data.append(current_section)
                    current_section = {
                        "title": text,
                        "style": style_name,
                        "paragraphs": [],
                        "tables": []
                    }
                else:
                    # Only attach paragraphs to an existing section.
                    if current_section is not None:
                        current_section["paragraphs"].append({
                            "text": text,
                            "style": style_name,
                            "bold": any(run.bold for run in para.runs),
                            "italic": any(run.italic for run in para.runs),
                            "font_size": para.runs[0].font.size.pt if para.runs and para.runs[0].font.size else None
                        })

            elif block.tag.endswith("tbl"):
                table = docx.table.Table(block, doc)
                if current_section:
                    current_section["tables"].append(self.table_to_dict_list(table))

        if current_section:
            sections_data.append(current_section)

        return sections_data


def _print_sections(sections):
    for sec in sections:
        print(f"\n=== Section: {sec['title']} ({sec['style']}) ===")

        if sec["paragraphs"]:
            print("\nParagraphs:")
            for para in sec["paragraphs"]:
                print(f"- {para['text']} (Style={para['style']}, Bold={para['bold']}, Italic={para['italic']}, FontSize={para['font_size']})")

        if sec["tables"]:
            print("\nTables:")
            for table in sec["tables"]:
                for row in table:
                    print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a DOCX file.")
    parser.add_argument("--docx-path", type=str, required=True, help="Path to the DOCX file.")
    args = parser.parse_args()

    reader = DocxReader(args.docx_path)
    sections = reader.parse()
    _print_sections(sections)
