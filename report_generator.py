import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

def generate_pdf_report(df, chart_fig, summary_text):
    """
    Generates a PDF report with a summary, a data table, and a chart.

    Args:
        df (pd.DataFrame): The dataframe to include in the report.
        chart_fig (matplotlib.figure.Figure): The chart figure to include.
        summary_text (str): The summary text to include.

    Returns:
        str: The file path of the generated PDF.
    """
    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=(11, 8.5))
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("AI Data Analyst Report", styles['h1'])
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))

    # Section 1: Summary
    summary_header = Paragraph("Summary", styles['h2'])
    story.append(summary_header)
    summary = Paragraph(summary_text.replace("\n", "<br/>"), styles['BodyText'])
    story.append(summary)
    story.append(Spacer(1, 0.2 * inch))

    # Section 2: Data Table
    if not df.empty:
        table_header = Paragraph("Data Sample (First 10 Rows)", styles['h2'])
        story.append(table_header)
        df_sample = df.head(10)
        data = [df_sample.columns.to_list()] + df_sample.values.tolist()
        
        table = Table(data)
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        table.setStyle(style)
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

    # Section 3: Chart
    if chart_fig:
        chart_header = Paragraph("Chart", styles['h2'])
        story.append(chart_header)
        
        img_buffer = io.BytesIO()
        chart_fig.savefig(img_buffer, format='PNG', dpi=300)
        img_buffer.seek(0)
        
        chart_image = Image(img_buffer, width=6*inch, height=4*inch)
        story.append(chart_image)

    doc.build(story)
    return file_path
