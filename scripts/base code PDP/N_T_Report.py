"""
N_T_Report.py
FUNCTIONALITY
    Creates PDF reports with visualizations for PDP analysis
EXPLANATION
    Generates multi-page PDF reports containing static visualizations, 
    inequality matrices, heat maps, cluster maps, hierarchical clustering,
    MDS visualizations, and Top-K analyses
INPUT
    Generated PNG images from other visualization modules
OUTPUT
    PDF report file (report_moving_objects_PDP_*.pdf)
"""
# TODO: 
# - Add page numbers
# - Make layout more customizable
# - Finetune visualization sizes based on content

from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import Flowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
import av
import os
import time

# Start time
t_start = time.time()

# Define custom page size
width_cm = 29.7
height_cm = 21
custom_page_size = (width_cm * cm, height_cm * cm)

# Define styles for the document
styles = getSampleStyleSheet()
title_style = styles['Title']
body_style = styles['Normal']

# Create filename
if av.PDPg_fundamental_active == 1:
    filename = os.path.join(os.getcwd(), "report_moving_objects_PDP_fundamental.pdf")
elif av.PDPg_buffer_active == 1:
    filename = os.path.join(os.getcwd(), "report_moving_objects_PDP_buffer.pdf")
elif av.PDPg_rough_active == 1:
    filename = os.path.join(os.getcwd(), "report_moving_objects._PDP_rough.pdf")
elif av.PDPg_bufferrough_active == 1:
    filename = os.path.join(os.getcwd(), "report_moving_objects_PDP_bufferrough.pdf")
else:
    print("Variable a does not hold an appropriate value.")
    filename = os.path.join(os.getcwd(), "report_moving_objects_PDP.pdf")

doc = SimpleDocTemplate(filename, pagesize=custom_page_size, rightMargin=0.5*cm, leftMargin=0.5*cm, topMargin=0.5*cm, bottomMargin=0.5*cm)

# Create the story for the document
story = []

if av.N_VA_DynamicAbsolute == 1:
    # PAGE WITH DYNAMIC VISUALISATION (ABSOLUTE)
    title = Paragraph("Dynamic visualizations", title_style)
    story.append(title)
    subtitle_text = "Dynamic visualizations: see N_Moving_Objects_Results or Powerpoint (still to do) or after running code"
    subtitle_style = ParagraphStyle(name='Subtitle', parent=title_style, fontname="Arial", fontsize=10)
    subtitle = Paragraph(subtitle_text, subtitle_style)
    story.append(subtitle)
    story.append(PageBreak())

if av.N_VA_StaticAbsolute == 1:
    # PAGE WITH STATIC VISUALISATIONS (ABSOLUTE)
    title = Paragraph("Static Visualizations (absolute)", title_style)
    story.append(title)
    subtitle_text = "Absolute visualizations of the static data"
    subtitle_style = ParagraphStyle(name='Subtitle', parent=title_style, fontname="Arial", fontsize=10)
    subtitle = Paragraph(subtitle_text, subtitle_style)
    story.append(subtitle)
    file_paths = [os.path.join(os.getcwd(), 'N_C_Csa' + str(i) + '.png') for i in range(av.con)]
    images = [Image(fp, width=240, height=240) for fp in file_paths]
    image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
    page_width = custom_page_size[0]
    margin = 0.5 * cm
    table_width = page_width - 2 * margin
    col_width = table_width / 3
    table = Table(image_rows, colWidths=[col_width]*3)
    story.append(table)
    story.append(PageBreak())

if av.N_VA_StaticRelative == 1:
    # PAGE WITH STATIC VISUALISATIONS (RELATIVE)
    title = Paragraph("Static Visualizations (relative)", title_style)
    story.append(title)
    subtitle_text = "Relative visualizations of the static data"
    subtitle_style = ParagraphStyle(name='Subtitle', parent=title_style, fontname="Arial", fontsize=10)
    subtitle = Paragraph(subtitle_text, subtitle_style)
    story.append(subtitle)
    file_paths = [os.path.join(os.getcwd(), 'N_C_Csr' + str(i) + '.png') for i in range(av.con)]
    images = [Image(fp, width=240, height=240) for fp in file_paths]
    image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
    page_width = custom_page_size[0]
    margin = 0.5 * cm
    table_width = page_width - 2 * margin
    col_width = table_width / 3
    table = Table(image_rows, colWidths=[col_width]*3)
    story.append(table)
    story.append(PageBreak())

if av.N_VA_StaticFinetuned == 1:
    # PAGE WITH STATIC VISUALIZATIONS (FINETUNED)
    title = Paragraph("Static visualizations (finetuned)", title_style)
    story.append(title)
    subtitle_text = "Finetuned visualizations of the static data"
    subtitle_style = ParagraphStyle(name='Subtitle', parent=title_style, fontname="Arial", fontsize=10)
    subtitle = Paragraph(subtitle_text, subtitle_style)
    story.append(subtitle)
    file_paths = [os.path.join(os.getcwd(), 'N_C_Csf' + str(i) + '.png') for i in range(av.con)]
    images = [Image(fp, width=105, height=270) for fp in file_paths]
    image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
    page_width = custom_page_size[0]
    margin = 0.5 * cm
    table_width = page_width - 2 * margin
    col_width = table_width / 3
    table = Table(image_rows, colWidths=[col_width]*3)
    story.append(table)
    story.append(PageBreak())

if av.N_VA_InequalityMatrices == 1: 
    # PAGE WITH INEQUALITY MATRICES
    if av.PDPg_fundamental_active == 1:
        title = Paragraph("Inequality Matrices (fundamental)", title_style)
        story.append(title)
        file_paths = []
        for i in range(av.con):
            for w in range(av.tst-(av.window_length_tst-1)):
                for d in range(av.DD):
                    file_path = os.path.join(os.getcwd(), 'N_C_PDPg_fundamental_InequalityMatrix' + '_c' + str(i) + '_t' + str(w) + '_d' + str(d) + '.png')        
                    file_paths.append(file_path)
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        image_rows = [images[i:i+av.DD] for i in range(0, len(images), av.DD)]
        page_width = custom_page_size[0]
        margin = 0.5 * cm
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        table = Table(image_rows, colWidths=[col_width]*3)
        story.append(table)
        story.append(PageBreak())
    elif av.PDPg_buffer_active == 1:
        title = Paragraph("Inequality Matrices (buffer)", title_style)
        story.append(title)
        file_paths = []
        for i in range(av.con):
            for d in range(av.DD):
                file_path = os.path.join(os.getcwd(), 'N_C_PDPg_buffer_InequalityMatrix' + '_c' + str(i) + '_t0' + '_d' + str(d) + '.png')  
                file_paths.append(file_path)
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        image_rows = [images[i:i+av.DD] for i in range(0, len(images), av.DD)]
        page_width = custom_page_size[0]
        margin = 0.5 * cm
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        table = Table(image_rows, colWidths=[col_width]*3)
        story.append(table)
        story.append(PageBreak())
    elif av.PDPg_rough_active == 1:
        title = Paragraph("Inequality Matrices (rough)", title_style)
        story.append(title)
        file_paths = []
        for i in range(av.con):
            for d in range(av.DD):
                file_path = os.path.join(os.getcwd(), 'N_C_PDPg_rough_InequalityMatrix' + '_c' + str(i) + '_t0' + '_d' + str(d) + '.png')        
                file_paths.append(file_path)
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        image_rows = [images[i:i+av.DD] for i in range(0, len(images), av.DD)]
        page_width = custom_page_size[0]
        margin = 0.5 * cm
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        table = Table(image_rows, colWidths=[col_width]*3)
        story.append(table)
        story.append(PageBreak())
    elif av.PDPg_bufferrough_active == 1:
        title = Paragraph("Inequality Matrices (bufferrough)", title_style)
        story.append(title)
        file_paths = []
        for i in range(av.con):
            for d in range(av.DD):
                file_path = os.path.join(os.getcwd(), 'N_C_PDPg_bufferrough_InequalityMatrix' + '_c' + str(i) + '_t0' + '_d' + str(d) + '.png')        
                file_paths.append(file_path)
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        image_rows = [images[i:i+av.DD] for i in range(0, len(images), av.DD)]
        page_width = custom_page_size[0]
        margin = 0.5 * cm
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        table = Table(image_rows, colWidths=[col_width]*3)
        story.append(table)
        story.append(PageBreak())

if av.N_VA_HeatMap == 1:  
    # PAGE WITH HEAT MAP
    if av.PDPg_fundamental_active == 1:
        title = Paragraph("Heat Map (fundamental)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_fundamental_HeatMap.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_buffer_active == 1:
        title = Paragraph("Heat Map (buffer)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_buffer_HeatMap.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_rough_active == 1:
        title = Paragraph("Heat Map (rough)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_rough_HeatMap.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_bufferrough_active == 1:
        title = Paragraph("Heat Map (bufferrough)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_bufferrough_HeatMap.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())

if av.N_VA_ClusterMap == 1: 
    # PAGE WITH CLUSTER MAP
    if av.PDPg_fundamental_active == 1:
        title = Paragraph("Cluster Map (fundamental)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_fundamental_ClusterMap.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_buffer_active == 1:
        title = Paragraph("Cluster Map (buffer)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_buffer_ClusterMap.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_rough_active == 1:
        title = Paragraph("Cluster Map (rough)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_rough_ClusterMap.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_bufferrough_active == 1:
        title = Paragraph("Cluster Map (bufferrough)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_bufferrough_ClusterMap.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())

if av.N_VA_HClust == 1:  
    # PAGE WITH HIERARCHICAL CLUSTER TREE
    if av.PDPg_fundamental_active == 1:
        title = Paragraph("Hierarchical Clustering (fundamental)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_fundamental_HClust.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_buffer_active == 1:
        title = Paragraph("Hierarchical Clustering (buffer)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_buffer_HClust.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_rough_active == 1:
        title = Paragraph("Hierarchical Clustering (rough)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_rough_HClust.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_bufferrough_active == 1:
        title = Paragraph("Hierarchical Clustering (bufferrough)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_bufferrough_HClust.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())

if av.N_VA_Mds == 1: 
    # PAGE WITH MDS
    if av.PDPg_fundamental_active == 1:
        title = Paragraph("Dimensionality Reduction (MDS) (fundamental)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_fundamental_Mds.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_buffer_active == 1:
        title = Paragraph("Dimensionality Reduction (MDS) (buffer)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_buffer_Mds.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_rough_active == 1:
        title = Paragraph("Dimensionality Reduction (MDS) (rough)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_rough_Mds.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())
    elif av.PDPg_bufferrough_active == 1:
        title = Paragraph("Dimensionality Reduction (MDS) (bufferrough)", title_style)
        story.append(title)
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_bufferrough_Mds.png")
        pil_image = PILImage.open(file_path)
        image = Image(file_path, width=400, height=400)
        story.append(image)
        story.append(PageBreak())

if av.N_VA_TopK == 1: 
    # PAGE WITH TOPK VISUALISATIONS
    if av.PDPg_fundamental_active == 1:
        title = Paragraph("Top-K (fundamental)", title_style)
        story.append(title)
        file_paths = [os.path.join(os.getcwd(), 'N_C_PDPg_fundamental_TopK_c' + str(i) + '.png') for i in range(av.con)]
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
        page_width = custom_page_size[0]
        margin = 0.5 * cm
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        table = Table(image_rows, colWidths=[col_width]*3)
        story.append(table)
        story.append(PageBreak())
    elif av.PDPg_buffer_active == 1:
        title = Paragraph("Top-K (buffer)", title_style)
        story.append(title)
        file_paths = [os.path.join(os.getcwd(), 'N_C_PDPg_buffer_TopK_c' + str(i) + '.png') for i in range(av.con)]
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
        page_width = custom_page_size[0]
        margin = 0.5 * cm
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        table = Table(image_rows, colWidths=[col_width]*3)
        story.append(table)
        story.append(PageBreak())
    elif av.PDPg_rough_active == 1:
        title = Paragraph("Top-K (rough)", title_style)
        story.append(title)
        file_paths = [os.path.join(os.getcwd(), 'N_C_PDPg_rough_TopK_c' + str(i) + '.png') for i in range(av.con)]
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
        page_width = custom_page_size[0]
        margin = 0.5 * cm
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        table = Table(image_rows, colWidths=[col_width]*3)
        story.append(table)
        story.append(PageBreak())
    elif av.PDPg_bufferrough_active == 1:
        title = Paragraph("Top-K (bufferrough)", title_style)
        story.append(title)
        file_paths = [os.path.join(os.getcwd(), 'N_C_PDPg_bufferrough_TopK_c' + str(i) + '.png') for i in range(av.con)]
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
        page_width = custom_page_size[0]
        margin = 0.5 * cm
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        table = Table(image_rows, colWidths=[col_width]*3)
        story.append(table)
        story.append(PageBreak())

# Build the PDF document
doc.build(story)

# End and print time
print('Time elapsed for running module "N_T_Report": {:.3f} sec.'.format(time.time() - t_start))
