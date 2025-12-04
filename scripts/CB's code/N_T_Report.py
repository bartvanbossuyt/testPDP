#alles wat ik al kan meerdere keren erop
#visualisaties duidelijker maken: grootte...
#andere drie configuraties op een volgende pagina toevoegen, ... toevoegen tot zoveel configuraties er zijn.
#*kan ik zo een template maken met kolommen en grootte van figuren en dat dan bij de verschillende gebruiken?!

#not yet perfect, but ok so far... finetune lay-out based on other pages...
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

# Create the "dir" subdirectory if it doesn't exist
# subdir = os.path.join(os.getcwd(), "dir")
# os.makedirs(av.dir, exist_ok=True)
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

doc = SimpleDocTemplate(filename, pagesize=custom_page_size, rightMargin=0.5*cm, leftMargin=0.5*cm, topMargin=0.5*cm, bottomMargin=0.5*cm)
# Create the story for the document
story = []

if av.N_VA_DynamicAbsolute == 1:
    # PAGE WITH DYNAMIC VISUALISATION (ABSOLUTE)
    # Add the title
    title = Paragraph("Dynamic visualizations", title_style)
    story.append(title)
    # Add a subtitle
    subtitle_text = "Dynamic visualizations: see N_Moving_Objects_Results or Powerpoint (still to do) or after running code"
    subtitle_style = ParagraphStyle(name='Subtitle', parent=title_style,fontname="monospace", fontsize=10)
    subtitle = Paragraph(subtitle_text, subtitle_style)
    story.append(subtitle)
    #Open the image using PIL
    #file_path = os.path.join(av.dir, "N_C_Static.png")
    #pil_image = PILImage.open(file_path)
    # Add the image
    #image = Image(file_path, width=400, height=400)
    #story.append(image)
    # Add a page break to start a new page
    story.append(PageBreak())

if av.N_VA_StaticAbsolute == 1:
    # PAGE WITH STATIC VISUALISATIONS(ABSOLUTE)
    # Create a title paragraph using the predefined style 'title_style'
    title = Paragraph("Static Visualizations (absolute)", title_style)
    # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
    story.append(title)
    # Add a subtitle
    subtitle_text = "Absolute visualizations of the static data"
    subtitle_style = ParagraphStyle(name='Subtitle', parent=title_style,fontname="monospace", fontsize=10)
    subtitle = Paragraph(subtitle_text, subtitle_style)
    story.append(subtitle)
    file_paths = [os.path.join(os.getcwd(), 'N_C_Csa' + str(i) + '.png') for i in range(av.con)]
    # Create a list of Image objects from the file paths
    #images = [Image(fp, width=250, height=166) for fp in file_paths]
    images = [Image(fp, width=240, height=240) for fp in file_paths]
    # Split the images list into sublists of 3 images each
    image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
    # Calculate the width of each column
    page_width = custom_page_size[0]
    margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
    table_width = page_width - 2 * margin
    col_width = table_width / 3
    # Create a table with rows using the list of image rows
    table = Table(image_rows)
    # Specify the width of each column
    table = Table(image_rows, colWidths=[col_width]*3)
    # Append the table to the 'story' list
    story.append(table)
    # Add a page break to start a new page
    story.append(PageBreak())

if av.N_VA_StaticRelative == 1:
    # PAGE WITH STATIC VISUALISATIONS(RELATIVE)
    # Create a title paragraph using the predefined style 'title_style'
    title = Paragraph("Static Visualizations (relative)", title_style)
    # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
    story.append(title)
    # Add a subtitle
    subtitle_text = "Relative visualizations of the static data"
    subtitle_style = ParagraphStyle(name='Subtitle', parent=title_style,fontname="monospace", fontsize=10)
    subtitle = Paragraph(subtitle_text, subtitle_style)
    story.append(subtitle)
    # Create a list of file paths for the images
    #file_paths = [os.path.join(av.dir, 'N_C_Csr' + str(i) + '.png') for i in range(av.con)] 
    file_paths = [os.path.join(os.getcwd(), 'N_C_Csr' + str(i) + '.png') for i in range(av.con)]
    # Create a list of Image objects from the file paths
    #images = [Image(fp, width=250, height=166) for fp in file_paths]
    images = [Image(fp, width=240, height=240) for fp in file_paths]
    # Split the images list into sublists of 3 images each
    image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
    # Calculate the width of each column
    page_width = custom_page_size[0]
    margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
    table_width = page_width - 2 * margin
    col_width = table_width / 3
    # Create a table with rows using the list of image rows
    table = Table(image_rows)
    # Specify the width of each column
    table = Table(image_rows, colWidths=[col_width]*3)
    # Append the table to the 'story' list
    story.append(table)
    # Add a page break to start a new page
    story.append(PageBreak())

if av.N_VA_StaticFinetuned == 1:
    # PAGE WITH STATIC VISUALIZATIONS(FINETUNED)
    # Create a title paragraph using the predefined style 'title_style'
    title = Paragraph("Static visualizations (finetuned)", title_style)
    # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
    story.append(title)
    # Add a subtitle
    subtitle_text = "Finetuned visualizations of the static data"
    subtitle_style = ParagraphStyle(name='Subtitle', parent=title_style,fontname="monospace", fontsize=10)
    subtitle = Paragraph(subtitle_text, subtitle_style)
    story.append(subtitle)
    # Create a list of file paths for the images
    #file_paths = [os.path.join(av.dir, 'N_C_Csr' + str(i) + '.png') for i in range(av.con)] 
    file_paths = [os.path.join(os.getcwd(), 'N_C_Csf' + str(i) + '.png') for i in range(av.con)]
    # Create a list of Image objects from the file paths
    #images = [Image(fp, width=250, height=166) for fp in file_paths]
    images = [Image(fp, width=105, height=270) for fp in file_paths]
    # Split the images list into sublists of 3 images each
    image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
    # Calculate the width of each column
    page_width = custom_page_size[0]
    margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
    table_width = page_width - 2 * margin
    col_width = table_width / 3
    # Create a table with rows using the list of image rows
    table = Table(image_rows)
    # Specify the width of each column
    table = Table(image_rows, colWidths=[col_width]*3)
    # Append the table to the 'story' list
    story.append(table)
    # Add a page break to start a new page
    story.append(PageBreak())

if av.N_VA_InequalityMatrices == 1: 
    # PAGE WITH INEQUALITY MATRICES PDPg_FUNDAMENTAL
    if av.PDPg_fundamental_active == 1:
        # Create a title paragraph using the predefined style 'title_style'
        title = Paragraph("Inequality Matrices (fundamental)", title_style)
        # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
        story.append(title)
        # Create an empty list to store the file paths
        file_paths = []
        # Iterate over the range of av.con and av.DD
        for i in range(av.con):
            #for w in range(av.window_length_tst + 1):
            for w in range(av.tst-(av.window_length_tst-1)):
                for d in range(av.DD):
                    # Create the file path using the working directory
                    file_path = os.path.join(os.getcwd(), 'N_C_PDPg_fundamental_InequalityMatrix' + '_c' + str(i) + '_t' + str(w) + '_d' + str(d)  +  '.png')        
                    # Append the file path to the list
                    file_paths.append(file_path)
        # Create a list of Image objects from the file paths
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        # Split the images list into sublists of 3 images each
        image_rows = [images[i:i+av.DD] for i in range(0, len(images), av.DD)]
        # Calculate the width of each column
        page_width = custom_page_size[0]
        margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        # Create a table with rows using the list of image rows
        table = Table(image_rows, colWidths=[col_width]*3)
        # Append the table to the 'story' list
        story.append(table)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH INEQUALITY MATRICES PDPg_BUFFER
    elif av.PDPg_buffer_active == 1:
        # Create a title paragraph using the predefined style 'title_style'
        title = Paragraph("Inequality Matrices (buffer)", title_style)
        # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
        story.append(title)
        # Create a list of file paths for the images
        # file_paths = []
        # for i in range(av.con):
        #     for d in range(av.DD):
        #         file_path = os.path.join(av.dir, 'N_C_Inequality_matrix' + '_c' + str(i) + '_d' + str(d+1) + '.png')
        #         file_paths.append(file_path)
        # Create an empty list to store the file paths
        file_paths = []
        # Iterate over the range of av.con and av.DD
        for i in range(av.con):
            for d in range(av.DD):
                # Create the file path using the working directory
                #file_path = os.path.join(os.getcwd(), 'N_C_Inequality_matrix' + '_c' + str(i) + '_d' + str(d+1) + "N_C_Dataset_g_buffer" + '.png')
                file_path = os.path.join(os.getcwd(), 'N_C_PDPg_buffer_InequalityMatrix' + '_c' + str(i) + '_t0' + '_d' + str(d)  +  '.png')  
                
                # Append the file path to the list
                file_paths.append(file_path)
        # Create a list of Image objects from the file paths
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        # Split the images list into sublists of 3 images each
        image_rows = [images[i:i+av.DD] for i in range(0, len(images), av.DD)]
        # Calculate the width of each column
        page_width = custom_page_size[0]
        margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        # Create a table with rows using the list of image rows
        table = Table(image_rows, colWidths=[col_width]*3)
        # Append the table to the 'story' list
        story.append(table)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH INEQUALITY MATRICES PDPg_ROUGH
    elif av.PDPg_rough_active == 1:
        # Create a title paragraph using the predefined style 'title_style'
        title = Paragraph("Inequality Matrices (rough)", title_style)
        # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
        story.append(title)
        # Create an empty list to store the file paths
        file_paths = []
        # Iterate over the range of av.con and av.DD
        for i in range(av.con):
            for d in range(av.DD):
                # Create the file path using the working directory
                file_path = os.path.join(os.getcwd(), 'N_C_PDPg_rough_InequalityMatrix' + '_c' + str(i) + '_t0' + '_d' + str(d)  +  '.png')        
                # Append the file path to the list
                file_paths.append(file_path)
        # Create a list of Image objects from the file paths
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        # Split the images list into sublists of 3 images each
        image_rows = [images[i:i+av.DD] for i in range(0, len(images), av.DD)]
        # Calculate the width of each column
        page_width = custom_page_size[0]
        margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        # Create a table with rows using the list of image rows
        table = Table(image_rows, colWidths=[col_width]*3)
        # Append the table to the 'story' list
        story.append(table)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH INEQUALITY MATRICES PDPg_BUFFERROUGH
    elif av.PDPg_bufferrough_active == 1:
        # Create a title paragraph using the predefined style 'title_style'
        title = Paragraph("Inequality Matrices (bufferrough)", title_style)
        # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
        story.append(title)
        # Create an empty list to store the file paths
        file_paths = []
        # Iterate over the range of av.con and av.DD
        for i in range(av.con):
            for d in range(av.DD):
                # Create the file path using the working directory
                file_path = os.path.join(os.getcwd(), 'N_C_PDPg_bufferrough_InequalityMatrix' + '_c' + str(i) + '_t0' + '_d' + str(d)  +  '.png')        
                # Append the file path to the list
                file_paths.append(file_path)
        # Create a list of Image objects from the file paths
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        # Split the images list into sublists of 3 images each
        image_rows = [images[i:i+av.DD] for i in range(0, len(images), av.DD)]
        # Calculate the width of each column
        page_width = custom_page_size[0]
        margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        # Create a table with rows using the list of image rows
        table = Table(image_rows, colWidths=[col_width]*3)
        # Append the table to the 'story' list
        story.append(table)
        # Add a page break to start a new page
        story.append(PageBreak())

if av.N_VA_HeatMap == 1:  
    # PAGE WITH HEAT MAP PDPg_FUNDAMENTAL
    if av.PDPg_fundamental_active == 1:
        # Add the title
        title = Paragraph("Heat Map (fundamental)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_fundamental_HeatMap.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH HEAT MAP PDPg_BUFFER
    elif av.PDPg_buffer_active == 1:
        # Add the title
        title = Paragraph("Heat Map (buffer)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_buffer_HeatMap.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH HEAT MAP PDPg_ROUGH
    elif av.PDPg_rough_active == 1:
        # Add the title
        title = Paragraph("Heat Map (rough)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_rough_HeatMap.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH HEAT MAP PDPg_BUFFERROUGH
    elif av.PDPg_rough_active == 1:
        # Add the title
        title = Paragraph("Heat Map (bufferrough)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_bufferrough_HeatMap.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())

if av.N_VA_ClusterMap == 1: 
# PAGE WITH CLUSTER MAP PDPG_fundamental
    if av.PDPg_fundamental_active == 1:
        # Add the title
        title = Paragraph("Cluster Map (fundamental)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_fundamental_ClusterMap.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH CLUSTER MAP PDPG_buffer
    elif av.PDPg_buffer_active == 1:
        # Add the title
        title = Paragraph("Cluster Map (buffer)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_ClusterMapN_C_Dataset_g_buffer.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH CLUSTER MAP PDPG_rough
    elif av.PDPg_rough_active == 1:
        # Add the title
        title = Paragraph("Cluster Map (rough: " + str(av.rough) + "m)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPPg_rough_ClusterMap.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())

if av.N_VA_HClust == 1:  
    # PAGE WITH HIERARCHICAL CLUSTER TREE PDPg_FUNDAMENTAL
    if av.PDPg_fundamental_active == 1:
        # Add the title
        title = Paragraph("Hierarchical Clustering (fundamental)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_fundamental_HClust.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH HIERARCHICAL CLUSTER TREE PDPg_BUFFER
    elif av.PDPg_buffer_active == 1:
        # Add the title
        title = Paragraph("Hierarchical Clustering (buffer)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_buffer_HClust.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH HIERARCHICAL CLUSTER TREE PDPg_ROUGH
    elif av.PDPg_rough_active == 1:
        # Add the title
        title = Paragraph("Hierarchical Clustering (rough)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_rough_HClust.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH HIERARCHICAL CLUSTER TREE PDPg_bufferrough
    elif av.PDPg_bufferrough_active == 1:
        # Add the title
        title = Paragraph("Hierarchical Clustering (bufferrough)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_bufferrough_HClust.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())

if av.N_VA_Mds == 1: 
    # PAGE WITH MDS PDPg_FUNDAMENTAL
    if av.PDPg_fundamental_active == 1:
        # Add the title
        title = Paragraph("Dimensionality Reduction (MDS) (fundamental)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_fundamental_Mds.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH MDS PDPg_BUFFER
    elif av.PDPg_buffer_active == 1:
        # Add the title
        title = Paragraph("Dimensionality Reduction (MDS) (buffer)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_buffer_Mds.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH MDS PDPg_ROUGH
    elif av.PDPg_rough_active == 1:
        # Add the title
        title = Paragraph("Dimensionality Reduction (MDS) (rough)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_rough_Mds.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH MDS PDPg_BUFFERROUGH
    elif av.PDPg_bufferrough_active == 1:
        # Add the title
        title = Paragraph("Dimensionality Reduction (MDS) (bufferrough)", title_style)
        story.append(title)
        # Create the file path using the working directory
        file_path = os.path.join(os.getcwd(), "N_C_PDPg_bufferrough_Mds.png")
        # Open the image using PIL
        pil_image = PILImage.open(file_path)
        # Add the image
        image = Image(file_path, width=400, height=400)
        story.append(image)
        # Add a page break to start a new page
        story.append(PageBreak())

if av.N_VA_TopK == 1: 
    # PAGE WITH TOPK VISUALISATIONS PDPg_FUNDAMENTAL
    if av.PDPg_fundamental_active == 1:
        # Create a title paragraph using the predefined style 'title_style'
        title = Paragraph("Top-K (fundamental)", title_style)
        # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
        story.append(title)
        # Create a list of file paths for the images in the working directory
        file_paths = [os.path.join(os.getcwd(), 'N_C_PDPg_fundamental_TopK_c' + str(i) + '.png') for i in range(av.con)]
        # Create a list of Image objects from the file paths
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        # Split the images list into sublists of 5 images each
        image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
        # Calculate the width of each column
        page_width = custom_page_size[0]
        margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        # Create a table with rows using the list of image rows
        table = Table(image_rows, colWidths=[col_width]*3)
        # Append the table to the 'story' list
        story.append(table)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH TOPK VISUALISATIONS PDPg_BUFFER
    elif av.PDPg_buffer_active == 1:
        # Create a title paragraph using the predefined style 'title_style'
        title = Paragraph("Top-K (buffer)", title_style)
        # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
        story.append(title)
        # Create a list of file paths for the images in the working directory
        file_paths = [os.path.join(os.getcwd(), 'N_C_PDPg_buffer_TopK_c' + str(i) + '.png') for i in range(av.con)]
        # Create a list of Image objects from the file paths
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        # Split the images list into sublists of 5 images each
        image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
        # Calculate the width of each column
        page_width = custom_page_size[0]
        margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        # Create a table with rows using the list of image rows
        table = Table(image_rows, colWidths=[col_width]*3)
        # Append the table to the 'story' list
        story.append(table)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH TOPK VISUALISATIONS PDPg_ROUGH
    elif av.PDPg_rough_active == 1:
        # Create a title paragraph using the predefined style 'title_style'
        title = Paragraph("Top-K (rough)", title_style)
        # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
        story.append(title)
        # Create a list of file paths for the images in the working directory
        file_paths = [os.path.join(os.getcwd(), 'N_C_PDPg_rough_TopK_c' + str(i) + '.png') for i in range(av.con)]
        # Create a list of Image objects from the file paths
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        # Split the images list into sublists of 5 images each
        image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
        # Calculate the width of each column
        page_width = custom_page_size[0]
        margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        # Create a table with rows using the list of image rows
        table = Table(image_rows, colWidths=[col_width]*3)
        # Append the table to the 'story' list
        story.append(table)
        # Add a page break to start a new page
        story.append(PageBreak())
    # PAGE WITH TOPK VISUALISATIONS PDPg_BUFFERROUGH
    elif av.PDPg_bufferrough_active == 1:
        # Create a title paragraph using the predefined style 'title_style'
        title = Paragraph("Top-K (bufferrough)", title_style)
        # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
        story.append(title)
        # Create a list of file paths for the images in the working directory
        file_paths = [os.path.join(os.getcwd(), 'N_C_PDPg_bufferrough_TopK_c' + str(i) + '.png') for i in range(av.con)]
        # Create a list of Image objects from the file paths
        images = [Image(fp, width=250, height=166) for fp in file_paths]
        # Split the images list into sublists of 5 images each
        image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
        # Calculate the width of each column
        page_width = custom_page_size[0]
        margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
        table_width = page_width - 2 * margin
        col_width = table_width / 3
        # Create a table with rows using the list of image rows
        table = Table(image_rows, colWidths=[col_width]*3)
        # Append the table to the 'story' list
        story.append(table)
        # Add a page break to start a new page
        story.append(PageBreak())

"""
# PAGE WITH TOPK VISUALISATIONS PDPG_buffer
if av.PDPg_buffer == 1:
    # Create a title paragraph using the predefined style 'title_style'
    title = Paragraph("Top-K (buffer)", title_style)
    # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
    story.append(title)
    # Create a list of file paths for the images in the working directory
    file_paths = [os.path.join(os.getcwd(), 'N_C_TopKN_C_Dataset_g_buffer' + str(i+1) + '.png') for i in range(av.con)]
    # file_paths = [os.path.join(os.getcwd(), 'N_C_TopK' + str(i+1) + '.png') for i in range(av.con)]
    # Create a list of Image objects from the file paths
    images = [Image(fp, width=250, height=166) for fp in file_paths]
    # Split the images list into sublists of 5 images each
    image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
    # Calculate the width of each column
    page_width = custom_page_size[0]
    margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
    table_width = page_width - 2 * margin
    col_width = table_width / 3
    # Create a table with rows using the list of image rows
    table = Table(image_rows, colWidths=[col_width]*3)
    # Append the table to the 'story' list
    story.append(table)
    # Add a page break to start a new page
    story.append(PageBreak())
# PAGE WITH TOPK VISUALISATIONS PDPG_rough
if av.PDPg_rough == 1:
    # Create a title paragraph using the predefined style 'title_style'
    title = Paragraph("Top-K (rough): " + str(av.rough) + "m)", title_style)
    # Append the title to the 'story' list. This list will contain all the elements to be added to the final document
    story.append(title)
    # Create a list of file paths for the images in the working directory
    file_paths = [os.path.join(os.getcwd(), 'N_C_TopKN_C_Dataset_g_rough' + str(i+1) + '.png') for i in range(av.con)]
    # Create a list of Image objects from the file paths
    images = [Image(fp, width=250, height=166) for fp in file_paths]
    # Split the images list into sublists of 5 images each
    image_rows = [images[i:i+3] for i in range(0, len(images), 3)]
    # Calculate the width of each column
    page_width = custom_page_size[0]
    margin = 0.5 * cm  # total margin is 1 cm (0.5 cm on each side)
    table_width = page_width - 2 * margin
    col_width = table_width / 3
    # Create a table with rows using the list of image rows
    table = Table(image_rows, colWidths=[col_width]*3)
    # Append the table to the 'story' list
    story.append(table)
    # Add a page break to start a new page
    story.append(PageBreak())
"""

# Build the PDF document
doc.build(story)

# End and print time
print('Time elapsed for running module "N_T_Report": {:.3f} sec.'.format(time.time() - t_start))
