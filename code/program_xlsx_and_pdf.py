from PIL import Image
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# Function to cut the PNG image into horizontal parts based on the height of each part
def cut_image(input_image_path, part_height):
    # Open the image using PIL
    img = Image.open(input_image_path)
    width, height = img.size
    
    num_parts = height // part_height  # Calculate number of parts
    remainder = height % part_height  # Check if there is any remainder for the last part
    
    parts = []
    for i in range(num_parts):
        top = i * part_height
        bottom = (i + 1) * part_height
        
        # Slice the image horizontally
        part = img.crop((0, top, width, bottom))
        parts.append(part)
    
    if remainder > 0:  # If there's leftover height, add the final part
        top = num_parts * part_height
        bottom = height
        part = img.crop((0, top, width, bottom))
        parts.append(part)
    
    return parts

# Function to save the sliced parts into a PDF
def save_parts_to_pdf(parts, output_pdf_path):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)

    for idx, part in enumerate(parts):
        part_width, part_height = part.size 
        
        # Save each part as a unique temporary image file
        part_path = f"temp_part_{idx}.png"
        part.save(part_path)  # Save part as a temporary image file
        
        # Place the image on the PDF page
        c.drawImage(part_path, 0, 0, width=part_width / 2, height=part_height / 2)
        c.showPage()  # Move to the next page after drawing the image
        
        # Delete the temporary part image after it's added to the PDF
        os.remove(part_path)
    
    c.save()

def csv_to_xlsx(csv_file_path, output_path):

    df = pd.read_csv(csv_file_path)

    df.to_excel(output_path, index=False, engine='openpyxl')

def main():
    input_image_path = 'results image//graphs.png'
    output_pdf_path = 'results image//graphs.pdf'  
    part_height = 1510 
    
    parts = cut_image(input_image_path, part_height)
    save_parts_to_pdf(parts, output_pdf_path)
    #os.remove('temp_part.png')

    input_csv = "results table//results.csv"
    output_excel_csv = "results table//results.xlsx"
    csv_to_xlsx(input_csv, output_excel_csv)

if __name__ == '__main__':
    main()