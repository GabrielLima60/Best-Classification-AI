from PIL import Image
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def cut_image(input_image_path, part_height):
    # Open the image using PIL
    img = Image.open(input_image_path)
    width, height = img.size
    
    num_parts = height // part_height 
    remainder = height % part_height  
    
    parts = []
    for i in range(num_parts):
        top = i * part_height
        bottom = (i + 1) * part_height
        
        part = img.crop((0, top, width, bottom))
        parts.append(part)
    
    if remainder > 0:  
        top = num_parts * part_height
        bottom = height
        part = img.crop((0, top, width, bottom))
        parts.append(part)
    
    return parts

def save_parts_to_pdf(parts, output_pdf_path):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)

    for idx, part in enumerate(parts):
        part_width, part_height = part.size 
        
        part_path = f"temp_part_{idx}.png"
        part.save(part_path)  
        
        c.drawImage(part_path, 0, 0, width=part_width / 2, height=part_height / 2)
        c.showPage()  
        
        os.remove(part_path)
    
    c.save()

def csv_to_xlsx(csv_file_path, output_path):

    df = pd.read_csv(csv_file_path)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    grouped = df.groupby(df.select_dtypes(exclude=[np.number]).columns.tolist()).agg({
        col: ["mean", "std"] for col in numeric_cols
    })

    grouped.columns = ['_'.join(col).strip() for col in grouped.columns]

    for col in numeric_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        grouped[f"{col} ±"] = grouped[mean_col].round(2).astype(str) + " ± " + grouped[std_col].round(2).astype(str)
        grouped.drop(columns=[mean_col, std_col], inplace=True)

    grouped.reset_index(inplace=True)

    df.to_excel(output_path, index=False, engine='openpyxl')
    grouped.to_excel("results table//resumed results.xlsx", index=False, engine='openpyxl')

def main():
    input_image_path = 'results image//graphs.png'
    output_pdf_path = 'results image//graphs.pdf'  
    part_height = 1510 
    
    parts = cut_image(input_image_path, part_height)
    save_parts_to_pdf(parts, output_pdf_path)

    input_csv = "results table//results.csv"
    output_excel_csv = "results table//results.xlsx"
    csv_to_xlsx(input_csv, output_excel_csv)

if __name__ == '__main__':
    main()