#!/usr/bin/env python3
"""
Simple Research Paper to PDF Converter
Converts markdown research papers to PDF using basic HTML conversion
"""

import os
import sys
import markdown
from pathlib import Path

def convert_markdown_to_html_pdf(markdown_file_path: str, output_pdf_path: str = None) -> str:
    """
    Convert a markdown file to a styled HTML file that can be printed to PDF
    """
    try:
        # Read the markdown file
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Generate output path if not provided
        if output_pdf_path is None:
            base_path = Path(markdown_file_path).stem
            output_dir = Path(markdown_file_path).parent
            output_pdf_path = output_dir / f"{base_path}_printable.html"
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_content,
            extensions=['tables', 'fenced_code', 'toc', 'codehilite']
        )
        
        # Add comprehensive CSS styling for PDF printing
        css_style = """
        <style>
        @media print {
            body { margin: 0.5in; }
            .no-print { display: none; }
            h1, h2, h3 { page-break-after: avoid; }
            pre, blockquote { page-break-inside: avoid; }
        }
        
        body {
            font-family: 'Times New Roman', 'Georgia', serif;
            font-size: 12pt;
            line-height: 1.6;
            margin: 1in;
            color: #333;
            max-width: 8.5in;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        
        h1 { 
            font-size: 24pt; 
            border-bottom: 2px solid #3498db; 
            padding-bottom: 0.3em;
            text-align: center;
        }
        
        h2 { 
            font-size: 18pt; 
            border-bottom: 1px solid #bdc3c7; 
            padding-bottom: 0.2em;
        }
        
        h3 { font-size: 14pt; }
        h4 { font-size: 12pt; }
        
        p {
            margin-bottom: 1em;
            text-align: justify;
        }
        
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 10pt;
        }
        
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            overflow-x: auto;
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 10pt;
            line-height: 1.4;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            margin-left: 0;
            padding-left: 20px;
            font-style: italic;
            color: #555;
            background-color: #f9f9f9;
            padding: 10px 20px;
            margin: 1em 0;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 11pt;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        
        ul, ol {
            margin: 1em 0;
            padding-left: 2em;
        }
        
        li {
            margin-bottom: 0.5em;
        }
        
        strong, b {
            font-weight: bold;
            color: #2c3e50;
        }
        
        em, i {
            font-style: italic;
        }
        
        .header-info {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 2em;
            font-size: 11pt;
        }
        
        .print-instructions {
            background-color: #e8f5e8;
            border: 1px solid #4caf50;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 2em;
        }
        
        @media print {
            .print-instructions { display: none; }
        }
        </style>
        """
        
        # Create print instructions
        print_instructions = """
        <div class="print-instructions no-print">
            <h3>📄 How to Save as PDF:</h3>
            <ol>
                <li><strong>Press Ctrl+P</strong> (or Cmd+P on Mac) to open print dialog</li>
                <li><strong>Select "Save as PDF"</strong> as the destination</li>
                <li><strong>Choose "More settings"</strong> and select:
                    <ul>
                        <li>Paper size: A4 or Letter</li>
                        <li>Margins: Default or Custom (0.5 inch)</li>
                        <li>Options: ✓ Background graphics</li>
                    </ul>
                </li>
                <li><strong>Click "Save"</strong> and choose your desired location</li>
            </ol>
            <p><em>This instruction box will not appear in the printed PDF.</em></p>
        </div>
        """
        
        # Combine everything
        full_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Research Paper - PDF Ready</title>
            {css_style}
        </head>
        <body>
            {print_instructions}
            {html_content}
        </body>
        </html>
        """
        
        # Save the HTML file
        with open(output_pdf_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"✅ HTML file created: {output_pdf_path}")
        print(f"📄 Open this file in your browser and press Ctrl+P to save as PDF")
        
        return str(output_pdf_path)
    
    except Exception as e:
        print(f"❌ Error converting markdown to HTML: {e}")
        return None

def main():
    """Main function for standalone PDF conversion"""
    
    # Default file path
    default_file = r"C:\Users\nayak\Documents\Agent_Fly\research_outputs\research_conversation_20250831_103320.md"
    
    print("📄 Research Paper to PDF Converter")
    print("=" * 50)
    
    # Ask for file path
    file_path = input(f"📁 Enter markdown file path\n(default: {default_file}): ").strip()
    
    if not file_path:
        file_path = default_file
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n🔄 Converting to printable HTML...")
    print(f"📄 Input: {file_path}")
    
    # Convert to HTML
    result_path = convert_markdown_to_html_pdf(file_path)
    
    if result_path:
        print(f"\n🎉 Conversion successful!")
        print(f"📄 HTML file: {result_path}")
        print(f"\n📋 Next steps:")
        print(f"   1. Open the HTML file in your web browser")
        print(f"   2. Press Ctrl+P (or Cmd+P on Mac)")
        print(f"   3. Select 'Save as PDF' as destination")
        print(f"   4. Choose your preferred settings and save")
        
        # Ask if user wants to open the file
        open_file = input(f"\n🌐 Open HTML file in browser now? (y/n): ").strip().lower()
        if open_file == 'y':
            try:
                os.startfile(result_path)  # Windows
            except:
                try:
                    os.system(f'open "{result_path}"')  # macOS
                except:
                    os.system(f'xdg-open "{result_path}"')  # Linux
    else:
        print(f"\n❌ Conversion failed!")

if __name__ == "__main__":
    main()
